"""Dynamic partial optimal transport (DPOT) pseudo-labels with lower bounds on the class marginals.

Core functions of the method of the manuscript "Dynamic Partial Optimal Transport for Cross-Scene Hyperspectral Image
Classification".  Equation and section numbers refer to the manuscript.  Requires PyTorch only; no target label is used.

Overview (one epoch e > e0 of training):
    pbar  = update_teacher(pbar, p, alpha)                      # temporal ensemble of the target posteriors, eq. (7)
    pi0   = initial_shares(pbar)                                # once, at the first epoch of the plan
    rho   = mass_schedule(e, e0, E, rho0, rho1)                 # growing transported mass, eq. (3)
    plan  = dpot_plan(pbar, rho, b, pi0, cost='classifier')     # Algorithm 1 (gate x rows, with the class bounds)
    ... next epoch: consistency_loss(logits_t, plan[idx]) (eq. (8)), lmmd_class_weights(plan[idx]) and
        contrastive_labels(plan[idx], idx, K) replace the pseudo-label-dependent parts of the base objective.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------- schedule and teacher
def mass_schedule(epoch, e0, epochs, rho0, rho1):
    """Transported mass rho(e) = rho0 + (rho1 - rho0) (e - e0) / (E - e0) for e0 < e <= E, eq. (3)."""
    t = min(1.0, max(0.0, (epoch - e0) / max(1, epochs - e0)))
    return min(1.0, rho0 + (rho1 - rho0) * t)


def consistency_ramp(epoch, e0, length=20):
    """Ramp r(e) = min{1, (e - e0) / 20} of the consistency weight, eq. (9)."""
    return min(1.0, max(0.0, (epoch - e0) / float(length)))


@torch.no_grad()
def update_teacher(pbar, p, alpha=0.9):
    """Temporal ensemble of the target posteriors, eq. (7): pbar <- alpha pbar + (1 - alpha) p.
    pbar: N x K teacher (None at the first epoch of the plan, which returns p), p: N x K posteriors of this epoch
    (evaluation mode, running normalization statistics)."""
    return p.clone() if pbar is None else alpha * pbar + (1.0 - alpha) * p


@torch.no_grad()
def initial_shares(pbar):
    """pi^0: share of every class among the hard predictions of the teacher at the first epoch of the plan, eq. (5)."""
    K = pbar.shape[1]
    return torch.bincount(pbar.argmax(1), minlength=K).double() / len(pbar)


# ----------------------------------------------------------------------------------------------- costs
@torch.no_grad()
def source_prototypes(src_feat, src_labels, K):
    """nu_k: mean of the l2-normalized source features of class k, normalized (Section III-E)."""
    fs = F.normalize(src_feat.float(), dim=1)
    return F.normalize(torch.stack([fs[src_labels == k].mean(0) for k in range(K)]), dim=1)


@torch.no_grad()
def prototype_similarity(feat, protos, pseudo):
    """cos(z_j, nu_{yhat_j}) = 1 - d_j: similarity of every target feature to the source prototype of its pseudo-class."""
    ft = F.normalize(feat.float(), dim=1)
    return (ft @ protos.t()).gather(1, pseudo[:, None])[:, 0]


# ----------------------------------------------------------------------------------------------- the plan
@torch.no_grad()
def dpot_plan(pbar, mass, bound, pi0, cost='classifier', feat=None, protos=None, eps=0.05):
    """Plan of one epoch (Algorithm 1): two-stage computation of the bounded partial plan.

    pbar   : N x K teacher posteriors.
    mass   : rho(e), the transported mass (fraction of the N target samples).
    bound  : b in [0, 1); every class k keeps at least b * rho * N * pi0_k samples (eq. (5)); b = 0: no bounds.
    pi0    : K initial predicted class shares (initial_shares).
    cost   : 'classifier' (DPOT-C, c_jk = -log pbar_jk: samples admitted in the order of their confidence) or
             'prototype' (DPOT-P, kappa -> infinity: admitted in the order of the similarity of the feature to the source
             prototype of the pseudo-class; needs feat (N x d) and protos (K x d, source_prototypes)).
    eps    : entropic weight of the rows, q_jk proportional to pbar_jk^(1 / eps) (Theorem 1).

    Returns the plan N x K with rows g_j q_j (g_j in {0, 1}: the gate in the limit eps -> 0 of Theorem 1, item 3; the rows of
    the samples outside the plan are zero) and the boolean vector of the samples in the plan.  The pseudo-label of a sample
    in the plan is plan.argmax(1).  As in our experiments the bound of a class is counted in whole samples, int(b rho N pi0_k)
    (within one sample per class of the bound b rho N pi0_k of eq. (5))."""
    N, K = pbar.shape
    conf, pseudo = pbar.max(1)
    if cost == 'classifier':
        score = conf                                                    # smallest m_j = -log max_k pbar_jk first
    elif cost == 'prototype':
        score = prototype_similarity(feat, protos, pseudo)              # smallest d_j first
    else:
        raise ValueError(cost)
    keep = torch.zeros(N, dtype=torch.bool, device=pbar.device)
    keep[score.topk(max(1, int(mass * N))).indices] = True             # gate: the floor(rho N) cheapest samples
    # rows: sharpened teacher posteriors.  As in our experiments the rows are normalized by F.normalize (p = 1, eps = 1e-12):
    # for a sample with sum_k pbar_jk^(1/eps) < 1e-12 (all posteriors below about 0.25 at eps = 0.05) the row stays below one
    # and the sample counts as outside the plan in the loss terms (plan.sum(1) <= 0.5).
    rows = F.normalize(pbar.double().clamp_min(1e-12) ** (1.0 / eps), p=1, dim=1)
    if bound > 0:
        for k in range(K):                                              # top-up of every class below its bound
            need = int(bound * mass * N * float(pi0[k])) - int((keep & (rows.argmax(1) == k)).sum())
            cand = (~keep).nonzero(as_tuple=True)[0]
            if need > 0 and len(cand):
                add = cand[pbar[cand, k].topk(min(need, len(cand))).indices]
                keep[add] = True
                rows[add] = 0
                rows[add, k] = 1
    return (rows * keep[:, None]).float(), keep


@torch.no_grad()
def closed_form_plan(C, mass, eps=0.05, iters=200):
    """Entropic partial plan of problem (2) for a finite eps, by Theorem 1: gamma_jk = (1/N) g_j q_jk with the rows
    q_jk = softmax_k(-C_jk / eps), the soft minima s_j = -eps log sum_k exp(-C_jk / eps), the gates
    g_j = min{1, exp((tau - s_j) / eps)} and tau the root of (1/N) sum_j g_j = mass (bisection).  C: N x K costs.
    Returns (gamma, g, q, tau).  For reference: dpot_plan uses the limit eps -> 0 of the gate."""
    C = C.double(); N = C.shape[0]
    q = torch.softmax(-C / eps, dim=1)
    s = -eps * torch.logsumexp(-C / eps, dim=1)
    lo, hi = float(s.min()) - 50 * eps, float(s.max()) + 50 * eps
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(torch.clamp(torch.exp((mid - s) / eps), max=1.0).mean()) < mass:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    g = torch.clamp(torch.exp((tau - s) / eps), max=1.0)
    return g[:, None] * q / N, g, q, tau


@torch.no_grad()
def bounded_plan_exact(C, R, L, sweeps=50, steps=40):
    """Exact solution of the bounded partial transport problem in the limit eps -> 0 (supplementary material, Section S-III):
    choose R of the N samples and one class for each, with at least L[k] samples in class k, at minimal total cost
    sum_j C[j, k(j)].  By Theorem 2 the solution takes the R samples of smallest min_k (C_jk - beta_k) and labels them by
    that arg min, with class offsets beta >= 0; the thresholds of the bounded classes are found by bisection.
    C: N x K costs, R: number of transported samples, L: K integer bounds with sum(L) <= R.
    Returns (beta, keep (bool N), labels (long N), number of sweeps)."""
    N, K = C.shape
    L = torch.as_tensor(L, device=C.device).long()
    gen = torch.Generator(device='cpu').manual_seed(0)      # ties broken by a fixed perturbation far below the cost scale
    C = C + (1e-9 * (float(C.max() - C.min()) + 1.0)) * torch.rand(N, K, generator=gen, dtype=C.dtype).to(C.device)
    top = float(C.max()) + 1.0
    A = torch.zeros(K, dtype=torch.bool, device=C.device)
    theta = torch.zeros(K, dtype=C.dtype, device=C.device)

    def assign(th, A_):
        inf = torch.full((N,), float('inf'), dtype=C.dtype, device=C.device)
        aA = (C - th[None, :]).masked_fill(~A_[None, :], float('inf')).min(1)[0] if bool(A_.any()) else inf
        mU = C.masked_fill(A_[None, :], float('inf')).min(1)[0] if not bool(A_.all()) else inf
        inA = aA <= 0
        rest = R - int(inA.sum())
        if rest > 0:
            cand = mU.masked_fill(inA, float('inf'))
            tau = float(torch.kthvalue(cand, rest).values) if rest <= N - int(inA.sum()) else top
        else:
            tau = float(C.min()) - 1.0
        t = torch.where(A_, th, torch.full_like(th, tau))
        m, lab = (C - t[None, :]).min(1)
        keep = m <= 0
        if int(keep.sum()) > R:
            keep = torch.zeros(N, dtype=torch.bool, device=C.device)
            keep[torch.topk(m, R, largest=False).indices] = True
        return keep, lab, torch.bincount(lab[keep], minlength=K), tau

    keep, lab, cnt, tau = assign(theta, A)
    n = 0
    tol = 1e-9 * max(1.0, top)
    while n < sweeps:
        bad = []
        for k in range(K):
            if int(cnt[k]) < int(L[k]):
                bad.append(k)
            elif bool(A[k]) and int(cnt[k]) > int(L[k]) and float(theta[k]) > tau + tol:
                th = theta.clone(); th[k] = max(tau, float(theta[k]) - 10 * tol)
                if int(assign(th, A)[2][k]) >= int(L[k]):
                    bad.append(k)
        if not bad:
            break
        n += 1
        for k in bad:
            if not bool(A[k]):
                A[k] = True; theta[k] = tau
            lo, hi = tau, top
            for _ in range(steps):
                mid = 0.5 * (lo + hi); th = theta.clone(); th[k] = mid
                if int(assign(th, A)[2][k]) >= int(L[k]):
                    hi = mid
                else:
                    lo = mid
            theta[k] = hi
            keep, lab, cnt, tau = assign(theta, A)
            if float(theta[k]) <= tau + 1e-12 and int(cnt[k]) >= int(L[k]):
                A[k] = False; theta[k] = 0.0
                keep, lab, cnt, tau = assign(theta, A)
    beta = torch.where(A, theta - tau, torch.zeros_like(theta)).clamp_min(0.0)
    return beta, keep, lab, n


# ----------------------------------------------------------------------------------------------- losses driven by the plan
def consistency_loss(logits_t, plan_rows):
    """Transport-network consistency, eq. (8): -(1/|B_t|) sum_j g_j sum_k q_jk log p_jk on the target minibatch.
    logits_t: B x K logits of the training pass, plan_rows: B x K rows g_j q_j of the current plan (zero outside the plan)."""
    return -(plan_rows * F.log_softmax(logits_t, dim=1)).sum(1).mean()


def lmmd_class_weights(plan_rows):
    """Class weights of the target samples in the class-conditional alignment term (LMMD): g_j q_jk instead of the soft
    predictions p_jk; a sample outside the plan has zero weight for every class (Section IV-B, DPOT-P)."""
    return plan_rows


def contrastive_labels(plan_rows, idx, K):
    """Labels that define the positives of the target contrastive term (Section IV-B, DPOT-P): two samples of the plan with
    the same pseudo-label are positives; a sample outside the plan gets a label of its own (K + its index), so that its only
    positive is its own augmented view.  idx: indices of the minibatch samples in the target set."""
    on = plan_rows.sum(1) > 0.5
    return torch.where(on, plan_rows.argmax(1), K + idx.to(plan_rows.device))


# ----------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def reestimate_bn(model, target_x, batch_size=256, max_batches=64, seed=0):
    """Copy of the model whose normalization statistics are re-estimated on unlabeled target samples (Section IV-D): the
    running statistics are reset and accumulated over random target minibatches, the weights are unchanged.
    target_x: tensor of target inputs on the device of the model.  Used for DPOT-C on Houston, PU->PC and SH->HZ."""
    m2 = copy.deepcopy(model)
    for m in m2.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None      # cumulative average
    m2.train()
    gen = torch.Generator().manual_seed(seed)
    n = len(target_x); nb = min(max_batches, max(1, n // batch_size))
    perm = torch.cat([torch.randperm(n, generator=gen) for _ in range(nb * batch_size // n + 1)])[:nb * batch_size]
    for a in range(0, nb * batch_size, batch_size):
        m2(target_x[perm[a:a + batch_size].to(target_x.device)])
    return m2.eval()


# ----------------------------------------------------------------------------------------------- a small example
if __name__ == '__main__':
    torch.manual_seed(0)
    N, K = 2000, 3
    shares = torch.tensor([0.75, 0.2, 0.05])                         # an imbalanced target
    y = torch.multinomial(shares, N, replacement=True)
    logits = 2.0 * F.one_hot(y, K).float() + torch.randn(N, K)        # a teacher that is right most of the time
    logits[:, 0] += 0.8                                               # ... and biased towards the large class
    pbar = torch.softmax(logits, 1)
    pi0 = initial_shares(pbar)
    for e in (21, 60, 100):
        rho = mass_schedule(e, 20, 100, 0.3, 0.9)
        for b in (0.0, 0.7):
            plan, keep = dpot_plan(pbar, rho, b, pi0)
            lab = plan.argmax(1)
            print('epoch %3d  rho %.3f  b %.1f  in plan %4d  pseudo-label accuracy %.3f  per class %s' % (
                e, rho, b, int(keep.sum()), float((lab[keep] == y[keep]).float().mean()),
                torch.bincount(lab[keep], minlength=K).tolist()))
