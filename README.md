# Dynamic Partial Optimal Transport for Cross-Scene Hyperspectral Image Classification

Core implementation of **dynamic partial optimal transport (DPOT)** with lower bounds on the class marginals, the method of
the manuscript "Dynamic Partial Optimal Transport for Cross-Scene Hyperspectral Image Classification".

Author of the code: Zheng-Yang Ren.

## What is here

`dpot.py` holds the pseudo-label generation of DPOT and the terms that it drives. In every epoch a growing fraction of the
target mass is transported to the classes; samples that receive no mass are left out of the pseudo-label-dependent terms,
and every class keeps at least a fraction of its initially predicted share. It needs PyTorch only and uses no target label.

| Function | Step of the method |
|---|---|
| `mass_schedule`, `consistency_ramp` | Growing transported mass rho(e), eq. (3); ramp r(e) of the consistency weight, eq. (9) |
| `update_teacher`, `initial_shares` | Temporal ensemble of the target posteriors (teacher), eq. (7); initial predicted class shares pi^0 |
| `source_prototypes`, `prototype_similarity` | Source prototypes and the feature-space term of the cost, eq. (6) |
| `dpot_plan` | Plan of one epoch, Algorithm 1: gate (the floor(rho N) cheapest samples), rows (sharpened teacher posteriors), top-up of the classes below their bounds, eq. (5); classifier cost (DPOT-C) or prototype cost (DPOT-P) |
| `closed_form_plan` | Entropic partial plan for a finite epsilon by the closed form of Theorem 1 (rows, soft gates, threshold) |
| `bounded_plan_exact` | Exact solution of the bounded problem in the limit epsilon -> 0 (Theorem 2; supplementary material) |
| `consistency_loss` | Transport-network consistency, eq. (8) |
| `lmmd_class_weights`, `contrastive_labels` | Plan-weighted class-conditional alignment and plan-restricted contrastive positives (DPOT-P) |
| `reestimate_bn` | Inference with normalization statistics re-estimated on the unlabeled target samples (Section IV-D) |

## Use

```python
import torch
from dpot import mass_schedule, update_teacher, initial_shares, dpot_plan, consistency_loss

# p: N x K posteriors of the network on the N target samples, computed at the end of epoch e (evaluation mode)
pbar = update_teacher(pbar, p, alpha=0.9)               # teacher, eq. (7)   (pbar = None before the first plan)
if pi0 is None:
    pi0 = initial_shares(pbar)                           # at the first epoch of the plan (e = e0 + 1)
rho = mass_schedule(e, e0=20, epochs=100, rho0=0.3, rho1=0.9)
plan, in_plan = dpot_plan(pbar, rho, bound=0.7, pi0=pi0, cost='classifier')

# next epoch, for a target minibatch with indices idx and logits logits_t of the training pass:
loss = loss_base + lam * r_e * consistency_loss(logits_t, plan[idx])
```

`python dpot.py` runs a small synthetic example and prints the size, the pseudo-label accuracy and the class counts of the
plan with and without the bounds.

## Status

This repository currently holds the core functions of the method. The complete code will be released here when the paper
is published: the adaptation network and its training with the DPOT plan, the data pipeline of the cross-scene tasks, the
evaluation scripts, the per-run results, and the scripts that generate the tables of the paper.
