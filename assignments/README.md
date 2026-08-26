# Assignments

This directory contains three assignments from a Spring 2025 reinforcement
learning course.

| Assignment | Topic | Algorithms or tools |
| --- | --- | --- |
| [HW1](hw01-policy-gradient-foundations/) | Policy-gradient foundations | REINFORCE, value baselines, GAE, and D4RL |
| [HW2](hw02-ddpg-continuous-control/) | Deterministic continuous control | DDPG for Pendulum and HalfCheetah |
| [HW3](hw03-soft-actor-critic/) | Entropy-regularized continuous control | Soft Actor-Critic for Pendulum and HalfCheetah |

## Repository Convention

Each assignment presents its canonical portfolio implementation under `src/`,
project dependencies in `requirements.txt`, and compact representative
measurements under `results/`. Its `submission/README.md` records exact private
grading provenance without copying identity-bearing or redistribution-restricted
artifacts into the portfolio tree.

Canonical sources are cleaned derivatives rather than byte-exact submissions.
Exact grading packages, reports, course-owned material, raw logs, redundant
checkpoints, and experiment sweeps remain recoverable from the private
`legacy` history. Historical figures and tables are not presented as fresh
benchmarks of the portfolio source.

Several workloads require legacy Gym, Box2D, D4RL, MuJoCo, downloaded datasets,
or accelerator hardware. Each assignment README distinguishes what was
executed from static, mocked, or provenance-only validation.
