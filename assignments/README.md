> Specification update (2026-10-06): original course specifications are now included; see [SPECS.md](../SPECS.md). Earlier handout-exclusion statements below describe the previous curation scope. Other exclusions remain unchanged.

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
grading provenance, and `submission/report.pdf` preserves the original report
without modification.

Canonical sources are cleaned derivatives rather than byte-exact submissions.
Exact reports are public with their original identity information; grading
ZIPs, course-owned material, raw logs, redundant checkpoints, and experiment
sweeps remain recoverable from the private `legacy` history. Historical figures
and tables are not presented as fresh benchmarks of the portfolio source.

Several workloads require legacy Gym, Box2D, D4RL, MuJoCo, downloaded datasets,
or accelerator hardware. Each assignment README distinguishes what was
executed from static, mocked, or provenance-only validation.
