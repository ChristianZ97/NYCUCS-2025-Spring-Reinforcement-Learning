# Reinforcement Learning Portfolio

Coursework from a Spring 2025 reinforcement learning course. The repository
collects canonical implementations and compact experiment records for
policy-gradient methods, deterministic continuous control, entropy-regularized
control, and vision-language-model alignment.

## Projects

| Project | Topic | Canonical entry point |
| --- | --- | --- |
| [HW1: Policy-Gradient Foundations](assignments/hw01-policy-gradient-foundations/) | REINFORCE, value baselines, GAE, and D4RL | `src/reinforce.py` |
| [HW2: DDPG Continuous Control](assignments/hw02-ddpg-continuous-control/) | Pendulum and HalfCheetah with DDPG | `src/pendulum/main.py` |
| [HW3: Soft Actor-Critic](assignments/hw03-soft-actor-critic/) | Pendulum and HalfCheetah with SAC | `src/sac_pendulum.py` |
| [Final Project: VLM Action Alignment](final-project/) | CLIP rewards, RLOO, and supervised action tuning | `src/rloo_train.py` |

## Repository Layout

```text
assignments/   Three portfolio-ready homework projects
final-project/ Team VLM alignment pipeline and selected historical results
```

Each project exposes its cleaned portfolio implementation under `src/`,
dependencies in `requirements.txt`, representative measurements under
`results/`, and exact student-authored reports plus a provenance manifest under
`submission/`.
Measurements are labeled **historical** unless they were regenerated from the
cleaned source.

The original reports and final-project milestone documents are intentionally
published byte-for-byte and retain their historical identity and contact
details. Source archives, course-owned handouts and starter bundles, raw
TensorBoard/W&B output, tuning sweeps, redundant checkpoints, and presentation
feedback remain on `legacy`. See [`SUBMISSIONS.md`](SUBMISSIONS.md) for the
document inventory and hashes. Cleaned files under `src/` are derivatives and
are never represented as exact submissions.

## Reproducibility

The assignments span legacy Gym/D4RL stacks, modern Gymnasium/MuJoCo, PyTorch,
and Transformers. Create a separate environment for each project and follow its
README rather than installing every dependency into one environment. HW1
targets a legacy Python 3.8 stack, while the later projects target Python
3.10–3.11.

GPU-, MuJoCo-, dataset-, and checkpoint-dependent experiments are not portable
without their external runtime assets. Project READMEs distinguish historical
evidence from checks that can be run locally and document unavailable
grading-time data or hardware. See [`VALIDATION.md`](VALIDATION.md) for the
executed, static, provenance, and residual validation record.

## Publication

This public repository combines sanitized portfolio source with explicitly
identified, unredacted original submission documents. The private source
repository keeps a separate `legacy` branch with restricted course artifacts
and the remaining provenance material; that branch and its history are not
published here. Publication permission for the team and course-derived
portions was confirmed with the relevant collaborators and course staff.

## Academic Use

This repository is shared as a portfolio and learning reference. Current
students should follow their institution's academic-integrity rules and should
not submit this work as their own. Unless a file states otherwise, no license
is granted for copying or redistribution.
