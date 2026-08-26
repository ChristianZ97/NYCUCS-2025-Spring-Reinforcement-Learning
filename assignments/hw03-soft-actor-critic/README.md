# Soft Actor-Critic for Continuous Control

This project implements the original value-network variant of Soft Actor-Critic
(SAC) for `Pendulum-v1` and `HalfCheetah-v5`. The portfolio source keeps the
training strategy that produced the coursework results: a tanh-squashed
Gaussian actor, twin Q critics, an explicit value network and target value
network, automatic entropy-temperature tuning, replay memory, and soft target
updates.

## Repository layout

```text
assignments/hw03-soft-actor-critic/
├── src/
│   ├── sac_pendulum.py
│   └── sac_halfcheetah.py
├── results/
│   ├── README.md
│   ├── pendulum-return.png
│   ├── pendulum-summary.json
│   ├── halfcheetah-return.png
│   ├── halfcheetah-summary.json
│   └── halfcheetah-hyperparameters.json
├── submission/
│   └── README.md
├── requirements.txt
└── README.md
```

`src/` is the cleaned portfolio implementation. Exact grading artifacts are a
separate provenance product and are intentionally not published; their hashes
and private recovery point are recorded in
[`submission/README.md`](submission/README.md).

## Redistribution status

The canonical sources remain derivatives of course-provided starter
scaffolding. Exact starter and grading artifacts remain excluded; permission to
publish this cleaned portfolio version was confirmed before release.

## Setup and use

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r assignments/hw03-soft-actor-critic/requirements.txt

python assignments/hw03-soft-actor-critic/src/sac_pendulum.py
python assignments/hw03-soft-actor-critic/src/sac_halfcheetah.py
```

HalfCheetah requires a working MuJoCo installation supplied by the
`gymnasium[mujoco]` extra. PyTorch automatically uses CUDA when available and
otherwise falls back to CPU. The historical hardware and exact package
versions were not recorded, so the dependency file specifies the required
packages rather than claiming an exact historical environment.

Both programs run without external experiment tracking by default. To send
metrics to Weights & Biases, install `wandb`, authenticate separately, and add
`--track`. Tracking does not upload source code.

```bash
python -m pip install wandb
python assignments/hw03-soft-actor-critic/src/sac_halfcheetah.py --track
```

Use `--help` for every option. Invalid sizes, update frequencies, learning
rates, discount factors, and target-update coefficients are rejected before
training.

## Retained configurations

| Setting | Pendulum-v1 | HalfCheetah-v5 |
| --- | ---: | ---: |
| Hidden layers | 2 × 128 | 2 × 512 |
| Learning rate | `3e-4` | `0.0006344571440300171` |
| Discount factor | `0.99` | `0.98` |
| Target update coefficient | `5e-3` | `0.002244255539895659` |
| Batch size | 256 | 512 |
| Replay capacity | 1,000,000 | 1,000,000 |
| Initial random steps | 1,000 | 10,000 |
| Training steps | 100,000 | 1,000,000 |
| Policy update frequency | 1 | 1 |
| Seed | 77 | 77 |

The HalfCheetah defaults are the tuned configuration associated with the
retained one-million-step result. The former sweep runner and its duplicate
300,000-step source were experiment infrastructure, not separate algorithms,
and are not part of the portfolio surface.

## Historical results

The retained W&B exports report a final Pendulum return of `-117.8241` after
100,000 steps and a final HalfCheetah return of `7490.5371` after 1,000,000
steps. The corresponding runs lasted about 17.6 minutes and 5.35 hours.

These are **historical measurements** from the development sources, not fresh
benchmarks of the cleaned files. Only representative return plots, terminal
summaries, and the selected HalfCheetah hyperparameters are retained. See
[`results/README.md`](results/README.md) for exact hashes and provenance.

## Preserved behavioral choices

The cleanup intentionally did not redesign the training algorithm:

- both environment termination and time-limit truncation set the replay
  terminal mask;
- every episode reset reuses the configured seed;
- the actor mean is bounded before Gaussian sampling, matching the historical
  implementation;
- during Pendulum's initial random-exploration phase, the wrapper samples the
  native `[-2, 2]` action space and then interprets those samples as normalized
  `[-1, 1]` inputs. This inherited behavior saturates some environment actions
  and stores values outside the actor's output range in replay;
- the value/target-value formulation is retained rather than replacing it with
  a newer SAC variant.

These choices matter when comparing against the historical plots. Change them
only as a separately documented experiment.
