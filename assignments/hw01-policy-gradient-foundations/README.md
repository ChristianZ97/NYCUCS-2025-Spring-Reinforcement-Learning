# Homework 1: Policy-Gradient Foundations

This portfolio version presents the five implementation deliverables from the
first reinforcement-learning assignment:

- vanilla REINFORCE on `CartPole-v0`;
- REINFORCE with a learned state-value baseline on `LunarLander-v2`;
- REINFORCE with Generalized Advantage Estimation (GAE) on `LunarLander-v2`;
- a D4RL installation sanity check; and
- a compact D4RL dataset-format inspector.

The policy-gradient targets retain their submitted network layouts, losses,
learning rates, scheduler boundaries, and stopping conventions. The portfolio
cleanup adds explicit train/evaluation commands, argument validation, finite
run limits, safe device remapping, strict state-dictionary loading, headless
evaluation by default, Gym API compatibility helpers, and reliable resource
cleanup.

## Repository layout

```text
src/                         canonical portfolio implementations
results/historical_metrics.csv
results/checkpoints/         representative historical state dictionaries
submission/README.md         private-provenance manifest and omission record
requirements.txt             historical compatibility environment
```

The exact grading ZIP, report, course handout, and starter-derived submitted
sources remain recoverable from the private `legacy` branch at commit
`14c78350cc80c564ce9b1f7abedb4c44075c0626`. They are intentionally absent
from the portfolio-facing tree because the report contains identity-bearing
material and the redistribution status of the course handout and starter
scaffold is unclear. See
[`submission/README.md`](submission/README.md) for the complete hash manifest.

The canonical sources are cleaned derivatives of that grading-time work and
may retain elements of the course scaffold. The exact handout, starter bundle,
and grading artifacts remain excluded; permission to publish this cleaned
portfolio version was confirmed before release.

## Environment

The historical code used legacy Gym environment IDs and the legacy
MuJoCo/D4RL stack. Python 3.8 is the most conservative common environment for
the pinned packages:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Box2D may require native build tools. D4RL additionally needs a working
MuJoCo 2.1 installation and may download large datasets on first use. The
policy-gradient programs support CPU, CUDA, and Apple MPS through
`--device`; evaluation maps the historical CUDA/MPS checkpoints onto the
selected device.

## Run and evaluate

Every program has `--help`. Evaluation is headless unless `--render` is
provided.

```bash
# Evaluate retained historical checkpoints.
python src/reinforce.py eval
python src/reinforce_baseline.py eval
python src/reinforce_gae.py eval --lambda 0.8
python src/reinforce_gae.py eval --lambda 0.92 \
  --checkpoint results/checkpoints/LunarLander_0.0001_lambda_0.92_pc.pth

# Train without overwriting retained historical results.
python src/reinforce.py train --checkpoint-out runs/CartPole_0.008.pth
python src/reinforce_baseline.py train \
  --checkpoint-out runs/LunarLander_0.0001.pth
python src/reinforce_gae.py train --lambda 0.98 \
  --checkpoint-out runs/LunarLander_0.0001_lambda_0.98.pth

# Inspect the two datasets discussed in the report.
python src/d4rl_sanity_check.py hopper-random-v2
python src/d4rl_format.py maze2d-umaze-v1 hopper-medium-v2
```

Training metrics are written only when `--log-dir` is supplied. This avoids
creating large TensorBoard logs during ordinary imports or evaluations.

## Historical results

[`results/historical_metrics.csv`](results/historical_metrics.csv) records
compact measurements parsed from the original TensorBoard event files. These
measurements predate the portfolio refactor; the cleaned sources were not
retrained or benchmarked.

The CartPole run stopped after 919 episodes at EWMA reward 195.110. The
LunarLander baseline and reported GAE sweep stopped when a custom EWMA reward
crossed `+120`. That historical threshold is below the assignment handout's
testing-return criterion of `200`, so these records are not presented as proof
that LunarLander was solved.

Five representative checkpoints are retained: CartPole, the primary
LunarLander baseline, and GAE with λ values 0.8, 0.92, and 0.98. The CartPole
file is a cleaned derivative of the submitted checkpoint: all genuine PyTorch
records are byte-identical, but 13 unrelated files accidentally embedded in
the submitted checkpoint were removed. It is therefore not labeled an exact
submission.

### GAE source selection

The exact grading ZIP, report, and historical logs agree on the submitted ReLU
architecture and custom `+120` EWMA threshold. A separate loose working-tree
copy at the pre-curation commit had later diverged to `tanh` activations and
the environment's configured reward threshold. That unsubmitted variant was
omitted; the canonical GAE target follows the exact grading source and retained
results.

## Limitations

- Training is stochastic and sensitive to legacy Gym, Box2D, PyTorch, and
  device behavior.
- The historical GAE implementation treats the end of every collected episode
  as terminal; it does not bootstrap across a time-limit truncation.
- Vanilla REINFORCE retains the submitted auxiliary value loss on the shared
  representation even though its policy loss uses unbaselined returns.
- The D4RL commands require external datasets and the legacy MuJoCo runtime.
- The exact pinned environment was unavailable, but a Python 3.8 environment
  with Gym 0.23.1 and PyTorch 2.2.2 strictly loaded all five checkpoints and
  confirmed finite tensors. Deterministic loss/GAE probes passed, and the
  retained CartPole policy completed 10 headless episodes at return 200. Gym's
  import required a minimal in-memory pygame shim; no pygame API was exercised.
  Box2D, D4RL, and MuJoCo remained unavailable, so LunarLander and dataset
  commands received static rather than end-to-end validation.
