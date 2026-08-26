# Homework 2: DDPG for Continuous Control

This portfolio contains the implementation work for the continuous-control
portion of a reinforcement-learning assignment. It presents one canonical
Pendulum target and one canonical HalfCheetah target while keeping grading
provenance recoverable from the private `legacy` branch.

## Repository layout

- `src/pendulum/` is the post-submission Pendulum refinement that produced the
  retained checkpoint.
- `src/halfcheetah/ddpg.py` is a cleaned derivative of the submitted
  HalfCheetah implementation.
- `results/pendulum/` contains one representative actor/critic pair and a
  portfolio-time checkpoint comparison.
- `results/historical/` contains compact, grading-era evidence from an
  unsuccessful Pendulum hyperparameter search.
- `submission/README.md` records the exact private grading manifest and its
  hashes. Restricted and identity-bearing originals are not published here.

The theoretical answers, assignment handout, identity-bearing report, exact
grading archive, course starter scaffold, raw TensorBoard events, and redundant
experiment trees are intentionally absent from the portfolio tree.

The canonical sources are cleaned derivatives of grading-time code that
included course scaffolding. Exact starter and grading artifacts remain
excluded; permission to publish this cleaned portfolio version was confirmed
before release.

## Lineage and scope

The grading archive and submitted source first appeared in Git commit
`6102089c4159f0260726e8c42f41f4bd863d664c` on April 25, 2025. The modular
Pendulum rewrite began in commit `e059498c9905c5b9c39b0b0c10297a388580adc4`.
Its source is byte-for-byte unchanged from checkpoint commit
`cea37dd612e8cf00faa96649f730b88a08e1fe78` through the private pre-curation
commit `14c78350cc80c564ce9b1f7abedb4c44075c0626`.

The old directory named `New Try/HalfCheetah` was not retained: its final entry
point selected `Pendulum-v1`, so it was not evidence of a distinct
HalfCheetah run. The canonical HalfCheetah target instead starts from the exact
submitted `ddpg_cheetah.py`.

The submitted deliverables did not contain a clipped-double-Q implementation,
a HalfCheetah checkpoint, or a clipped-double-Q result. This portfolio does not
invent those missing deliverables and makes no claim that HalfCheetah reached
the assignment target.

## Pendulum

The canonical target uses `Pendulum-v1`, a two-hidden-layer ReLU actor, a
two-hidden-layer layer-normalized critic, target networks, replay memory, and
Ornstein-Uhlenbeck action noise. Its retained tuned defaults are:

| Parameter | Value |
|---|---:|
| discount factor | `0.999855778577656` |
| target update rate | `0.02988295604632515` |
| initial OU noise scale | `1.4068608877080406` |
| actor learning rate | `0.0011115818565635618` |
| critic learning rate | `0.0026822139156196297` |
| batch size | `512` |
| replay capacity | `100000` |
| warm-up transitions | `5000` |
| reward scale | `0.1` |
| hidden width | `128` |
| updates after each episode | `1` |

“Updates after each episode” is deliberate wording. The historical code called
this setting `updates_per_step`, but performed the update only after a complete
episode. The canonical code preserves that execution strategy while naming it
accurately.

The actor and critic in `results/pendulum/` were trained under seed
`1745986213`. Their SHA-256 hashes are:

- actor: `a14c3a3c5359324b00bf3abf25e56abec524422962263028bf853ea53bedf6a3`
- critic: `8232bb709e16a0739ecabe2881984e8826ebc85a592883434b36c9a9b12e6f3f`

An independent, portfolio-time implementation of the standard Pendulum
equations compared all 35 saved post-submission actors on 1,000 fixed initial
states for 200 steps. The retained actor had the best mean return:

| Checkpoint | Mean | Median | 10th percentile | 90th percentile |
|---|---:|---:|---:|---:|
| retained post-submission actor | `-156.994` | `-129.855` | `-273.357` | `-1.603` |
| submitted actor | `-1634.012` | `-1647.364` | `-1728.107` | `-1526.799` |

These are checkpoint evaluations, not regenerated training benchmarks and not
grading-time measurements. The complete compact comparison is in
`results/pendulum/checkpoint-evaluation.csv`.

## HalfCheetah

The canonical HalfCheetah file preserves the submitted three-hidden-layer
actor and critic, LayerNorm placement, parameter-space exploration, target
updates, critic regularization, and once-per-episode update schedule. It fixes
two clear submitted-code defects:

- deterministic evaluation had used the action lower bound as both bounds;
- training-function actor and critic learning-rate arguments were accepted but
  never passed to the agent.

The portfolio CLI uses modern Gymnasium and defaults to `HalfCheetah-v5`; the
submission targeted legacy Gym `HalfCheetah-v3`. Version-to-version scores must
not be compared without controlling the environment version. The
`500000`-transition default budget counts training transitions; evaluation
episodes are logged separately.

## Setup and use

Python 3.10 or newer is required:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Run Pendulum from this directory:

```bash
PYTHONPATH=src python -m pendulum.main --seed 1745986213
```

No files are written by default. Supply an explicit output directory to record
TensorBoard data, and add `--save-model` to save checkpoints:

```bash
PYTHONPATH=src python -m pendulum.main \
  --output-dir /tmp/hw2-pendulum --save-model
```

HalfCheetah requires the Gymnasium MuJoCo extra and a functioning MuJoCo
runtime:

```bash
python src/halfcheetah/ddpg.py \
  --max-training-steps 500000 \
  --output-dir /tmp/hw2-halfcheetah --save-model
```

Use `--help` on either command for short smoke-test budgets and device
selection. Both CLIs validate negative or zero sizes and report missing
environment/runtime support explicitly.

## Historical results and validation limits

The grading-era Pendulum Bayesian search was unsuccessful. Its best recorded
score was `-1348.142`; the retained plot and text tables are historical
artifacts and do not benchmark the cleaned source. Moreover, the submitted
training function ignored the proposed actor/critic learning rates and did not
use its OU-noise scale, so those search dimensions were ineffective. See
`results/historical/pendulum-bo-summary.md`.

During curation, all canonical Python files passed Python 3 AST parsing, the
grading manifest was independently hashed, and the retained checkpoint pair
was checked for container integrity, compatible state-dict shapes, and finite
tensors. The selected actor received the independent dynamics evaluation
described above. The local curation environment did not contain Gymnasium or a
MuJoCo runtime, so no canonical training run or real-environment HalfCheetah
execution is claimed.
