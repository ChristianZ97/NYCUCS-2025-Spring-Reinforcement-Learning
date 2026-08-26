# Historical results

These files are a compact, public-safe subset of the original experiment
exports. They were moved without changing their bytes. The measurements
predate the portfolio cleanup and must not be presented as benchmarks of the
cleaned source.

| File | Meaning | SHA-256 |
| --- | --- | --- |
| `pendulum-return.png` | W&B return curve through 100,000 steps | `fc3c1aeb965f2a21c2aca2217a07449d386094a31ccba4620e4859754bf3a61f` |
| `pendulum-summary.json` | Terminal W&B metrics | `6e1b8b931c15b15e1824b8648f62a364813ac96d3f0e1a0a43b9372026896e69` |
| `halfcheetah-return.png` | W&B return curve through 1,000,000 steps | `1e440925394ffddac46c6232836e8e1b7452d05d5e91bf62ad2cbf50ca61f8b3` |
| `halfcheetah-summary.json` | Terminal W&B metrics | `dcd6c9e30797e94dc7ab0f0e081af6ae7b67788bc538616509856fc6344747a2` |
| `halfcheetah-hyperparameters.json` | Selected sweep configuration | `d35bd442412e13d84f7ce11ac5de0e898da9c9e0b3c158c1f89c2d6c5f88e822` |

The summaries report:

- Pendulum: return `-117.82414871545531`, episode 500, 100,000 steps, and
  runtime `1055.35` seconds.
- HalfCheetah: return `7490.537058457859`, episode 1000, 1,000,000 steps, and
  runtime `19245.51` seconds.

The original loss plots were redundant with the terminal loss values in the
JSON summaries and were omitted. Raw W&B histories and model checkpoints were
not preserved, so the curves cannot be regenerated exactly from this
repository.
