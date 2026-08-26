# Historical submitted-era Pendulum search

These files record an unsuccessful Bayesian hyperparameter search performed
with the submitted Pendulum implementation:

- `pendulum-bo-history.txt`: exact 100-trial table;
- `pendulum-bo-best.txt`: exact recorded best-parameter summary;
- `pendulum-bo-convergence.png`: exact Matplotlib convergence plot.

They predate the canonical source and must not be interpreted as benchmarks of
that source.

| Statistic | Historical value |
|---|---:|
| trials | `100` |
| best trial | `46` |
| best trial seed | `88` |
| best recorded score | `-1348.142` |
| mean recorded score | `-1517.085528` |
| worst recorded score | `-1756.9107` |

The recorded best configuration was gamma `0.98`, tau `0.001`, OU noise scale
`0.05`, actor learning rate `0.0001`, critic learning rate `0.0006`, hidden
width `128`, batch size `64`, one update after each episode, and 3,000 warm-up
steps.

Important limitation: the submitted training function did not pass the actor
or critic learning-rate arguments into the DDPG agent, and its exploration path
did not consume the OU-noise scale. Consequently, those three search dimensions
did not affect the run as their labels imply.

Original SHA-256 hashes:

- history: `1f4b6c41c21c0e377cfb3cd7e938a561d0538cc91fc5e21acf7560863cd8d174`
- best parameters: `7e53ffa73ece55985de8d2fcb8e84cecead77bf2ead2b514596528c1ac980476`
- plot: `d8add50ba31d2e6d0c262130624011d4f97b36cc587a36327886d425b00b67f7`
