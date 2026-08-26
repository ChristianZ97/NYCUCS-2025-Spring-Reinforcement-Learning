# Private grading provenance

Exact grading artifacts are intentionally not copied into this portfolio-facing
directory. They remain recoverable from the private `legacy` branch at exact
pre-curation commit
`14c78350cc80c564ce9b1f7abedb4c44075c0626`.

The omission is deliberate:

- the report contains a name, student identifier, and email address;
- the handout is course-owned and has no redistribution license;
- the grading ZIP contains a derivative of the supplied course scaffold and
  has no redistribution license;
- the ZIP's original filename contains a student identifier.

## Exact grading ZIP manifest

The ZIP SHA-256 is:

`0fda09df676ad5056aad9cadf28a043f1f4fe6bbd2a5ded63dea52926ef23bfd`

It contains exactly four entries:

| Entry | Uncompressed bytes | SHA-256 |
|---|---:|---|
| `ddpg.py` | 17118 | `a3336d47ca1f5b9b24f652bc53e01aeb46a8a3c314b898290511ff66b8503ecd` |
| `ddpg_cheetah.py` | 18175 | `54cea83a9e4bf46394251761f46142788c5e1c06597f9435ca95a1f46e5ea394` |
| `ddpg_actor_Pendulum-v0_04252025_182503_.pth` | 137871 | `9f552b42c247c225ab37bf0561a3df29b1bd324af81e3133574c5a69ea48fb50` |
| `ddpg_critic_Pendulum-v0_04252025_182503_.pth` | 138393 | `abb5489e658e28f286b9d9db16c0cf310605976e0e5a29080e96d0376ddc306a` |

The two submitted Python entries were byte-identical to the corresponding
working files at the pre-curation commit. The submitted actor and critic were
byte-identical to the grading-era Pendulum checkpoint pair. `unzip -t` reported
no archive errors.

For completeness, the private report SHA-256 is
`6e7ba9144fa7a0e85cd3d7a35a627948a671f7dfd6f88951bab69da3d7858227`
and the private handout SHA-256 is
`7aac52eab3437810ed3a76f30ebdf63e6671205832494b7c95d7899f3257fae8`.

## Missing grading deliverables

The actual ZIP did not contain:

- a HalfCheetah actor or critic checkpoint;
- `ddpg_cdq_cheetah.py` or another clipped-double-Q implementation;
- a clipped-double-Q result.

Those omissions are recorded rather than filled retroactively.
