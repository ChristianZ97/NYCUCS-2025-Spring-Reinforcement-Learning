> Specification update (2026-10-06): original course specifications are now included; see [SPECS.md](../../../SPECS.md). Earlier handout-exclusion statements below describe the previous curation scope. Other exclusions remain unchanged.

# Submission provenance

The exact submitted report is published here as [`report.pdf`](report.pdf).
It is byte-for-byte identical to the report on the private `legacy` branch,
whose tip is the exact pre-curation commit:

```text
14c78350cc80c564ce9b1f7abedb4c44075c0626
```

The report intentionally retains the student name, identifier, email address,
personal repository URL, and local user paths visible in screenshots. The
course handout and starter-derived source have unclear redistribution
permission, so the exact handout, ZIP, and submitted source members stay only
in the private legacy history. Files under `src/` are cleaned portfolio
derivatives, not exact submissions.

## Grading artifacts

Identity-bearing filenames are generalized here; the private commit records
the exact paths.

| Artifact | SHA-256 |
|---|---|
| Submitted source ZIP | `6f396afd43aaa2f831d0bbfc25c0dd7fe58122c744a17bede3638496d29d6cf0` |
| Submitted report PDF ([`report.pdf`](report.pdf)) | `fdeb3772727b01e60e09c36d5c9155dbc8e8ed8b6dc2152f50840e11b7b2ee80` |
| Course handout PDF | `4f75bd01a04a2db87ba0679cc2d5651862b69b71d205f884a899869e3d2fbed1` |

The ZIP passed a complete archive-integrity check and contains exactly 14
root-level members:

| ZIP member | Bytes | SHA-256 |
|---|---:|---|
| `CartPole_0.008.pth` | 23,732,953 | `1fe53c06cfcc5cd22cc3cad7c1044fe06b61ac3877c24a9b427904e2c9cfd0a1` |
| `LunarLander_0.0001.pth` | 3,223,554 | `a321b48798fd031f9c7d3c27e2fe2a8c72cf47084790ff142e7ff6c2fb0a9d7e` |
| `LunarLander_0.0001_10000episodes.pth` | 3,223,547 | `058f39810a49b2b101affbbdafe03f7ae9d15cab564a5df47467c1aa55718007` |
| `LunarLander_0.0001_lambda_0.8_pc.pth` | 3,223,985 | `cae133504c300cc4c6aec747e249c321b3653c9d2c2dcf087e3d560cecd46525` |
| `LunarLander_0.0001_lambda_0.92_lab.pth` | 3,224,019 | `6af04af2a08be3aaec925ae137ae39cc0c976a2eaee921bd3b90bc0c69758049` |
| `LunarLander_0.0001_lambda_0.92_pc.pth` | 3,224,019 | `d8a5c6bb68afe7c03838ef3d6c36592069c752abc535a4449448c00f5153b2cc` |
| `LunarLander_0.0001_lambda_0.98_pc.pth` | 3,224,019 | `3f04f6e5f31e637552493563ae45b1f9104e7c27d5b69a098baf17bbec6d4196` |
| `LunarLander_0.0001_lambda_0.995.pth` | 3,224,086 | `c0f97530d0884b49387565619e2b8c7196e941bb44fa936d17f5abfcc1a625b8` |
| `LunarLander_0.0001_lambda_0.995_lab.pth` | 3,224,053 | `3cd9f82112ceebfdbeb28556ce7d36eaa0663fb92dfa256d2f1b25d7bdd8039f` |
| `d4rl_format.py` | 1,186 | `d90ef5156207c7e8ed26306876bf7b8e6fb04a801db2662f147ec632a19e9b22` |
| `d4rl_sanity_check.py` | 824 | `5722683acece3659b2edf87f785b73f4aea85a100492323af3c657bfebc8c6ac` |
| `reinforce.py` | 10,774 | `17407cf634dab1b8dc860fa74b75edf2be39d0cbae178bb0fc3dde29fc833578` |
| `reinforce_baseline.py` | 11,467 | `95a35d615950421c7a8ac0f060ab2b224ebad2a43c3e0544e44151b21dcecf1e` |
| `reinforce_gae.py` | 12,600 | `02e66ad200960d85c8dd4c70d5a2e0b5e40b91978d393067c9a3d981126f44d6` |

The ZIP's GAE source uses ReLU activations and the custom `+120` EWMA threshold,
matching the report and historical logs. A separate loose working-tree copy at
the pre-curation commit had diverged to `tanh` and the environment threshold;
the project README records why that unsubmitted variant was omitted.

## Submitted CartPole checkpoint defect

The submitted `CartPole_0.008.pth` is a valid PyTorch ZIP checkpoint plus 13
unrelated files accidentally nested under `Assignment/`. Those 13 records are
byte-identical duplicates of every other ZIP member. The retained portfolio
checkpoint removes only those unrelated records; all ten genuine PyTorch
checkpoint records are byte-identical to the submitted file.

The cleaned derivative has SHA-256
`932cff303ffd2556fb3ce70124feb5a3b3291b72372ce3deb9829e654c11bbf0`
and is stored at `results/checkpoints/CartPole_0.008.pth`.
