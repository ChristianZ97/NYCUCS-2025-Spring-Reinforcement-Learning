# Submission provenance

The exact grading artifacts are intentionally excluded from the portfolio
tree. The report contains personal identity and contact information, while the
source archive includes substantial course-provided starter scaffolding and
originally had an identity-bearing filename. Editing either artifact would
destroy its provenance, so both remain byte-for-byte recoverable only from the
private `legacy` branch at the exact pre-curation commit:

```text
14c78350cc80c564ce9b1f7abedb4c44075c0626
```

## Exact artifact ledger

| Private artifact | Size | SHA-256 |
| --- | ---: | --- |
| Submitted report PDF | 12,202,678 bytes | `0db39a672d9f0fdaabf4da8e5ba1b73548e7cf246e8e9cc75c4359ecc953c800` |
| Submitted source ZIP | 8,869 bytes | `af8614a22ec91a7d2f4e5cac2c46673934c3599ef3e9c858e236a9a3346461ea` |
| Course handout PDF (not a submission) | 189,799 bytes | `b48d859ea0940c6b7ba13848c47b948f83d8b8443223ed5fa677364ea1c018de` |

The grading ZIP contained exactly two regular files:

| Archive member | Size | CRC-32 | SHA-256 |
| --- | ---: | --- | --- |
| `sac.py` | 15,972 bytes | `af7767fc` | `0f77c942b194fe14f5cdf19ff00c930085bc54e003221ad797ee3fd286abfd83` |
| `sac_halfcheetah.py` | 16,739 bytes | `9f6554c0` | `1ccf511f3213ada71f8458bc9bebd7468bff88eac81b9a8298531e6a9e50531e` |

No file under this portfolio-facing `submission/` directory is represented as
an exact grading submission. The implementations in `../src/` are explicitly
cleaned portfolio derivatives.
