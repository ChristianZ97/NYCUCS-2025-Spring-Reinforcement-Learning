# Curation validation

This record separates executed behavior from static and provenance checks. The
private `legacy` branch points to the exact curation baseline
`14c78350cc80c564ce9b1f7abedb4c44075c0626`.

## Executed checks

- Under Python 3.8 with Gym 0.23.1 and PyTorch 2.2.2, all five retained HW1
  checkpoints loaded strictly on CPU and every tensor was finite.
- The retained CartPole policy completed ten headless evaluation episodes at
  return 200 each. Gym imported through an in-memory pygame shim because
  pygame was absent; the headless path exercised no pygame API.
- A one-step CartPole training run exercised rollout, loss, optimizer,
  submitted pre-update scheduler placement, temporary checkpoint writing, and
  cleanup without creating repository output. PyTorch emitted its expected
  warning about the historical scheduler order.
- A hand-computed two-step GAE oracle matched the canonical implementation, and
  empty or mismatched trajectories were rejected.
- The retained HW2 Pendulum actor and critic loaded strictly, produced finite
  forward outputs, and respected the action bound. Mock environment probes
  confirmed that both DDPG collectors reset on termination or truncation but
  bootstrap through time-limit truncation.
- Dependency-free probes exercised numeric validators, seed bounds, derived
  checkpoint names, Final Project label schemas and action extraction, HW3
  transition storage, Final Project precision selection, and RLOO output
  collision protection.

## Static, documentation, and artifact checks

- All 17 canonical Python files pass Ruff, formatting checks, AST parsing, and
  compilation with the local interpreter. HW1 additionally passes Python 3.8
  grammar parsing.
- All CSV and JSON files parse, have consistent row shapes, and retain their
  documented historical/result provenance. Every local Markdown link resolves.
- The HW1 dependency set resolves under a Python 3.8 dry run. HW2, HW3, and the
  Final Project dependency sets resolve for Python 3.11 without installation.
- Raw Git-object bytes at the baseline verified the SHA-256 hashes of 20
  private grading/course artifacts. CRC and documented uncompressed hashes
  verified 26 grading-archive members.
- Fourteen retained result files are byte-identical to their baseline sources.
  The cleaned CartPole checkpoint retains all ten genuine PyTorch records
  byte-for-byte and removes only 13 unrelated files accidentally embedded in
  the submitted container.
- All seven retained PyTorch ZIP containers pass CRC and path-safety checks.
  The three retained PNG figures have valid headers and were visually
  inspected.
- During the original curation, portfolio filenames, source, checkpoint
  metadata, and result artifacts were scanned for identifiers, email addresses,
  credentials, private keys, workstation names, and private paths. No
  configured sensitive pattern remained in that sanitized surface. The exact
  reports and presentations added on 2026-09-05 are an explicit, documented
  exception and intentionally retain their submitted identity metadata.
- Independent read-only reviewers covered every canonical project. Their
  findings were fixed and narrowly re-reviewed before staging.

## Runtime and publication limits

- LunarLander, D4RL, and legacy MuJoCo execution remain unavailable because
  Box2D, datasets, and the required legacy runtime are absent.
- HW2 and HW3 were not trained or evaluated in real Gymnasium/MuJoCo
  environments. Their canonical sources received static, checkpoint, oracle,
  and mocked boundary-path validation.
- The Final Project was not executed end-to-end: its private BDD-OIA data,
  reward/VLM checkpoints, model downloads, and suitable CUDA runtime are not
  available. Its source received static, schema, lineage, and device-selection
  review.
- Historical figures and tables were not regenerated from the cleaned source
  and are labeled accordingly.
- This public release combines sanitized portfolio source with the exact
  student-authored documents inventoried in [`SUBMISSIONS.md`](SUBMISSIONS.md).
  The `legacy` branch remains on the separate private source remote and is not
  published here. Collaborator and course-staff permission for this portfolio
  publication was confirmed before release.
