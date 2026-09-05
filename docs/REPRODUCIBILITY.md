# Reproducibility instructions — QA-057

## Canonical operational QA replay

The frozen command for the QA-057 development snapshot is:

```bash
python -m gilttpy.engineering.operational_reproduction --project-root . --evidence QA_REPLAY_EVIDENCE.json
```

The command reads `reproduction/qa057_qa_replay.toml`, verifies exact assignment of all QA test files, checks the expected collection count for each partition, applies the frozen thread/hash environment and executes the partitions sequentially.

The machine-readable plan contains 14 partitions and 384 expected tests. Partial partition execution is available for operational diagnostics with `--partition NAME`, but a partial replay is not a complete QA-057 reproduction.

## Scope boundary

This entrypoint reproduces the governed QA suite. It does not yet reproduce final manuscript tables/figures, because final manuscript-associated output configurations are not frozen. That remains an explicit HOLD for the next provenance/release stage.

Likewise, this document does not resolve software authorship, ORCIDs, software license, DOI, public semantic version, release date, Zenodo record, or unverified data redistribution rights.

## Coverage scope

QA-057 freezes coverage evidence for the new operational reproduction modules only. Full-package coverage remains HOLD and is not inferred from this focused report.
