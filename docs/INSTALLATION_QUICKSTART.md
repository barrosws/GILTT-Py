# Installation quickstart — QA-057 development snapshot

This quickstart is for the governed development snapshot. It does not imply a public release or cross-platform CI PASS.

## 1. Create an isolated environment

```bash
python -m venv .venv
```

Activate it using the command appropriate for your shell/operating system.

## 2. Install the project and test dependencies

From the extracted project root:

```bash
pip3 install ".[test]"
```

For an offline engineering replay in an environment where the declared dependencies are already installed, the frozen local QA workflow may instead build/install without dependency resolution; this is not a universal installation recommendation.

## 3. Verify the QA replay plan without executing tests

```bash
python -m gilttpy.engineering.operational_reproduction --project-root . --dry-run
```

## 4. Execute the canonical governed QA replay

```bash
python -m gilttpy.engineering.operational_reproduction --project-root . --evidence QA_REPLAY_EVIDENCE.json
```

The QA-057 plan expects 384 tests. A successful local replay is evidence only for the executed environment; it is not cross-platform proof and does not reproduce manuscript tables/figures.
