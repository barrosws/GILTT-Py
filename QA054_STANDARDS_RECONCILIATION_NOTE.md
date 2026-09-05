# QA-054 standards reconciliation note

Two QA-054 staging branches existed before freeze. The more complete scientific-metadata/citation/licensing stage was selected as the canonical candidate because it includes an explicit citation template, license-policy document and machine-readable metadata-state file.

Before final regression, one useful standards detail from the alternate staging branch was incorporated: current PyPA guidance records setuptools 77.0.3 as the first setuptools release supporting the PEP 639 `license` string and `license-files` syntax. This change affects metadata governance only. It does not select a license, change `pyproject.toml`, modify scientific code, change tests outside QA-054, or alter any scientific claim.

The alternate stage is retained as non-canonical development provenance and is not a frozen release artifact.
