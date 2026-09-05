# GILTT-Py QA-054 — Scientific Metadata, Citation and Licensing Contract

## Canonical gate

`PASS_SCIENTIFIC_METADATA_CITATION_LICENSING_CONTRACT`

## Scope

QA-054 governs release metadata, citation metadata, software-authorship review state and licensing state only. It changes no scientific equation, physical closure, numerical tolerance, benchmark target, sensitivity design, uncertainty distribution or model-form conclusion. NO TARGET TUNING remains binding.

## Inheritance

- QA-053 Python files inherited: 92
- byte-identical: 92/92
- modified inherited Python files: 0
- new QA-054 Python files: `engineering/release_metadata.py`, `validation/release_metadata_campaign.py`, `test_qa054_release_metadata.py`.

## Metadata decisions

- package remains development version `0.0.0.dev1`;
- target Citation File Format is 1.2.0;
- `CITATION.cff.template` is explicitly non-release metadata; no public `CITATION.cff` is activated;
- software authorship and ORCIDs remain HOLD and may not be inferred from authorship of source papers, dissertations or legacy code;
- software license selection remains HOLD;
- benchmark-data redistribution rights remain a separate HOLD;
- no DOI, release date, repository URL or public release version is fabricated;
- current PEP 639 activation syntax is recorded as SPDX `license` plus `license-files`; for setuptools the current PyPA support table begins at 77.0.3. The active build requirement is deliberately not upgraded by QA-054 because license metadata is not yet activated.

## Verification

- dedicated QA-054: 12/12 PASS;
- complete source regression QA029–QA054: 348/348 PASS;
- installed-wheel replay outside source tree: 348/348 PASS;
- two direct wheels: bitwise identical;
- wheel SHA-256: `23fb1ad0277762c2303ff4bb782d66d7b8e2b60baa4b7d447e633540503bcdfd`;
- two normalized sdists: bitwise identical;
- normalized sdist SHA-256: `eea625afb8c0bee4e5f2dde1bf31d0c67be1469644bbb3fcdba6a115b88d51b6`;
- wheel rebuilt from normalized sdist: exact match.

## Controlled HOLDs

- software authorship review;
- author ORCID review;
- software license selection;
- data redistribution-rights audit;
- PEP 639 backend upgrade at license activation;
- public version/date/DOI/repository metadata;
- final CFF schema validation on the reviewed release candidate;
- cross-platform CI and hermetic wheelhouse HOLDs inherited from QA-051–053 remain unaffected.

## Prohibitions

No guessed ORCID, inferred software author, unreviewed license, fabricated DOI, placeholder in public release metadata, development version presented as public release, engineering gate presented as new scientific validation, or target tuning is permitted.
