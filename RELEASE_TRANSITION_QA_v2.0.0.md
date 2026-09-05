# GILTT-Py 2.0.0 — release-transition QA contract

The frozen QA-058 tree contains 396 pre-release tests. Six of those tests deliberately assert that the package is still `0.0.0.dev1` or that software licensing remains unresolved. After explicit approval of version `2.0.0` and `BSD-3-Clause`, those six assertions are historical provenance, not valid release-state assertions.

They are not edited or deleted. The release CI deselects exactly these six node IDs and adds exactly six STEP F2 tests under `03_RELEASE_TESTS/`, preserving a 396-test release-transition suite.

Deselected historical assertions:

1. `test_qa050_02_governed_tree_matches_pyproject`
2. `test_qa054_02_known_project_metadata_matches_pyproject`
3. `test_qa054_06_pyproject_does_not_guess_license`
4. `test_qa054_10_metadata_evidence_reports_hold_not_release_ready`
5. `test_qa056_07_citation_and_license_remain_qa054_holds_not_fabricated`
6. `test_qa058_05_public_version_authorship_license_and_citation_remain_hold`

The replacement STEP F2 tests verify: release version/backend activation; PEP 639 BSD-3-Clause metadata and license-file inclusion; non-inference of software authorship/ORCIDs/DOIs; non-redistribution of raw Copenhagen/Hanford data; exact 12-cell CI declaration; and continued non-promotion of repository/CI/Zenodo evidence before real platform readback.

This transition changes release metadata and licensing only. It does not modify `01_SRC/` or `02_TESTS/`.


## STEP F6 public-metadata transition

After explicit review of the GILTT-Py 2.0 software-author list, the active
`CITATION.cff` and `AUTHORS.md` are no longer inferred or placeholder metadata.
They are reviewed release-transition metadata. The QA-054/055 assertions that
require `CITATION.cff` to be absent, the QA-054 assertion that requires
`pyproject.toml` software authors to be absent, and the STEP F2 assertion that
requires active citation metadata to remain absent are therefore historical
pre-F6 provenance.

Those four assertions remain unchanged in their historical files and are
deselected by node ID in release CI:

1. `test_qa054_05_template_placeholders_block_public_release_readiness`
2. `test_qa054_07_pyproject_does_not_guess_software_authors`
3. `test_qa055_11_unresolved_qa054_metadata_still_blocks_public_release`
4. `test_authorship_orcid_and_doi_are_not_inferred`

Four STEP F6 replacement tests under
`03_RELEASE_TESTS/test_step_f6_public_metadata.py` verify the reviewed author
order, active placeholder-free CFF metadata, preservation of the historical F2
non-inference state, explicit omission of ORCID/DOI until separately resolved,
and correct citation/DOI HOLD semantics.

The release-transition suite remains exactly 396 selected tests:
408 collected, 12 historical/runtime-local assertions deselected.

This transition changes public release metadata and packaging only. It does not
change any scientific equation, physical closure, numerical tolerance,
benchmark target, uncertainty design, claim envelope, or NO TARGET TUNING rule.
