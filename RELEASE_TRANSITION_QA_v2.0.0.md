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
