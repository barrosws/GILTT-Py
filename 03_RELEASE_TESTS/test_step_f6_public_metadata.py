from pathlib import Path
import json
import tomllib

from gilttpy.engineering.archival_readiness import (
    ReadinessStatus,
    audit_archival_readiness,
    evidence_map,
    public_release_blockers,
)
from gilttpy.engineering.release_metadata import (
    assert_no_public_citation_with_placeholders,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUTHORS = [
    "Jorge Luís Braga Ribes",
    "Maicon Nardino",
    "Régis Sperotto de Quadros",
    "Daniela Buske",
    "Luís Carlos Timm",
    "Willian Silva Barros",
]


def _project():
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def _f6_state():
    return json.loads(
        (ROOT / "metadata/step_f6_public_metadata_state.json").read_text(
            encoding="utf-8"
        )
    )


def test_step_f6_01_reviewed_software_authorship_is_explicit_not_inferred():
    project = _project()
    assert [a["name"] for a in project["authors"]] == EXPECTED_AUTHORS
    state = _f6_state()
    assert state["software_authorship"] == EXPECTED_AUTHORS
    assert state["software_authorship_reviewed"] is True

    # Historical F2 provenance remains untouched: authorship was not inferred there.
    f2 = json.loads(
        (ROOT / "metadata/step_f2_release_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert f2["software_authorship"] is None
    assert f2["author_orcids"] is None


def test_step_f6_02_active_citation_is_placeholder_free_and_consistent():
    final = ROOT / "CITATION.cff"
    template = ROOT / "CITATION.cff.template"
    assert final.exists() and template.exists()
    assert_no_public_citation_with_placeholders(ROOT)

    text = final.read_text(encoding="utf-8")
    for token in (
        "cff-version: 1.2.0",
        'title: "GILTT-Py 2.0"',
        "type: software",
        "version: 2.0.0",
        "license: BSD-3-Clause",
        'repository-code: "https://github.com/barrosws/GILTT-Py"',
    ):
        assert token in text
    for name in (
        "Jorge Luís", "Braga Ribes", "Maicon", "Nardino", "Régis",
        "Sperotto de Quadros", "Daniela", "Buske", "Luís Carlos", "Timm",
        "Willian Silva", "Barros",
    ):
        assert name in text
    assert "orcid:" not in text.lower()
    assert "\ndoi:" not in text.lower()
    assert "date-released:" not in text.lower()
    assert not any(x in text for x in ("TO_REVIEW", "TO_SELECT", "TO_RELEASE"))

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include CITATION.cff\n" in manifest
    assert "include AUTHORS.md\n" in manifest
    assert "include CITATION.cff.template" not in manifest


def test_step_f6_03_release_state_records_real_evidence_without_identifier_inference():
    state = _f6_state()
    assert state["release_version"] == "2.0.0"
    assert state["software_license_spdx"] == "BSD-3-Clause"
    assert state["repository_created"] is True
    assert state["repository_url"] == "https://github.com/barrosws/GILTT-Py"
    assert state["canonical_main_ci_required_before_tag"] is True
    assert state["ci_evidence_policy"].startswith("EXTERNAL_READBACK_FOR_EXACT_RELEASE_SHA")
    pre = state["pre_merge_ci_evidence"]
    assert pre["branch"] == "step-f6b1-windows-utf8-metadata"
    assert pre["commit_sha"] == "8c805ccb0e9a0df3ef1056886403997aef9728ea"
    assert pre["release_ci_run_id"] == 33980580933
    assert pre["release_ci_conclusion"] == "success"
    assert pre["citation_metadata"] == "PASS"
    assert pre["matrix"] == "12/12 PASS"
    assert state["author_orcids"] is None
    assert state["zenodo_version_doi"] is None
    assert state["zenodo_concept_doi"] is None
    assert state["release_date"] is None


def test_step_f6_04_active_cff_does_not_falsely_resolve_doi_archival_role():
    m = evidence_map(audit_archival_readiness(ROOT))
    assert m["public_software_version"].status is ReadinessStatus.READY
    assert m["software_authorship_orcids"].status is ReadinessStatus.READY
    assert m["software_license"].status is ReadinessStatus.READY
    assert m["citation_doi_metadata"].status is ReadinessStatus.HOLD
    assert m["external_ci_matrix"].status is ReadinessStatus.HOLD
    assert "CITATION.cff" in m["citation_doi_metadata"].paths
    blockers = public_release_blockers(m.values())
    assert "citation_doi_metadata" in blockers
    assert "external_ci_matrix" in blockers
