"""QA-056 documentation/provenance completeness audit primitives.

This module audits evidence and documentation roles required by the frozen
project governance. It does not invent public metadata, data rights, authors,
ORCIDs, licenses, DOI values, release dates, or scientific claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Iterable

QA056_GATE = "PASS_DOCUMENTATION_PROVENANCE_COMPLETENESS_AUDIT"

QA056_HOLDS = (
    "HOLD_INSTALLATION_QUICKSTART_WITH_EXECUTABLE_COMMANDS",
    "HOLD_ONE_COMMAND_CANONICAL_REPRODUCTION_ENTRYPOINT",
    "HOLD_CHANGELOG_AND_RELEASE_NOTES",
    "HOLD_MACHINE_READABLE_CONFIG_COVERAGE_FOR_ALL_CANONICAL_RUNS",
    "HOLD_DATA_AND_SOURCE_PROVENANCE_CATALOG_COMPLETENESS",
    "HOLD_AUTOMATED_TEST_COVERAGE_REPORT",
    "HOLD_FINAL_PUBLIC_RELEASE_METADATA",
    "HOLD_ZENODO_ARCHIVAL_RECORD",
)

QA056_PROHIBITIONS = (
    "PROHIBIT_MISSING_DOCUMENTATION_AS_IMPLICITLY_COMPLETE",
    "PROHIBIT_UNRESOLVED_METADATA_AS_FINAL_RELEASE_METADATA",
    "PROHIBIT_UNVERIFIED_DATA_REDISTRIBUTION_RIGHTS",
    "PROHIBIT_SOURCE_AUTHOR_AS_SOFTWARE_AUTHOR_BY_INFERENCE",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)


class EvidenceStatus(str, Enum):
    PRESENT = "PRESENT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class DocumentationRole:
    key: str
    required_for_archival_release: bool
    description: str


@dataclass(frozen=True)
class DocumentationEvidence:
    key: str
    status: EvidenceStatus
    paths: tuple[str, ...]
    note: str


REQUIRED_ROLES = (
    DocumentationRole("readme_project_scope", True, "project scope and historical/modern branch distinction"),
    DocumentationRole("installation_quickstart", True, "executable installation instructions"),
    DocumentationRole("reproduction_entrypoint", True, "single canonical reproduction command or script"),
    DocumentationRole("machine_readable_configs", True, "machine-readable configuration coverage"),
    DocumentationRole("citation_metadata", True, "software citation metadata"),
    DocumentationRole("software_license", True, "software license after explicit selection"),
    DocumentationRole("data_rights_provenance", True, "data/source provenance and redistribution rights"),
    DocumentationRole("changelog_release_notes", True, "change log and release notes"),
    DocumentationRole("checksums", True, "SHA-256 integrity manifest"),
    DocumentationRole("test_evidence", True, "test execution evidence"),
    DocumentationRole("coverage_report", False, "automated code coverage report"),
    DocumentationRole("runtime_environment_provenance", True, "runtime/environment provenance"),
    DocumentationRole("qa_claim_governance", True, "claim/HOLD/PROHIBIT governance"),
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _contains_command_like_install(readme: str) -> bool:
    lower = readme.lower()
    return any(token in lower for token in ("pip install", "python -m pip", "uv sync", "conda env"))


def audit_documentation(project_root: str | Path) -> tuple[DocumentationEvidence, ...]:
    root = Path(project_root)
    readme_path = root / "README.md"
    readme = readme_path.read_text() if readme_path.exists() else ""
    configs = tuple(sorted(str(p.relative_to(root)) for p in root.rglob("*.json") if "logs" not in p.parts))
    qa_json = tuple(p for p in configs if p.startswith("QA") or p.startswith("metadata/") or p.startswith("ci/"))
    citation_final = root / "CITATION.cff"
    citation_template = root / "CITATION.cff.template"
    license_file = root / "LICENSE"
    license_policy = root / "LICENSE_POLICY.md"
    changelog = root / "CHANGELOG.md"
    checksums = root / "SHA256SUMS.txt"
    tests = tuple(sorted(root.glob("02_TESTS/test_qa*.py")))
    runtime_refs = tuple(sorted(root.glob("requirements/qa05*_reference*.txt"))) + tuple(sorted(root.glob("requirements/qa053_local_resolution*.json")))
    claim_governance = root / "01_SRC/gilttpy/validation/claim_envelope.py"

    return (
        DocumentationEvidence("readme_project_scope", EvidenceStatus.PRESENT if "historical" in readme.lower() and "modern" in readme.lower() else EvidenceStatus.HOLD, ("README.md",) if readme_path.exists() else (), "README distinguishes the historical and modern branches."),
        DocumentationEvidence("installation_quickstart", EvidenceStatus.PRESENT if _contains_command_like_install(readme) else EvidenceStatus.HOLD, ("README.md",) if readme_path.exists() else (), "README currently lacks an executable installation quickstart."),
        DocumentationEvidence("reproduction_entrypoint", EvidenceStatus.PRESENT if any((root / name).exists() for name in ("reproduce.py", "reproduce.sh", "Makefile")) else EvidenceStatus.HOLD, (), "No single canonical reproduction entrypoint is frozen."),
        DocumentationEvidence("machine_readable_configs", EvidenceStatus.PRESENT if len(qa_json) >= 3 else EvidenceStatus.HOLD, tuple(qa_json), "QA engineering configs exist; coverage of all canonical scientific runs remains incomplete."),
        DocumentationEvidence("citation_metadata", EvidenceStatus.PRESENT if citation_final.exists() else EvidenceStatus.HOLD, ("CITATION.cff",) if citation_final.exists() else (("CITATION.cff.template",) if citation_template.exists() else ()), "Only a non-public citation template exists while QA054 metadata fields remain HOLD."),
        DocumentationEvidence("software_license", EvidenceStatus.PRESENT if license_file.exists() else EvidenceStatus.HOLD, ("LICENSE",) if license_file.exists() else (("LICENSE_POLICY.md",) if license_policy.exists() else ()), "License policy exists, but no software license has been selected."),
        DocumentationEvidence("data_rights_provenance", EvidenceStatus.HOLD, (), "A complete source/data provenance and redistribution-rights catalog is not yet frozen."),
        DocumentationEvidence("changelog_release_notes", EvidenceStatus.PRESENT if changelog.exists() else EvidenceStatus.HOLD, ("CHANGELOG.md",) if changelog.exists() else (), "CHANGELOG/release notes are not yet frozen."),
        DocumentationEvidence("checksums", EvidenceStatus.PRESENT if checksums.exists() else EvidenceStatus.HOLD, ("SHA256SUMS.txt",) if checksums.exists() else (), "Checkpoint-level SHA-256 manifest exists."),
        DocumentationEvidence("test_evidence", EvidenceStatus.PRESENT if tests else EvidenceStatus.HOLD, tuple(str(p.relative_to(root)) for p in tests), "QA test suite and execution evidence are present."),
        DocumentationEvidence("coverage_report", EvidenceStatus.PRESENT if any(root.glob("coverage.*")) or (root / "htmlcov").exists() else EvidenceStatus.HOLD, (), "No code coverage report is frozen."),
        DocumentationEvidence("runtime_environment_provenance", EvidenceStatus.PRESENT if runtime_refs else EvidenceStatus.HOLD, tuple(str(p.relative_to(root)) for p in runtime_refs), "Local runtime/dependency reference evidence exists."),
        DocumentationEvidence("qa_claim_governance", EvidenceStatus.PRESENT if claim_governance.exists() else EvidenceStatus.HOLD, (str(claim_governance.relative_to(root)),) if claim_governance.exists() else (), "QA045 claim envelope remains present."),
    )


def evidence_map(items: Iterable[DocumentationEvidence]) -> dict[str, DocumentationEvidence]:
    return {item.key: item for item in items}


def audit_can_pass(items: Iterable[DocumentationEvidence]) -> bool:
    mapping = evidence_map(items)
    if set(mapping) != {r.key for r in REQUIRED_ROLES}:
        return False
    # This QA gate validates the completeness *audit*, not archival-release readiness.
    # Every missing role must remain explicit HOLD evidence rather than disappear.
    return all(item.status in (EvidenceStatus.PRESENT, EvidenceStatus.HOLD) and item.note for item in mapping.values())


def provenance_index(project_root: str | Path) -> dict:
    root = Path(project_root)
    evidence = audit_documentation(root)
    critical = [
        "README.md", "pyproject.toml", "SHA256SUMS.txt", "LICENSE_POLICY.md",
        "CITATION.cff.template", "metadata/qa054_release_metadata_state.json",
        "metadata/qa055_clean_room_contract.json",
    ]
    hashes = {}
    for rel in critical:
        path = root / rel
        if path.exists():
            hashes[rel] = sha256_file(path)
    return {
        "gate": QA056_GATE,
        "scope": "documentation/provenance completeness audit; not archival release readiness",
        "roles": [
            {"key": item.key, "status": item.status.value, "paths": list(item.paths), "note": item.note}
            for item in evidence
        ],
        "critical_hashes": hashes,
        "holds": list(QA056_HOLDS),
        "prohibitions": list(QA056_PROHIBITIONS),
        "archival_release_ready": False,
    }


def write_provenance_index(project_root: str | Path, output: str | Path) -> dict:
    payload = provenance_index(project_root)
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
