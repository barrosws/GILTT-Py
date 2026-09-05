"""QA-058 final archival-release-readiness audit primitives.

This layer audits release readiness without inventing authorship, ORCIDs,
licenses, data rights, DOI values, public versions, Zenodo records, or
scientific/manuscript provenance that has not been frozen.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Iterable

QA058_GATE = "PASS_ARCHIVAL_RELEASE_READINESS_AUDIT"
QA058_HOLDS = (
    "HOLD_PUBLIC_SOFTWARE_VERSION",
    "HOLD_SOFTWARE_AUTHORSHIP_AND_ORCID_REVIEW",
    "HOLD_SOFTWARE_LICENSE_SELECTION",
    "HOLD_DATA_REDISTRIBUTION_RIGHTS_AUDIT",
    "HOLD_FINAL_CITATION_CFF_AND_DOI",
    "HOLD_CANONICAL_DATA_RELEASE",
    "HOLD_MANUSCRIPT_ASSOCIATED_OUTPUT_FREEZE",
    "HOLD_MACHINE_READABLE_CONFIG_COVERAGE_FOR_ALL_CANONICAL_SCIENTIFIC_RUNS",
    "HOLD_FULL_PACKAGE_COVERAGE_REPORT",
    "HOLD_EXTERNAL_CI_MATRIX_EXECUTION",
    "HOLD_CROSS_PLATFORM_HERMETIC_LOCK",
    "HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION",
    "HOLD_ZENODO_DEPOSITION",
)
QA058_PROHIBITIONS = (
    "PROHIBIT_AUDIT_PASS_AS_ARCHIVAL_RELEASE_READY",
    "PROHIBIT_GUESSED_AUTHORSHIP_OR_ORCID",
    "PROHIBIT_UNREVIEWED_LICENSE_SELECTION",
    "PROHIBIT_UNVERIFIED_DATA_REDISTRIBUTION_RIGHTS",
    "PROHIBIT_PLACEHOLDER_CITATION_AS_FINAL_METADATA",
    "PROHIBIT_QA_REPLAY_AS_MANUSCRIPT_OUTPUT_REPRODUCTION_PROOF",
    "PROHIBIT_LOCAL_REPLAY_AS_CROSS_PLATFORM_PROOF",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

class ReadinessStatus(str, Enum):
    READY = "READY"
    HOLD = "HOLD"

@dataclass(frozen=True)
class ArchivalRole:
    key: str
    required_for_archival_release: bool
    description: str

@dataclass(frozen=True)
class ArchivalEvidence:
    key: str
    status: ReadinessStatus
    paths: tuple[str, ...]
    note: str

REQUIRED_ARCHIVAL_ROLES = (
    ArchivalRole("distribution_reproducibility", True, "deterministic wheel/sdist and exact package replay evidence"),
    ArchivalRole("operational_qa_replay", True, "single governed QA replay command and machine-readable plan"),
    ArchivalRole("integrity_manifest", True, "SHA-256 manifest/checksum evidence"),
    ArchivalRole("runtime_environment_provenance", True, "runtime and dependency provenance"),
    ArchivalRole("installation_reproduction_docs", True, "installation and reproduction instructions"),
    ArchivalRole("development_change_history", True, "development changelog and release notes"),
    ArchivalRole("public_software_version", True, "reviewed public semantic software version"),
    ArchivalRole("software_authorship_orcids", True, "reviewed software creators/contributors and ORCIDs where applicable"),
    ArchivalRole("software_license", True, "explicit reviewed software license"),
    ArchivalRole("citation_doi_metadata", True, "final citation metadata and DOI/repository identifiers"),
    ArchivalRole("canonical_data_release", True, "frozen canonical data release"),
    ArchivalRole("data_rights_provenance", True, "source/data provenance and redistribution rights"),
    ArchivalRole("scientific_config_coverage", True, "machine-readable configs for all manuscript-associated scientific runs"),
    ArchivalRole("manuscript_output_reproduction", True, "regeneration of manuscript tables/figures from frozen release"),
    ArchivalRole("full_package_coverage", False, "automated full-package code coverage evidence"),
    ArchivalRole("external_ci_matrix", True, "executed supported Python/platform CI evidence"),
    ArchivalRole("cross_platform_hermetic_lock", True, "target-specific hash-locked dependency evidence"),
    ArchivalRole("zenodo_archival_record", True, "final archival deposition and persistent identifier"),
)

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def _project_version(root: Path) -> str:
    with (root/"pyproject.toml").open("rb") as f:
        return str(tomllib.load(f)["project"]["version"])

def audit_archival_readiness(project_root: str | Path) -> tuple[ArchivalEvidence, ...]:
    root=Path(project_root)
    def has(rel: str) -> bool: return (root/rel).exists()
    qa057_build=has("QA057_BUILD_REPRODUCIBILITY.json")
    qa057_source=has("QA057_SOURCE_ONE_COMMAND_REPLAY.json")
    plan=has("reproduction/qa057_qa_replay.toml")
    install=has("docs/INSTALLATION_QUICKSTART.md")
    repro=has("docs/REPRODUCIBILITY.md")
    change=has("docs/CHANGELOG.md") and has("docs/RELEASE_NOTES_DEV.md")
    checks=has("SHA256SUMS.txt") and has("manifest.json")
    runtime=has("requirements/qa053_local_resolution_py313_linux_x86_64.json")
    final_cff=has("CITATION.cff")
    license_file=has("LICENSE")
    version=_project_version(root)
    coverage=has("coverage/qa057_coverage.json")
    return (
        ArchivalEvidence("distribution_reproducibility", ReadinessStatus.READY if qa057_build else ReadinessStatus.HOLD, ("QA057_BUILD_REPRODUCIBILITY.json",) if qa057_build else (), "QA057 deterministic distribution evidence is frozen locally."),
        ArchivalEvidence("operational_qa_replay", ReadinessStatus.READY if qa057_source and plan else ReadinessStatus.HOLD, tuple(x for x in ("QA057_SOURCE_ONE_COMMAND_REPLAY.json","reproduction/qa057_qa_replay.toml") if has(x)), "QA057 one-command QA replay is frozen; this is not manuscript-output reproduction."),
        ArchivalEvidence("integrity_manifest", ReadinessStatus.READY if checks else ReadinessStatus.HOLD, tuple(x for x in ("manifest.json","SHA256SUMS.txt") if has(x)), "Checkpoint integrity manifests are present."),
        ArchivalEvidence("runtime_environment_provenance", ReadinessStatus.READY if runtime else ReadinessStatus.HOLD, ("requirements/qa053_local_resolution_py313_linux_x86_64.json",) if runtime else (), "Executed local runtime/dependency provenance is present; cross-platform lock remains separate HOLD."),
        ArchivalEvidence("installation_reproduction_docs", ReadinessStatus.READY if install and repro else ReadinessStatus.HOLD, tuple(x for x in ("docs/INSTALLATION_QUICKSTART.md","docs/REPRODUCIBILITY.md") if has(x)), "Operational installation/reproduction instructions are present."),
        ArchivalEvidence("development_change_history", ReadinessStatus.READY if change else ReadinessStatus.HOLD, tuple(x for x in ("docs/CHANGELOG.md","docs/RELEASE_NOTES_DEV.md") if has(x)), "Development change history is present but is not a public release."),
        ArchivalEvidence("public_software_version", ReadinessStatus.HOLD, ("pyproject.toml",), f"Current version is {version}; public release version remains an explicit decision."),
        ArchivalEvidence("software_authorship_orcids", ReadinessStatus.HOLD, ("metadata/qa054_release_metadata_state.json",) if has("metadata/qa054_release_metadata_state.json") else (), "Software authorship and ORCIDs remain under explicit review and may not be inferred."),
        ArchivalEvidence("software_license", ReadinessStatus.READY if license_file else ReadinessStatus.HOLD, ("LICENSE",) if license_file else (("LICENSE_POLICY.md",) if has("LICENSE_POLICY.md") else ()), "License policy exists; no software license has been selected."),
        ArchivalEvidence("citation_doi_metadata", ReadinessStatus.READY if final_cff else ReadinessStatus.HOLD, ("CITATION.cff",) if final_cff else (("CITATION.cff.template",) if has("CITATION.cff.template") else ()), "Only a template exists; DOI and final citation metadata remain HOLD."),
        ArchivalEvidence("canonical_data_release", ReadinessStatus.HOLD, (), "No canonical public data release is frozen by QA058."),
        ArchivalEvidence("data_rights_provenance", ReadinessStatus.HOLD, ("LICENSE_POLICY.md",) if has("LICENSE_POLICY.md") else (), "Complete data/source provenance and redistribution-rights adjudication remains HOLD."),
        ArchivalEvidence("scientific_config_coverage", ReadinessStatus.HOLD, ("reproduction/qa057_qa_replay.toml",) if plan else (), "QA replay config exists; complete configs for manuscript-associated scientific runs remain HOLD."),
        ArchivalEvidence("manuscript_output_reproduction", ReadinessStatus.HOLD, (), "No frozen end-to-end manuscript table/figure regeneration proof exists yet."),
        ArchivalEvidence("full_package_coverage", ReadinessStatus.HOLD, ("coverage/qa057_coverage.json",) if coverage else (), "Only QA057 operational-layer coverage is frozen; full-package coverage remains HOLD."),
        ArchivalEvidence("external_ci_matrix", ReadinessStatus.HOLD, (), "Unexecuted Python/platform cells may not be promoted to PASS."),
        ArchivalEvidence("cross_platform_hermetic_lock", ReadinessStatus.HOLD, (), "Local dependency resolution is not a universal cross-platform hermetic lock."),
        ArchivalEvidence("zenodo_archival_record", ReadinessStatus.HOLD, (), "Zenodo deposition is downstream of final release audit and metadata decisions."),
    )

def evidence_map(items: Iterable[ArchivalEvidence]) -> dict[str, ArchivalEvidence]:
    return {x.key:x for x in items}

def audit_can_pass(items: Iterable[ArchivalEvidence]) -> bool:
    m=evidence_map(items)
    return set(m)=={r.key for r in REQUIRED_ARCHIVAL_ROLES} and all(x.note and x.status in (ReadinessStatus.READY,ReadinessStatus.HOLD) for x in m.values())

def archival_release_ready(items: Iterable[ArchivalEvidence]) -> bool:
    m=evidence_map(items)
    required={r.key for r in REQUIRED_ARCHIVAL_ROLES if r.required_for_archival_release}
    return bool(required) and all(m[k].status is ReadinessStatus.READY for k in required)

def public_release_blockers(items: Iterable[ArchivalEvidence]) -> tuple[str, ...]:
    m=evidence_map(items)
    required={r.key for r in REQUIRED_ARCHIVAL_ROLES if r.required_for_archival_release}
    return tuple(sorted(k for k in required if m[k].status is ReadinessStatus.HOLD))

def readiness_state(project_root: str | Path) -> dict:
    root=Path(project_root)
    items=audit_archival_readiness(root)
    critical=(
        "pyproject.toml","README.md","MANIFEST.in","QA057_SCOPE_GOVERNANCE.md",
        "docs/INSTALLATION_QUICKSTART.md","docs/REPRODUCIBILITY.md","docs/CHANGELOG.md","docs/RELEASE_NOTES_DEV.md",
        "reproduction/qa057_qa_replay.toml","metadata/qa054_release_metadata_state.json",
    )
    hashes={rel:sha256_file(root/rel) for rel in critical if (root/rel).exists()}
    return {
        "gate":QA058_GATE,
        "scope":"final archival-release-readiness audit; audit PASS is not archival release readiness",
        "roles":[{"key":x.key,"status":x.status.value,"paths":list(x.paths),"note":x.note} for x in items],
        "ready_roles":sorted(x.key for x in items if x.status is ReadinessStatus.READY),
        "hold_roles":sorted(x.key for x in items if x.status is ReadinessStatus.HOLD),
        "public_release_blockers":list(public_release_blockers(items)),
        "archival_release_ready":archival_release_ready(items),
        "critical_hashes":hashes,
        "holds":list(QA058_HOLDS),
        "prohibitions":list(QA058_PROHIBITIONS),
    }

def write_readiness_state(project_root: str | Path, output: str | Path) -> dict:
    payload=readiness_state(project_root)
    Path(output).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload
