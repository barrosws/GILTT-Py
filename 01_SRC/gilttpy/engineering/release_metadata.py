"""QA-054 scientific software release-metadata governance.

This module validates metadata readiness without inventing authorship, ORCIDs,
licenses, DOI values, release dates, or repository URLs.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import tomllib

QA054_GATE = "PASS_SCIENTIFIC_METADATA_CITATION_LICENSING_CONTRACT"
CFF_SCHEMA_VERSION = "1.2.0"
PEP639_MIN_SETUPTOOLS = "77.0.3"
CORE_METADATA_LICENSE_VERSION = "2.4"
QA054_HOLDS = (
    "HOLD_SOFTWARE_AUTHORSHIP_REVIEW",
    "HOLD_AUTHOR_ORCID_REVIEW",
    "HOLD_SOFTWARE_LICENSE_SELECTION",
    "HOLD_DATA_REDISTRIBUTION_RIGHTS_AUDIT",
    "HOLD_PEP639_BACKEND_UPGRADE_UNTIL_LICENSE_ACTIVATION",
    "HOLD_RELEASE_DATE_VERSION_AND_DOI_TO_QA055",
    "HOLD_FINAL_CITATION_CFF_SCHEMA_VALIDATION_TO_RELEASE_CANDIDATE",
)
QA054_PROHIBITIONS = (
    "PROHIBIT_SOURCE_AUTHOR_AS_SOFTWARE_AUTHOR_BY_INFERENCE",
    "PROHIBIT_GUESSED_ORCID",
    "PROHIBIT_UNREVIEWED_LICENSE_SELECTION",
    "PROHIBIT_DATA_LICENSE_AS_SOFTWARE_LICENSE_BY_DEFAULT",
    "PROHIBIT_TEMPLATE_PLACEHOLDER_IN_PUBLIC_RELEASE_METADATA",
    "PROHIBIT_PLACEHOLDER_DOI_AS_REAL_IDENTIFIER",
    "PROHIBIT_DEVELOPMENT_VERSION_AS_PUBLIC_RELEASE",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

class MetadataStatus(str, Enum):
    KNOWN = "KNOWN"
    HOLD = "HOLD"

@dataclass(frozen=True)
class MetadataField:
    name: str
    status: MetadataStatus
    value: str | None = None

RELEASE_CRITICAL_FIELDS = (
    "software_authorship",
    "author_orcids",
    "software_license_spdx",
    "release_version",
    "release_date",
)

def load_state(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())

def load_project_metadata(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)["project"]

def unresolved_fields(state: dict) -> tuple[str, ...]:
    return tuple(sorted(k for k, v in state["fields"].items() if v["status"] == "HOLD"))

def public_release_ready(state: dict) -> bool:
    fields = state["fields"]
    for name in RELEASE_CRITICAL_FIELDS:
        item = fields.get(name)
        if item is None or item.get("status") != "KNOWN" or not item.get("value"):
            return False
    return True

def citation_template_has_placeholder(path: str | Path) -> bool:
    text = Path(path).read_text()
    return any(x in text for x in ("TO_REVIEW", "TO_SELECT", "TO_RELEASE"))

def assert_no_public_citation_with_placeholders(project_root: str | Path) -> None:
    root = Path(project_root)
    final = root / "CITATION.cff"
    if final.exists() and citation_template_has_placeholder(final):
        raise ValueError("public CITATION.cff contains unresolved placeholders")
