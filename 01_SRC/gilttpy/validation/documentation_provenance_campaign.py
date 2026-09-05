"""QA-056 documentation/provenance completeness campaign."""
from __future__ import annotations
from pathlib import Path
from gilttpy.engineering.provenance_catalog import audit_documentation, provenance_index


def qa056_documentation_evidence(project_root: str | Path) -> dict:
    root = Path(project_root)
    index = provenance_index(root)
    roles = {item["key"]: item for item in index["roles"]}
    return {
        "gate": index["gate"],
        "present_roles": sorted(k for k, v in roles.items() if v["status"] == "PRESENT"),
        "hold_roles": sorted(k for k, v in roles.items() if v["status"] == "HOLD"),
        "role_count": len(roles),
        "critical_hash_count": len(index["critical_hashes"]),
        "archival_release_ready": index["archival_release_ready"],
        "holds": index["holds"],
        "prohibitions": index["prohibitions"],
    }
