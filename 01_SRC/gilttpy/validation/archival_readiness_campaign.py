"""QA-058 archival-release-readiness audit campaign."""
from __future__ import annotations
from pathlib import Path
from gilttpy.engineering.archival_readiness import readiness_state

def qa058_archival_readiness_evidence(project_root: str | Path) -> dict:
    s=readiness_state(project_root)
    return {
        "gate":s["gate"],
        "role_count":len(s["roles"]),
        "ready_role_count":len(s["ready_roles"]),
        "hold_role_count":len(s["hold_roles"]),
        "ready_roles":s["ready_roles"],
        "hold_roles":s["hold_roles"],
        "public_release_blockers":s["public_release_blockers"],
        "archival_release_ready":s["archival_release_ready"],
        "critical_hash_count":len(s["critical_hashes"]),
        "holds":s["holds"],
        "prohibitions":s["prohibitions"],
    }
