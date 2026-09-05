"""QA-053 dependency-lock and environment-resolution governance."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib, json
from pathlib import Path

QA053_GATE = "PASS_DEPENDENCY_LOCK_AND_ENVIRONMENT_RESOLUTION_CONTRACT"
QA053_HOLDS = (
    "HOLD_CROSS_PLATFORM_HASH_LOCK_GENERATION",
    "HOLD_OFFLINE_WHEELHOUSE_COMPLETENESS",
    "HOLD_MINIMUM_SUPPORTED_DEPENDENCY_EXECUTION",
    "HOLD_EXTERNAL_CI_LOCK_REPLAY",
)
QA053_PROHIBITIONS = (
    "PROHIBIT_PIP_FREEZE_AS_UNIVERSAL_CROSS_PLATFORM_LOCK",
    "PROHIBIT_UNHASHED_REQUIREMENT_AS_HERMETIC_PROOF",
    "PROHIBIT_HOST_EXTRANEOUS_PACKAGES_AS_RUNTIME_REQUIREMENTS",
    "PROHIBIT_UNEXECUTED_LOCK_TARGET_AS_PASS",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

class LockEvidence(str, Enum):
    LOCAL_RESOLUTION = "LOCAL_RESOLUTION"
    HERMETIC_HASH_LOCK = "HERMETIC_HASH_LOCK"
    CI_REQUIRED = "CI_REQUIRED"

@dataclass(frozen=True)
class ResolvedPackage:
    name: str
    version: str

def canonical_resolution(packages):
    rows=sorted((ResolvedPackage(str(n).lower().replace('_','-'),str(v)) for n,v in packages.items()), key=lambda x:x.name)
    return tuple(rows)

def resolution_fingerprint(rows):
    raw=json.dumps([r.__dict__ for r in rows],sort_keys=True,separators=(",",":" )).encode()
    return hashlib.sha256(raw).hexdigest()

def validate_hash_lock_lines(lines):
    content=[x.strip() for x in lines if x.strip() and not x.lstrip().startswith('#')]
    if not content: raise ValueError("empty lock")
    for line in content:
        if '==' not in line or '--hash=sha256:' not in line:
            raise ValueError("hermetic lock entries require exact version and sha256 hash")
    return True

def write_local_resolution(path, *, python, platform, packages, note):
    rows=canonical_resolution(packages)
    payload={"evidence":"LOCAL_RESOLUTION","python":python,"platform":platform,"packages":[r.__dict__ for r in rows],"fingerprint":resolution_fingerprint(rows),"note":note,"hermetic":False,"cross_platform":False}
    Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload
