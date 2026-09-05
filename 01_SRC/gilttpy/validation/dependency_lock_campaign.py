"""QA-053 dependency resolution campaign."""
from __future__ import annotations
import importlib.metadata as md
import platform, sys
from pathlib import Path
from gilttpy.engineering.dependency_lock import QA053_GATE, QA053_HOLDS, QA053_PROHIBITIONS, write_local_resolution

QA053_RUNTIME_DISTRIBUTIONS=("numpy","scipy","mpmath","pytest","threadpoolctl")

def installed_versions():
    return {n: md.version(n) for n in QA053_RUNTIME_DISTRIBUTIONS}

def local_resolution_payload(path: str|Path):
    return write_local_resolution(path, python=platform.python_version(), platform=f"{platform.system()} {platform.machine()}", packages=installed_versions(), note="Executed local resolution evidence only; not a cross-platform or hash-locked dependency proof.")
