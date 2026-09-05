"""Non-personal reproducibility provenance helpers.

These helpers intentionally avoid host names, user names, home directories, working
paths and wall-clock timestamps.  The resulting fingerprint identifies one runtime
snapshot; it is not a cross-platform lock or a proof of hermetic dependency resolution.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: str | Path, *, suffix: str = ".py") -> dict[str, str]:
    base = Path(root)
    return {
        p.relative_to(base).as_posix(): sha256_file(p)
        for p in sorted(base.rglob(f"*{suffix}"))
        if p.is_file()
    }


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def canonical_runtime_snapshot(
    *,
    required: Iterable[str] = ("numpy", "scipy", "mpmath"),
    optional: Iterable[str] = ("pytest", "threadpoolctl"),
) -> dict[str, object]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "packages": {
            "required": {name: _version(name) for name in required},
            "optional": {name: _version(name) for name in optional},
        },
        "thread_controls": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def environment_fingerprint(snapshot: dict[str, object] | None = None) -> str:
    payload = canonical_runtime_snapshot() if snapshot is None else snapshot
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
