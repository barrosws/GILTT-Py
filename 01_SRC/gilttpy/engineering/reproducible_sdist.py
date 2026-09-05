"""Deterministic normalization of an sdist tar.gz container.

The normalizer preserves member bytes, type, mode and link target while replacing
container provenance fields (mtime, uid/gid, owner/group labels and gzip timestamp)
with deterministic values.  It is deliberately separate from the raw PEP-517 backend
output and must not be described as byte-identical raw setuptools output.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
import tarfile


def normalize_sdist(source: str | Path, destination: str | Path, *, epoch: int) -> None:
    src = Path(source)
    dst = Path(destination)
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(src, "r:gz") as tf:
        for old in tf.getmembers():
            data = None
            if old.isfile():
                fh = tf.extractfile(old)
                data = b"" if fh is None else fh.read()
            info = tarfile.TarInfo(old.name)
            info.mode = old.mode
            info.type = old.type
            info.linkname = old.linkname
            info.size = len(data) if data is not None else old.size
            info.mtime = int(epoch)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.pax_headers = {}
            members.append((info, data))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=int(epoch), compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as out:
                for info, data in sorted(members, key=lambda item: item[0].name):
                    out.addfile(info, io.BytesIO(data) if data is not None else None)
