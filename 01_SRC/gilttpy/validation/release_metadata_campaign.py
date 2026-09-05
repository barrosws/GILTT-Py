"""QA-054 metadata/citation/licensing campaign."""
from __future__ import annotations
from pathlib import Path
from gilttpy.engineering.release_metadata import (
    CFF_SCHEMA_VERSION, CORE_METADATA_LICENSE_VERSION, PEP639_MIN_SETUPTOOLS,
    load_project_metadata, load_state, public_release_ready, unresolved_fields,
)

def qa054_metadata_evidence(project_root: str | Path) -> dict:
    root=Path(project_root)
    state=load_state(root/'metadata/qa054_release_metadata_state.json')
    project=load_project_metadata(root/'pyproject.toml')
    return {
        'project_name':project['name'],
        'development_version':project['version'],
        'requires_python':project['requires-python'],
        'license_declared_in_pyproject':'license' in project,
        'authors_declared_in_pyproject':'authors' in project,
        'cff_schema_target':CFF_SCHEMA_VERSION,
        'pep639_min_setuptools':PEP639_MIN_SETUPTOOLS,
        'core_metadata_license_version':CORE_METADATA_LICENSE_VERSION,
        'unresolved_fields':list(unresolved_fields(state)),
        'public_release_ready':public_release_ready(state),
    }
