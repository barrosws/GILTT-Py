"""QA-057 operational reproduction/documentation campaign evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gilttpy.engineering.operational_reproduction import (
    QA057_GATE, QA057_HOLDS, QA057_PROHIBITIONS, THREAD_ENVIRONMENT,
    assert_exact_test_file_coverage, expected_total, load_replay_plan, plan_partitions,
)


def qa057_operational_documentation_evidence(project_root: str | Path) -> dict:
    root=Path(project_root)
    plan_path=root/'reproduction/qa057_qa_replay.toml'
    plan=load_replay_plan(plan_path)
    assert_exact_test_file_coverage(root,plan)
    coverage_json=root/'coverage/qa057_coverage.json'
    coverage_payload=json.loads(coverage_json.read_text()) if coverage_json.exists() else None
    return {
        'gate':QA057_GATE,
        'qa_replay_command':'python -m gilttpy.engineering.operational_reproduction --project-root .',
        'expected_total_tests':expected_total(plan),
        'partition_count':len(plan_partitions(plan)),
        'quickstart_present':(root/'docs/INSTALLATION_QUICKSTART.md').exists(),
        'reproducibility_doc_present':(root/'docs/REPRODUCIBILITY.md').exists(),
        'development_changelog_present':(root/'docs/CHANGELOG.md').exists(),
        'development_release_notes_present':(root/'docs/RELEASE_NOTES_DEV.md').exists(),
        'coverage_contract_available':importlib.util.find_spec('coverage') is not None,
        'coverage_evidence_present':coverage_payload is not None,
        'coverage_total_percent':None if coverage_payload is None else coverage_payload.get('totals',{}).get('percent_covered'),
        'thread_environment':dict(THREAD_ENVIRONMENT),
        'holds':list(QA057_HOLDS),
        'prohibitions':list(QA057_PROHIBITIONS),
        'archival_release_ready':False,
        'manuscript_output_reproduction_ready':False,
    }
