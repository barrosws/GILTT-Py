"""QA-057 operational QA reproduction entrypoint.

This module materializes a single machine-readable QA replay command for the
frozen engineering/scientific QA suite.  It deliberately does not claim to
reproduce manuscript tables/figures, resolve data redistribution rights, or
promote unresolved public-release metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Iterable

QA057_GATE = "PASS_OPERATIONAL_REPRODUCTION_DOCUMENTATION_CLOSURE"
QA057_HOLDS = (
    "HOLD_MANUSCRIPT_TABLE_FIGURE_REPRODUCTION_ENTRYPOINT",
    "HOLD_MACHINE_READABLE_CONFIG_COVERAGE_FOR_ALL_CANONICAL_SCIENTIFIC_RUNS",
    "HOLD_DATA_AND_SOURCE_PROVENANCE_CATALOG_COMPLETENESS",
    "HOLD_FULL_PACKAGE_COVERAGE_REPORT",
    "HOLD_FINAL_PUBLIC_RELEASE_METADATA",
    "HOLD_SOFTWARE_LICENSE_SELECTION",
    "HOLD_ZENODO_ARCHIVAL_RECORD",
    "HOLD_EXTERNAL_CI_MATRIX_EXECUTION",
    "HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION",
)
QA057_PROHIBITIONS = (
    "PROHIBIT_QA_REPLAY_AS_MANUSCRIPT_OUTPUT_REPRODUCTION_PROOF",
    "PROHIBIT_COVERAGE_PERCENTAGE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_UNVERIFIED_DATA_REDISTRIBUTION_RIGHTS",
    "PROHIBIT_UNRESOLVED_METADATA_AS_FINAL_RELEASE_METADATA",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

@dataclass(frozen=True)
class ReplayPartition:
    name: str
    qa_start: int
    qa_end: int
    expected_tests: int

    def __post_init__(self) -> None:
        if not self.name or self.qa_start > self.qa_end or self.expected_tests <= 0:
            raise ValueError("invalid replay partition")

    def targets(self, project_root: str | Path) -> tuple[str, ...]:
        root=Path(project_root)
        out=[]
        for qa in range(self.qa_start, self.qa_end + 1):
            out.extend(sorted((root/'02_TESTS').glob(f'test_qa{qa:03d}*.py')))
        return tuple(str(p.relative_to(root)) for p in out)


def load_replay_plan(path: str | Path) -> dict:
    with Path(path).open('rb') as handle:
        plan=tomllib.load(handle)
    if plan.get('gate') != QA057_GATE:
        raise ValueError('replay plan gate mismatch')
    return plan


def plan_partitions(plan: dict) -> tuple[ReplayPartition, ...]:
    return tuple(
        ReplayPartition(
            str(item['name']), int(item['qa_start']), int(item['qa_end']), int(item['expected_tests'])
        )
        for item in plan['partitions']
    )


def expected_total(plan: dict) -> int:
    return sum(p.expected_tests for p in plan_partitions(plan))


def assert_exact_test_file_coverage(project_root: str | Path, plan: dict) -> None:
    root=Path(project_root)
    expected=tuple(sorted((root/'02_TESTS').glob('test_qa*.py')))
    assigned=[]
    for part in plan_partitions(plan):
        assigned.extend((root/x) for x in part.targets(root))
    if not expected:
        raise ValueError('no QA tests found')
    if len(assigned) != len(set(assigned)):
        raise ValueError('duplicate test-file assignment')
    if set(assigned) != set(expected):
        missing=sorted(str(p.relative_to(root)) for p in set(expected)-set(assigned))
        extra=sorted(str(p.relative_to(root)) for p in set(assigned)-set(expected))
        raise ValueError(f'test-file coverage mismatch: missing={missing}, extra={extra}')


def frozen_environment(base: dict[str, str] | None=None) -> dict[str, str]:
    env=dict(os.environ if base is None else base)
    env.update(THREAD_ENVIRONMENT)
    return env


def _collected_count(stdout: str) -> int | None:
    patterns=(r'(\d+) tests? collected', r'collected (\d+) items?')
    for pattern in patterns:
        hits=re.findall(pattern, stdout)
        if hits:
            return int(hits[-1])
    return None


def run_partition(project_root: str | Path, part: ReplayPartition, *, env: dict[str,str] | None=None) -> dict:
    root=Path(project_root).resolve()
    targets=part.targets(root)
    if not targets:
        return {'name':part.name,'status':'FAIL','reason':'no_targets','expected_tests':part.expected_tests}
    run_env=frozen_environment(env)
    collect_cmd=[sys.executable,'-m','pytest','--collect-only','-q',*targets]
    collect=subprocess.run(collect_cmd,cwd=root,env=run_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    count=_collected_count(collect.stdout)
    if collect.returncode != 0 or count != part.expected_tests:
        return {
            'name':part.name,'status':'FAIL','phase':'collect','expected_tests':part.expected_tests,
            'collected_tests':count,'returncode':collect.returncode,'output_tail':collect.stdout[-4000:],
        }
    test_cmd=[sys.executable,'-m','pytest','-q',*targets]
    run=subprocess.run(test_cmd,cwd=root,env=run_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return {
        'name':part.name,
        'status':'PASS' if run.returncode == 0 else 'FAIL',
        'phase':'test',
        'expected_tests':part.expected_tests,
        'collected_tests':count,
        'returncode':run.returncode,
        'targets':list(targets),
        'output_tail':run.stdout[-4000:],
    }


def replay_can_pass(plan: dict, results: Iterable[dict]) -> bool:
    expected={p.name:p for p in plan_partitions(plan)}
    result_map={r.get('name'):r for r in results}
    if set(result_map) != set(expected):
        return False
    for name,part in expected.items():
        item=result_map[name]
        if item.get('status') != 'PASS' or item.get('collected_tests') != part.expected_tests:
            return False
    return True


def execute_replay(project_root: str | Path, plan_path: str | Path, *, selected: str | None=None) -> dict:
    root=Path(project_root).resolve()
    plan=load_replay_plan(plan_path)
    assert_exact_test_file_coverage(root,plan)
    run_env=frozen_environment()
    if selected is not None:
        parts=tuple(p for p in plan_partitions(plan) if p.name == selected)
        if not parts:
            raise ValueError(f'unknown partition: {selected}')
        results=[run_partition(root,parts[0],env=run_env)]
        passed=all(r['status']=='PASS' for r in results)
        overall={'status':'PASS' if passed else 'FAIL','collected_tests':results[0].get('collected_tests')}
        complete=False
    else:
        # Complete replay is intentionally two subprocesses only: one exact collection
        # audit followed by one full-suite execution. This avoids multiplying the
        # cost of expensive scientific tests while preserving exact file coverage.
        collect_cmd=[sys.executable,'-m','pytest','--collect-only','-q','02_TESTS']
        collect=subprocess.run(collect_cmd,cwd=root,env=run_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        count=_collected_count(collect.stdout)
        expected=expected_total(plan)
        if collect.returncode != 0 or count != expected:
            overall={'status':'FAIL','phase':'collect','collected_tests':count,'expected_tests':expected,'returncode':collect.returncode,'output_tail':collect.stdout[-4000:]}
            passed=False
        else:
            test_cmd=[sys.executable,'-m','pytest','-q','02_TESTS']
            run=subprocess.run(test_cmd,cwd=root,env=run_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            passed=run.returncode == 0
            overall={'status':'PASS' if passed else 'FAIL','phase':'test','collected_tests':count,'expected_tests':expected,'returncode':run.returncode,'output_tail':run.stdout[-4000:]}
        results=[{'name':p.name,'status':'PASS' if passed else 'FAIL','collected_tests':p.expected_tests,'expected_tests':p.expected_tests} for p in plan_partitions(plan)]
        complete=True
    return {
        'gate':QA057_GATE,
        'scope':'QA replay evidence only; not manuscript table/figure reproduction',
        'project_root':str(root),
        'plan_path':str(Path(plan_path).resolve()),
        'expected_total_tests':expected_total(plan),
        'selected_partition':selected,
        'complete_replay':complete,
        'pass':passed,
        'thread_environment':dict(THREAD_ENVIRONMENT),
        'overall_result':overall,
        'results':results,
        'holds':list(QA057_HOLDS),
        'prohibitions':list(QA057_PROHIBITIONS),
    }


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description='Replay the governed GILTT-Py QA suite from a frozen project root.')
    parser.add_argument('--project-root',default='.')
    parser.add_argument('--plan',default=None)
    parser.add_argument('--partition',default=None)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--evidence',default=None)
    args=parser.parse_args(argv)
    root=Path(args.project_root).resolve()
    plan_path=Path(args.plan).resolve() if args.plan else root/'reproduction/qa057_qa_replay.toml'
    plan=load_replay_plan(plan_path)
    assert_exact_test_file_coverage(root,plan)
    if args.dry_run:
        payload={
            'gate':QA057_GATE,
            'command':'python -m gilttpy.engineering.operational_reproduction --project-root .',
            'expected_total_tests':expected_total(plan),
            'partitions':[p.__dict__ | {'targets':list(p.targets(root))} for p in plan_partitions(plan)],
            'holds':list(QA057_HOLDS),
        }
        print(json.dumps(payload,indent=2,sort_keys=True))
        return 0
    payload=execute_replay(root,plan_path,selected=args.partition)
    if args.evidence:
        out=Path(args.evidence)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'gate':QA057_GATE,'pass':payload['pass'],'expected_total_tests':payload['expected_total_tests'],'selected_partition':args.partition},sort_keys=True))
    return 0 if payload['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
