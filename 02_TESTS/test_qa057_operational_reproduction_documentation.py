from pathlib import Path
import json
import subprocess
import sys

from gilttpy.engineering.operational_reproduction import (
    QA057_GATE, QA057_HOLDS, QA057_PROHIBITIONS, THREAD_ENVIRONMENT,
    assert_exact_test_file_coverage, expected_total, load_replay_plan,
    plan_partitions, replay_can_pass,
)
from gilttpy.validation.operational_reproduction_campaign import qa057_operational_documentation_evidence

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'reproduction/qa057_qa_replay.toml'


def test_qa057_01_gate_holds_prohibitions_explicit():
    assert QA057_GATE=='PASS_OPERATIONAL_REPRODUCTION_DOCUMENTATION_CLOSURE'
    assert 'HOLD_MANUSCRIPT_TABLE_FIGURE_REPRODUCTION_ENTRYPOINT' in QA057_HOLDS
    assert 'HOLD_FINAL_PUBLIC_RELEASE_METADATA' in QA057_HOLDS
    assert 'HOLD_FULL_PACKAGE_COVERAGE_REPORT' in QA057_HOLDS
    assert 'PROHIBIT_QA_REPLAY_AS_MANUSCRIPT_OUTPUT_REPRODUCTION_PROOF' in QA057_PROHIBITIONS
    assert 'PROHIBIT_TARGET_TUNING' in QA057_PROHIBITIONS


def test_qa057_02_plan_is_machine_readable_and_total_is_384():
    plan=load_replay_plan(PLAN)
    assert plan['gate']==QA057_GATE
    assert expected_total(plan)==384
    assert len(plan_partitions(plan))==14


def test_qa057_03_plan_covers_every_qa_test_file_exactly_once():
    plan=load_replay_plan(PLAN)
    assert_exact_test_file_coverage(ROOT,plan)


def test_qa057_04_partition_counts_preserve_prior_counts_and_add_qa056_qa057():
    parts={p.name:p for p in plan_partitions(load_replay_plan(PLAN))}
    assert parts['qa029_qa042'].expected_tests==210
    assert parts['qa043_qa045'].expected_tests==30
    assert parts['qa056'].expected_tests==12
    assert parts['qa057'].expected_tests==12


def test_qa057_05_frozen_thread_environment_is_exact():
    assert THREAD_ENVIRONMENT=={
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONHASHSEED':'0'
    }


def test_qa057_06_dry_run_entrypoint_is_executable_and_reports_384():
    cmd=[sys.executable,'-m','gilttpy.engineering.operational_reproduction','--project-root',str(ROOT),'--dry-run']
    out=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert out.returncode==0
    payload=json.loads(out.stdout)
    assert payload['expected_total_tests']==384
    assert len(payload['partitions'])==14


def test_qa057_07_operational_docs_are_materialized():
    for rel in ('docs/INSTALLATION_QUICKSTART.md','docs/REPRODUCIBILITY.md','docs/CHANGELOG.md','docs/RELEASE_NOTES_DEV.md'):
        assert (ROOT/rel).exists() and (ROOT/rel).read_text().strip()
    assert 'Operational documentation' in (ROOT/'README.md').read_text()


def test_qa057_08_quickstart_contains_executable_qa_replay_and_scope_boundary():
    text=(ROOT/'docs/INSTALLATION_QUICKSTART.md').read_text()
    assert 'pip3 install ".[test]"' in text
    assert 'python -m gilttpy.engineering.operational_reproduction' in text
    assert 'does not reproduce manuscript tables/figures' in text


def test_qa057_09_development_changelog_and_sdist_manifest_preserve_operational_artifacts():
    text=(ROOT/'docs/CHANGELOG.md').read_text()
    assert '0.0.0.dev1' in text
    assert 'does not create a public semantic release' in text
    manifest=(ROOT/'MANIFEST.in').read_text()
    for token in ('recursive-include 02_TESTS','recursive-include docs','recursive-include reproduction'):
        assert token in manifest


def test_qa057_10_campaign_preserves_archival_and_manuscript_holds():
    ev=qa057_operational_documentation_evidence(ROOT)
    assert ev['expected_total_tests']==384
    assert ev['partition_count']==14
    assert ev['quickstart_present'] and ev['reproducibility_doc_present']
    assert ev['archival_release_ready'] is False
    assert ev['manuscript_output_reproduction_ready'] is False


def test_qa057_11_replay_pass_requires_complete_exact_partition_result_set():
    plan=load_replay_plan(PLAN)
    good=[{'name':p.name,'status':'PASS','collected_tests':p.expected_tests} for p in plan_partitions(plan)]
    assert replay_can_pass(plan,good)
    assert not replay_can_pass(plan,good[:-1])
    bad=list(good); bad[-1]=dict(bad[-1]); bad[-1]['collected_tests']-=1
    assert not replay_can_pass(plan,bad)


def test_qa057_12_new_engineering_layer_has_no_target_specific_logic():
    text=(ROOT/'01_SRC/gilttpy/engineering/operational_reproduction.py').read_text().lower()
    text+=(ROOT/'01_SRC/gilttpy/validation/operational_reproduction_campaign.py').read_text().lower()
    for forbidden in ('observed concentration','historical concentration','copenhagen','hanford'):
        assert forbidden not in text
