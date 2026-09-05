from pathlib import Path
import os
import io
import json
import tarfile
import gzip

from gilttpy.engineering.reproducibility import canonical_runtime_snapshot, environment_fingerprint, sha256_file, sha256_tree
from gilttpy.engineering.reproducible_sdist import normalize_sdist
from gilttpy.validation.reproducibility_campaign import QA050_GATE, QA050_HOLDS, QA050_PROHIBITIONS, load_packaging_contract

ROOT = Path(os.environ.get("GILTTPY_QA050_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def test_qa050_01_gate_holds_prohibitions_explicit():
    assert QA050_GATE == "PASS_ENGINEERING_REPRODUCIBILITY_BASELINE"
    assert "HOLD_HERMETIC_DEPENDENCY_RESOLUTION_TO_QA051" in QA050_HOLDS
    assert "PROHIBIT_TARGET_TUNING" in QA050_PROHIBITIONS


def test_qa050_02_governed_tree_matches_pyproject():
    c = load_packaging_contract(ROOT / "pyproject.toml")
    assert c.package_dir == "01_SRC"
    assert c.test_dir == "02_TESTS"
    assert c.version == "0.0.0.dev1"


def test_qa050_03_runtime_dependencies_are_explicit():
    c = load_packaging_contract(ROOT / "pyproject.toml")
    assert c.runtime_dependencies == ("numpy>=1.26", "scipy>=1.11", "mpmath>=1.3")


def test_qa050_04_runtime_snapshot_excludes_personal_host_fields():
    s = canonical_runtime_snapshot()
    flat = json.dumps(s).lower()
    for forbidden in ("hostname", "username", "home directory", "cwd", "working_directory"):
        assert forbidden not in flat


def test_qa050_05_environment_fingerprint_is_deterministic():
    s = canonical_runtime_snapshot()
    assert environment_fingerprint(s) == environment_fingerprint(s)
    assert len(environment_fingerprint(s)) == 64


def test_qa050_06_sha256_file_matches_known_vector(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_qa050_07_sha256_tree_is_relative_and_stable(tmp_path):
    (tmp_path / "b").mkdir(); (tmp_path / "b" / "z.py").write_text("z=1\n")
    (tmp_path / "a.py").write_text("a=1\n")
    d = sha256_tree(tmp_path)
    assert list(d) == ["a.py", "b/z.py"]
    assert d == sha256_tree(tmp_path)


def _make_tar(path, mtime):
    payload=b"science-bytes\n"
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                i=tarfile.TarInfo("pkg/a.txt"); i.size=len(payload); i.mtime=mtime; i.mode=0o644; i.uid=123; i.gid=456
                tf.addfile(i, io.BytesIO(payload))


def test_qa050_08_sdist_normalization_removes_timestamp_variability(tmp_path):
    a=tmp_path/"a.tar.gz"; b=tmp_path/"b.tar.gz"; na=tmp_path/"na.tar.gz"; nb=tmp_path/"nb.tar.gz"
    _make_tar(a, 100); _make_tar(b, 200)
    assert sha256_file(a) != sha256_file(b)
    normalize_sdist(a,na,epoch=1234); normalize_sdist(b,nb,epoch=1234)
    assert sha256_file(na) == sha256_file(nb)


def test_qa050_09_sdist_normalization_preserves_member_bytes(tmp_path):
    a=tmp_path/"a.tar.gz"; n=tmp_path/"n.tar.gz"; _make_tar(a,100); normalize_sdist(a,n,epoch=1234)
    with tarfile.open(a,"r:gz") as x, tarfile.open(n,"r:gz") as y:
        assert x.getnames() == y.getnames()
        assert x.extractfile("pkg/a.txt").read() == y.extractfile("pkg/a.txt").read()


def test_qa050_10_readme_declares_historical_modern_separation():
    text=(ROOT/"README.md").read_text().lower()
    assert "historical" in text and "modern giltt-py 2.0" in text


def test_qa050_11_pyproject_uses_setuptools_pep517_backend():
    text=(ROOT/"pyproject.toml").read_text()
    assert 'build-backend = "setuptools.build_meta"' in text


def test_qa050_12_engineering_layer_does_not_import_transport_targets():
    base=ROOT/"01_SRC"/"gilttpy"/"engineering"
    text="\n".join(p.read_text() for p in base.glob("*.py")).lower()
    for forbidden in ("copenhagen", "hanford", "observed concentration", "historical concentration"):
        assert forbidden not in text
