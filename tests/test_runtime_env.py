"""Pinned-runtime bootstrap against real local git repositories and interpreters."""
import shutil
import subprocess
import sys
from importlib.metadata import version

import pytest

from emilia_studio.runtime_env import ensure_checkout, read_pins, requirements_drift

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "node.py").write_text("value = 1\n")
    _git(repo, "add", "node.py")
    _git(repo, "commit", "-qm", "pinned")
    pinned = _git(repo, "rev-parse", "HEAD")
    (repo / "node.py").write_text("value = 2\n")
    _git(repo, "commit", "-qam", "newer")
    newer = _git(repo, "rev-parse", "HEAD")
    patch = tmp_path / "node.patch"
    patch.write_text(
        "diff --git a/node.py b/node.py\n--- a/node.py\n+++ b/node.py\n@@ -1 +1 @@\n-value = 1\n+value = 'patched'\n"
    )
    return repo, pinned, newer, patch


def test_bootstrap_clones_pins_patches_and_is_idempotent(tmp_path, upstream):
    repo, pinned, newer, patch = upstream
    target = tmp_path / "app" / "custom_nodes" / "node"
    log = []
    ensure_checkout(target, str(repo), pinned, patch, log.append)
    assert _git(target, "rev-parse", "HEAD") == pinned
    assert (target / "node.py").read_text().strip() == "value = 'patched'"
    ensure_checkout(target, str(repo), pinned, patch, log.append)
    assert log[-1].endswith("already applied")
    assert (target / "node.py").read_text().strip() == "value = 'patched'"


def test_bootstrap_moves_clean_checkout_but_never_discards_local_changes(tmp_path, upstream):
    repo, pinned, newer, patch = upstream
    clean = tmp_path / "clean"
    _git(tmp_path, "clone", "-q", str(repo), str(clean))
    assert _git(clean, "rev-parse", "HEAD") == newer
    ensure_checkout(clean, str(repo), pinned, patch, lambda _: None)
    assert _git(clean, "rev-parse", "HEAD") == pinned

    dirty = tmp_path / "dirty"
    _git(tmp_path, "clone", "-q", str(repo), str(dirty))
    (dirty / "node.py").write_text("value = 'my edit'\n")
    with pytest.raises(RuntimeError, match="local changes"):
        ensure_checkout(dirty, str(repo), pinned, patch, lambda _: None)
    assert _git(dirty, "rev-parse", "HEAD") == newer
    assert (dirty / "node.py").read_text() == "value = 'my edit'\n"

    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "file.txt").write_text("x")
    with pytest.raises(RuntimeError, match="not a git checkout"):
        ensure_checkout(stray, str(repo), pinned, patch, lambda _: None)


def test_requirements_drift_reports_wrong_and_missing_pins(tmp_path):
    pins = tmp_path / "pins.txt"
    pins.write_text(f"# header\npytest=={version('pytest')}\nPillow==0.0.1  # wrong\nnot-a-real-package-xyz==1.0\n")
    drift = requirements_drift(sys.executable, read_pins(pins))
    assert drift == ["not-a-real-package-xyz (have none, want 1.0)", f"Pillow (have {version('Pillow')}, want 0.0.1)"]
    pins.write_text("numpy>=1.0\n")
    with pytest.raises(ValueError, match="exact pin"):
        read_pins(pins)
