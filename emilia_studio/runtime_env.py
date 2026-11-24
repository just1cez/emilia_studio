"""Pinned ComfyUI runtime: git checkouts, source patches and the inference interpreter.

Follows the AUTOMATIC1111 ``launch_utils`` pattern: clone each repository if missing,
check out the pinned commit, and install exact-pinned requirements only when the
installed versions differ. ComfyUI-Manager snapshots record the same three layers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

TORCH_INDEX_ENV = "EMILIA_TORCH_INDEX_URL"

_VERSIONS_PROBE = (
    "import importlib.metadata as m, json, sys\n"
    "out = {}\n"
    "for name in json.loads(sys.argv[1]):\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        out[name] = None\n"
    "print(json.dumps(out))\n"
)


def read_pins(path: Path) -> dict[str, str]:
    """Parse a requirements file in which every requirement is ``name==version``."""
    pins: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep or not name.strip() or not version.strip():
            raise ValueError(f"{path}:{number}: expected an exact pin `name==version`, got {raw!r}")
        pins[name.strip()] = version.strip()
    return pins


def pin_list(requirements: list[str]) -> dict[str, str]:
    return {name: version for name, _, version in (r.partition("==") for r in requirements)}


def requirements_drift(python: Path | str, pins: dict[str, str]) -> list[str]:
    """Pins the interpreter does not satisfy, as ``name (have X, want Y)``.

    Local version labels are ignored, so ``torch==2.13.0`` accepts ``2.13.0+cu130``.
    """
    probe = subprocess.run(
        [str(python), "-c", _VERSIONS_PROBE, json.dumps(sorted(pins))],
        capture_output=True, text=True, timeout=120, check=True,
    )
    installed = json.loads(probe.stdout)
    drift = []
    for name, wanted in sorted(pins.items(), key=lambda item: item[0].lower()):
        have = installed.get(name)
        if have is None or have.split("+", 1)[0] != wanted:
            drift.append(f"{name} (have {have or 'none'}, want {wanted})")
    return drift


def _git(*arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to set up the pinned ComfyUI runtime")
    command = [git, *arguments] if cwd is None else [git, "-C", str(cwd), *arguments]
    return subprocess.run(command, capture_output=True, text=True)


def _git_ok(*arguments: str, cwd: Path | None = None) -> str:
    result = _git(*arguments, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed in {cwd}: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def ensure_checkout(directory: Path, repo: str, commit: str, patch: Path | None, log: Callable[[str], None]) -> None:
    """Clone ``repo`` into ``directory`` at ``commit`` and apply ``patch``; idempotent.

    An existing checkout on another commit is moved only when its tracked files are
    clean, so a patched or locally edited tree is never discarded.
    """
    name = directory.name
    if not directory.exists() or (directory.is_dir() and not any(directory.iterdir())):
        log(f"{name}: cloning {repo}")
        _git_ok("clone", "--filter=blob:none", "--no-checkout", repo, str(directory))
        _git_ok("checkout", "--detach", commit, cwd=directory)
    elif not (directory / ".git").exists():
        raise RuntimeError(f"{directory} exists but is not a git checkout; move it away or point app_dir elsewhere")
    else:
        head = _git_ok("rev-parse", "HEAD", cwd=directory)
        if head != commit:
            if _git_ok("status", "--porcelain", "--untracked-files=no", cwd=directory):
                raise RuntimeError(
                    f"{directory} has local changes on {head[:12]}; commit, stash or revert them "
                    f"before moving to the pinned {commit[:12]}"
                )
            if _git("cat-file", "-e", f"{commit}^{{commit}}", cwd=directory).returncode != 0:
                log(f"{name}: fetching {commit[:12]}")
                _git_ok("fetch", "origin", commit, cwd=directory)
            log(f"{name}: {head[:12]} -> {commit[:12]}")
            _git_ok("checkout", "--detach", commit, cwd=directory)
    if patch is None:
        return
    if _git("apply", "--reverse", "--check", str(patch), cwd=directory).returncode == 0:
        log(f"{name}: {patch.name} already applied")
        return
    check = _git("apply", "--check", str(patch), cwd=directory)
    if check.returncode != 0:
        raise RuntimeError(f"{patch.name} does not apply to {directory}: {check.stderr.strip()}")
    _git_ok("apply", str(patch), cwd=directory)
    log(f"{name}: applied {patch.name}")


def _pinned_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [manifest["comfyui"]] if manifest.get("comfyui") else []
    return entries + [node for node in manifest.get("custom_nodes", []) if node.get("pinned_commit")]


def bootstrap_checkouts(config: Any, manifest: dict[str, Any], log: Callable[[str], None]) -> None:
    """Bring ComfyUI and every pinned custom node to its manifest commit plus patch."""
    for entry in _pinned_entries(manifest):
        patch = config.resources_dir / entry["patch"] if entry.get("patch") else None
        ensure_checkout((config.app_dir / entry["directory"]).resolve(), entry["upstream_repo"], entry["pinned_commit"], patch, log)


def install_inference_env(config: Any, manifest: dict[str, Any], log: Callable[[str], None]) -> None:
    """Install the pinned torch build and requirements into ``conda_python`` where they drift."""
    python = config.conda_python
    if python is None or not Path(python).is_file():
        raise RuntimeError("Set conda_python (emilia config init --python ...) to the inference interpreter first")
    env = manifest["python_env"]
    version = subprocess.run(
        [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout.strip()
    if version != env["python"]:
        log(f"warning: {python} is Python {version}; the pins were verified on Python {env['python']}")
    torch_pins = pin_list(env["torch_packages"])
    if requirements_drift(python, torch_pins):
        index_url = os.environ.get(TORCH_INDEX_ENV) or env["torch_index_url"]
        log(f"installing {' '.join(env['torch_packages'])} from {index_url}")
        subprocess.run([str(python), "-m", "pip", "install", *env["torch_packages"], "--index-url", index_url], check=True)
    requirements = config.resources_dir / env["requirements"]
    if requirements_drift(python, read_pins(requirements)):
        log(f"installing {requirements.name}")
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
    log("inference requirements satisfied")


def checkout_checks(config: Any, manifest: dict[str, Any]) -> list[tuple[str, str, str, bool]]:
    """Compare runtime git checkouts with their pinned commits and tracked source patches.

    Returns ``(check, status, details, ok)`` rows. A checkout that git cannot inspect is
    reported UNKNOWN without failing; a different commit or an unapplied patch fails,
    because every recorded timing and pixel hash refers to exactly this patched source.
    """
    rows: list[tuple[str, str, str, bool]] = []
    git = shutil.which("git")
    for entry in _pinned_entries(manifest):
        directory = (config.app_dir / entry["directory"]).resolve()
        pinned = entry["pinned_commit"]

        def run(*arguments: str) -> subprocess.CompletedProcess[str] | None:
            if git is None or not (directory / ".git").exists():
                return None
            try:
                return subprocess.run([git, "-C", str(directory), *arguments], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                return None

        head = run("rev-parse", "HEAD")
        if head is None or head.returncode != 0:
            rows.append((f"{entry['name']} commit", "UNKNOWN", f"not a readable git checkout; expected {pinned[:12]}", True))
        else:
            actual = head.stdout.strip()
            matches = actual.startswith(pinned) or pinned.startswith(actual)
            rows.append((f"{entry['name']} commit", "OK" if matches else "DRIFT", actual[:12] if matches else f"{actual[:12]} (pinned {pinned[:12]})", matches))
        patch = entry.get("patch")
        if patch:
            patch_path = config.resources_dir / patch
            applied = run("apply", "--reverse", "--check", str(patch_path)) if patch_path.is_file() else None
            if applied is None:
                rows.append((f"{entry['name']} patch", "UNKNOWN", patch, True))
            else:
                rows.append((f"{entry['name']} patch", "OK" if applied.returncode == 0 else "NOT APPLIED", patch, applied.returncode == 0))
    return rows


def environment_check(config: Any, manifest: dict[str, Any]) -> tuple[str, str, str, bool] | None:
    """Doctor row comparing the inference interpreter with every pin; None without one."""
    python = config.conda_python
    if python is None or not Path(python).is_file() or "python_env" not in manifest:
        return None
    env = manifest["python_env"]
    try:
        pins = {**pin_list(env["torch_packages"]), **read_pins(config.resources_dir / env["requirements"])}
        drift = requirements_drift(python, pins)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return ("Inference requirements", "UNKNOWN", str(exc)[:300], False)
    if not drift:
        return ("Inference requirements", "OK", f"{len(pins)} pins ({env['requirements']}, torch from manifest)", True)
    shown = ", ".join(drift[:4]) + (f", +{len(drift) - 4} more" if len(drift) > 4 else "")
    return ("Inference requirements", "DRIFT", f"{len(drift)} differ: {shown}; run `emilia setup --install-deps`", False)
