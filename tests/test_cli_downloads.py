"""Offline HTTP and terminal regressions for safe CLI model downloads."""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emilia_studio import tui


PAYLOAD = b"emilia model bytes\x00\xff" * 128
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def download_server():
    state = {"mode": "normal", "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            state["ranges"].append(requested)
            mode = state["mode"]
            offset = int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
            if mode.startswith("416") or offset >= len(PAYLOAD):
                self.send_response(416)
                if mode != "416-missing":
                    size = len(PAYLOAD) + (1 if mode == "416-wrong" else 0)
                    self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            partial = bool(requested) and mode != "ignore"
            body = PAYLOAD[offset:] if partial else PAYLOAD
            self.send_response(206 if partial else 200)
            if partial and mode != "range-missing":
                start = offset + (1 if mode == "range-wrong" else 0)
                end = len(PAYLOAD) - 1
                if mode == "range-short":
                    end -= 5
                    body = body[:-5]
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
            if mode != "unknown":
                size = len(body) + (1 if mode == "range-length" else 0)
                self.send_header("Content-Length", str(size))
            if mode == "truncated":
                body = body[:-5]
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # shutdown() waits one poll interval; the 0.5s default cost half a second per test.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/model", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_full_download_validates_checksum_and_uses_stderr(download_server, tmp_path, capsys):
    url, state = download_server
    target = tmp_path / "model.bin"
    result = tui.download_file_with_progress(
        url, target, expected_bytes=len(PAYLOAD), expected_sha256=SHA256.upper(), chunk_size=64
    )
    assert result == target
    assert target.read_bytes() == PAYLOAD
    assert not target.with_suffix(".bin.part").exists()
    assert state["ranges"] == [None]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "downloaded" in captured.err
    assert "\x1b" not in captured.err
    assert "\r" not in captured.err


@pytest.mark.parametrize("mode", ["normal", "ignore"])
def test_resume_and_ignored_range(download_server, tmp_path, mode):
    url, state = download_server
    state["mode"] = mode
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(PAYLOAD[:37])
    tui.download_file_with_progress(url, target, expected_sha256=SHA256, chunk_size=64)
    assert state["ranges"] == ["bytes=37-"]
    assert target.read_bytes() == PAYLOAD


@pytest.mark.parametrize("mode", ["range-missing", "range-wrong", "range-length", "range-short"])
def test_invalid_range_never_promotes(download_server, tmp_path, capsys, mode):
    url, state = download_server
    state["mode"] = mode
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous good model")
    part = target.with_suffix(".bin.part")
    part.write_bytes(PAYLOAD[:37])
    with pytest.raises(RuntimeError):
        tui.download_file_with_progress(url, target, chunk_size=64)
    assert target.read_bytes() == b"previous good model"
    assert part.exists()
    assert "downloaded" not in capsys.readouterr().err


def test_truncated_response_retains_partial_without_completion(download_server, tmp_path, capsys):
    url, state = download_server
    state["mode"] = "truncated"
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous good model")
    with pytest.raises(RuntimeError, match="Truncated|Download failed"):
        tui.download_file_with_progress(url, target, chunk_size=64)
    assert target.read_bytes() == b"previous good model"
    assert target.with_suffix(".bin.part").exists()
    output = capsys.readouterr().err
    assert "downloaded" not in output
    assert "100.0%" not in output


def test_expected_length_mismatch_does_not_destroy_partial(download_server, tmp_path):
    url, _ = download_server
    target = tmp_path / "model.bin"
    part = target.with_suffix(".bin.part")
    part.write_bytes(PAYLOAD[:37])
    with pytest.raises(RuntimeError, match="size mismatch"):
        tui.download_file_with_progress(url, target, expected_bytes=len(PAYLOAD) + 1)
    assert part.read_bytes() == PAYLOAD[:37]
    assert not target.exists()


@pytest.mark.parametrize("resume", [False, True])
def test_bad_checksum_preserves_destination(download_server, tmp_path, capsys, resume):
    url, _ = download_server
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous good model")
    if resume:
        target.with_suffix(".bin.part").write_bytes(b"X" * 37)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        tui.download_file_with_progress(
            url, target, expected_sha256=SHA256 if resume else "0" * 64, chunk_size=64
        )
    assert target.read_bytes() == b"previous good model"
    assert "downloaded" not in capsys.readouterr().err


def test_complete_partial_accepts_validated_416(download_server, tmp_path):
    url, _ = download_server
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(PAYLOAD)
    tui.download_file_with_progress(url, target, expected_bytes=len(PAYLOAD), expected_sha256=SHA256)
    assert target.read_bytes() == PAYLOAD


@pytest.mark.parametrize("mode", ["416", "416-missing", "416-wrong"])
def test_416_cannot_promote_incomplete_partial(download_server, tmp_path, mode):
    url, state = download_server
    state["mode"] = mode
    target = tmp_path / "model.bin"
    part = target.with_suffix(".bin.part")
    part.write_bytes(PAYLOAD[:37])
    with pytest.raises(RuntimeError, match="HTTP 416"):
        tui.download_file_with_progress(url, target)
    assert not target.exists()
    assert part.read_bytes() == PAYLOAD[:37]


def test_416_still_checks_hash(download_server, tmp_path, capsys):
    url, _ = download_server
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(b"X" * len(PAYLOAD))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        tui.download_file_with_progress(url, target, expected_sha256=SHA256)
    assert not target.exists()
    assert "downloaded" not in capsys.readouterr().err


def test_unknown_length_never_invents_percentage(download_server, tmp_path, capsys):
    url, state = download_server
    state["mode"] = "unknown"
    target = tmp_path / "model.bin"
    tui.download_file_with_progress(url, target, expected_sha256=SHA256, chunk_size=64)
    assert target.read_bytes() == PAYLOAD
    assert "%" not in capsys.readouterr().err


def test_failed_promotion_never_announces_success(download_server, tmp_path, capsys, monkeypatch):
    url, _ = download_server
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous good model")

    def fail_replace(self, destination):
        raise PermissionError("destination locked")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError):
        tui.download_file_with_progress(url, target)
    assert target.read_bytes() == b"previous good model"
    assert "downloaded" not in capsys.readouterr().err


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_redirected_output_never_enables_ansi(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("WT_SESSION", "present")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert not tui.supports_ansi(io.StringIO())


def test_empty_no_color_disables_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert not tui.supports_ansi(Terminal())


def test_narrow_progress_does_not_slice_ansi(monkeypatch):
    terminal = Terminal()
    monkeypatch.setattr(tui, "supports_ansi", lambda stream=None: True)
    monkeypatch.setattr(tui.shutil, "get_terminal_size", lambda fallback: os.terminal_size((24, 10)))
    bar = tui.ProgressBar("A very long label for a narrow terminal", stream=terminal)
    bar.update(20)
    output = terminal.getvalue()
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    assert "\x1b" not in plain
    assert len(plain.removeprefix("\r")) <= 23
    assert output.endswith(tui.Palette.RESET)


def test_416_still_checks_expected_size(download_server, tmp_path):
    url, _ = download_server
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.part").write_bytes(PAYLOAD)
    with pytest.raises(RuntimeError, match="size mismatch"):
        tui.download_file_with_progress(url, target, expected_bytes=len(PAYLOAD) + 1)
    assert not target.exists()


@pytest.mark.parametrize("options", [{"expected_bytes": -1}, {"chunk_size": 0}, {"expected_sha256": "bad"}])
def test_invalid_integrity_options_do_not_request(download_server, tmp_path, options):
    url, state = download_server
    with pytest.raises(ValueError):
        tui.download_file_with_progress(url, tmp_path / "model.bin", **options)
    assert state["ranges"] == []
