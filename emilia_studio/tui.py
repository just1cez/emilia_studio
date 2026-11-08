"""Rich terminal UI, slant ASCII banner, progress bars, and model downloader for Emilia Studio."""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Slant ASCII art spelling "EMILIA" using --, \, /, _, and |
EMILIA_SLANT_ASCII: tuple[str, ...] = (
    r"    ______   __  ___   ____   __     ____   ___ ",
    r"   / ____/  /  |/  /  /  _/  / /    /  _/  /   |",
    r"  / __/    / /|_/ /   / /   / /     / /   / /| |",
    r" / /___   / /  / /  _/ /   / /___ _/ /   / ___ |",
    r"/_____/  /_/  /_/  /___/  /_____//___/  /_/  |_|",
)

EMILIA_TAGLINE: str = (
    "  *+.  E M I L I A   S T U D I O   v2.1  --  Re:Zero Decorative SDXL Pipeline  .+*"
)


def supports_ansi(stream: Any = None) -> bool:
    """Use color only on a real terminal, respecting NO_COLOR even when empty."""
    target = stream if stream is not None else sys.stdout
    if "NO_COLOR" in os.environ or not getattr(target, "isatty", lambda: False)():
        return False
    if os.environ.get("TERM") == "dumb" and not os.environ.get("FORCE_COLOR"):
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-12 if target is sys.stderr else -11)
            mode = ctypes.c_ulong()
            return bool(
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                and kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            )
        except Exception:
            return False
    return True


def supports_unicode(stream: Any = None) -> bool:
    """Return True if stream encoding supports Unicode box-drawing characters."""
    target = stream if stream is not None else sys.stdout
    enc = getattr(target, "encoding", None) or "utf-8"
    try:
        "━╺─╭╮╰╯✓✗✦".encode(enc)
        return True
    except Exception:
        return False


class Palette:
    """Amethyst/Silver/Ice-Crystal ANSI 256-color palette for Emilia."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    SILVER = "\033[38;5;255m"
    LAVENDER = "\033[38;5;183m"
    VIOLET = "\033[38;5;141m"
    DEEP_VIOLET = "\033[38;5;135m"
    AMETHYST = "\033[38;5;177m"
    ICE_BLUE = "\033[38;5;159m"
    EMERALD = "\033[38;5;120m"
    GOLD = "\033[38;5;221m"
    ROSE = "\033[38;5;204m"
    MUTED = "\033[38;5;245m"


def style(text: str, *codes: str, enabled: bool | None = None) -> str:
    """Apply ANSI color/style codes when supported."""
    use_color = supports_ansi() if enabled is None else enabled
    if not use_color or not codes:
        return text
    return "".join(codes) + text + Palette.RESET


def render_emilia_banner(*, color: bool | None = None, subtitle: str | None = None) -> str:
    """Render the signature -- / \\ / slant 'EMILIA' ASCII banner with silver-to-violet gradient."""
    use_color = supports_ansi() if color is None else color
    gradient = (
        Palette.SILVER,
        Palette.LAVENDER,
        Palette.AMETHYST,
        Palette.VIOLET,
        Palette.DEEP_VIOLET,
    )
    rule_top = "  +" + "--/\\--" * 11 + "+"
    rule_bot = "  +" + "--\\/--" * 11 + "+"

    lines: list[str] = [style(rule_top, Palette.MUTED, enabled=use_color)]
    for idx, raw_line in enumerate(EMILIA_SLANT_ASCII):
        tone = gradient[idx % len(gradient)]
        lines.append("  " + style(raw_line, Palette.BOLD, tone, enabled=use_color))
    lines.append(style(rule_bot, Palette.MUTED, enabled=use_color))
    lines.append(style(EMILIA_TAGLINE, Palette.BOLD, Palette.ICE_BLUE, enabled=use_color))
    if subtitle:
        lines.append(style(f"  >> {subtitle}", Palette.LAVENDER, enabled=use_color))
    return "\n".join(lines)


def print_banner(*, stream: Any = None, subtitle: str | None = None) -> None:
    """Print the Emilia ASCII banner to the target stream."""
    out = stream if stream is not None else sys.stdout
    use_color = supports_ansi(out)
    print(render_emilia_banner(color=use_color, subtitle=subtitle), file=out, flush=True)


def format_bytes(num_bytes: int | float) -> str:
    """Format byte count into human-readable KiB/MiB/GiB."""
    val = float(max(0, num_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            return f"{val:.2f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024.0
    return f"{val:.2f} TiB"


def format_duration(seconds: float) -> str:
    """Format seconds as mm:ss or X.Xs."""
    if seconds < 0 or not (seconds < 1e9):
        return "--:--"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(round(seconds)), 60)
    if mins < 60:
        return f"{mins:02d}:{secs:02d}"
    hours, mins = divmod(mins, 60)
    return f"{hours:d}:{mins:02d}:{secs:02d}"


def render_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    title: str | None = None,
    color: bool | None = None,
) -> str:
    """Render an aligned ASCII/Unicode table for CLI output."""
    use_color = supports_ansi() if color is None else color
    import unicodedata

    def cell_width(text: str) -> int:
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in text)

    def pad(text: str, width: int) -> str:
        return text + ' ' * max(0, width - cell_width(text))

    col_count = len(headers)
    widths = [cell_width(h) for h in headers]
    for row in rows:
        for i in range(min(col_count, len(row))):
            widths[i] = max(widths[i], cell_width(str(row[i])))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out_lines: list[str] = []
    if title:
        out_lines.append(style(f"=== {title} ===", Palette.BOLD, Palette.LAVENDER, enabled=use_color))
    out_lines.append(style(sep, Palette.MUTED, enabled=use_color))
    header_cells = [
        style(pad(headers[i], widths[i]), Palette.BOLD, Palette.ICE_BLUE, enabled=use_color)
        for i in range(col_count)
    ]
    out_lines.append("| " + " | ".join(header_cells) + " |")
    out_lines.append(style(sep, Palette.MUTED, enabled=use_color))
    for row in rows:
        padded = [pad(str(row[i]), widths[i]) if i < len(row) else " " * widths[i] for i in range(col_count)]
        out_lines.append("| " + " | ".join(padded) + " |")
    out_lines.append(style(sep, Palette.MUTED, enabled=use_color))
    return "\n".join(out_lines)


class ProgressBar:
    """Terminal progress bar supporting both step-based ComfyUI sampling and byte-based downloads."""

    def __init__(
        self,
        label: str,
        total: int = 100,
        *,
        unit: str = "step",
        width: int = 28,
        stream: Any = None,
        enabled: bool = True,
    ) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.current = 0
        self.unit = unit
        self.width = width
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self.is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.use_color = supports_ansi(self.stream)
        self.use_unicode = supports_unicode(self.stream)
        self.start_time = time.monotonic()
        self._last_logged_step = -1
        self._last_line_width = 0
        self.initial_current = 0

    def update(self, current: int, total: int | None = None, *, sub_label: str | None = None) -> None:
        if not self.enabled:
            return
        if total is not None:
            self.total = max(0, int(total))
        self.current = max(0, int(current))
        elapsed = max(1e-4, time.monotonic() - self.start_time)
        frac = min(1.0, self.current / self.total) if self.total else None
        tag = sub_label or self.label
        if self.unit == "bytes":
            count = format_bytes(self.current)
            detail = f"{count}/{format_bytes(self.total)}" if self.total else count
            speed = max(0, self.current - self.initial_current) / elapsed
            eta = format_duration(max(0, self.total - self.current) / speed) if self.total and speed else '--:--'
            detail = f"ETA {eta} {detail} ({format_bytes(speed)}/s)"
        else:
            detail = f"{self.current}/{self.total}" if self.total else str(self.current)

        if self.is_tty:
            filled = int(round(self.width * frac)) if frac is not None else 0
            fill, empty = ("━", "─") if self.use_unicode else ("#", "-")
            bar = fill * filled + empty * (self.width - filled)
            percentage = f"{frac * 100:5.1f}%" if frac is not None else "   --%"
            plain = f"  {tag[:18]:<18} [{bar}] {percentage} {detail}"
            cols = max(1, shutil.get_terminal_size((100, 24)).columns - 1)
            # Truncate plain text, never an ANSI sequence. Pad before coloring.
            plain = plain[:cols].ljust(min(cols, self._last_line_width))
            self._last_line_width = len(plain)
            self.stream.write("\r" + style(plain, Palette.LAVENDER, enabled=self.use_color))
            self.stream.flush()
        else:
            interval = max(1, self.total // 5) if self.total else 1024 * 1024
            if self._last_logged_step < 0 or self.current - self._last_logged_step >= interval:
                self._last_logged_step = self.current
                percentage = f"{frac * 100:.1f}% " if frac is not None else ""
                print(f"{tag}: {percentage}{detail}", file=self.stream, flush=True)

    def finish(self, message: str | None = None) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.start_time
        if self.is_tty:
            self.stream.write("\r" + " " * self._last_line_width + "\r")
            self.stream.flush()
        done_msg = message or f"{self.label}: completed in {elapsed:.1f}s"
        mark = "✓" if self.use_unicode else "[OK]"
        print(
            f"  {style(mark, Palette.BOLD, Palette.EMERALD, enabled=self.use_color)} {done_msg}",
            file=self.stream,
            flush=True,
        )


def download_file_with_progress(
    url: str,
    dest_path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    label: str | None = None,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 256,
) -> Path:
    """Resume into .part; validate HTTP framing, size and optional SHA256 before replacing.

    Failed transfers retain their partial file and never replace the destination.
    A 416 response is only complete when its advertised total matches the partial.
    Without an expected digest, HTTP length validation cannot establish file identity.
    """
    if expected_bytes is not None and expected_bytes < 0:
        raise ValueError("expected_bytes must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        expected_sha256 = expected_sha256.lower()
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    existing_size = part_path.stat().st_size if part_path.is_file() else 0
    headers = {"User-Agent": "EmiliaStudio-CLI/2.1", "Accept-Encoding": "identity"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    bar = ProgressBar(label or dest_path.name, total=expected_bytes or 0, unit="bytes")
    total_bytes = expected_bytes
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP {status} downloading {url}")
            if resp.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RuntimeError("Cannot validate an encoded download response")
            content_length = resp.headers.get("Content-Length")
            if content_length is not None and not re.fullmatch(r"[0-9]+", content_length):
                raise RuntimeError("Invalid Content-Length in download response")
            response_bytes = int(content_length) if content_length is not None else None
            if status == 206:
                match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", resp.headers.get("Content-Range", ""))
                if match is None:
                    raise RuntimeError("Missing or invalid Content-Range in download response")
                start, end, total_bytes = map(int, match.groups())
                if start != existing_size or end < start or end >= total_bytes:
                    raise RuntimeError("Content-Range does not match the requested download offset")
                range_bytes = end - start + 1
                if response_bytes is not None and response_bytes != range_bytes:
                    raise RuntimeError("Content-Length does not match Content-Range")
                response_bytes = range_bytes
            else:
                # An ignored Range is a full response, never append it to old bytes.
                existing_size = 0
                total_bytes = response_bytes if response_bytes is not None else expected_bytes
            if expected_bytes is not None and total_bytes is not None and total_bytes != expected_bytes:
                raise RuntimeError(f"Downloaded size mismatch: expected {expected_bytes} B, server reports {total_bytes} B")
            downloaded = existing_size
            bar.initial_current = existing_size
            bar.start_time = time.monotonic()
            received = 0
            with part_path.open("ab" if existing_size else "wb") as out_f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    received += len(chunk)
                    downloaded += len(chunk)
                    if response_bytes is not None and received > response_bytes:
                        raise RuntimeError("Download exceeds the advertised response length")
                    if total_bytes is not None and downloaded > total_bytes:
                        raise RuntimeError("Download exceeds the expected file length")
                    out_f.write(chunk)
                    # Reserve completion output until validation and atomic promotion.
                    if total_bytes is None or downloaded < total_bytes:
                        bar.update(downloaded, total=total_bytes or 0)
            if response_bytes is not None and received != response_bytes:
                raise RuntimeError(f"Truncated download: expected {response_bytes} response bytes, got {received}")
    except urllib.error.HTTPError as exc:
        with exc:
            match = re.fullmatch(r"bytes \*/([0-9]+)", exc.headers.get("Content-Range", ""))
            if exc.code != 416 or not existing_size or match is None or int(match[1]) != existing_size:
                raise RuntimeError(f"HTTP {exc.code} downloading {url}: {exc.reason}") from exc
            total_bytes = int(match[1])
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    actual = part_path.stat().st_size
    for required_size in (total_bytes, expected_bytes):
        if required_size is not None and actual != required_size:
            raise RuntimeError(f"Downloaded size mismatch for {dest_path.name}: expected {required_size} B, got {actual} B")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with part_path.open("rb") as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch for {dest_path.name}; remove {part_path} before retrying")
    part_path.replace(dest_path)
    bar.finish(f"{label or dest_path.name}: downloaded {format_bytes(actual)} -> {dest_path}")
    return dest_path
