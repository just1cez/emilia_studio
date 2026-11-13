"""SQLite-backed experiment ledger for Emilia Studio."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StudioConfig, load_config


_INITIALIZED_DBS: set[str] = set()

_GIB = 1024**3


def _file_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _receipt_row(data: dict[str, Any]) -> dict[str, Any]:
    """Ledger columns derivable from a pipeline ``.run.json`` receipt."""
    finishing = data.get("finishing") or {}
    inspection = data.get("quality_inspection") or {}
    native = inspection.get("native") or {}
    final = inspection.get("final") or {}
    eye_repair = inspection.get("eye_repair") or data.get("eye_repair") or {}
    # Every Stage 1 attempt counts; receipts older than gate retries only have `generation`.
    generations = [a.get("generation") or {} for a in data.get("attempts") or []] or [data.get("generation") or {}]
    native_seconds = sum(float(g.get("seconds") or 0.0) for g in generations)
    finish_seconds = float(finishing.get("seconds") or 0.0)
    repair_seconds = float(eye_repair.get("seconds") or 0.0) if isinstance(eye_repair, dict) else 0.0
    profile = str(data.get("profile") or "quality")
    pag = bool((data.get("selected_settings") or {}).get("pag")) or "PAG" in str(data.get("guidance") or "")
    return {
        "timestamp": data.get("timestamp"),
        "seed": data.get("seed"),
        "mode": f"{profile}_pag" if pag and not profile.endswith("_pag") else profile,
        "guidance": data.get("guidance", "baseline"),
        "prompt_idea": data.get("user_idea"),
        "native_path": data.get("native"),
        "final_path": data.get("final"),
        "native_sha256": native.get("pixel_sha256"),
        "final_sha256": final.get("pixel_sha256"),
        "native_seconds": native_seconds,
        "finish_seconds": finish_seconds,
        "total_seconds": native_seconds + finish_seconds + repair_seconds,
        "face_count": native.get("face_count"),
        "hand_count": native.get("hand_count"),
        "heterochromia_score": native.get("heterochromia_score"),
        "eye_drift_detected": native.get("eye_drift_detected"),
        "quality_action": (
            "reject_seed"
            if data.get("status") == "quality_rejected"
            else (final.get("recommended_action") or native.get("recommended_action") or "accept")
        ),
    }


def _benchmark_row(data: dict[str, Any], round_num: int | None) -> dict[str, Any]:
    """Ledger columns of one ``scripts/benchmark_architecture_iterations.py`` record."""
    receipt = data.get("receipt") or {}
    hashes = data.get("pixel_hashes") or {}
    peak_rss = data.get("peak_sampled_rss")
    min_ram = data.get("min_available_ram")
    return {
        "seed": data.get("seed") or receipt.get("seed"),
        "mode": "benchmark",
        "guidance": receipt.get("guidance", "baseline"),
        "native_path": receipt.get("native"),
        "final_path": receipt.get("final"),
        "native_sha256": hashes.get("native"),
        "final_sha256": hashes.get("final"),
        "native_seconds": (receipt.get("generation") or {}).get("seconds"),
        "finish_seconds": (receipt.get("finishing") or {}).get("seconds"),
        "total_seconds": data.get("wall_seconds"),
        "peak_rss_gib": peak_rss / _GIB if peak_rss is not None else None,
        "min_avail_ram_gib": min_ram / _GIB if min_ram is not None else None,
        "round_num": round_num,
    }


class ExperimentLedger:
    """SQLite ledger tracking Emilia Studio renders and benchmark rounds."""

    _RUN_COLS: tuple[str, ...] = (
        "timestamp", "seed", "mode", "guidance", "prompt_idea", "positive_prompt",
        "native_path", "final_path", "native_sha256", "final_sha256",
        "native_seconds", "finish_seconds", "total_seconds",
        "peak_rss_gib", "min_avail_ram_gib", "face_count", "hand_count",
        "heterochromia_score", "eye_drift_detected", "quality_action",
        "receipt_path", "round_num",
    )

    def __init__(self, db_path: str | Path | None = None, config: StudioConfig | None = None) -> None:
        self.config = config or load_config()
        self.db_path = Path(db_path) if db_path else self.config.ledger_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.db_path.resolve())
        if key not in _INITIALIZED_DBS or not self.db_path.is_file() or self.db_path.stat().st_size == 0:
            self._init_db()
            _INITIALIZED_DBS.add(key)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    seed INTEGER,
                    mode TEXT,
                    guidance TEXT,
                    prompt_idea TEXT,
                    positive_prompt TEXT,
                    native_path TEXT,
                    final_path TEXT,
                    native_sha256 TEXT,
                    final_sha256 TEXT,
                    native_seconds REAL,
                    finish_seconds REAL,
                    total_seconds REAL,
                    peak_rss_gib REAL,
                    min_avail_ram_gib REAL,
                    face_count INTEGER,
                    hand_count INTEGER,
                    heterochromia_score REAL,
                    eye_drift_detected INTEGER,
                    quality_action TEXT,
                    receipt_path TEXT UNIQUE,
                    round_num INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_seed ON runs(seed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_round ON runs(round_num)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mode ON runs(mode)")

    def _normalize_row(self, values: dict[str, Any]) -> dict[str, Any]:
        row = {column: values.get(column) for column in self._RUN_COLS}
        if not row["timestamp"]:
            row["timestamp"] = datetime.now(timezone.utc).isoformat()
        if isinstance(row["eye_drift_detected"], bool):
            row["eye_drift_detected"] = int(row["eye_drift_detected"])
        for column, value in row.items():
            if isinstance(value, dict):
                row[column] = value.get("path") or value.get("image") or json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (list, tuple)):
                row[column] = json.dumps(value, ensure_ascii=False)
        return row

    def _upsert(self, conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
        """Insert rows; a row whose receipt is already indexed is updated in place.

        Updating keeps the row id (so `recent` ordering is stable) and keeps existing
        values the new source cannot provide, such as the prompt of a live render.
        """
        columns = ", ".join(self._RUN_COLS)
        placeholders = ", ".join(f":{column}" for column in self._RUN_COLS)
        updates = ", ".join(
            f"{column} = COALESCE(excluded.{column}, {column})"
            for column in self._RUN_COLS if column != "receipt_path"
        )
        conn.executemany(
            f"INSERT INTO runs ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(receipt_path) DO UPDATE SET {updates}",
            [self._normalize_row(row) for row in rows],
        )

    def record_run(self, **values: Any) -> None:
        """Index one finished or rejected render."""
        with closing(self._connect()) as conn, conn:
            self._upsert(conn, [values])

    def ingest_existing_artifacts(self, *, refresh: bool = False) -> int:
        """Index receipts and benchmark records found on disk; return how many were found.

        Already indexed files are skipped unless ``refresh`` re-reads all of them.
        """
        app_dir = self.config.app_dir
        assert app_dir is not None
        output_dir = self.config.output_dir or app_dir / "output"
        # Each source carries the function that turns its JSON into ledger columns.
        sources: list[tuple[Path, Any]] = [
            (path, _receipt_row) for path in sorted((output_dir / "emilia").rglob("*.run.json"))
        ]
        for round_dir in sorted((app_dir / "temp" / "architecture_iterations").glob("round_*")):
            if not round_dir.is_dir():
                continue
            try:
                round_num: int | None = int(round_dir.name.split("_")[-1])
            except ValueError:
                round_num = None
            sources.extend(
                (path, lambda data, round_num=round_num: _benchmark_row(data, round_num))
                for path in sorted(round_dir.glob("*.json")) if path.name != "summary.json"
            )

        with closing(self._connect()) as conn, conn:
            indexed = set() if refresh else {
                row[0] for row in conn.execute("SELECT receipt_path FROM runs WHERE receipt_path IS NOT NULL")
            }
            rows: list[dict[str, Any]] = []
            found = 0
            for path, to_row in sources:
                resolved = str(path.resolve())
                if resolved in indexed:
                    found += 1
                    continue
                try:
                    row = to_row(json.loads(path.read_text(encoding="utf-8")))
                    row["timestamp"] = row.get("timestamp") or _file_timestamp(path)
                except (OSError, ValueError, TypeError, AttributeError):
                    # A truncated or foreign JSON file is not a receipt; it must not block the rest.
                    continue
                row["receipt_path"] = resolved
                rows.append(row)
                found += 1
            if rows:
                self._upsert(conn, rows)
            return found

    def get_summary_stats(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    AVG(total_seconds) as avg_total_seconds,
                    AVG(native_seconds) as avg_native_seconds,
                    AVG(finish_seconds) as avg_finish_seconds,
                    COUNT(DISTINCT seed) as unique_seeds
                FROM runs
            """).fetchone()
            return dict(row) if row else {}

    def compare_rounds(self, round_a: int, round_b: int) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows_a = {r["seed"]: dict(r) for r in conn.execute("SELECT * FROM runs WHERE round_num = ?", (round_a,)).fetchall()}
            rows_b = {r["seed"]: dict(r) for r in conn.execute("SELECT * FROM runs WHERE round_num = ?", (round_b,)).fetchall()}

            common_seeds = sorted(set(rows_a.keys()) & set(rows_b.keys()))
            exact_matches = 0
            comparisons = []
            for s in common_seeds:
                ra, rb = rows_a[s], rows_b[s]
                match = (ra.get("final_sha256") == rb.get("final_sha256")) and (ra.get("final_sha256") is not None)
                if match:
                    exact_matches += 1
                comparisons.append({
                    "seed": s,
                    "pixel_match": match,
                    "round_a_sec": ra.get("total_seconds"),
                    "round_b_sec": rb.get("total_seconds")
                })
            return {
                "round_a": round_a,
                "round_b": round_b,
                "common_seeds": len(common_seeds),
                "exact_pixel_matches": exact_matches,
                "details": comparisons
            }

    def list_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
