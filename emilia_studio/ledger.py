"""SQLite-backed experiment ledger for Emilia Studio."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StudioConfig, load_config


class ExperimentLedger:
    """SQLite ledger tracking Emilia Studio renders and benchmark rounds."""

    def __init__(self, db_path: str | Path | None = None, config: StudioConfig | None = None) -> None:
        self.config = config or load_config()
        self.db_path = Path(db_path) if db_path else self.config.ledger_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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

    def record_run(self, **kwargs: Any) -> int:
        if "timestamp" not in kwargs or not kwargs["timestamp"]:
            kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        cols = [
            "timestamp", "seed", "mode", "guidance", "prompt_idea", "positive_prompt",
            "native_path", "final_path", "native_sha256", "final_sha256",
            "native_seconds", "finish_seconds", "total_seconds",
            "peak_rss_gib", "min_avail_ram_gib", "face_count", "hand_count",
            "heterochromia_score", "eye_drift_detected", "quality_action",
            "receipt_path", "round_num"
        ]
        
        filtered = {k: kwargs.get(k) for k in cols if k in kwargs}
        if "eye_drift_detected" in filtered and isinstance(filtered["eye_drift_detected"], bool):
            filtered["eye_drift_detected"] = 1 if filtered["eye_drift_detected"] else 0

        placeholders = ", ".join(["?"] * len(filtered))
        columns = ", ".join(filtered.keys())
        values = list(filtered.values())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT OR REPLACE INTO runs ({columns}) VALUES ({placeholders})",
                values,
            )
            return cursor.lastrowid or 0

    def ingest_existing_artifacts(self, app_dir: str | Path | None = None) -> int:
        app_path = Path(app_dir) if app_dir else self.config.app_dir
        assert app_path is not None
        count = 0

        # 1. Scan output/emilia/quality/*.run.json
        quality_dir = app_path / "output" / "emilia" / "quality"
        if quality_dir.is_dir():
            for run_file in quality_dir.glob("*.run.json"):
                try:
                    data = json.loads(run_file.read_text(encoding="utf-8"))
                    gen = data.get("generation", {})
                    fin = data.get("finishing", {})
                    n_sec = gen.get("seconds", 0.0)
                    f_sec = fin.get("seconds", 0.0)
                    self.record_run(
                        seed=data.get("seed"),
                        mode="quality",
                        guidance=data.get("guidance", "baseline"),
                        native_path=data.get("native"),
                        final_path=data.get("final"),
                        native_seconds=n_sec,
                        finish_seconds=f_sec,
                        total_seconds=n_sec + f_sec,
                        receipt_path=str(run_file.resolve()),
                    )
                    count += 1
                except Exception:
                    continue

        # 2. Scan temp/architecture_iterations/round_*/*.json
        iter_dir = app_path / "temp" / "architecture_iterations"
        if iter_dir.is_dir():
            for round_dir in iter_dir.glob("round_*"):
                if not round_dir.is_dir():
                    continue
                try:
                    round_num = int(round_dir.name.split("_")[-1])
                except ValueError:
                    round_num = None
                for rec_file in round_dir.glob("*.json"):
                    if rec_file.name == "summary.json":
                        continue
                    try:
                        data = json.loads(rec_file.read_text(encoding="utf-8"))
                        receipt = data.get("receipt", {})
                        hashes = data.get("pixel_hashes", {})
                        gen = receipt.get("generation", {})
                        fin = receipt.get("finishing", {})
                        self.record_run(
                            seed=data.get("seed") or receipt.get("seed"),
                            mode="benchmark",
                            guidance=receipt.get("guidance", "baseline"),
                            native_path=receipt.get("native"),
                            final_path=receipt.get("final"),
                            native_sha256=hashes.get("native"),
                            final_sha256=hashes.get("final"),
                            native_seconds=gen.get("seconds"),
                            finish_seconds=fin.get("seconds"),
                            total_seconds=data.get("wall_seconds"),
                            peak_rss_gib=data.get("peak_rss_bytes", 0) / (1024**3) if "peak_rss_bytes" in data else None,
                            min_avail_ram_gib=data.get("min_available_ram_bytes", 0) / (1024**3) if "min_available_ram_bytes" in data else None,
                            receipt_path=str(rec_file.resolve()),
                            round_num=round_num,
                        )
                        count += 1
                    except Exception:
                        continue
        return count

    def get_summary_stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
