"""Unified rendering engine for Emilia Studio with quality gate and memory lifecycle management."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

from .config import StudioConfig, load_config
from .inspector import InspectionReport, QualityInspector
from .ledger import ExperimentLedger
from .workflows import EmiliaWorkflowBuilder


def _output_path(app_dir: Path, meta: dict[str, Any]) -> Path:
    subfolder = meta.get("subfolder", "")
    filename = meta["filename"]
    return (app_dir / "output" / subfolder / filename) if subfolder else (app_dir / "output" / filename)


@dataclass
class StageResult:
    """Result of a single stage execution in ComfyUI."""

    stage: str
    prompt_id: str
    seconds: float
    outputs: dict[str, Any]
    submitted_graph: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "seconds": self.seconds,
            "outputs": self.outputs,
            "submitted_graph": self.submitted_graph,
        }


class EmiliaRenderEngine:
    """Orchestrates two-stage high-res generation, quality inspection, and finishing."""

    def __init__(
        self,
        config: StudioConfig | None = None,
        builder: EmiliaWorkflowBuilder | None = None,
        inspector: QualityInspector | None = None,
        ledger: ExperimentLedger | None = None,
    ) -> None:
        self.config = config or load_config()
        self.builder = builder or EmiliaWorkflowBuilder(self.config)
        self.inspector = inspector or QualityInspector(app_dir=self.config.app_dir)
        self.ledger = ledger or ExperimentLedger(config=self.config)

    async def _upload_image(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        image_path: Path,
        upload_filename: str = "emilia_quality_selected_native.png",
    ) -> str:
        with image_path.open("rb") as image_file:
            data = aiohttp.FormData()
            data.add_field(
                "image",
                image_file,
                filename=upload_filename,
                content_type="image/png",
            )
            data.add_field("overwrite", "false")
            async with session.post(f"{base_url}/upload/image", data=data) as response:
                response.raise_for_status()
                uploaded = await response.json()
        return "/".join(part for part in (uploaded.get("subfolder", ""), uploaded["name"]) if part)

    async def _execute_graph_on_session(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        graph: dict[str, Any],
        stage: str,
        log_dir: Path | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> StageResult:
        if stage in ("finish", "eye_repair"):
            selected = Path(graph["1"]["inputs"]["image"])
            uploaded_ref = await self._upload_image(session, base_url, selected)
            graph["1"]["inputs"]["image"] = uploaded_ref

        client_id = str(uuid.uuid4())
        ws_url = base_url.replace("http:", "ws:") + "/ws"
        async with session.ws_connect(
            ws_url,
            params={"clientId": client_id},
            max_msg_size=32 * 1024**2,
        ) as ws:
            started = time.monotonic()
            payload_body: dict[str, Any] = {"prompt": graph, "client_id": client_id}
            if extra_pnginfo:
                payload_body["extra_data"] = {"extra_pnginfo": extra_pnginfo}
            async with session.post(f"{base_url}/prompt", json=payload_body) as response:
                submitted = await response.json()
                if response.status != 200:
                    raise RuntimeError(submitted)
            prompt_id = submitted["prompt_id"]
            print(f"{stage}: queued {prompt_id}", flush=True)
            while True:
                message = await ws.receive()
                if message.type == aiohttp.WSMsgType.BINARY:
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(f"{stage}: WebSocket closed before completion")
                event = json.loads(message.data)
                kind, payload = event["type"], event["data"]
                if payload.get("prompt_id") != prompt_id:
                    continue
                if kind in ("execution_error", "execution_interrupted"):
                    raise RuntimeError(f"{stage}: {payload}")
                if kind == "progress" and (
                    payload["value"] == 1
                    or payload["value"] == payload["max"]
                    or payload["value"] % 8 == 0
                ):
                    print(f'{stage}: {payload["value"]}/{payload["max"]}', flush=True)
                if kind == "executing" and payload.get("node") is None:
                    break

            async with session.get(f"{base_url}/history/{prompt_id}") as response:
                response.raise_for_status()
                history = (await response.json())[prompt_id]
            if history["status"]["status_str"] != "success":
                raise RuntimeError(f'{stage}: {history["status"]}')
            elapsed = time.monotonic() - started
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
            print(f"{stage}: completed in {elapsed:.1f}s", flush=True)
            return StageResult(
                stage=stage,
                prompt_id=prompt_id,
                seconds=elapsed,
                outputs=history["outputs"],
                submitted_graph=graph,
            )

    async def free_server_memory(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
    ) -> bool:
        """Explicitly instruct ComfyUI server to unload all models and free VRAM/RAM."""
        try:
            async with session.post(
                f"{base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def render_stage_isolated(
        self,
        graph: dict[str, Any],
        stage: str,
        run_dir: Path,
        port: int,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single stage in an isolated ComfyUI child process and exit cleanly."""
        assert self.config.app_dir is not None
        base = f"http://127.0.0.1:{port}"
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"Port {port} is occupied; existing ComfyUI was not stopped."
                )
        home = run_dir / stage
        (home / "user").mkdir(parents=True, exist_ok=True)
        (home / "temp").mkdir(parents=True, exist_ok=True)
        python_exe = str(self.config.conda_python or sys.executable)
        command = [
            python_exe,
            "-u",
            str(self.config.app_dir / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--reserve-vram",
            "0.8",
            "--fp32-vae",
            "--disable-pinned-memory",
            "--preview-method",
            "none",
            "--disable-api-nodes",
            "--disable-auto-launch",
            "--user-directory",
            str(home / "user"),
            "--temp-directory",
            str(home / "temp"),
            "--database-url",
            "sqlite:///" + (home / "user" / "render.db").as_posix(),
            "--log-stdout",
        ]
        environment = os.environ.copy()
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1",
        )
        with (home / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self.config.app_dir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60),
                    trust_env=False,
                ) as session:
                    deadline = time.monotonic() + 300
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f'{stage} server exited; see {home / "server.log"}'
                            )
                        try:
                            async with session.get(
                                base + "/system_stats",
                                timeout=aiohttp.ClientTimeout(total=3),
                            ) as response:
                                if response.status == 200:
                                    break
                        except (aiohttp.ClientError, TimeoutError):
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f'{stage} server did not start; see {home / "server.log"}'
                            )
                        await asyncio.sleep(0.5)

                    stage_res = await self._execute_graph_on_session(
                        session=session,
                        base_url=base,
                        graph=graph,
                        stage=stage,
                        log_dir=home,
                        extra_pnginfo=extra_pnginfo,
                    )
                    return stage_res.to_dict()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait, timeout=10)

    async def render_quality_pipeline(
        self,
        *,
        seed: int = 20261027,
        port: int = 8199,
        pag: bool = False,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        ipadapter_weight: float = 0.4,
        user_idea: str | None = None,
        strict_seed_lock: bool = True,
        auto_eye_repair: bool = False,
        max_seed_retries: int = 0,
        isolation_mode: str = "process",
    ) -> dict[str, Any]:
        """Execute full HighRes Native -> Quality Gate -> Finish 2x pipeline and record to ledger."""
        assert self.config.app_dir is not None
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.config.app_dir / "temp" / ("quality_" + run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        current_seed = seed
        attempt = 0
        generation: dict[str, Any] = {}
        selected: Path | None = None
        native_report: InspectionReport | None = None

        while True:
            attempt += 1
            native_graph = self.builder.build_highres_native(
                seed=current_seed,
                pag=pag,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                ipadapter_weight=ipadapter_weight,
            )
            generation = await self.render_stage_isolated(
                native_graph,
                "native" if attempt == 1 else f"native_retry_{attempt}",
                run_dir,
                port,
            )
            selected = _output_path(
                self.config.app_dir,
                generation["outputs"]["9"]["images"][0],
            )
            with Image.open(selected) as image:
                if image.size != (1152, 1728):
                    raise RuntimeError(
                        "Native output dimensions differ from the quality workflow contract."
                    )

            native_report = self.inspector.inspect_image(
                selected,
                expected_size=(1152, 1728),
                stage="native",
                solo_prompt=True,
                strict_seed_lock=strict_seed_lock,
            )

            if (
                not strict_seed_lock
                and native_report.recommended_action == "reject_seed"
                and attempt <= max_seed_retries
            ):
                print(
                    f"quality-gate: seed {current_seed} rejected ({native_report.reasons}); retrying with new seed...",
                    flush=True,
                )
                current_seed += 1000
                continue
            break

        assert selected is not None and native_report is not None

        eye_repair_result: dict[str, Any] | None = None
        if (
            not strict_seed_lock
            and auto_eye_repair
            and native_report.recommended_action == "repair_eyes"
        ):
            print(
                f"quality-gate: eye drift detected ({native_report.reasons}); running targeted eye repair...",
                flush=True,
            )
            repair_graph = self.builder.build_eye_repair(
                image_path=str(selected),
                seed=current_seed + 9,
            )
            eye_repair_result = await self.render_stage_isolated(
                repair_graph,
                "eye_repair",
                run_dir,
                port,
            )
            selected = _output_path(
                self.config.app_dir,
                eye_repair_result["outputs"]["22"]["images"][0],
            )

        finish_graph = self.builder.build_highres_finish(
            image_path=str(selected),
            seed=current_seed,
            pag=pag,
        )
        finishing = await self.render_stage_isolated(
            finish_graph,
            "finish",
            run_dir,
            port,
        )
        final = _output_path(
            self.config.app_dir,
            finishing["outputs"]["16"]["images"][0],
        )
        with Image.open(final) as image:
            if image.size != (2304, 3456):
                raise RuntimeError(
                    "Finished output dimensions differ from the quality workflow contract."
                )

        final_report = self.inspector.inspect_image(
            final,
            expected_size=(2304, 3456),
            stage="final",
            solo_prompt=True,
            strict_seed_lock=strict_seed_lock,
        )

        receipt: dict[str, Any] = {
            "seed": current_seed,
            "guidance": "PAG scale 1.0" if pag else "baseline",
            "python": sys.executable,
            "isolation": "Native process exited before finishing process started; no model overlap across stages.",
            "temperature_power_software_stop": False,
            "native": str(selected),
            "final": str(final),
            "generation": generation,
            "finishing": finishing,
            "logs": str(run_dir),
            "quality_inspection": {
                "native": native_report.to_dict(),
                "final": final_report.to_dict(),
                "eye_repair": eye_repair_result,
            },
        }
        if user_idea:
            receipt["user_idea"] = user_idea

        receipt_path = final.with_suffix(".run.json")
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        try:
            self.ledger.record_run(
                seed=current_seed,
                mode="quality_pag" if pag else "quality",
                guidance=receipt["guidance"],
                prompt_idea=user_idea,
                positive_prompt=positive_prompt or self.config.base_positive_prompt,
                native_path=str(selected),
                final_path=str(final),
                native_sha256=native_report.pixel_sha256,
                final_sha256=final_report.pixel_sha256,
                native_seconds=generation.get("seconds", 0.0),
                finish_seconds=finishing.get("seconds", 0.0),
                total_seconds=float(generation.get("seconds", 0.0))
                + float(finishing.get("seconds", 0.0)),
                face_count=native_report.face_count,
                hand_count=native_report.hand_count,
                heterochromia_score=native_report.heterochromia_score,
                eye_drift_detected=native_report.eye_drift_detected,
                quality_action=native_report.recommended_action,
                receipt_path=str(receipt_path.resolve()),
            )
        except Exception:
            pass

        print(f"FINISHED: {final}", flush=True)
        print(f"RECEIPT: {receipt_path}", flush=True)
        return receipt
