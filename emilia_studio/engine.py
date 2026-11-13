"""Two-stage isolated rendering engine: process lifecycle, quality-gate routing and receipts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .config import StudioConfig, load_config
from .inspector import InspectionReport, QualityInspector
from .ledger import ExperimentLedger
from .workflows import (
    EYE_REPAIR_OUTPUT_NODE,
    FINAL_OUTPUT_NODE,
    NATIVE_OUTPUT_NODE,
    LIGHTNING_PREVIEW_UNET,
    LIGHTNING_PREVIEW_STEPS,
    EmiliaWorkflowBuilder,
)

FP16_VAE_NAME = "sdxl-vae-fp16-fix.safetensors"

# Custom nodes each stage may load. Stage 1 and Stage 2 models never share a process.
_STAGE_CUSTOM_NODES: dict[str, tuple[str, ...]] = {
    "native": ("comfyui-ipadapter-plus", "emilia_runtime_patches"),
    "finish": ("comfyui-impact-pack", "comfyui-impact-subpack", "emilia_runtime_patches"),
}

# Output sizes, seeds and file paths remain controlled by the pipeline contract.
_NATIVE_OPTION_KEYS = frozenset({
    "pag_scale", "base_width", "base_height", "base_steps", "hires_steps",
    "cfg_base", "cfg_hires", "hires_denoise", "rescale_cfg_multiplier",
    "lora_name", "lora_strength_base", "lora_strength_hires",
    "skirt_ipadapter_weight", "ipadapter_start_at", "ipadapter_end_at",
    "framing", "camera_angle", "reference_mode", "hires_bridge",
    "checkpoint_name", "downscale_factor",
})
_FINISH_OPTION_KEYS = frozenset({
    "lora_name", "lora_strength", "face_denoise", "hand_denoise",
    "face_steps", "hand_steps", "face_guide_size", "face_max_size",
    "hand_guide_size", "hand_max_size", "checkpoint_name", "garment_region",
    "garment_denoise", "garment_steps", "upscale_blend_strength",
    "upscaler_name", "secondary_upscaler_name",
})


def _output_path(output_dir: Path, meta: dict[str, Any]) -> Path:
    """Resolve SaveImage metadata against the directory passed to ComfyUI."""
    return output_dir / meta.get("subfolder", "") / meta["filename"]


def _assert_port_available(host: str, port: int) -> None:
    """Check port availability in <1 ms via exclusive socket bind instead of 2s Winsock SYN retry."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is occupied; existing ComfyUI was not stopped."
            ) from exc


_RUN_LOCK_NAME = ".lock"


def _acquire_exclusive(handle: Any) -> None:
    """Take a non-blocking exclusive OS lock on an open file; OSError when someone else holds it."""
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Mark a run directory as in use for as long as its pipeline is running.

    The OS releases the lock when the process ends, however it ends, so a crashed
    run never looks active afterwards.
    """
    with (run_dir / _RUN_LOCK_NAME).open("a+b") as handle:
        _acquire_exclusive(handle)
        yield


def run_is_active(run_dir: Path) -> bool:
    """Whether a live pipeline still holds the directory (see :func:`run_lock`)."""
    lock_path = run_dir / _RUN_LOCK_NAME
    if not lock_path.is_file():
        return False
    try:
        with lock_path.open("a+b") as handle:
            _acquire_exclusive(handle)
    except OSError:
        return True
    return False


class QualityGateRejected(RuntimeError):
    """Generation stopped by the gate; its completed work has a receipt."""

    def __init__(self, reason: str, receipt_path: Path, receipt: dict[str, Any]) -> None:
        super().__init__(f"{reason} Receipt: {receipt_path}")
        self.receipt_path = receipt_path
        self.receipt = receipt


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


@dataclass(frozen=True)
class RuntimeFlags:
    """Opt-in, non-bit-exact accelerations applied to every stage server of one run."""

    fp16_upscaler: bool = False
    fp16_vae: bool = False
    fast_tiled_vae: bool = False


@dataclass
class StageServer:
    """One live isolated ComfyUI child process and the HTTP session bound to it."""

    session: Any
    base_url: str
    home: Path
    temp_dir: Path
    process: subprocess.Popen[Any] | None


@dataclass(frozen=True)
class _RenderPlan:
    """Validated settings shared by every seed of one pipeline run."""

    profile: str
    port: int
    pag: bool
    positive_prompt: str | None
    negative_prompt: str | None
    ipadapter_weight: float
    user_idea: str | None
    strict_seed_lock: bool
    auto_eye_repair: bool
    max_seed_retries: int
    native_options: dict[str, Any]
    finish_options: dict[str, Any]
    native_steps: int | None
    face_steps: int | None
    hand_steps: int | None
    flags: RuntimeFlags
    framing: str
    require_pose: bool
    preview: bool = False

    @property
    def is_ultra(self) -> bool:
        return self.profile == "ultra"

    @property
    def gated(self) -> bool:
        """Whether inspection findings may change the stage sequence."""
        return not self.strict_seed_lock


@dataclass
class _RenderItem:
    """Mutable progress of one requested seed through both stages."""

    requested_seed: int
    seed: int
    attempts: list[dict[str, Any]] = field(default_factory=list)
    native_graph: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    native: Path | None = None
    selected: Path | None = None
    native_report: InspectionReport | None = None
    eye_repair: dict[str, Any] | None = None
    repaired_report: InspectionReport | None = None
    finishing: dict[str, Any] = field(default_factory=dict)
    final: Path | None = None
    final_report: InspectionReport | None = None
    rejection: str | None = None
    receipt: dict[str, Any] | None = None

    @property
    def settled(self) -> bool:
        """Finished or rejected: the only states that get a receipt."""
        return self.rejection is not None or self.final_report is not None

    @property
    def needs_eye_repair(self) -> bool:
        return (
            self.rejection is None
            and self.native_report is not None
            and self.native_report.recommended_action == "repair_eyes"
        )


def _native_rejection(plan: _RenderPlan, report: InspectionReport) -> str | None:
    if not report.dimensions_match:
        return "Native output dimensions differ from the quality workflow contract."
    if not plan.gated:
        return None
    if report.recommended_action == "reject_seed":
        return "Quality gate rejected all permitted seed attempts."
    if not report.passed and report.recommended_action != "repair_eyes":
        return "Native image did not pass the quality gate."
    return None


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
        self.inspector = inspector or QualityInspector(
            app_dir=self.config.app_dir, python_executable=self.config.conda_python
        )
        self.ledger = ledger or ExperimentLedger(config=self.config)

    @property
    def _output_dir(self) -> Path:
        assert self.config.app_dir is not None
        return (self.config.output_dir or self.config.app_dir / "output").resolve()

    def _python_executable(self) -> str:
        if self.config.conda_python is not None:
            executable = str(self.config.conda_python)
            if getattr(sys, "frozen", False):
                resolved = shutil.which(executable) or executable
                if Path(resolved).resolve() == Path(sys.executable).resolve():
                    raise RuntimeError("conda_python must be an external Python interpreter, not Emilia.")
            return executable
        if getattr(sys, "frozen", False):
            raise RuntimeError("Configure conda_python with an external Python interpreter to launch ComfyUI.")
        return sys.executable

    def _stage_custom_node_whitelist(self, stage: str) -> list[str]:
        """Return the stage-minimal custom_nodes whitelist when all of its directories exist."""
        if self.config.app_dir is None:
            return []
        cn_root = self.config.app_dir / "custom_nodes"
        desired = _STAGE_CUSTOM_NODES[stage]
        missing = [name for name in desired if not (cn_root / name).exists()]
        if missing:
            # ComfyUI then reports the missing node class itself; say why the boot is slower.
            print(
                f"{stage}: {', '.join(missing)} not found in {cn_root}; loading every custom node instead of the stage whitelist.",
                file=sys.stderr, flush=True,
            )
            return []
        return list(desired)

    def _prewarm_stage2_files_in_ram(self, flags: RuntimeFlags) -> None:
        """Sequentially read Stage 2 exclusive model files into OS standby RAM cache while HDD is idle during Stage 1."""
        if self.config.app_dir is None:
            return
        models = self.config.app_dir / "models"
        candidates = [
            models / "sams" / "sam_vit_b_01ec64.pth",
            models / "upscale_models" / "RealESRGAN_x4plus_anime_6B.pth",
            models / "ultralytics" / "bbox" / "face_yolov8n.pt",
            models / "ultralytics" / "bbox" / "hand_yolov8n.pt",
        ]
        if flags.fp16_vae:
            candidates.append(models / "vae" / FP16_VAE_NAME)
        view = memoryview(bytearray(4 * 1024 * 1024))
        for path in candidates:
            if not path.is_file():
                continue
            try:
                with path.open("rb", buffering=0) as stream:
                    while stream.readinto(view):
                        pass
            except OSError:
                pass

    def _resolve_local_load_image_ref(self, image_path: Path, stage_temp: Path | None = None) -> str | None:
        """Return zero-copy ComfyUI LoadImage annotated reference if image_path is inside local output/input/temp dirs."""
        if self.config.app_dir is None or not image_path.is_file():
            return None
        resolved = image_path.resolve()
        temp_dir = stage_temp or self.config.temp_dir or self.config.app_dir / "temp"
        candidates = [
            (self._output_dir, "[output]"),
            ((self.config.app_dir / "input").resolve(), "[input]"),
            (temp_dir.resolve(), "[temp]"),
        ]
        for base_dir, tag in candidates:
            try:
                rel = resolved.relative_to(base_dir).as_posix()
                return f"{rel} {tag}" if tag != "[input]" else rel
            except ValueError:
                continue
        return None

    def _load_image_ref(self, image_path: Path, server: StageServer) -> str:
        reference = self._resolve_local_load_image_ref(image_path, server.temp_dir)
        if reference is None:
            raise FileNotFoundError(f"Stage input is not inside a ComfyUI output/input/temp directory: {image_path}")
        return reference

    async def _execute_graph(
        self,
        server: StageServer,
        *,
        graph: dict[str, Any],
        stage: str,
        log_dir: Path | None = None,
        on_first_progress: Callable[[], None] | None = None,
    ) -> StageResult:
        """Bound connect, submission, execution and history by one per-graph deadline."""
        timeout_s = self.config.stage_timeout_s
        deadline = asyncio.timeout(timeout_s if timeout_s > 0 else None)
        try:
            async with deadline:
                return await self._execute_graph_session(
                    server, graph=graph, stage=stage, log_dir=log_dir,
                    on_first_progress=on_first_progress,
                )
        except TimeoutError as exc:
            if deadline.expired():
                raise TimeoutError(
                    f"{stage}: execution exceeded stage_timeout_s ({timeout_s:g}s); "
                    f"see {server.home / 'server.log'}"
                ) from exc
            raise

    async def _execute_graph_session(
        self,
        server: StageServer,
        *,
        graph: dict[str, Any],
        stage: str,
        log_dir: Path | None,
        on_first_progress: Callable[[], None] | None,
    ) -> StageResult:
        """Queue one graph on a live stage server and wait for its outputs."""
        session, base_url = server.session, server.base_url
        client_id = str(uuid.uuid4())
        ws_url = base_url.replace("http:", "ws:") + "/ws"
        async with session.ws_connect(
            ws_url,
            params={"clientId": client_id},
            max_msg_size=32 * 1024**2,
        ) as ws:
            started = time.monotonic()
            async with session.post(
                f"{base_url}/prompt", json={"prompt": graph, "client_id": client_id}
            ) as response:
                submitted = await response.json()
                if response.status != 200:
                    # Missing models and invalid inputs surface here, named per node.
                    raise RuntimeError(
                        f"{stage}: ComfyUI rejected the workflow: {json.dumps(submitted, ensure_ascii=False)}"
                    )
            prompt_id = submitted["prompt_id"]
            print(f"{stage}: queued {prompt_id}", file=sys.stderr, flush=True)
            from .tui import ProgressBar

            pbar: ProgressBar | None = None
            current_max: int = 0
            current_node: str | None = None
            progress_node: str | None = None
            previous_value = -1
            pass_number = 0
            reset_progress = False
            stall_timeout = self.config.stall_timeout_s
            last_activity = started
            while True:
                if server.process is not None and server.process.poll() is not None:
                    raise RuntimeError(
                        f"{stage}: ComfyUI process exited mid-execution (code {server.process.returncode}); see {server.home / 'server.log'}"
                    )
                now = time.monotonic()
                wait_s: float | None = None
                if stall_timeout > 0:
                    rem_stall = (last_activity + stall_timeout) - now
                    if rem_stall <= 0:
                        raise TimeoutError(
                            f"{stage}: WebSocket stalled with no progress for {stall_timeout:.1f}s; see {server.home / 'server.log'}"
                        )
                    wait_s = rem_stall if wait_s is None else min(wait_s, rem_stall)
                recv_step = 5.0 if server.process is not None else wait_s
                if wait_s is not None and recv_step is not None:
                    recv_step = min(wait_s, recv_step)
                try:
                    message = (
                        await asyncio.wait_for(ws.receive(), timeout=recv_step)
                        if recv_step is not None
                        else await ws.receive()
                    )
                except TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.BINARY:
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(
                        f"{stage}: WebSocket closed before completion; see {server.home / 'server.log'}"
                    )
                event = json.loads(message.data)
                kind, payload = event["type"], event["data"]
                if payload.get("prompt_id") != prompt_id:
                    continue
                last_activity = time.monotonic()
                if kind in ("execution_error", "execution_interrupted"):
                    raise RuntimeError(f"{stage}: {payload}")
                if kind == "executing":
                    node = payload.get("node")
                    if node is None:
                        break
                    current_node = str(node)
                    reset_progress = True
                if kind == "progress":
                    if on_first_progress is not None:
                        on_first_progress()
                        on_first_progress = None
                    val = int(payload["value"])
                    mx = int(payload["max"])
                    node = payload.get("node", current_node)
                    node = str(node) if node is not None else None
                    if (
                        pbar is None or reset_progress or node != progress_node
                        or mx != current_max or val < previous_value
                        or (previous_value >= current_max and val <= previous_value)
                    ):
                        if pbar is not None and pbar.is_tty:
                            print(file=sys.stderr, flush=True)
                        pass_number = pass_number + 1 if node == progress_node else 1
                        current_max = mx
                        progress_node = node
                        label = f"{stage} node {node}" if node is not None else stage
                        pbar = ProgressBar(
                            f"{label} pass {pass_number}", total=mx, unit="step", stream=sys.stderr
                        )
                        reset_progress = False
                    pbar.update(val, total=mx)
                    previous_value = val

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
            if pbar is not None and pbar.is_tty:
                pbar.finish(f"{stage}: completed in {elapsed:.1f}s")
            else:
                print(f"{stage}: completed in {elapsed:.1f}s", file=sys.stderr, flush=True)
            return StageResult(
                stage=stage,
                prompt_id=prompt_id,
                seconds=elapsed,
                outputs=history["outputs"],
                submitted_graph=graph,
            )

    @asynccontextmanager
    async def _stage_server(
        self, stage: str, run_dir: Path, port: int, flags: RuntimeFlags
    ) -> AsyncIterator[StageServer]:
        """Launch one isolated ComfyUI process for every graph of a stage; it exits with the context."""
        assert self.config.app_dir is not None
        base = f"http://127.0.0.1:{port}"
        _assert_port_available("127.0.0.1", port)
        home = run_dir / stage
        (home / "user").mkdir(parents=True, exist_ok=True)
        stage_temp = (home / "temp").resolve()
        stage_temp.mkdir(parents=True, exist_ok=True)
        output_dir = self._output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self._python_executable(),
            "-u",
            str((self.config.app_dir / "main.py").resolve()),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--reserve-vram",
            "0.8",
            "--fp16-vae" if flags.fp16_vae else "--fp32-vae",
            "--disable-pinned-memory",
            "--mmap-torch-files",
            "--preview-method",
            "none",
            "--disable-api-nodes",
            "--disable-auto-launch",
            "--user-directory",
            str((home / "user").resolve()),
            "--temp-directory",
            str(stage_temp),
            "--output-directory",
            str(output_dir),
            "--database-url",
            "sqlite:///:memory:",
            "--log-stdout",
        ]
        whitelist = self._stage_custom_node_whitelist(stage)
        if whitelist:
            command.extend(["--disable-all-custom-nodes", "--whitelist-custom-nodes", *whitelist])
        environment = os.environ.copy()
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1",
            EMILIA_HEADLESS_FAST_BOOT="1",
        )
        if flags.fp16_upscaler:
            environment["EMILIA_FP16_UPSCALER"] = "1"
        if flags.fast_tiled_vae:
            environment["EMILIA_FAST_TILED_VAE"] = "1"
        server_log_path = home / "server.log"
        with server_log_path.open("w", encoding="utf-8") as log:
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
                    log_ready_deadline = time.monotonic() + 25.0
                    log_seen_ready = False
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"{stage} server exited; see {server_log_path}"
                            )
                        if not log_seen_ready and time.monotonic() < log_ready_deadline:
                            try:
                                if server_log_path.is_file() and "To see the GUI go to:" in server_log_path.read_text(encoding="utf-8", errors="ignore"):
                                    log_seen_ready = True
                            except OSError:
                                pass
                            if not log_seen_ready:
                                await asyncio.sleep(0.025)
                                continue
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
                                f"{stage} server did not start; see {server_log_path}"
                            )
                        await asyncio.sleep(0.05)
                    yield StageServer(
                        session=session, base_url=base, home=home, temp_dir=stage_temp, process=process
                    )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait, timeout=10)

    # ------------------------------------------------------------------ pipeline

    async def render_quality_pipeline(
        self, *, seed: int = 20261027, profile: str = "quality", **options: Any
    ) -> dict[str, Any]:
        """Render one seed; single-seed form of :meth:`render_batch_pipeline` with the same options.

        Raises :class:`QualityGateRejected`, carrying the saved receipt, when the gate stops the seed.
        """
        receipt = (await self.render_batch_pipeline(seeds=[seed], profile=profile, **options))[0]
        if receipt["status"] == "quality_rejected":
            raise QualityGateRejected(receipt["rejection_reason"], Path(receipt["receipt_path"]), receipt)
        return receipt

    async def render_batch_pipeline(
        self,
        *,
        seeds: list[int],
        port: int = 8199,
        pag: bool = False,
        profile: str = "ultra",
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        ipadapter_weight: float | None = None,
        user_idea: str | None = None,
        strict_seed_lock: bool = True,
        auto_eye_repair: bool = False,
        max_seed_retries: int = 0,
        native_options: dict[str, Any] | None = None,
        finish_options: dict[str, Any] | None = None,
        native_steps: int | None = None,
        face_steps: int | None = None,
        hand_steps: int | None = None,
        fp16_upscaler: bool = False,
        fp16_vae: bool = False,
        fast_tiled_vae: bool = False,
        preview: bool = False,
    ) -> list[dict[str, Any]]:
        """Render seeds as one Stage 1 process, a CPU gate, then one Stage 2 process.

        Every native runs in the Stage 1 process, which exits before the Stage 2
        process (eye repairs, then finishing) starts, so the two model sets never
        overlap. Nonempty, explicitly whitelisted tuning dictionaries are ultra-only.
        Strict seed lock keeps the baseline stage sequence and only records
        advisory findings. Auto-gating never finishes a rejected native or an
        unaccepted eye repair.

        Returns one receipt per seed, in order; rejected seeds have
        ``status == "quality_rejected"``. A seed gets its receipt and ledger row as
        soon as it is finished or rejected, including when a later seed fails;
        ledger failures propagate after the receipt is saved.
        """
        assert self.config.app_dir is not None
        if not seeds:
            raise ValueError("seeds must not be empty")
        plan = self._plan(
            port=port, pag=pag, profile=profile, positive_prompt=positive_prompt,
            negative_prompt=negative_prompt, ipadapter_weight=ipadapter_weight, user_idea=user_idea,
            strict_seed_lock=strict_seed_lock, auto_eye_repair=auto_eye_repair,
            max_seed_retries=max_seed_retries, native_options=native_options,
            finish_options=finish_options, native_steps=native_steps, face_steps=face_steps,
            hand_steps=hand_steps, preview=bool(preview),
            flags=RuntimeFlags(bool(fp16_upscaler), bool(fp16_vae), bool(fast_tiled_vae)),
        )
        temp_dir = self.config.temp_dir or self.config.app_dir / "temp"
        run_dir = temp_dir.resolve() / f"{profile}_{uuid.uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        items = [_RenderItem(requested_seed=int(seed), seed=int(seed)) for seed in seeds]
        with run_lock(run_dir):
            try:
                await self._native_wave(plan, items, run_dir)
                if plan.preview:
                    for item in items:
                        if plan.gated and item.needs_eye_repair:
                            item.rejection = "Preview requires eye repair; run the full pipeline to repair and reinspect it."
                        if item.rejection is None:
                            item.final = item.selected
                            item.final_report = item.native_report
                else:
                    if not plan.auto_eye_repair:
                        for item in items:
                            if plan.gated and item.needs_eye_repair:
                                item.rejection = "Eye repair is required but automatic eye repair is disabled."
                    if any(item.rejection is None for item in items):
                        await self._finish_wave(plan, items, run_dir)
            finally:
                for item in items:
                    if item.settled:
                        self._persist(plan, item, run_dir)
        return [item.receipt for item in items if item.receipt is not None]

    def _plan(
        self,
        *,
        port: int,
        pag: bool,
        profile: str,
        positive_prompt: str | None,
        negative_prompt: str | None,
        ipadapter_weight: float | None,
        user_idea: str | None,
        strict_seed_lock: bool,
        auto_eye_repair: bool,
        max_seed_retries: int,
        native_options: dict[str, Any] | None,
        finish_options: dict[str, Any] | None,
        native_steps: int | None,
        face_steps: int | None,
        hand_steps: int | None,
        preview: bool = False,
        flags: RuntimeFlags,
    ) -> _RenderPlan:
        """Validate a request before any process is launched."""
        assert self.config.app_dir is not None
        if profile not in {"quality", "ultra"}:
            raise ValueError("profile must be quality or ultra")
        is_ultra = profile == "ultra"
        if not isinstance(max_seed_retries, int) or max_seed_retries < 0:
            raise ValueError("max_seed_retries must be a nonnegative integer")
        for name, options, allowed in (
            ("native_options", native_options, _NATIVE_OPTION_KEYS),
            ("finish_options", finish_options, _FINISH_OPTION_KEYS),
        ):
            if options is not None and not isinstance(options, dict):
                raise TypeError(f"{name} must be a dictionary")
            if options and not is_ultra:
                raise ValueError(f"{name} is only supported by the ultra profile")
            unknown = set(options or {}) - allowed
            if unknown:
                raise ValueError(f"Unsupported {name}: {', '.join(sorted(map(str, unknown)))}")
        for name, value in (("native_steps", native_steps), ("face_steps", face_steps), ("hand_steps", hand_steps)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if preview:
            if pag or (native_steps is not None and native_steps != LIGHTNING_PREVIEW_STEPS):
                raise ValueError("Lightning preview requires exactly 8 steps and does not support PAG")
            if finish_options or face_steps is not None or hand_steps is not None or flags.fp16_upscaler:
                raise ValueError("Preview omits Stage 2; refinement and upscaler options are not supported")
            preview_keys = {"base_width", "base_height", "framing", "camera_angle", "checkpoint_name"}
            if ipadapter_weight is not None or set(native_options or {}) - preview_keys:
                raise ValueError("Lightning preview is text-only; identity LoRA/IPAdapter and hires tuning are not supported")
            lightning_path = self.config.app_dir / "models" / "diffusion_models" / LIGHTNING_PREVIEW_UNET
            if not lightning_path.is_file():
                raise FileNotFoundError(
                    f"Preview requires {lightning_path}; run `emilia pull sdxl-lightning-8step-unet`."
                )
        if flags.fp16_vae:
            vae_path = self.config.app_dir / "models" / "vae" / FP16_VAE_NAME
            if not vae_path.is_file():
                raise FileNotFoundError(
                    f"fp16_vae requires {vae_path}; run `emilia pull sdxl-vae-fp16-fix` or disable --fp16-vae."
                )
        native_options = dict(native_options or {})
        finish_options = dict(finish_options or {})
        if is_ultra and not preview:
            if native_steps is not None:
                native_options.setdefault("base_steps", native_steps)
            if face_steps is not None:
                finish_options.setdefault("face_steps", face_steps)
            if hand_steps is not None:
                finish_options.setdefault("hand_steps", hand_steps)
        framing = str(native_options.get("framing", "full_body")) if is_ultra else "full_body"
        return _RenderPlan(
            profile=profile, port=port, pag=pag,
            positive_prompt=positive_prompt, negative_prompt=negative_prompt,
            ipadapter_weight=0.0 if preview else ((0.18 if is_ultra else 0.40) if ipadapter_weight is None else ipadapter_weight),
            user_idea=user_idea, strict_seed_lock=strict_seed_lock, auto_eye_repair=auto_eye_repair,
            max_seed_retries=max_seed_retries, native_options=native_options, finish_options=finish_options,
            native_steps=native_steps, face_steps=face_steps, hand_steps=hand_steps, flags=flags,
            framing=framing,
            require_pose=bool(is_ultra and not strict_seed_lock and framing == "full_body"),
            preview=bool(preview),
        )

    def _with_runtime_vae(self, plan: _RenderPlan, graph: dict[str, Any]) -> dict[str, Any]:
        return self.builder.inject_external_vae(graph, FP16_VAE_NAME) if plan.flags.fp16_vae else graph

    def _native_expected_size(self, plan: _RenderPlan) -> tuple[int, int]:
        if plan.preview:
            if plan.is_ultra:
                return (
                    int(plan.native_options.get("base_width", 832)),
                    int(plan.native_options.get("base_height", 1216)),
                )
            return QualityInspector.SAFE_NATIVE_SIZE
        return QualityInspector.NATIVE_SIZE

    def _native_graph(self, plan: _RenderPlan, seed: int) -> dict[str, Any]:
        if plan.preview:
            if plan.is_ultra:
                preview_opts = dict(plan.native_options)
                steps = int(plan.native_steps or LIGHTNING_PREVIEW_STEPS)
                graph = self.builder.build_ultra_preview(
                    seed=seed, pag=plan.pag, steps=steps,
                    positive_prompt=plan.positive_prompt, negative_prompt=plan.negative_prompt,
                    **preview_opts,
                )
            else:
                w, h = QualityInspector.SAFE_NATIVE_SIZE
                graph = self.builder.build_highres_preview(
                    seed=seed, pag=plan.pag, positive_prompt=plan.positive_prompt,
                    negative_prompt=plan.negative_prompt, width=w, height=h,
                    steps=int(plan.native_steps or LIGHTNING_PREVIEW_STEPS),
                )
            return self._with_runtime_vae(plan, graph)
        if plan.is_ultra:
            graph = self.builder.build_ultra_native(
                seed=seed, pag=plan.pag, positive_prompt=plan.positive_prompt,
                negative_prompt=plan.negative_prompt, ipadapter_weight=plan.ipadapter_weight,
                **plan.native_options,
            )
        else:
            kwargs: dict[str, Any] = {
                "seed": seed, "pag": plan.pag, "positive_prompt": plan.positive_prompt,
                "negative_prompt": plan.negative_prompt, "ipadapter_weight": plan.ipadapter_weight,
            }
            if plan.native_steps is not None:
                kwargs["steps"] = plan.native_steps
            graph = self.builder.build_highres_native(**kwargs)
        return self._with_runtime_vae(plan, graph)

    def _eye_repair_graph(self, plan: _RenderPlan, item: _RenderItem, image_ref: str) -> dict[str, Any]:
        if plan.is_ultra:
            graph = self.builder.build_ultra_eye_repair(
                image_path=image_ref, seed=item.seed + 9, native_graph=item.native_graph,
                lora_name=plan.finish_options.get("lora_name"),
                lora_strength=plan.finish_options.get("lora_strength"),
            )
        else:
            graph = self.builder.build_eye_repair(image_path=image_ref, seed=item.seed + 9, upscale=False)
        return self._with_runtime_vae(plan, graph)

    def _finish_graph(self, plan: _RenderPlan, item: _RenderItem, image_ref: str) -> dict[str, Any]:
        if plan.is_ultra:
            graph = self.builder.build_ultra_finish(
                image_path=image_ref, seed=item.seed, pag=plan.pag, **plan.finish_options,
            )
        else:
            kwargs: dict[str, Any] = {"image_path": image_ref, "seed": item.seed, "pag": plan.pag}
            if plan.face_steps is not None:
                kwargs["face_steps"] = plan.face_steps
            if plan.hand_steps is not None:
                kwargs["hand_steps"] = plan.hand_steps
            graph = self.builder.build_highres_finish(**kwargs)
        return self._with_runtime_vae(plan, graph)

    def _inspect(
        self, plan: _RenderPlan, path: Path, size: tuple[int, int], stage: str, *, strict: bool
    ) -> asyncio.Task[InspectionReport]:
        """Start CPU inspection on a worker thread so it overlaps GPU work or process teardown."""
        return asyncio.create_task(asyncio.to_thread(
            self.inspector.inspect_image, path, expected_size=size, stage=stage, solo_prompt=True,
            strict_seed_lock=strict, framing=plan.framing, require_full_body_pose=plan.require_pose,
        ))

    def _warm_cpu_side(self, plan: _RenderPlan) -> None:
        """Best-effort work for the idle CPU and disk while Stage 1 samples on the GPU."""
        self.inspector.warmup(run_pose=plan.framing == "full_body")
        if not plan.preview:
            self._prewarm_stage2_files_in_ram(plan.flags)
    async def _native_wave(self, plan: _RenderPlan, items: list[_RenderItem], run_dir: Path) -> None:
        """Stage 1: every seed (and its gate retries) in one process, inspection overlapping the next render."""
        numbered = len(items) > 1
        # A retry decision needs the report first; otherwise inspection runs behind the next render.
        can_retry = plan.gated and plan.max_seed_retries > 0
        warmup: asyncio.Task[None] | None = None
        pending: tuple[_RenderItem, asyncio.Task[InspectionReport]] | None = None

        def start_warmup() -> None:
            nonlocal warmup
            warmup = asyncio.create_task(asyncio.to_thread(self._warm_cpu_side, plan))

        def record(item: _RenderItem, report: InspectionReport) -> None:
            item.native_report = report
            item.attempts[-1]["inspection"] = report.to_dict()

        async def settle() -> None:
            nonlocal pending
            if pending is None:
                return
            (item, task), pending = pending, None
            report = await task
            record(item, report)
            item.rejection = _native_rejection(plan, report)

        try:
            async with self._stage_server("native", run_dir, plan.port, plan.flags) as server:
                for index, item in enumerate(items):
                    base = f"native_{index + 1}" if numbered else "native"
                    while True:
                        attempt = len(item.attempts) + 1
                        stage = base if attempt == 1 else f"{base}_retry_{attempt}"
                        graph = self._native_graph(plan, item.seed)
                        result = await self._execute_graph(
                            server, graph=graph, stage=stage, log_dir=server.home / stage,
                            on_first_progress=start_warmup if warmup is None else None,
                        )
                        await settle()
                        generation = result.to_dict()
                        item.native_graph, item.generation = graph, generation
                        item.native = item.selected = _output_path(
                            self._output_dir, generation["outputs"][NATIVE_OUTPUT_NODE]["images"][0]
                        )
                        item.attempts.append({"seed": item.seed, "generation": generation, "inspection": {}})
                        task = self._inspect(
                            plan, item.native, self._native_expected_size(plan), "native", strict=plan.strict_seed_lock
                        )
                        if not can_retry:
                            pending = (item, task)
                            break
                        report = await task
                        record(item, report)
                        if (
                            report.dimensions_match
                            and report.recommended_action == "reject_seed"
                            and attempt <= plan.max_seed_retries
                        ):
                            print(
                                f"quality-gate: seed {item.seed} rejected ({report.reasons}); retrying with new seed...",
                                file=sys.stderr, flush=True,
                            )
                            item.seed += 1000
                            continue
                        item.rejection = _native_rejection(plan, report)
                        break
        finally:
            # The last inspection has been running while the Stage 1 process shut down.
            try:
                await settle()
            finally:
                if warmup is not None:
                    # Warmup is an optimization; a real detector failure surfaces from inspection.
                    with contextlib.suppress(Exception):
                        await warmup

    async def _finish_wave(self, plan: _RenderPlan, items: list[_RenderItem], run_dir: Path) -> None:
        """Stage 2: eye repairs, then finishing, sharing one process and one checkpoint load."""
        numbered = len(items) > 1
        pending: tuple[_RenderItem, asyncio.Task[InspectionReport]] | None = None

        async def settle() -> None:
            nonlocal pending
            if pending is None:
                return
            (item, task), pending = pending, None
            report = await task
            item.final_report = report
            if not report.dimensions_match:
                item.rejection = "Finished output dimensions differ from the quality workflow contract."
            elif plan.gated and not report.passed:
                item.rejection = "Finished image did not pass the quality gate."
            self._persist(plan, item, run_dir)

        try:
            async with self._stage_server("finish", run_dir, plan.port, plan.flags) as server:
                for index, item in enumerate(items):
                    if not (plan.gated and item.needs_eye_repair):
                        continue
                    assert item.selected is not None
                    stage = f"eye_repair_{index + 1}" if numbered else "eye_repair"
                    graph = self._eye_repair_graph(plan, item, self._load_image_ref(item.selected, server))
                    result = await self._execute_graph(server, graph=graph, stage=stage, log_dir=server.home / stage)
                    item.eye_repair = result.to_dict()
                    item.selected = _output_path(
                        self._output_dir, item.eye_repair["outputs"][EYE_REPAIR_OUTPUT_NODE]["images"][0]
                    )
                    item.repaired_report = await self._inspect(
                        plan, item.selected, QualityInspector.NATIVE_SIZE, "eye_repair", strict=False
                    )
                    if not item.repaired_report.passed:
                        item.rejection = "Repaired native image did not pass the quality gate."
                for index, item in enumerate(items):
                    if item.rejection is not None:
                        continue
                    assert item.selected is not None
                    stage = f"finish_{index + 1}" if numbered else "finish"
                    graph = self._finish_graph(plan, item, self._load_image_ref(item.selected, server))
                    result = await self._execute_graph(server, graph=graph, stage=stage, log_dir=server.home / stage)
                    await settle()
                    item.finishing = result.to_dict()
                    item.final = _output_path(
                        self._output_dir, item.finishing["outputs"][FINAL_OUTPUT_NODE]["images"][0]
                    )
                    pending = (item, self._inspect(
                        plan, item.final, QualityInspector.FINAL_SIZE, "final", strict=plan.strict_seed_lock
                    ))
        finally:
            await settle()

    def _persist(self, plan: _RenderPlan, item: _RenderItem, run_dir: Path) -> None:
        """Write the receipt beside the image it describes, then index it in the ledger (once)."""
        if item.receipt is not None:
            return
        native_report = item.native_report
        assert native_report is not None and item.native is not None and item.selected is not None
        guidance = (
            f"{plan.profile} 832x1216 preview"
            if plan.preview
            else (
                ("Ultra Multi-Scale + PAG" if plan.pag else "Ultra Multi-Scale (832->1152->2304)")
                if plan.is_ultra else ("PAG scale 1.0" if plan.pag else "baseline")
            )
        )
        fast_overrides = {
            key: value
            for key, value in {
                "preview": plan.preview,
                "native_steps": plan.native_steps, "face_steps": plan.face_steps,
                "hand_steps": plan.hand_steps, "fp16_upscaler": plan.flags.fp16_upscaler,
                "fp16_vae": plan.flags.fp16_vae, "fast_tiled_vae": plan.flags.fast_tiled_vae,
            }.items()
            if value
        }
        receipt_path = (item.final or item.selected).with_suffix(".run.json")
        receipt: dict[str, Any] = {
            "seed": item.seed,
            "requested_seed": item.requested_seed,
            "profile": plan.profile,
            "preview": plan.preview,
            "status": "quality_rejected" if item.rejection else "completed",
            "rejection_reason": item.rejection,
            "guidance": guidance,
            "python": self._python_executable(),
            "isolation": (
                "Single-stage 832x1216 preview process; Stage 2 refinement omitted."
                if plan.preview
                else "Stage 1 process exited before the Stage 2 process started; no model overlap across stages."
            ),
            "native": str(item.native),
            "selected_native": str(item.selected),
            "final": str(item.final) if item.final is not None else None,
            "generation": item.generation,
            "finishing": item.finishing,
            "attempts": item.attempts,
            "logs": str(run_dir),
            "receipt_path": str(receipt_path),
            "selected_settings": {
                "pag": plan.pag,
                "ipadapter_weight": plan.ipadapter_weight,
                "strict_seed_lock": plan.strict_seed_lock,
                "auto_eye_repair": plan.auto_eye_repair,
                "max_seed_retries": plan.max_seed_retries,
                "native_options": plan.native_options,
                "finish_options": plan.finish_options,
                **({"fast_overrides": fast_overrides} if fast_overrides else {}),
            },
            "quality_inspection": {
                "native": native_report.to_dict(),
                "final": item.final_report.to_dict() if item.final_report is not None else {},
                "eye_repair": item.eye_repair,
                "repaired": item.repaired_report.to_dict() if item.repaired_report is not None else {},
            },
        }
        if plan.user_idea:
            receipt["user_idea"] = plan.user_idea
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        item.receipt = receipt
        native_seconds = sum(float(attempt["generation"].get("seconds", 0.0)) for attempt in item.attempts)
        finish_seconds = float(item.finishing.get("seconds", 0.0))
        repair_seconds = float((item.eye_repair or {}).get("seconds", 0.0))
        self.ledger.record_run(
            seed=item.seed,
            mode=f"{plan.profile}_preview" if plan.preview else (f"{plan.profile}_pag" if plan.pag else plan.profile),
            guidance=guidance,
            prompt_idea=plan.user_idea,
            positive_prompt=plan.positive_prompt or (
                self.config.ultra_positive_prompt if plan.is_ultra else self.config.base_positive_prompt
            ),
            native_path=str(item.native),
            final_path=str(item.final) if item.final is not None else None,
            native_sha256=native_report.pixel_sha256,
            final_sha256=item.final_report.pixel_sha256 if item.final_report is not None else None,
            native_seconds=native_seconds,
            finish_seconds=finish_seconds,
            total_seconds=native_seconds + finish_seconds + repair_seconds,
            face_count=native_report.face_count,
            hand_count=native_report.hand_count,
            heterochromia_score=native_report.heterochromia_score,
            eye_drift_detected=native_report.eye_drift_detected,
            quality_action="reject_seed" if item.rejection else (item.final_report or native_report).recommended_action,
            receipt_path=str(receipt_path.resolve()),
        )
        if item.final is not None:
            print(f"FINISHED: {item.final}", file=sys.stderr, flush=True)
        print(f"RECEIPT: {receipt_path}", file=sys.stderr, flush=True)
