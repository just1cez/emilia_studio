"""Unified CLI entrypoint for Emilia Studio."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .compiler import BridgeError, OllamaPromptCompiler, outfit_stage_options
from .config import (
    BUILTIN_SCENE_PRESETS,
    EMILIA_ANGLE_SPECS,
    EMILIA_MAGIC_TAGS,
    EMILIA_OUTFIT_MODES,
    StudioConfig,
    load_config,
)
from .commands import add_commands, handle
from .engine import EmiliaRenderEngine, QualityGateRejected
from .inspector import QualityInspector
from .ledger import ExperimentLedger
from .workflows import EmiliaWorkflowBuilder

DEFAULT_SEED = 20261027
DEFAULT_SCENE_PRESET = "rose-garden"

# Ultra-only tuning flags: CLI attribute -> (stage, builder option).
_ULTRA_TUNING: dict[str, tuple[tuple[str, str], ...]] = {
    "reference_mode": (("native", "reference_mode"),),
    "hires_bridge": (("native", "hires_bridge"),),
    "checkpoint": (("native", "checkpoint_name"), ("finish", "checkpoint_name")),
    "garment_region": (("finish", "garment_region"),),
    "upscaler": (("finish", "upscaler_name"),),
    "secondary_upscaler": (("finish", "secondary_upscaler_name"),),
    "upscale_blend": (("finish", "upscale_blend_strength"),),
}
_CREATIVE_ARGS = ("outfit", "angle", "magic", *_ULTRA_TUNING)


def _add_pipeline_args(parser: argparse.ArgumentParser, *, default_profile: str, idea: bool = False) -> None:
    """Arguments shared by every command that drives the two-stage pipeline."""
    parser.add_argument("--port", type=int, default=None, help="Override configured rendering port")
    parser.add_argument("--pag", action="store_true", help="Enable PerturbedAttentionGuidance (scale 1.0)")
    parser.add_argument(
        "--profile",
        choices=("quality", "ultra"),
        default=default_profile,
        help="quality: bit-exact 1152x1728 baseline; ultra: multi-scale 832->1152->2304 Hires.fix + canonical outfit lock",
    )
    # `idea` always routes an outfit; elsewhere an unset flag means "no creative routing".
    parser.add_argument("--outfit", choices=tuple(EMILIA_OUTFIT_MODES), default="royal" if idea else None, help="Emilia outfit mode (ultra profile)")
    parser.add_argument("--angle", choices=tuple(EMILIA_ANGLE_SPECS), default="auto" if idea else None, help="Camera angle & pose reference routing (ultra profile)")
    parser.add_argument("--puck", action="store_true", help="Include Puck fairy cat companion (ultra profile)")
    parser.add_argument("--magic", choices=tuple(EMILIA_MAGIC_TAGS), default="none" if idea else None, help="Ice magic effect (ultra profile)")
    parser.add_argument("--auto-gate", action="store_true", help="Enable automatic seed retry, full-body pose gate, and eye-repair gate")
    parser.add_argument("--max-retries", type=int, default=2, help="Max seed retries when auto-gate rejects anatomy")

    parser.add_argument("--reference-mode", choices=("single", "paired"), default=None, help="Face reference mode (ultra profile)")
    parser.add_argument("--hires-bridge", choices=("lanczos", "anime"), default=None, help="Pass 1->2 hires bridge (ultra profile)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename override (ultra profile)")
    parser.add_argument(
        "--garment-region",
        choices=("brooch_and_collar", "bodice_and_peplum", "full_garment"),
        default=None,
        help="Optional Stage 2 face-relative garment refinement region (ultra profile)",
    )
    parser.add_argument("--upscaler", type=str, default=None, help="Primary Stage 2 upscaler filename (ultra profile)")
    parser.add_argument("--secondary-upscaler", type=str, default=None, help="Secondary Stage 2 upscaler filename for blending (ultra profile)")
    parser.add_argument("--upscale-blend", type=float, default=None, help="Secondary upscaler blend strength 0..1 (ultra profile)")

    parser.add_argument("--steps", type=int, default=None, help="Native/base KSampler step count override")
    parser.add_argument("--face-steps", type=int, default=None, help="Stage 2 FaceDetailer step count override")
    parser.add_argument("--hand-steps", type=int, default=None, help="Stage 2 HandDetailer step count override")
    parser.add_argument("--fp16-upscaler", action="store_true", help="Run Stage 2 RealESRGAN 4x upscaler in FP16 TensorCore mode (~10x faster)")
    parser.add_argument("--fp16-vae", action="store_true", help="Use standalone sdxl-vae-fp16-fix.safetensors in FP16 mode")
    parser.add_argument("--fast-tiled-vae", action="store_true", help="Use single-pass feathered tiled VAE decode/encode instead of 3-pass")
    parser.add_argument("--fast", action="store_true", help="Preset enabling --fp16-upscaler --fp16-vae --fast-tiled-vae and reduced steps (32/16)")


def _add_scene_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", type=str, default=None, help="Custom positive English prompt")
    parser.add_argument("--preset", choices=tuple(BUILTIN_SCENE_PRESETS))


def _acceleration_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    native_steps, face_steps, hand_steps = args.steps, args.face_steps, args.hand_steps
    if args.fast:
        if native_steps is None:
            native_steps = 24 if args.profile == "ultra" else 32
        if hand_steps is None:
            hand_steps = 16
    out: dict[str, Any] = {}
    for key, value in (("native_steps", native_steps), ("face_steps", face_steps), ("hand_steps", hand_steps)):
        if value is not None:
            out[key] = int(value)
    for key in ("fp16_upscaler", "fp16_vae", "fast_tiled_vae"):
        if args.fast or getattr(args, key):
            out[key] = True
    return out


def _apply_ultra_tuning(
    args: argparse.Namespace,
    native_options: dict[str, Any] | None,
    finish_options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if all(getattr(args, attr) is None for attr in _ULTRA_TUNING):
        return native_options, finish_options
    stages = {"native": dict(native_options or {}), "finish": dict(finish_options or {})}
    for attr, targets in _ULTRA_TUNING.items():
        value = getattr(args, attr)
        if value is not None:
            for stage, option in targets:
                stages[stage][option] = value
    return stages["native"], stages["finish"]


def _pipeline_kwargs(args: argparse.Namespace, config: StudioConfig) -> dict[str, Any]:
    """Engine options that depend only on the shared pipeline flags."""
    return {
        "port": args.port if args.port is not None else config.quality_port,
        "pag": args.pag,
        "profile": args.profile,
        "strict_seed_lock": not args.auto_gate,
        "auto_eye_repair": args.auto_gate,
        "max_seed_retries": args.max_retries if args.auto_gate else 0,
        **_acceleration_kwargs(args),
    }


def _scene_kwargs(args: argparse.Namespace, config: StudioConfig) -> dict[str, Any]:
    """Prompt and outfit routing shared by `render` and `batch`."""
    creative = args.puck or any(getattr(args, attr) is not None for attr in _CREATIVE_ARGS)
    if creative and args.profile != "ultra":
        raise ValueError("Creative and ultra tuning flags (--outfit, --angle, --puck, --magic, --garment-region, etc.) require --profile ultra")
    if args.preset and args.prompt:
        raise ValueError("Use --preset or --prompt, not both")
    prompt, negative, ipadapter_weight = args.prompt, None, None
    native_options: dict[str, Any] | None = None
    finish_options: dict[str, Any] | None = None
    if args.preset or creative:
        outfit, angle, framing = args.outfit or "royal", args.angle or "auto", "full_body"
        scene = args.prompt
        if scene is None:
            preset = BUILTIN_SCENE_PRESETS[args.preset or DEFAULT_SCENE_PRESET]
            scene, framing = preset["tags"], preset["framing"]
        compiler = OllamaPromptCompiler(config)
        prompt = compiler.build_emilia_ultra_prompt(
            scene, framing, outfit=outfit, companion="puck" if args.puck else "none",
            magic=args.magic or "none", angle=angle,
        )
        if args.profile == "ultra":
            negative = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            ipadapter_weight, native_options, finish_options = outfit_stage_options(outfit, framing, angle)
    native_options, finish_options = _apply_ultra_tuning(args, native_options, finish_options)
    return {
        "positive_prompt": prompt,
        "negative_prompt": negative,
        "ipadapter_weight": ipadapter_weight,
        "native_options": native_options,
        "finish_options": finish_options,
    }


def _base_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return args.seed
    return BUILTIN_SCENE_PRESETS[args.preset]["default_seed"] if args.preset else DEFAULT_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emilia-studio",
        description="Emilia Studio — High-Res SDXL + IPAdapter + Detailer + Upscale Pipeline & Ledger",
    )
    parser.add_argument('--version', action='version', version=f'emilia-studio {__version__}')
    parser.add_argument('--config', type=Path, help='Explicit TOML configuration file')
    sub = parser.add_subparsers(dest="command")
    add_commands(sub)

    p_render = sub.add_parser(
        "render",
        help="Run the 2-stage HighRes Quality (1152x1728 -> 2304x3456) render pipeline.",
    )
    p_render.add_argument("--seed", type=int, default=None, help=f"Generation seed (defaults to preset seed or {DEFAULT_SEED})")
    _add_scene_args(p_render)
    _add_pipeline_args(p_render, default_profile="quality")

    p_batch = sub.add_parser(
        "batch",
        help="Render multiple seeds using same-stage process reuse while keeping Stage 1 and Stage 2 isolated.",
    )
    p_batch.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit list of seeds")
    p_batch.add_argument("--seed", type=int, default=None, help=f"Starting seed when --count is used (defaults to preset seed or {DEFAULT_SEED})")
    p_batch.add_argument("--count", type=int, default=2, help="Number of seeds to generate when --seeds is omitted")
    _add_scene_args(p_batch)
    _add_pipeline_args(p_batch, default_profile="ultra")

    p_idea = sub.add_parser(
        "idea",
        help="Compile a Chinese idea with Ollama and render it through the 2-stage pipeline.",
    )
    p_idea.add_argument("idea", nargs="+", help="Chinese image description / idea")
    p_idea.add_argument("--seed", type=int, default=None, help="Random seed if omitted")
    p_idea.add_argument("--style", choices=("anime", "illustration", "cinematic"), default="anime")
    _add_pipeline_args(p_idea, default_profile="ultra", idea=True)

    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect an existing PNG for violet iris consistency, heterochromia, full-body pose, and face/hand counts.",
    )
    p_inspect.add_argument("image", type=Path, help="Path to PNG image")
    p_inspect.add_argument("--stage", choices=("native", "final"), default=None)
    p_inspect.add_argument("--framing", choices=("full_body", "upper_body", "portrait"), default="full_body")
    p_inspect.add_argument("--require-full-body-pose", action="store_true", help="Fail gate if full-body standing/ankle completeness is not met")

    p_ledger = sub.add_parser(
        "ledger",
        help="Query or ingest historical render receipts and benchmark rounds in SQLite.",
    )
    p_ledger.add_argument(
        "action",
        choices=("ingest", "stats", "recent", "compare"),
        default="stats",
        nargs="?",
    )
    p_ledger.add_argument("--round-a", type=int, default=1)
    p_ledger.add_argument("--round-b", type=int, default=5)
    p_ledger.add_argument("--limit", type=int, default=10)
    p_ledger.add_argument("--refresh", action="store_true", help="With ingest: re-read receipts that are already indexed")

    p_wf = sub.add_parser(
        "workflows",
        help="Verify the builder reproduces the canonical API graphs exactly; audit the workflow directory; optionally re-sync GUI workflows.",
    )
    p_wf.add_argument("--sync-visual", action="store_true", help="Rewrite the GUI workflows from the builder's API graphs")

    p_setup = sub.add_parser(
        "setup",
        help="Verify installed model weights, reference assets, and custom nodes against models_manifest.json.",
    )
    p_setup.add_argument(
        "--sync",
        action="store_true",
        help="Sync top-level assets/, patches/, workflows/, and tests/ into the app/ ComfyUI runtime directory.",
    )

    return parser


def _cmd_render(args: argparse.Namespace, config: StudioConfig) -> int:
    request = {**_scene_kwargs(args, config), **_pipeline_kwargs(args, config)}
    engine = EmiliaRenderEngine(config=config)
    asyncio.run(engine.render_quality_pipeline(seed=_base_seed(args), **request))
    return 0


def _cmd_batch(args: argparse.Namespace, config: StudioConfig) -> int:
    if args.seeds:
        seeds = [int(seed) for seed in args.seeds]
    else:
        if args.count < 1:
            raise ValueError("--count must be positive")
        seeds = [_base_seed(args) + offset for offset in range(args.count)]
    request = {**_scene_kwargs(args, config), **_pipeline_kwargs(args, config)}
    engine = EmiliaRenderEngine(config=config)
    receipts = asyncio.run(engine.render_batch_pipeline(seeds=seeds, **request))
    from .tui import render_table

    rows = [
        [
            str(r["requested_seed"]),
            str(r["seed"]),
            r["status"],
            f"{float(r['generation'].get('seconds', 0.0)):.1f}s",
            f"{float(r['finishing'].get('seconds', 0.0)):.1f}s",
            str(r["final"] or r["selected_native"]),
        ]
        for r in receipts
    ]
    print(render_table(["Req Seed", "Used Seed", "Status", "Native", "Finish", "Output"], rows, title="Emilia Batch Render Summary"))
    return 0 if all(r["status"] == "completed" for r in receipts) else 2


def _cmd_idea(args: argparse.Namespace, config: StudioConfig) -> int:
    idea_text = " ".join(args.idea).strip()
    compiler = OllamaPromptCompiler(config=config)
    compiler.ensure_ollama()
    is_ultra = args.profile == "ultra"
    spec = compiler.compile_idea(
        idea_text,
        style=args.style,
        highres_quality_mode=True,
        ultra_mode=is_ultra,
        outfit=args.outfit,
        companion="puck" if args.puck else "none",
        magic=args.magic,
        angle=args.angle,
    )
    print(
        f"编译完成 [{args.profile}/{spec.framing}, outfit={args.outfit}, angle={args.angle}, IPAdapter={spec.recommended_ipadapter_weight}]:\n{spec.positive_prompt}",
        flush=True,
    )
    ipadapter_weight = spec.recommended_ipadapter_weight
    native_options: dict[str, Any] | None = None
    finish_options: dict[str, Any] | None = None
    if is_ultra:
        outfit_weight, native_options, finish_options = outfit_stage_options(args.outfit, spec.framing, args.angle)
        ipadapter_weight = max(ipadapter_weight, outfit_weight)
        native_options, finish_options = _apply_ultra_tuning(args, native_options, finish_options)
    engine = EmiliaRenderEngine(config=config)
    asyncio.run(
        engine.render_quality_pipeline(
            seed=args.seed if args.seed is not None else secrets.randbelow(2**31 - 1),
            positive_prompt=spec.positive_prompt,
            negative_prompt=spec.negative_prompt,
            ipadapter_weight=ipadapter_weight,
            user_idea=idea_text,
            native_options=native_options,
            finish_options=finish_options,
            **_pipeline_kwargs(args, config),
        )
    )
    return 0


def _cmd_inspect(args: argparse.Namespace, config: StudioConfig) -> int:
    inspector = QualityInspector(app_dir=config.app_dir, python_executable=config.conda_python)
    report = inspector.inspect_image(
        args.image,
        stage=args.stage,
        framing=args.framing,
        require_full_body_pose=args.require_full_body_pose,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.passed else 2


def _cmd_ledger(args: argparse.Namespace, config: StudioConfig) -> int:
    ledger = ExperimentLedger(config=config)
    if args.action == "ingest":
        ingested = ledger.ingest_existing_artifacts(refresh=args.refresh)
        print(json.dumps({"ingested_records": ingested, "db_path": str(ledger.db_path)}, indent=2))
    elif args.action == "stats":
        ledger.ingest_existing_artifacts()
        print(json.dumps(ledger.get_summary_stats(), indent=2))
    elif args.action == "recent":
        print(json.dumps(ledger.list_recent_runs(limit=args.limit), indent=2, ensure_ascii=False))
    elif args.action == "compare":
        ledger.ingest_existing_artifacts()
        print(json.dumps(ledger.compare_rounds(args.round_a, args.round_b), indent=2))
    return 0


def _cmd_workflows(args: argparse.Namespace, config: StudioConfig) -> int:
    builder = EmiliaWorkflowBuilder(config=config)
    audit = builder.audit_workflow_directory()
    parity = builder.canonical_parity()
    summary: dict[str, object] = {
        "canonical_api_count": len(audit["canonical_workflows"]),
        "experiment_report_count": len(audit["experiment_reports"]),
        "historical_count": len(audit["historical_workflows"]),
        "canonical_workflows": sorted(audit["canonical_workflows"]),
        # Default builder calls must stay key-and-value identical to the tracked graphs.
        "builder_parity": parity,
    }
    if args.sync_visual and config.visual_workflows_dir.is_dir():
        visual_pairs = [
            (builder.build_highres_native(seed=DEFAULT_SEED, pag=False), "Emilia_08_HighRes_Quality_Base_8GB.json"),
            (builder.build_highres_finish("emilia_quality_selected_native.png", seed=DEFAULT_SEED, pag=False), "Emilia_09_Quality_Finish_2x.json"),
            (builder.build_highres_native(seed=DEFAULT_SEED, pag=True), "Emilia_10_Community_PAG_Quality_8GB.json"),
        ]
        synced_visuals: list[str] = []
        for api_graph, visual_name in visual_pairs:
            visual_path = config.visual_workflows_dir / visual_name
            if visual_path.is_file():
                synced = builder.sync_visual_workflow(api_graph, visual_path)
                visual_path.write_text(json.dumps(synced, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                synced_visuals.append(visual_name)
        summary["synced_visual_workflows"] = synced_visuals
    print(json.dumps(summary, indent=2))
    return 0 if all(parity.values()) else 2


def _sync_into_app(config: StudioConfig) -> list[str]:
    """Copy tracked assets, the patch plugin, workflows and runtime tests into the ComfyUI checkout."""
    app_dir = config.app_dir
    assert app_dir is not None
    synced: list[str] = []

    def copy(source: Path, relative: str) -> None:
        target = app_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        synced.append(relative)

    face_reference = config.assets_dir / "official_face_center_clean.png"
    if face_reference.is_file():
        copy(face_reference, f"input/{face_reference.name}")
    for reference in sorted((config.assets_dir / "official_references").glob("*.png")):
        copy(reference, f"input/{reference.name}")
    plugin = config.patches_dir / "emilia_runtime_patches" / "__init__.py"
    if plugin.is_file():
        copy(plugin, "custom_nodes/emilia_runtime_patches/__init__.py")
    for visual in sorted(config.visual_workflows_dir.glob("*.json")):
        copy(visual, f"user/default/workflows/{visual.name}")
    for api in sorted(config.workflows_dir.glob("*.api.json")):
        copy(api, f"emilia/{api.name}")
    decisions = config.resources_dir / "workflows" / "WORKFLOW_DECISIONS.txt"
    if decisions.is_file():
        copy(decisions, "emilia/WORKFLOW_DECISIONS.txt")
    if (app_dir / "tests-unit" / "comfy_test").is_dir():
        tests_dir = config.root_dir / "tests"
        for test in [tests_dir / "test_emilia_studio.py", *sorted((tests_dir / "comfy_core").glob("test_*.py"))]:
            if test.is_file():
                copy(test, f"tests-unit/comfy_test/{test.name}")
    return synced


def _cmd_setup(args: argparse.Namespace, config: StudioConfig) -> int:
    manifest_path = config.manifest_path
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert config.app_dir is not None
    synced_files = _sync_into_app(config) if args.sync and config.app_dir.is_dir() else []
    missing: list[str] = []
    verified_models: list[dict[str, object]] = []
    for item in manifest.get("models", []) + manifest.get("reference_assets", []):
        rel = item["relative_path"]
        target = config.app_dir / rel
        expected_size = item.get("size_bytes")
        exists = target.is_file()
        actual_size = target.stat().st_size if exists else 0
        size_ok = (expected_size is None) or (actual_size == expected_size)
        if not (exists and size_ok):
            missing.append(rel)
        verified_models.append({
            "name": item["name"],
            "relative_path": rel,
            "exists": exists,
            "size_match": size_ok,
            "actual_bytes": actual_size,
        })
    verified_nodes: list[dict[str, object]] = []
    for node in manifest.get("custom_nodes", []):
        installed = (config.app_dir / node["directory"]).is_dir()
        if not installed:
            missing.append(node["directory"])
        verified_nodes.append({
            "name": node["name"],
            "directory": node["directory"],
            "installed": installed,
            "pinned_commit": node.get("pinned_commit"),
        })
    print(json.dumps({
        "ready": not missing,
        "conda_python": str(config.conda_python),
        "ollama_exe": str(config.ollama_exe),
        "synced_to_app": synced_files,
        "models_and_assets": verified_models,
        "custom_nodes": verified_nodes,
        "missing_or_mismatched": missing,
    }, indent=2))
    return 0 if not missing else 2


_COMMANDS = {
    "render": _cmd_render,
    "batch": _cmd_batch,
    "idea": _cmd_idea,
    "inspect": _cmd_inspect,
    "ledger": _cmd_ledger,
    "workflows": _cmd_workflows,
    "setup": _cmd_setup,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        result = handle(args, config, main, parser)
        if result is not None:
            return result
        return _COMMANDS[args.command](args, config)
    except BridgeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except QualityGateRejected as exc:
        print(f"质检未通过：{exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Cancelled.', file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
