"""Unified CLI entrypoint for Emilia Studio."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import shutil
import sys
from pathlib import Path
from .compiler import BridgeError, OllamaPromptCompiler
from .config import (
    BUILTIN_SCENE_PRESETS,
    EMILIA_ANGLE_SPECS,
    EMILIA_COMPANION_TAGS,
    EMILIA_MAGIC_TAGS,
    EMILIA_OUTFIT_MODES,
    load_config,
)
from .commands import add_commands, handle
from .engine import EmiliaRenderEngine
from .inspector import QualityInspector
from .ledger import ExperimentLedger
from .workflows import EmiliaWorkflowBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emilia-studio",
        description="Emilia Studio — Declarative High-Res SDXL + IPAdapter + Detailer + Upscale Pipeline & Ledger",
    )
    parser.add_argument('--version', action='version', version='emilia-studio 2.1.0')
    parser.add_argument('--config', type=Path, help='Explicit TOML configuration file')
    sub = parser.add_subparsers(dest="command")
    add_commands(sub)

    # 1. render: quality pipeline (fixed seed or custom prompt)
    p_render = sub.add_parser(
        "render",
        help="Run the 2-stage HighRes Quality (1152x1728 -> 2304x3456) render pipeline.",
    )
    p_render.add_argument("--seed", type=int, default=20261027, help="Base generation seed")
    p_render.add_argument("--port", type=int, default=None, help="Override configured rendering port")
    p_render.add_argument("--pag", action="store_true", help="Enable PerturbedAttentionGuidance (scale 1.0)")
    p_render.add_argument(
        "--profile",
        choices=("quality", "ultra"),
        default="quality",
        help="quality: bit-exact 1152x1728 baseline; ultra: multi-scale 832->1152->2304 Hires.fix + canonical outfit lock",
    )
    p_render.add_argument("--prompt", type=str, default=None, help="Custom positive English prompt")
    p_render.add_argument('--preset', choices=tuple(BUILTIN_SCENE_PRESETS))
    p_render.add_argument('--outfit', choices=tuple(EMILIA_OUTFIT_MODES), default=None, help='Emilia outfit mode (ultra profile)')
    p_render.add_argument('--angle', choices=tuple(EMILIA_ANGLE_SPECS), default=None, help='Camera angle & pose reference routing (ultra profile)')
    p_render.add_argument('--puck', action='store_true', help='Include Puck fairy cat companion (ultra profile)')
    p_render.add_argument('--magic', choices=tuple(EMILIA_MAGIC_TAGS), default=None, help='Ice magic effect (ultra profile)')
    p_render.add_argument("--auto-gate", action="store_true", help="Enable automatic seed retry and eye-repair gate")
    p_render.add_argument("--max-retries", type=int, default=2, help="Max seed retries when auto-gate rejects anatomy")
    p_batch = sub.add_parser(
        "batch",
        help="Render multiple seeds using same-stage process reuse while keeping Stage 1 and Stage 2 isolated.",
    )
    p_batch.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit list of seeds")
    p_batch.add_argument("--seed", type=int, default=20261027, help="Starting seed when --count is used")
    p_batch.add_argument("--count", type=int, default=2, help="Number of seeds to generate when --seeds is omitted")
    p_batch.add_argument("--port", type=int, default=None, help="Override configured rendering port")
    p_batch.add_argument("--pag", action="store_true", help="Enable PerturbedAttentionGuidance (scale 1.0)")
    p_batch.add_argument("--profile", choices=("ultra", "quality"), default="ultra")
    p_batch.add_argument("--prompt", type=str, default=None, help="Custom positive English prompt")
    p_batch.add_argument("--preset", choices=tuple(BUILTIN_SCENE_PRESETS))
    p_batch.add_argument("--outfit", choices=tuple(EMILIA_OUTFIT_MODES), default=None, help="Emilia outfit mode (ultra profile)")
    p_batch.add_argument("--angle", choices=tuple(EMILIA_ANGLE_SPECS), default=None, help="Camera angle & pose reference routing (ultra profile)")
    p_batch.add_argument("--puck", action="store_true", help="Include Puck fairy cat companion (ultra profile)")
    p_batch.add_argument("--magic", choices=tuple(EMILIA_MAGIC_TAGS), default=None, help="Ice magic effect (ultra profile)")
    p_batch.add_argument("--auto-gate", action="store_true", help="Enable automatic seed retry, full-body pose gate, and eye-repair gate")
    p_batch.add_argument("--max-retries", type=int, default=2, help="Max seed retries when auto-gate rejects anatomy")
    p_idea = sub.add_parser(
        "idea",
        help="Compile a Chinese idea with Ollama and render via HighRes Quality or Fast LoRA.",
    )
    p_idea.add_argument("idea", nargs="+", help="Chinese image description / idea")
    p_idea.add_argument("--seed", type=int, default=None, help="Random seed if omitted")
    p_idea.add_argument("--port", type=int, default=None, help="Override configured rendering port")
    p_idea.add_argument("--style", choices=("anime", "illustration", "cinematic"), default="anime")
    p_idea.add_argument(
        "--profile",
        choices=("ultra", "quality"),
        default="ultra",
        help="ultra (default): multi-scale 832->1152->2304 with canonical outfit lock; quality: legacy 1152 direct",
    )
    p_idea.add_argument('--outfit', choices=tuple(EMILIA_OUTFIT_MODES), default='royal', help='Emilia outfit mode')
    p_idea.add_argument('--angle', choices=tuple(EMILIA_ANGLE_SPECS), default='auto', help='Camera angle & pose reference routing')
    p_idea.add_argument('--puck', action='store_true', help='Include Puck fairy cat companion')
    p_idea.add_argument('--magic', choices=tuple(EMILIA_MAGIC_TAGS), default='none', help='Ice magic effect')
    p_idea.add_argument("--pag", action="store_true", help="Enable PAG guidance")
    p_idea.add_argument("--auto-gate", action="store_true", help="Enable automatic eye-repair and anatomy gate")
    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect an existing PNG for violet iris consistency, heterochromia, full-body pose, and face/hand counts.",
    )
    p_inspect.add_argument("image", type=Path, help="Path to PNG image")
    p_inspect.add_argument("--stage", choices=("native", "final"), default=None)
    p_inspect.add_argument("--framing", choices=("full_body", "upper_body", "portrait"), default="full_body")
    p_inspect.add_argument("--require-full-body-pose", action="store_true", help="Fail gate if full-body standing/ankle completeness is not met")

    # 4. ledger: ingest, stats, list, compare rounds
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

    # 5. workflows: audit and verify workflow parity
    p_wf = sub.add_parser(
        "workflows",
        help="Audit app/emilia workflows and verify API/GUI builder parity.",
    )
    p_wf.add_argument("--sync-visual", action="store_true", help="Verify visual workflow sync")

    # 6. setup: verify models_manifest.json weights, reference assets, and custom nodes
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        result = handle(args, config, main, parser)
        if result is not None:
            return result
        if args.command == "render":
            has_creative_flags = any([
                args.outfit is not None,
                args.angle is not None,
                args.puck,
                args.magic is not None,
            ])
            if has_creative_flags and args.profile != "ultra":
                raise ValueError("Creative flags (--outfit, --angle, --puck, --magic) require --profile ultra")
            outfit = args.outfit or "royal"
            angle = args.angle or "auto"
            companion = "puck" if args.puck else "none"
            magic = args.magic or "none"
            framing = "full_body"
            neg_prompt: str | None = None
            native_opts: dict | None = None
            finish_opts: dict | None = None
            if args.preset:
                if args.prompt:
                    raise ValueError("Use --preset or --prompt, not both")
                preset = BUILTIN_SCENE_PRESETS[args.preset]
                framing = preset["framing"]
                compiler = OllamaPromptCompiler(config)
                args.prompt = compiler.build_emilia_ultra_prompt(
                    preset["tags"],
                    framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                if args.profile == "ultra":
                    neg_prompt = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            elif has_creative_flags and args.prompt is None:
                preset = BUILTIN_SCENE_PRESETS["rose-garden"]
                framing = preset["framing"]
                compiler = OllamaPromptCompiler(config)
                args.prompt = compiler.build_emilia_ultra_prompt(
                    preset["tags"],
                    framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                neg_prompt = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            elif has_creative_flags and args.prompt is not None:
                compiler = OllamaPromptCompiler(config)
                args.prompt = compiler.build_emilia_ultra_prompt(
                    args.prompt,
                    framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                neg_prompt = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            ipa_weight_override: float | None = None
            if args.profile == "ultra" and (has_creative_flags or args.preset):
                spec_outfit = EMILIA_OUTFIT_MODES[outfit]
                ipa_weight_override = float(spec_outfit.get("ipadapter_weight", 0.18))
                native_opts = {
                    "framing": framing,
                    "camera_angle": angle,
                    "reference_mode": spec_outfit.get("reference_mode", "single"),
                    "lora_strength_base": spec_outfit.get("lora_strength_base", spec_outfit["lora_strength"]),
                    "skirt_ipadapter_weight": 0.45 if spec_outfit["use_skirt_ipa"] else 0.0,
                }
                finish_opts = {"lora_strength": spec_outfit.get("lora_strength_finish", spec_outfit["lora_strength"])}
            engine = EmiliaRenderEngine(config=config)
            asyncio.run(
                engine.render_quality_pipeline(
                    seed=args.seed,
                    port=args.port if args.port is not None else config.quality_port,
                    pag=args.pag,
                    profile=args.profile,
                    positive_prompt=args.prompt,
                    negative_prompt=neg_prompt,
                    ipadapter_weight=ipa_weight_override,
                    strict_seed_lock=not args.auto_gate,
                    auto_eye_repair=args.auto_gate,
                    max_seed_retries=args.max_retries if args.auto_gate else 0,
                    native_options=native_opts,
                    finish_options=finish_opts,
                )
            )
            return 0
        if args.command == "batch":
            if args.seeds:
                seeds = [int(s) for s in args.seeds]
            else:
                if args.count < 1:
                    raise ValueError("--count must be positive")
                seeds = [int(args.seed) + i for i in range(args.count)]
            has_creative_flags = any([
                args.outfit is not None,
                args.angle is not None,
                args.puck,
                args.magic is not None,
            ])
            if has_creative_flags and args.profile != "ultra":
                raise ValueError("Creative flags (--outfit, --angle, --puck, --magic) require --profile ultra")
            outfit = args.outfit or "royal"
            angle = args.angle or "auto"
            companion = "puck" if args.puck else "none"
            magic = args.magic or "none"
            framing = "full_body"
            neg_prompt = None
            native_opts = None
            finish_opts = None
            ipa_weight_override = None
            if args.preset:
                if args.prompt:
                    raise ValueError("Use --preset or --prompt, not both")
                preset = BUILTIN_SCENE_PRESETS[args.preset]
                framing = preset["framing"]
                compiler = OllamaPromptCompiler(config)
                args.prompt = compiler.build_emilia_ultra_prompt(
                    preset["tags"],
                    framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                if args.profile == "ultra":
                    neg_prompt = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            elif has_creative_flags and args.prompt is None:
                preset = BUILTIN_SCENE_PRESETS["rose-garden"]
                framing = preset["framing"]
                compiler = OllamaPromptCompiler(config)
                args.prompt = compiler.build_emilia_ultra_prompt(
                    preset["tags"],
                    framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                neg_prompt = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            elif has_creative_flags and args.prompt is not None:
                compiler = OllamaPromptCompiler(config)
                args.prompt = compiler.build_emilia_ultra_prompt(
                    args.prompt,
                    framing,
                    outfit=outfit,
                    companion=companion,
                    magic=magic,
                    angle=angle,
                )
                neg_prompt = compiler.build_emilia_ultra_negative(outfit=outfit, framing=framing)
            if args.profile == "ultra" and (has_creative_flags or args.preset):
                spec_outfit = EMILIA_OUTFIT_MODES[outfit]
                ipa_weight_override = float(spec_outfit.get("ipadapter_weight", 0.18))
                native_opts = {
                    "framing": framing,
                    "camera_angle": angle,
                    "reference_mode": spec_outfit.get("reference_mode", "single"),
                    "lora_strength_base": spec_outfit.get("lora_strength_base", spec_outfit["lora_strength"]),
                    "skirt_ipadapter_weight": 0.45 if spec_outfit["use_skirt_ipa"] else 0.0,
                }
                finish_opts = {"lora_strength": spec_outfit.get("lora_strength_finish", spec_outfit["lora_strength"])}
            engine = EmiliaRenderEngine(config=config)
            receipts = asyncio.run(
                engine.render_batch_pipeline(
                    seeds=seeds,
                    port=args.port if args.port is not None else config.quality_port,
                    pag=args.pag,
                    profile=args.profile,
                    positive_prompt=args.prompt,
                    negative_prompt=neg_prompt,
                    ipadapter_weight=ipa_weight_override,
                    strict_seed_lock=not args.auto_gate,
                    auto_eye_repair=args.auto_gate,
                    max_seed_retries=args.max_retries if args.auto_gate else 0,
                    native_options=native_opts,
                    finish_options=finish_opts,
                )
            )
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
            return 0

        if args.command == "idea":
            idea_text = " ".join(args.idea).strip()
            compiler = OllamaPromptCompiler(config=config)
            compiler.ensure_ollama()
            is_ultra = args.profile == "ultra"
            companion = "puck" if args.puck else "none"
            spec = compiler.compile_idea(
                idea_text,
                style=args.style,
                highres_quality_mode=True,
                ultra_mode=is_ultra,
                outfit=args.outfit,
                companion=companion,
                magic=args.magic,
                angle=args.angle,
            )
            print(
                f"编译完成 [{args.profile}/{spec.framing}, outfit={args.outfit}, angle={args.angle}, IPAdapter={spec.recommended_ipadapter_weight}]:\n{spec.positive_prompt}",
                flush=True,
            )
            seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
            spec_outfit = EMILIA_OUTFIT_MODES[args.outfit]
            eff_ipa = (
                max(spec.recommended_ipadapter_weight, float(spec_outfit.get("ipadapter_weight", 0.18)))
                if is_ultra
                else spec.recommended_ipadapter_weight
            )
            native_opts = (
                {
                    "framing": spec.framing,
                    "camera_angle": args.angle,
                    "reference_mode": spec_outfit.get("reference_mode", "single"),
                    "lora_strength_base": spec_outfit.get("lora_strength_base", spec_outfit["lora_strength"]),
                    "skirt_ipadapter_weight": 0.45 if spec_outfit["use_skirt_ipa"] else 0.0,
                }
                if is_ultra
                else None
            )
            finish_opts = {"lora_strength": spec_outfit.get("lora_strength_finish", spec_outfit["lora_strength"])} if is_ultra else None
            engine = EmiliaRenderEngine(config=config)
            asyncio.run(
                engine.render_quality_pipeline(
                    seed=seed,
                    port=args.port if args.port is not None else config.quality_port,
                    pag=args.pag,
                    profile=args.profile,
                    positive_prompt=spec.positive_prompt,
                    negative_prompt=spec.negative_prompt,
                    ipadapter_weight=eff_ipa,
                    user_idea=idea_text,
                    strict_seed_lock=not args.auto_gate,
                    auto_eye_repair=args.auto_gate,
                    max_seed_retries=2 if args.auto_gate else 0,
                    native_options=native_opts,
                    finish_options=finish_opts,
                )
            )
            return 0

        if args.command == "inspect":
            inspector = QualityInspector(app_dir=config.app_dir, python_executable=config.conda_python)
            report = inspector.inspect_image(
                args.image,
                stage=args.stage,
                framing=args.framing,
                require_full_body_pose=args.require_full_body_pose,
            )
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            return 0 if report.passed else 2

        if args.command == "ledger":
            ledger = ExperimentLedger(config=config)
            if args.action == "ingest":
                ingested = ledger.ingest_existing_artifacts()
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

        if args.command == "workflows":
            builder = EmiliaWorkflowBuilder(config=config)
            audit = builder.audit_workflow_directory()
            summary = {
                "canonical_api_count": len(audit["canonical_workflows"]),
                "experiment_report_count": len(audit["experiment_reports"]),
                "historical_count": len(audit["historical_workflows"]),
                "canonical_workflows": sorted(audit["canonical_workflows"]),
            }
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "setup":
            manifest_path = config.manifest_path
            if not manifest_path.is_file():
                raise RuntimeError(f"Manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert config.app_dir is not None
            synced_files: list[str] = []
            if getattr(args, "sync", False) and config.app_dir.is_dir():
                # 1. Sync reference assets (canonical face + multi-angle official reference pack)
                ref_src = config.assets_dir / "official_face_center_clean.png"
                if ref_src.is_file():
                    ref_dst = config.app_dir / "input" / "official_face_center_clean.png"
                    ref_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ref_src, ref_dst)
                    synced_files.append("input/official_face_center_clean.png")
                ref_pack_dir = config.assets_dir / "official_references"
                if ref_pack_dir.is_dir():
                    input_dir = config.app_dir / "input"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    for rf in sorted(ref_pack_dir.glob("*.png")):
                        shutil.copy2(rf, input_dir / rf.name)
                        synced_files.append(f"input/{rf.name}")
                # 2. Sync emilia_runtime_patches custom node
                patch_src = config.patches_dir / "emilia_runtime_patches" / "__init__.py"
                if patch_src.is_file():
                    patch_dst = config.app_dir / "custom_nodes" / "emilia_runtime_patches" / "__init__.py"
                    patch_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(patch_src, patch_dst)
                    synced_files.append("custom_nodes/emilia_runtime_patches/__init__.py")
                # 3. Sync visual workflows into app/user/default/workflows
                vis_src_dir = config.visual_workflows_dir
                if vis_src_dir.is_dir():
                    vis_dst_dir = config.app_dir / "user" / "default" / "workflows"
                    vis_dst_dir.mkdir(parents=True, exist_ok=True)
                    for vf in sorted(vis_src_dir.glob("*.json")):
                        shutil.copy2(vf, vis_dst_dir / vf.name)
                        synced_files.append(f"user/default/workflows/{vf.name}")
                # 4. Sync API workflows and WORKFLOW_DECISIONS.txt into app/emilia
                api_dst_dir = config.app_dir / "emilia"
                if config.workflows_dir.is_dir():
                    api_dst_dir.mkdir(parents=True, exist_ok=True)
                    for af in sorted(config.workflows_dir.glob("*.api.json")):
                        shutil.copy2(af, api_dst_dir / af.name)
                        synced_files.append(f"emilia/{af.name}")
                decisions_src = config.resources_dir / "workflows" / "WORKFLOW_DECISIONS.txt"
                if decisions_src.is_file():
                    api_dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(decisions_src, api_dst_dir / "WORKFLOW_DECISIONS.txt")
                    synced_files.append("emilia/WORKFLOW_DECISIONS.txt")
                test_src = config.root_dir / "tests" / "test_emilia_studio.py"
                test_dst = config.app_dir / "tests-unit" / "comfy_test" / "test_emilia_studio.py"
                if test_src.is_file() and test_dst.parent.is_dir():
                    shutil.copy2(test_src, test_dst)
                    synced_files.append("tests-unit/comfy_test/test_emilia_studio.py")
            missing: list[str] = []
            verified_models: list[dict[str, object]] = []
            for item in manifest.get("models", []) + manifest.get("reference_assets", []):
                rel = item["relative_path"]
                target = config.app_dir / rel
                expected_size = item.get("size_bytes")
                exists = target.is_file()
                actual_size = target.stat().st_size if exists else 0
                size_ok = (expected_size is None) or (actual_size == expected_size)
                ok = exists and size_ok
                if not ok:
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
                ndir = config.app_dir / node["directory"]
                n_ok = ndir.is_dir()
                if not n_ok:
                    missing.append(node["directory"])
                verified_nodes.append({
                    "name": node["name"],
                    "directory": node["directory"],
                    "installed": n_ok,
                    "pinned_commit": node.get("pinned_commit"),
                })
            report_out = {
                "ready": len(missing) == 0,
                "conda_python": str(config.conda_python),
                "ollama_exe": str(config.ollama_exe),
                "synced_to_app": synced_files,
                "models_and_assets": verified_models,
                "custom_nodes": verified_nodes,
                "missing_or_mismatched": missing,
            }
            print(json.dumps(report_out, indent=2))
            return 0 if len(missing) == 0 else 2

    except BridgeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Cancelled.', file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
