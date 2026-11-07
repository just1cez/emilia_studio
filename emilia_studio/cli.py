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
from .config import load_config
from .engine import EmiliaRenderEngine
from .inspector import QualityInspector
from .ledger import ExperimentLedger
from .workflows import EmiliaWorkflowBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emilia-studio",
        description="Emilia Studio — Declarative High-Res SDXL + IPAdapter + Detailer + Upscale Pipeline & Ledger",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. render: quality pipeline (fixed seed or custom prompt)
    p_render = sub.add_parser(
        "render",
        help="Run the 2-stage HighRes Quality (1152x1728 -> 2304x3456) render pipeline.",
    )
    p_render.add_argument("--seed", type=int, default=20261027, help="Base generation seed")
    p_render.add_argument("--port", type=int, default=8199, help="ComfyUI server port")
    p_render.add_argument("--pag", action="store_true", help="Enable PerturbedAttentionGuidance (scale 1.0)")
    p_render.add_argument(
        "--profile",
        choices=("quality", "ultra"),
        default="quality",
        help="quality: bit-exact 1152x1728 baseline; ultra: multi-scale 832->1152->2304 Hires.fix + canonical outfit lock",
    )
    p_render.add_argument("--prompt", type=str, default=None, help="Custom positive English prompt")
    p_render.add_argument("--auto-gate", action="store_true", help="Enable automatic seed retry and eye-repair gate")
    p_render.add_argument("--max-retries", type=int, default=2, help="Max seed retries when auto-gate rejects anatomy")
    # 2. idea: Chinese natural language -> Ollama -> HighRes Quality or Fast LoRA
    p_idea = sub.add_parser(
        "idea",
        help="Compile a Chinese idea with Ollama and render via HighRes Quality or Fast LoRA.",
    )
    p_idea.add_argument("idea", nargs="+", help="Chinese image description / idea")
    p_idea.add_argument("--seed", type=int, default=None, help="Random seed if omitted")
    p_idea.add_argument("--port", type=int, default=8199, help="ComfyUI server port")
    p_idea.add_argument("--style", choices=("anime", "illustration", "cinematic"), default="anime")
    p_idea.add_argument(
        "--profile",
        choices=("ultra", "quality"),
        default="ultra",
        help="ultra (default): multi-scale 832->1152->2304 with canonical outfit lock; quality: legacy 1152 direct",
    )
    p_idea.add_argument("--pag", action="store_true", help="Enable PAG guidance")
    p_idea.add_argument("--auto-gate", action="store_true", help="Enable automatic eye-repair and anatomy gate")
    # 3. inspect: run CPU-only quality and violet-iris inspection on any PNG
    p_inspect = sub.add_parser(
        "inspect",
        help="Inspect an existing PNG for violet iris consistency, heterochromia, and face/hand counts.",
    )
    p_inspect.add_argument("image", type=Path, help="Path to PNG image")
    p_inspect.add_argument("--stage", choices=("native", "final"), default=None)

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
    config = load_config()

    try:
        if args.command == "render":
            engine = EmiliaRenderEngine(config=config)
            asyncio.run(
                engine.render_quality_pipeline(
                    seed=args.seed,
                    port=args.port,
                    pag=args.pag,
                    profile=args.profile,
                    positive_prompt=args.prompt,
                    strict_seed_lock=not args.auto_gate,
                    auto_eye_repair=args.auto_gate,
                    max_seed_retries=args.max_retries if args.auto_gate else 0,
                )
            )
            return 0

        if args.command == "idea":
            idea_text = " ".join(args.idea).strip()
            compiler = OllamaPromptCompiler(config=config)
            compiler.ensure_ollama()
            is_ultra = args.profile == "ultra"
            spec = compiler.compile_idea(
                idea_text,
                style=args.style,
                highres_quality_mode=True,
                ultra_mode=is_ultra,
            )
            print(
                f"编译完成 [{args.profile}/{spec.framing}, IPAdapter={spec.recommended_ipadapter_weight}]:\n{spec.positive_prompt}",
                flush=True,
            )
            seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
            engine = EmiliaRenderEngine(config=config)
            asyncio.run(
                engine.render_quality_pipeline(
                    seed=seed,
                    port=args.port,
                    pag=args.pag,
                    profile=args.profile,
                    positive_prompt=spec.positive_prompt,
                    negative_prompt=spec.negative_prompt,
                    ipadapter_weight=spec.recommended_ipadapter_weight,
                    user_idea=idea_text,
                    strict_seed_lock=not args.auto_gate,
                    auto_eye_repair=args.auto_gate,
                    max_seed_retries=2 if args.auto_gate else 0,
                )
            )
            return 0

        if args.command == "inspect":
            inspector = QualityInspector(app_dir=config.app_dir)
            report = inspector.inspect_image(args.image, stage=args.stage)
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
            manifest_path = config.root_dir / "models_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"Manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert config.app_dir is not None
            synced_files: list[str] = []
            if getattr(args, "sync", False) and config.app_dir.is_dir():
                # 1. Sync reference assets
                ref_src = config.root_dir / "assets" / "official_face_center_clean.png"
                if ref_src.is_file():
                    ref_dst = config.app_dir / "input" / "official_face_center_clean.png"
                    ref_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ref_src, ref_dst)
                    synced_files.append("input/official_face_center_clean.png")
                # 2. Sync emilia_runtime_patches custom node
                patch_src = config.root_dir / "patches" / "emilia_runtime_patches" / "__init__.py"
                if patch_src.is_file():
                    patch_dst = config.app_dir / "custom_nodes" / "emilia_runtime_patches" / "__init__.py"
                    patch_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(patch_src, patch_dst)
                    synced_files.append("custom_nodes/emilia_runtime_patches/__init__.py")
                # 3. Sync visual workflows into app/user/default/workflows
                vis_src_dir = config.root_dir / "workflows" / "visual"
                if vis_src_dir.is_dir():
                    vis_dst_dir = config.app_dir / "user" / "default" / "workflows"
                    vis_dst_dir.mkdir(parents=True, exist_ok=True)
                    for vf in sorted(vis_src_dir.glob("*.json")):
                        shutil.copy2(vf, vis_dst_dir / vf.name)
                        synced_files.append(f"user/default/workflows/{vf.name}")
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
    except Exception as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
