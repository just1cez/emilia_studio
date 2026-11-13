"""Interactive and maintenance commands for the installed CLI."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from datetime import datetime, timezone
import shutil
import time

from PIL import Image

from .config import (
    BUILTIN_SCENE_PRESETS,
    EMILIA_ANGLE_SPECS,
    EMILIA_COMPANION_TAGS,
    EMILIA_MAGIC_TAGS,
    EMILIA_OUTFIT_MODES,
    get_user_config_path,
)
from .tui import download_file_with_progress, format_bytes, print_banner, render_table

PATH_KEYS = (
    'root_dir', 'app_dir', 'output_dir', 'temp_dir',
    'workflows_dir', 'visual_workflows_dir', 'resources_dir',
    'conda_python', 'ollama_exe',
)
VALUE_KEYS = ('comfy_host', 'comfy_port', 'quality_port', 'ollama_base', 'ollama_model')


def checkout_checks(config, manifest: dict) -> list[tuple[str, str, str, bool]]:
    """Compare runtime git checkouts with their pinned commits and tracked source patches.

    Returns ``(check, status, details, ok)`` rows. A checkout that git cannot inspect is
    reported UNKNOWN without failing; a different commit or an unapplied patch fails,
    because every recorded timing and pixel hash refers to exactly this patched source.
    """
    rows: list[tuple[str, str, str, bool]] = []
    git = shutil.which('git')
    entries = [manifest['comfyui']] if manifest.get('comfyui') else []
    entries += [node for node in manifest.get('custom_nodes', []) if node.get('pinned_commit')]
    for entry in entries:
        directory = (config.app_dir / entry['directory']).resolve()
        pinned = entry['pinned_commit']

        def run(*arguments: str) -> subprocess.CompletedProcess[str] | None:
            if git is None or not (directory / '.git').exists():
                return None
            try:
                return subprocess.run([git, '-C', str(directory), *arguments], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                return None

        head = run('rev-parse', 'HEAD')
        if head is None or head.returncode != 0:
            rows.append((f"{entry['name']} commit", 'UNKNOWN', f'not a readable git checkout; expected {pinned[:12]}', True))
        else:
            actual = head.stdout.strip()
            matches = actual.startswith(pinned) or pinned.startswith(actual)
            rows.append((f"{entry['name']} commit", 'OK' if matches else 'DRIFT', actual[:12] if matches else f'{actual[:12]} (pinned {pinned[:12]})', matches))
        patch = entry.get('patch')
        if patch:
            patch_path = config.resources_dir / patch
            applied = run('apply', '--reverse', '--check', str(patch_path)) if patch_path.is_file() else None
            if applied is None:
                rows.append((f"{entry['name']} patch", 'UNKNOWN', patch, True))
            else:
                rows.append((f"{entry['name']} patch", 'OK' if applied.returncode == 0 else 'NOT APPLIED', patch, applied.returncode == 0))
    return rows


def add_commands(sub):
    p = sub.add_parser('config', help='Initialize, show or edit persistent configuration')
    p.add_argument('action', choices=('init', 'show', 'set', 'path'), nargs='?', default='show')
    p.add_argument('key', nargs='?', choices=PATH_KEYS + VALUE_KEYS)
    p.add_argument('value', nargs='?')
    p.add_argument('--root-dir', type=Path)
    p.add_argument('--app-dir', type=Path)
    p.add_argument('--python', type=Path)
    p.add_argument('--force', action='store_true', help='Replace an existing config on init')
    sub.add_parser('doctor', help='Check runtime, Python dependencies and installed models')
    p = sub.add_parser('pull', help='Download models safely; existing valid files are retained')
    p.add_argument('models', nargs='*', help='Manifest keys, or all')
    p.add_argument('--list', action='store_true')
    p = sub.add_parser('presets', help='List curated scene presets or print one prompt')
    p.add_argument('name', nargs='?', choices=tuple(BUILTIN_SCENE_PRESETS))
    p = sub.add_parser('outfits', help='List Emilia outfit modes or print one outfit prompt')
    p.add_argument('name', nargs='?', choices=tuple(EMILIA_OUTFIT_MODES))
    p.add_argument(
        '--framing',
        choices=('full_body', 'upper_body', 'portrait'),
        default='full_body',
        help='Framing when previewing a single outfit prompt',
    )
    p = sub.add_parser('gallery', help='List generated PNGs, optionally open one')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--open', type=int, metavar='INDEX')
    p = sub.add_parser('compare', help='Compare two rendered PNGs or receipts side-by-side')
    p.add_argument('image_a', type=Path, help='First PNG or .run.json receipt')
    p.add_argument('image_b', type=Path, help='Second PNG or .run.json receipt')
    p.add_argument('--json', action='store_true', help='Emit machine-readable JSON comparison')
    p = sub.add_parser('export', help='Export latest or specified render and receipt into a bundle directory')
    p.add_argument('destination', type=Path, help='Destination folder for exported bundle')
    p.add_argument('--image', type=Path, default=None, help='Specific PNG or .run.json to export (defaults to latest)')
    p = sub.add_parser('benchmark', help='Summarize historical render performance or run a 1-seed smoke check')
    p.add_argument('--limit', type=int, default=10, help='Number of recent runs to summarize')
    p.add_argument('--run', action='store_true', help='Execute a live quality pipeline render before summarizing')
    p.add_argument('--seed', type=int, default=20261027, help='Seed for live benchmark run')
    p.add_argument('--profile', choices=('quality', 'ultra'), default='quality')
    p = sub.add_parser('clean', help='Clean isolated stage temp folders in temp_dir')
    p.add_argument('--dry-run', action='store_true', help='List candidate temp folders without deleting')
    p.add_argument('--older-than-hours', type=float, default=1.0, help='Minimum age in hours before removal (0 for all finished stage dirs)')
    sub.add_parser('shell', help='Interactive command prompt (exit to leave)')
    sub.add_parser('banner', help='Display the EMILIA ASCII banner')

def handle(args, config, main, parser):
    cmd = args.command
    if cmd in (None, 'shell', 'banner'):
        print_banner()
        if cmd == 'banner':
            return 0
        print('Commands: config, doctor, pull, presets, outfits, render, batch, idea, inspect, gallery, compare, export, benchmark, clean, ledger, workflows, setup')
        print('Example: render --profile ultra --preset rose-garden --outfit royal --angle three-quarter\nUse help or COMMAND --help. Type exit to leave.')
        if not sys.stdin.isatty():
            return 0
        while True:
            try:
                line = input('emilia> ').strip()
                if line in ('exit', 'quit'):
                    return 0
                if not line:
                    continue
                if line == 'help':
                    parser.print_help()
                    continue
                words = shlex.split(line, posix=False)
                words = [w[1:-1] if len(w) >= 2 and w[0] == w[-1] and w[0] in '\"\'' else w for w in words]
                if words[0] == 'help' and len(words) > 1:
                    words = words[1:] + ['--help']
                if words[0] == 'shell':
                    print('Already in the interactive shell.')
                    continue
                try:
                    main((['--config', str(args.config)] if args.config else []) + words)
                except SystemExit:
                    pass
            except EOFError:
                return 0
            except KeyboardInterrupt:
                print('\nCancelled.')
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
    if cmd == 'config':
        target = args.config.expanduser().resolve() if args.config else get_user_config_path()
        if args.action == 'path':
            print(target)
            return 0
        if args.action == 'show':
            print(config.to_toml_string())
            print(f'# source: {config.config_source}')
            return 0
        if args.action == 'init':
            if target.exists() and not args.force:
                raise ValueError(f'Config exists: {target}; use config set or init --force')
            from .config import StudioConfig
            root = args.root_dir.expanduser().resolve() if args.root_dir else config.root_dir
            config = StudioConfig(root_dir=root, app_dir=args.app_dir, conda_python=args.python or config.conda_python)
        else:
            if args.key is None or args.value is None:
                raise ValueError('Usage: config set KEY VALUE')
            value = args.value
            if args.key in PATH_KEYS:
                value = Path(value).expanduser().resolve()
            elif args.key.endswith('_port'):
                value = int(value)
                if not 1 <= value <= 65535:
                    raise ValueError('Port must be between 1 and 65535')
            setattr(config, args.key, value)
        print(config.save_user_config(target))
        return 0
    if cmd == 'presets':
        if args.name:
            from .compiler import OllamaPromptCompiler
            preset = BUILTIN_SCENE_PRESETS[args.name]
            print(OllamaPromptCompiler(config).build_emilia_ultra_prompt(preset['tags'], preset['framing']))
        else:
            print(render_table(['Preset', 'Scene', 'Seed'], [[k, v['title'], str(v['default_seed'])] for k, v in BUILTIN_SCENE_PRESETS.items()]))
        return 0
    if cmd == 'outfits':
        if args.name:
            from .compiler import OllamaPromptCompiler
            spec = EMILIA_OUTFIT_MODES[args.name]
            compiler = OllamaPromptCompiler(config)
            pos = compiler.build_emilia_ultra_prompt('standing gracefully, gentle smile, looking at viewer', args.framing, outfit=args.name)
            neg = compiler.build_emilia_ultra_negative(outfit=args.name, framing=args.framing)
            print(json.dumps({
                'outfit': args.name,
                'title': spec['title'],
                'framing': args.framing,
                'lora_strength': spec['lora_strength'],
                'lora_strength_base': spec.get('lora_strength_base', spec['lora_strength']),
                'lora_strength_finish': spec.get('lora_strength_finish', spec['lora_strength']),
                'reference_mode': spec.get('reference_mode', 'single'),
                'use_skirt_ipa': spec['use_skirt_ipa'],
                'use_brooch_detailer': spec['use_brooch_detailer'],
                'positive_prompt': pos,
                'negative_prompt': neg,
            }, indent=2, ensure_ascii=False))
        else:
            rows = [
                [
                    k,
                    v['title'],
                    f"{v.get('lora_strength_base', v['lora_strength']):.2f} -> {v.get('lora_strength_finish', v['lora_strength']):.2f}",
                    'yes' if v['use_skirt_ipa'] else 'no',
                    'yes' if v['use_brooch_detailer'] else 'no',
                ]
                for k, v in EMILIA_OUTFIT_MODES.items()
            ]
            print(render_table(['Outfit', 'Description', 'LoRA (Base->Finish)', 'Skirt IPA', 'Brooch Detailer'], rows, title='Emilia Outfit Modes'))
        return 0
    if cmd == 'gallery':
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        files = sorted(config.output_dir.rglob('*.png'), key=lambda p: p.stat().st_mtime, reverse=True)[:args.limit]
        print(render_table(['Index', 'Image'], [[str(i), str(p)] for i, p in enumerate(files, 1)]))
        if args.open is not None:
            if not 1 <= args.open <= len(files):
                raise ValueError('Image index outside displayed gallery')
            path = str(files[args.open - 1])
            if sys.platform == 'win32':
                os.startfile(path)
            else:
                subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', path], check=True)
        return 0
    if cmd == 'compare':
        from .inspector import QualityInspector

        def _resolve_target(p: Path) -> tuple[Path, dict | None]:
            resolved = p.expanduser().resolve()
            if not resolved.is_file():
                raise ValueError(f'File not found: {resolved}')
            if resolved.name.endswith('.run.json') or resolved.suffix.lower() == '.json':
                rec = json.loads(resolved.read_text(encoding='utf-8'))
                img_str = rec.get('final') or rec.get('selected_native') or rec.get('native')
                if not img_str:
                    raise ValueError(f'Receipt does not reference an image: {resolved}')
                img_path = Path(img_str)
                if not img_path.is_file():
                    raise ValueError(f'Referenced image not found: {img_path}')
                return img_path, rec
            rec_candidate = resolved.with_suffix('.run.json')
            rec = json.loads(rec_candidate.read_text(encoding='utf-8')) if rec_candidate.is_file() else None
            return resolved, rec

        img_a, rec_a = _resolve_target(args.image_a)
        img_b, rec_b = _resolve_target(args.image_b)
        yolo_ready = bool(
            config.app_dir
            and (config.app_dir / 'models' / 'ultralytics' / 'bbox' / 'face_yolov8n.pt').is_file()
            and (config.app_dir / 'models' / 'ultralytics' / 'bbox' / 'hand_yolov8n.pt').is_file()
            and (not getattr(sys, 'frozen', False) or config.conda_python)
        )
        inspector = QualityInspector(app_dir=config.app_dir, python_executable=config.conda_python, use_yolo=yolo_ready)
        rep_a = inspector.inspect_image(img_a)
        rep_b = inspector.inspect_image(img_b)

        def _extract_metrics(img_path: Path, rep: object, rec: dict | None) -> dict:
            qi = (rec or {}).get('quality_inspection') or {}
            cached = qi.get('final') if (rec and rec.get('final') and Path(rec['final']).resolve() == img_path.resolve()) else qi.get('native')
            if isinstance(cached, dict) and cached.get('pixel_sha256') == rep.pixel_sha256:
                iris = cached.get('iris_analysis') or {}
                return {
                    'path': str(img_path),
                    'dimensions': [cached.get('width', rep.width), cached.get('height', rep.height)],
                    'pixel_sha256': rep.pixel_sha256,
                    'left_eye_hue_deg': iris.get('left_eye_hue_deg', cached.get('left_eye_hue_deg')),
                    'right_eye_hue_deg': iris.get('right_eye_hue_deg', cached.get('right_eye_hue_deg')),
                    'heterochromia_score': iris.get('heterochromia_score', cached.get('heterochromia_score')),
                    'passed': bool(cached.get('passed', rep.passed)),
                    'seed': rec.get('seed') if rec else None,
                    'profile': rec.get('profile') if rec else None,
                }
            return {
                'path': str(img_path),
                'dimensions': [rep.width, rep.height],
                'pixel_sha256': rep.pixel_sha256,
                'left_eye_hue_deg': rep.iris_analysis.get('left_eye_hue_deg'),
                'right_eye_hue_deg': rep.iris_analysis.get('right_eye_hue_deg'),
                'heterochromia_score': rep.iris_analysis.get('heterochromia_score'),
                'passed': rep.passed,
                'seed': rec.get('seed') if rec else None,
                'profile': rec.get('profile') if rec else None,
            }

        metrics_a = _extract_metrics(img_a, rep_a, rec_a)
        metrics_b = _extract_metrics(img_b, rep_b, rec_b)
        comparison = {
            'pixel_identical': rep_a.pixel_sha256 == rep_b.pixel_sha256,
            'image_a': metrics_a,
            'image_b': metrics_b,
        }
        if args.json:
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
        else:
            def _deg(value: object) -> str:
                return '-' if value is None else f'{value}°'

            rows = [
                ['Path', str(img_a.name), str(img_b.name)],
                ['Resolution', f'{rep_a.width}x{rep_a.height}', f'{rep_b.width}x{rep_b.height}'],
                ['Pixel SHA256', rep_a.pixel_sha256[:16] + '...', rep_b.pixel_sha256[:16] + '...'],
                ['Exact Match', 'YES' if comparison['pixel_identical'] else 'NO', 'YES' if comparison['pixel_identical'] else 'NO'],
                ['Seed', str(metrics_a['seed'] or '-'), str(metrics_b['seed'] or '-')],
                ['Iris Hue (L/R)', *(f"{_deg(m['left_eye_hue_deg'])} / {_deg(m['right_eye_hue_deg'])}" for m in (metrics_a, metrics_b))],
                ['Heterochromia', _deg(metrics_a['heterochromia_score']), _deg(metrics_b['heterochromia_score'])],
                ['Gate Passed', 'PASS' if metrics_a['passed'] else 'FAIL', 'PASS' if metrics_b['passed'] else 'FAIL'],
            ]
            print(render_table(['Metric', 'Image A', 'Image B'], rows, title='Emilia Render Comparison'))
        return 0
    if cmd == 'export':
        dest = args.destination.expanduser().resolve()
        if args.image is not None:
            candidate = args.image.expanduser().resolve()
            if not candidate.is_file():
                raise ValueError(f'Export source not found: {candidate}')
            if candidate.name.endswith('.run.json'):
                receipt_file = candidate
                rec_data = json.loads(receipt_file.read_text(encoding='utf-8'))
                img_file = Path(rec_data.get('final') or rec_data.get('native') or '')
            else:
                img_file = candidate
                receipt_file = candidate.with_suffix('.run.json')
        else:
            pngs = sorted(config.output_dir.rglob('*.png'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not pngs:
                raise ValueError(f'No rendered PNGs found in {config.output_dir}')
            img_file = pngs[0]
            receipt_file = img_file.with_suffix('.run.json')
        if not img_file.is_file():
            raise ValueError(f'Image file not found: {img_file}')
        dest.mkdir(parents=True, exist_ok=True)
        copied_files = []
        dst_img = dest / img_file.name
        shutil.copy2(img_file, dst_img)
        copied_files.append(str(dst_img))
        manifest_summary: dict = {'exported_at': datetime.now(timezone.utc).isoformat(), 'final_image': dst_img.name}
        if receipt_file.is_file():
            dst_rec = dest / receipt_file.name
            shutil.copy2(receipt_file, dst_rec)
            copied_files.append(str(dst_rec))
            rec_data = json.loads(receipt_file.read_text(encoding='utf-8'))
            manifest_summary['seed'] = rec_data.get('seed')
            manifest_summary['profile'] = rec_data.get('profile')
            native_src = Path(rec_data.get('native') or '')
            if native_src.is_file() and native_src.resolve() != img_file.resolve():
                dst_nat = dest / native_src.name
                shutil.copy2(native_src, dst_nat)
                copied_files.append(str(dst_nat))
                manifest_summary['native_image'] = dst_nat.name
        with Image.open(img_file) as im:
            manifest_summary['dimensions'] = list(im.size)
            if 'prompt' in im.info:
                (dest / 'workflow.api.json').write_text(im.info['prompt'], encoding='utf-8')
                copied_files.append(str(dest / 'workflow.api.json'))
        summary_path = dest / 'export_manifest.json'
        summary_path.write_text(json.dumps(manifest_summary, indent=2, ensure_ascii=False), encoding='utf-8')
        copied_files.append(str(summary_path))
        print(json.dumps({'destination': str(dest), 'files': copied_files}, indent=2, ensure_ascii=False))
        return 0
    if cmd == 'benchmark':
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        from .ledger import ExperimentLedger
        if args.run:
            import asyncio
            from .engine import EmiliaRenderEngine
            engine = EmiliaRenderEngine(config=config)
            asyncio.run(engine.render_quality_pipeline(seed=args.seed, port=config.quality_port, profile=args.profile, strict_seed_lock=True))
        ledger = ExperimentLedger(config=config)
        ledger.ingest_existing_artifacts()
        stats = ledger.get_summary_stats()
        recent = ledger.list_recent_runs(limit=args.limit)
        rows = [
            [
                str(r.get('seed') or '-'),
                str(r.get('mode') or '-'),
                f"{(r.get('native_seconds') or 0.0):.1f}s",
                f"{(r.get('finish_seconds') or 0.0):.1f}s",
                f"{(r.get('total_seconds') or 0.0):.1f}s",
                str(r.get('quality_action') or 'accept'),
            ]
            for r in recent
        ]
        print(render_table(['Seed', 'Mode', 'Native', 'Finish', 'Total', 'Gate'], rows, title=f"Benchmark Summary (Total Runs: {stats.get('total_runs', 0)})"))
        return 0
    if cmd == 'clean':
        if args.older_than_hours < 0:
            raise ValueError('--older-than-hours must be non-negative')
        temp_root = (config.temp_dir or (config.root_dir / 'temp')).resolve()
        if not temp_root.is_dir():
            print(json.dumps({'temp_dir': str(temp_root), 'removed_count': 0, 'freed_bytes': 0}, indent=2))
            return 0
        from .engine import run_is_active

        cutoff = time.time() - args.older_than_hours * 3600.0
        prefixes = ('quality_', 'ultra_')
        candidates: list[tuple[Path, int]] = []
        active: list[str] = []
        for entry in sorted(temp_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith(prefixes):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime > cutoff:
                continue
            # A directory's mtime does not move while its render runs; the run lock decides.
            if run_is_active(entry):
                active.append(str(entry))
                continue
            size = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
            candidates.append((entry, size))
        freed = sum(sz for _, sz in candidates)
        removed = []
        for folder, _ in candidates:
            removed.append(str(folder))
            if not args.dry_run:
                shutil.rmtree(folder, ignore_errors=True)
        print(json.dumps({
            'temp_dir': str(temp_root),
            'dry_run': bool(args.dry_run),
            'removed_count': len(removed),
            'freed_bytes': freed,
            'freed_human': format_bytes(freed),
            'folders': removed,
            'skipped_active': active,
        }, indent=2))
        return 0
    if cmd in ('doctor', 'pull'):
        manifest = json.loads(config.manifest_path.read_text(encoding='utf-8'))
        models = manifest['models']
        experimental = manifest.get('experimental_models', [])
        all_pullable = models + experimental
        if cmd == 'pull':
            if args.list or not args.models:
                rows_list = [
                    [m['key'], 'required', 'available' if m.get('download_url') else 'manual']
                    for m in models
                ] + [
                    [m['key'], 'experimental', 'available' if m.get('download_url') else 'manual']
                    for m in experimental
                ]
                print(render_table(['Key', 'Tier', 'Download'], rows_list))
                return 0
            keys = {m['key'] for m in all_pullable}
            if set(args.models) - keys - {'all'}:
                raise ValueError('Unknown model key; see pull --list')
            failed = False
            target_models = models if 'all' in args.models else [m for m in all_pullable if m['key'] in args.models]
            if 'all' in args.models:
                extra_explicit = [m for m in experimental if m['key'] in args.models]
                target_models = models + extra_explicit
            for item in target_models:
                dest = config.app_dir / item['relative_path']
                if dest.is_file() and dest.stat().st_size == item.get('size_bytes'):
                    if item.get('sha256'):
                        import hashlib
                        with dest.open('rb') as f:
                            valid = hashlib.file_digest(f, 'sha256').hexdigest() == item['sha256']
                        if valid:
                            print(f"{item['key']}: verified, skipped")
                            continue
                    else:
                        print(f"{item['key']}: size matches, skipped (no checksum available)")
                        continue
                if not item.get('download_url'):
                    print(f"{item['key']}: {item.get('manual_instructions', 'Install manually from the model author.')}", file=sys.stderr)
                    failed = True
                    continue
                download_file_with_progress(item['download_url'], dest, expected_bytes=item.get('size_bytes'), expected_sha256=item.get('sha256'), label=item['key'])
            return 2 if failed else 0
        rows = []
        ok = True
        comfy_main = (config.app_dir / 'main.py') if config.app_dir is not None else None
        for name, path in [('ComfyUI', comfy_main), ('Inference Python', config.conda_python)]:
            valid = path is not None and path.is_file()
            rows.append([name, 'OK' if valid else 'MISSING', str(path)])
            ok &= valid
        if config.conda_python and config.conda_python.is_file():
            try:
                probe = subprocess.run([str(config.conda_python), '-c', 'import torch, ultralytics, aiohttp; print("CUDA=" + str(torch.cuda.is_available()))'], capture_output=True, text=True, timeout=60)
                rows.append(['Runtime dependencies', 'OK' if probe.returncode == 0 else 'MISSING', (probe.stdout or probe.stderr).strip()[-300:]])
                ok &= probe.returncode == 0
            except (OSError, subprocess.SubprocessError) as exc:
                rows.append(['Runtime dependencies', 'MISSING', str(exc)[:300]])
                ok = False
        for item in models + manifest.get('reference_assets', []):
            path = config.app_dir / item['relative_path']
            valid = path.is_file() and (not item.get('size_bytes') or path.stat().st_size == item['size_bytes'])
            rows.append([item['name'], 'OK' if valid else 'MISSING', str(path)])
            ok &= valid
        for item in experimental:
            path = config.app_dir / item['relative_path']
            valid = path.is_file() and (not item.get('size_bytes') or path.stat().st_size == item['size_bytes'])
            rows.append([item['name'], 'OK' if valid else 'OPTIONAL', str(path)])
        for item in manifest.get('custom_nodes', []):
            valid = (config.app_dir / item['directory']).is_dir()
            rows.append([item['name'], 'OK' if valid else 'MISSING', item['directory']])
            ok &= valid
        for check, status, details, passed in checkout_checks(config, manifest):
            rows.append([check, status, details])
            ok &= passed
        print(render_table(['Check', 'Status', 'Details'], rows, title='Emilia doctor'))
        return 0 if ok else 2
    return None
