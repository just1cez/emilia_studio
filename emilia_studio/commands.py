"""Interactive and maintenance commands for the installed CLI."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .config import BUILTIN_SCENE_PRESETS, get_user_config_path
from .tui import download_file_with_progress, print_banner, render_table

PATH_KEYS = ('root_dir', 'app_dir', 'output_dir', 'temp_dir', 'conda_python', 'ollama_exe')
VALUE_KEYS = ('comfy_host', 'comfy_port', 'quality_port', 'ollama_base', 'ollama_model')


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
    p = sub.add_parser('gallery', help='List generated PNGs, optionally open one')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--open', type=int, metavar='INDEX')
    sub.add_parser('shell', help='Interactive command prompt (exit to leave)')
    sub.add_parser('banner', help='Display the EMILIA ASCII banner')


def handle(args, config, main, parser):
    cmd = args.command
    if cmd in (None, 'shell', 'banner'):
        print_banner()
        if cmd == 'banner':
            return 0
        print('Commands: config, doctor, pull, presets, render, idea, inspect, gallery, ledger, setup')
        print('Example: render --profile ultra --preset rose-garden\nUse help or COMMAND --help. Type exit to leave.')
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
    if cmd in ('doctor', 'pull'):
        manifest = json.loads(config.manifest_path.read_text(encoding='utf-8'))
        models = manifest['models']
        if cmd == 'pull':
            if args.list or not args.models:
                print(render_table(['Key', 'Download'], [[m['key'], 'available' if m.get('download_url') else 'manual'] for m in models]))
                return 0
            keys = {m['key'] for m in models}
            if set(args.models) - keys - {'all'}:
                raise ValueError('Unknown model key; see pull --list')
            failed = False
            for item in models:
                if 'all' not in args.models and item['key'] not in args.models:
                    continue
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
        for name, path in [('ComfyUI', config.app_dir / 'main.py'), ('Inference Python', config.conda_python)]:
            valid = path is not None and path.is_file()
            rows.append([name, 'OK' if valid else 'MISSING', str(path)])
            ok &= valid
        if config.conda_python and config.conda_python.is_file():
            probe = subprocess.run([str(config.conda_python), '-c', 'import torch, ultralytics, aiohttp; print("CUDA=" + str(torch.cuda.is_available()))'], capture_output=True, text=True, timeout=60)
            rows.append(['Runtime dependencies', 'OK' if probe.returncode == 0 else 'MISSING', (probe.stdout or probe.stderr).strip()[-300:]])
            ok &= probe.returncode == 0
        for item in models + manifest.get('reference_assets', []):
            path = config.app_dir / item['relative_path']
            valid = path.is_file() and (not item.get('size_bytes') or path.stat().st_size == item['size_bytes'])
            rows.append([item['name'], 'OK' if valid else 'MISSING', str(path)])
            ok &= valid
        for item in manifest.get('custom_nodes', []):
            valid = (config.app_dir / item['directory']).is_dir()
            rows.append([item['name'], 'OK' if valid else 'MISSING', item['directory']])
            ok &= valid
        print(render_table(['Check', 'Status', 'Details'], rows, title='Emilia doctor'))
        return 0 if ok else 2
    return None
