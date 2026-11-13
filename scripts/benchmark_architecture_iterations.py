"""Run one fixed 10-seed quality benchmark round through `emilia render`.

Records wall time, sampled RSS, minimum available RAM and decoded pixel hashes per seed
under ``<app_dir>/temp/architecture_iterations/round_NN/``; the ledger ingests these
records. Run with the inference Python (needs ``psutil``):

    python scripts/benchmark_architecture_iterations.py --round 6
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emilia_studio.config import load_config  # noqa: E402

SEEDS = range(20261027, 20261037)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    directory = load_config().app_dir / 'temp' / 'architecture_iterations' / f'round_{args.round:02}'
    directory.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        record_path = directory / f'{seed}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get('success') and Path(record['receipt']['final']).exists():
                print(f'SKIP verified seed {seed}', flush=True)
                continue
        started = time.monotonic()
        log_path = directory / f'{seed}.log'
        samples = []
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.Popen(
                [sys.executable, '-u', '-m', 'emilia_studio.cli', 'render', '--seed', str(seed), '--port', '8199'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            process = psutil.Process(child.pid)
            while child.poll() is None:
                rss = 0
                cpu = 0.0
                try:
                    for p in [process] + process.children(recursive=True):
                        try:
                            rss += p.memory_info().rss
                            times = p.cpu_times()
                            cpu += times.user + times.system
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                samples.append({'elapsed': time.monotonic() - started, 'rss': rss, 'cpu_seconds_live_processes': cpu, 'available_ram': psutil.virtual_memory().available})
                try:
                    child.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        elapsed = time.monotonic() - started
        text = log_path.read_text(encoding='utf-8')
        if child.returncode != 0:
            raise RuntimeError(f'Generation failed: {log_path}')
        receipt_line = next(line for line in text.splitlines() if line.startswith('RECEIPT: '))
        receipt_path = Path(receipt_line.removeprefix('RECEIPT: '))
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        hashes = {}
        qi = receipt.get('quality_inspection') or {}
        for stage, size in [('native', (1152, 1728)), ('final', (2304, 3456))]:
            stage_qi = qi.get(stage) or {}
            if stage_qi.get('pixel_sha256') and (stage_qi.get('width'), stage_qi.get('height')) == size:
                hashes[stage] = stage_qi['pixel_sha256']
            else:
                with Image.open(receipt[stage]) as im:
                    im.load()
                    if im.size != size:
                        raise RuntimeError(f'Unexpected {stage} dimensions: {im.size}')
                    hashes[stage] = hashlib.sha256(im.tobytes()).hexdigest()
        peak_rss = max((s['rss'] for s in samples), default=0)
        min_ram = min((s['available_ram'] for s in samples), default=psutil.virtual_memory().available)
        record = {'round': args.round, 'seed': seed, 'success': True, 'wall_seconds': elapsed, 'receipt_path': str(receipt_path), 'receipt': receipt, 'pixel_hashes': hashes, 'peak_sampled_rss': peak_rss, 'min_available_ram': min_ram, 'resource_samples': samples}
        record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f'ROUND {args.round} SEED {seed} COMPLETE {elapsed:.2f}s {receipt["final"]}', flush=True)
    print(f'ROUND {args.round} COMPLETE: {len(SEEDS)} complete final images', flush=True)


if __name__ == '__main__':
    main()
