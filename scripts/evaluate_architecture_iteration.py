"""Summarize one benchmark round against round 1: timings, memory, pixel parity, contact sheet.

    python scripts/evaluate_architecture_iteration.py --round 6
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emilia_studio.config import load_config  # noqa: E402

SEEDS = range(20261027, 20261037)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    rounds = load_config().app_dir / 'temp' / 'architecture_iterations'
    folder = rounds / f'round_{args.round:02}'
    records = [json.loads((folder / f'{seed}.json').read_text(encoding='utf-8')) for seed in SEEDS]
    if not all(r.get('success') for r in records):
        raise RuntimeError('Not all 10 benchmark seeds succeeded')
    if len({r['receipt']['final'] for r in records}) != 10:
        raise RuntimeError('Expected 10 distinct final images')
    contact = Image.new('RGB', (1200, 800), 'white')
    draw = ImageDraw.Draw(contact)
    comparisons = []
    for i, r in enumerate(records):
        with Image.open(r['receipt']['final']) as image:
            if image.size != (2304, 3456):
                raise RuntimeError(f"Unexpected final image dimensions: {image.size}")
            image.thumbnail((240, 360))
            x, y = (i % 5) * 240, (i // 5) * 400
            contact.paste(image, (x, y + 25))
            draw.text((x + 5, y + 5), str(r['seed']), fill='black')
        baseline_path = rounds / 'round_01' / f'{r["seed"]}.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        comparisons.append({'seed': r['seed'], 'native_exact': r['pixel_hashes']['native'] == baseline['pixel_hashes']['native'], 'final_exact': r['pixel_hashes']['final'] == baseline['pixel_hashes']['final']})
    contact.save(folder / 'contact_sheet.jpg', quality=95)
    summary = {'round': args.round, 'complete_final_images': len(records), 'median_wall_seconds': statistics.median(r['wall_seconds'] for r in records), 'total_wall_seconds': sum(r['wall_seconds'] for r in records), 'median_native_seconds': statistics.median(r['receipt']['generation']['seconds'] for r in records), 'median_finish_seconds': statistics.median(r['receipt']['finishing']['seconds'] for r in records), 'max_sampled_rss_gib': max(r['peak_sampled_rss'] for r in records) / 1024**3, 'min_available_ram_gib': min(r['min_available_ram'] for r in records) / 1024**3, 'pixel_comparisons': comparisons, 'contact_sheet': str(folder / 'contact_sheet.jpg')}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
