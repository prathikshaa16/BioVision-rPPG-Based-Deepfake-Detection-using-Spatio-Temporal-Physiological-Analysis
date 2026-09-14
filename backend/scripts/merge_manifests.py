"""Merge Celeb-DF v2 and DFDC manifests for multi-dataset training and evaluation."""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description='Merge Celeb-DF v2 and DFDC manifests')
    parser.add_argument('--celebdf', type=Path, default=Path('data/celebdf'), help='Celeb-DF manifest directory')
    parser.add_argument('--dfdc', type=Path, default=Path('data/dfdc'), help='DFDC manifest directory')
    parser.add_argument('--output', type=Path, default=Path('data/multidataset'), help='Output multi-dataset directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split in ('train', 'val', 'test'):
        c_records = load_jsonl(args.celebdf / f'{split}.jsonl')
        d_records = load_jsonl(args.dfdc / f'{split}.jsonl')

        combined = c_records + d_records
        out_path = args.output / f'{split}.jsonl'

        with open(out_path, 'w', encoding='utf-8') as f:
            for r in combined:
                f.write(json.dumps(r) + '\n')

        reals = sum(r['label'] == 0 for r in combined)
        fakes = sum(r['label'] == 1 for r in combined)

        summary[split] = {
            'total': len(combined),
            'celebdf_count': len(c_records),
            'dfdc_count': len(d_records),
            'real': reals,
            'fake': fakes,
        }

        print(f'[{split.upper()}] total={len(combined)} (Celeb-DF={len(c_records)}, DFDC={len(d_records)}) | REAL={reals}, FAKE={fakes} -> {out_path}')

    summary_path = args.output / 'multidataset_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'[OK] Multi-dataset manifests created at: {args.output}')


if __name__ == '__main__':
    main()
