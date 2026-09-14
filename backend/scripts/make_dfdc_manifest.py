"""Create deterministic identity-disjoint manifests from the DFDC / DFD dataset.

Scans DFD_original sequences (real) and DFD_manipulated_sequences (fake),
extracts actor IDs from filenames, and enforces strict identity-disjoint partitioning:
  - Train actors: 01 to 18 (18 actors)
  - Val actors:   19 to 23 (5 actors)
  - Test actors:  24 to 28 (5 actors)

Zero actor overlap between train, val, and test splits ensures zero facial memorization
and realistic out-of-distribution evaluation.
"""

import argparse
import json
import re
from pathlib import Path


TRAIN_ACTORS = {f'{i:02d}' for i in range(1, 19)}
VAL_ACTORS = {f'{i:02d}' for i in range(19, 24)}
TEST_ACTORS = {f'{i:02d}' for i in range(24, 29)}


def parse_video_info(path: Path):
    name = path.name
    # Fake pattern: XX_YY__*.mp4
    m_fake = re.match(r'^(\d+)_(\d+)__', name)
    if m_fake:
        actor_a, actor_b = m_fake.group(1), m_fake.group(2)
        return {
            'label': 1,
            'class': 'FAKE',
            'actors': [actor_a, actor_b],
            'original_actor': actor_a,
            'target_actor': actor_b,
        }
    # Real pattern: XX__*.mp4
    m_real = re.match(r'^(\d+)__', name)
    if m_real:
        actor = m_real.group(1)
        return {
            'label': 0,
            'class': 'REAL',
            'actors': [actor],
            'original_actor': actor,
            'target_actor': None,
        }
    return None


def assign_split(actors):
    actor_set = set(actors)
    if actor_set.issubset(TRAIN_ACTORS):
        return 'train'
    if actor_set.issubset(VAL_ACTORS):
        return 'val'
    if actor_set.issubset(TEST_ACTORS):
        return 'test'
    # Mixed cross-boundary pair
    return 'cross'


def main():
    parser = argparse.ArgumentParser(description='Create DFDC / DFD identity-disjoint manifests')
    parser.add_argument('root', type=Path, help='Root directory containing DFDC dataset')
    parser.add_argument('--output', type=Path, default=Path('data/dfdc'), help='Output manifest directory')
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f'DFDC directory not found at: {root}')

    print(f'Scanning DFDC dataset at: {root}')
    records = []
    skipped = []

    all_videos = sorted(root.rglob('*.mp4'))
    print(f'Found {len(all_videos)} total MP4 files.')

    for video_path in all_videos:
        info = parse_video_info(video_path)
        if not info:
            skipped.append(str(video_path))
            continue

        split = assign_split(info['actors'])
        rel_path = video_path.relative_to(root).as_posix()

        records.append({
            'video': str(video_path.resolve()),
            'relative_video': rel_path,
            'filename': video_path.name,
            'label': info['label'],
            'class': info['class'],
            'actors': info['actors'],
            'split': split,
            'dataset': 'DFDC',
        })

    if skipped:
        print(f'Warning: skipped {len(skipped)} files with unparsed naming:')
        for s in skipped[:5]:
            print(f'  {s}')

    args.output.mkdir(parents=True, exist_ok=True)

    summary = {}
    splits_to_write = ('train', 'val', 'test', 'cross')
    for split in splits_to_write:
        out_file = args.output / f'{split}.jsonl'
        split_records = [r for r in records if r['split'] == split]
        reals = [r for r in split_records if r['label'] == 0]
        fakes = [r for r in split_records if r['label'] == 1]
        actors = set()
        for r in split_records:
            actors.update(r['actors'])

        summary[split] = {
            'total': len(split_records),
            'real': len(reals),
            'fake': len(fakes),
            'actors': sorted(list(actors)),
        }

        with open(out_file, 'w', encoding='utf-8') as f:
            for r in split_records:
                f.write(json.dumps(r) + '\n')

        print(f'[{split.upper()}] total={len(split_records)} (REAL={len(reals)}, FAKE={len(fakes)}) | Actors={len(actors)} -> {out_file}')

    # Identity leakage verification across train, val, and test
    train_a = set(summary['train']['actors'])
    val_a = set(summary['val']['actors'])
    test_a = set(summary['test']['actors'])

    leak_train_val = train_a & val_a
    leak_train_test = train_a & test_a
    leak_val_test = val_a & test_a

    print('\n=== IDENTITY LEAKAGE AUDIT ===')
    print(f'Train vs Val actor overlap : {leak_train_val} (Leakage: {len(leak_train_val)})')
    print(f'Train vs Test actor overlap: {leak_train_test} (Leakage: {len(leak_train_test)})')
    print(f'Val vs Test actor overlap  : {leak_val_test} (Leakage: {len(leak_val_test)})')

    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError('IDENTITY LEAKAGE DETECTED across dataset splits!')

    print('[OK] ZERO IDENTITY LEAKAGE CONFIRMED across all evaluation splits!')

    summary_path = args.output / 'manifest_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
