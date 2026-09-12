"""Create deterministic raw-video manifests from an extracted Celeb-DF-v2 tree."""

import argparse
import hashlib
import json
from pathlib import Path


def load_test_paths(root: Path):
    test_file = root / 'List_of_testing_videos.txt'
    if not test_file.exists():
        raise FileNotFoundError(f'Official test list not found: {test_file}')
    return {line.split(maxsplit=1)[1].replace('\\', '/') for line in test_file.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description='Create Celeb-DF-v2 raw manifests')
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/celebdf'))
    args = parser.parse_args()
    root = args.root.resolve()
    test_paths = load_test_paths(root)
    records = []
    for category, label in (('Celeb-real', 0), ('YouTube-real', 0), ('Celeb-synthesis', 1)):
        for video in sorted((root / category).glob('*.mp4')):
            relative = video.relative_to(root).as_posix()
            split = 'test' if relative in test_paths else None
            if split is None:
                # Stable 10% validation split; keep the official test set untouched.
                bucket = int(hashlib.sha256(relative.encode()).hexdigest()[:8], 16) % 10
                split = 'val' if bucket == 0 else 'train'
            records.append({'video': str(video), 'relative_video': relative, 'label': label, 'split': split})
    if len(records) != 6529 or sum(record['split'] == 'test' for record in records) != 518:
        raise ValueError(f'Unexpected dataset inventory: {len(records)} videos, test={sum(record["split"] == "test" for record in records)}')
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        output = args.output / f'{split}.jsonl'
        selected = [record for record in records if record['split'] == split]
        output.write_text('\n'.join(json.dumps(record) for record in selected) + '\n')
        real = sum(record['label'] == 0 for record in selected)
        fake = sum(record['label'] == 1 for record in selected)
        print(f'{split}: total={len(selected)} real={real} fake={fake} -> {output}')


if __name__ == '__main__':
    main()