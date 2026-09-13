"""Audit BioVision manifests for split leakage and class balance."""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_manifest(path: Path) -> List[dict]:
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError(f'Manifest is empty: {path}')
    return records


def source_key(record: dict) -> str:
    value = record.get('relative_video') or record.get('source_video') or record.get('video')
    if value is None:
        raise ValueError(f'Manifest record has no source video identifier: {record}')
    return str(value).replace('\\', '/').lower()


def feature_keys(record: dict) -> List[str]:
    return [str(record[name]).replace('\\', '/').lower() for name in ('visual', 'rppg', 'mfcc', 'mouth') if name in record]


def summarize(records: List[dict]) -> Dict[str, object]:
    labels = [int(record['label']) for record in records]
    return {
        'samples': len(records),
        'real': labels.count(0),
        'fake': labels.count(1),
        'unique_source_videos': len({source_key(record) for record in records}),
        'duplicate_source_videos': len(records) - len({source_key(record) for record in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit train/validation/test manifest separation')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    manifests = {name: load_manifest(path) for name, path in (('train', args.train), ('val', args.val), ('test', args.test))}
    sources = {name: {source_key(record) for record in records} for name, records in manifests.items()}
    features = {name: set(key for record in records for key in feature_keys(record)) for name, records in manifests.items()}
    source_overlaps = {}
    feature_overlaps = {}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        source_overlaps[f'{left}_vs_{right}'] = sorted(sources[left] & sources[right])
        feature_overlaps[f'{left}_vs_{right}'] = sorted(features[left] & features[right])

    report = {
        'manifests': {name: str(path.resolve()) for name, path in (('train', args.train), ('val', args.val), ('test', args.test))},
        'splits': {name: summarize(records) for name, records in manifests.items()},
        'source_overlaps': source_overlaps,
        'feature_overlaps': feature_overlaps,
        'video_disjoint': not any(source_overlaps.values()),
        'feature_disjoint': not any(feature_overlaps.values()),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    if not report['video_disjoint'] or not report['feature_disjoint']:
        raise SystemExit('Manifest audit failed: overlapping source videos or feature paths detected')


if __name__ == '__main__':
    main()
