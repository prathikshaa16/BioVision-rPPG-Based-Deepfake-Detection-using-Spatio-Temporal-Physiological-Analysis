import json, random
from pathlib import Path
root = Path(r'C:\BioVision')
manifest = root / 'manifests' / 'celebdf' / 'train.jsonl'
feature_root = root / 'features' / 'celebdf'
rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
usable = []
for rec in rows:
    rel = rec.get('relative_video') or rec.get('video')
    if not rel:
        continue
    stem = Path(str(rel).replace('\\', '/')).with_suffix('').as_posix().replace('/', '_')
    visual = feature_root / f'{stem}_visual.npy'
    rppg = feature_root / f'{stem}_rppg.npy'
    if visual.exists() and rppg.exists():
        usable.append(rec)
random.Random(42).shuffle(usable)
by_label = {0: [], 1: []}
for rec in usable:
    label = int(float(rec.get('label', rec.get('y', 0))))
    by_label[label].append(rec)
train, val = [], []
for label, items in by_label.items():
    n = len(items)
    test_n = max(1, round(0.2 * n))
    val.extend(items[:test_n])
    train.extend(items[test_n:])
random.Random(42).shuffle(train)
random.Random(42).shuffle(val)
for path, subset in [(feature_root / 'train_valid_strat.jsonl', train), (feature_root / 'val_valid_strat.jsonl', val)]:
    with path.open('w', encoding='utf-8') as f:
        for rec in subset:
            f.write(json.dumps(rec) + '\n')
print('usable_total=', len(usable))
print('train_len=', len(train), 'val_len=', len(val))
print('train_counts=', {k: sum(int(float(r.get('label', r.get('y', 0)))) == k for r in train) for k in [0,1]})
print('val_counts=', {k: sum(int(float(r.get('label', r.get('y', 0)))) == k for r in val) for k in [0,1]})
