import json
from pathlib import Path
import os
from backend.scripts.train_multimodal import FeatureManifest, collate, compute_pos_weight
from backend.app.multimodal_model import build_biovision_visual_rppg_model
from torch.utils.data import DataLoader
import torch
from torch import nn

root = Path(r'C:\BioVision')
feature_root = root / 'features' / 'celebdf'
train_manifest = feature_root / 'train_valid_subset.jsonl'
val_manifest = feature_root / 'val_valid_subset.jsonl'

if not train_manifest.exists() or not val_manifest.exists():
    p = root / 'manifests' / 'celebdf' / 'train.jsonl'
    rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    usable = []
    for rec in rows:
        rel = rec.get('relative_video') or rec.get('video')
        if not rel:
            continue
        stem = Path(str(rel).replace('\\', '/')).with_suffix('').as_posix().replace('/', '_')
        v = feature_root / f'{stem}_visual.npy'
        r = feature_root / f'{stem}_rppg.npy'
        if v.exists() and r.exists():
            usable.append(rec)
    train_rows = usable[:1000]
    val_rows = usable[1000:1200]
    for name, subset in [('train_valid_subset.jsonl', train_rows), ('val_valid_subset.jsonl', val_rows)]:
        out = feature_root / name
        with out.open('w', encoding='utf-8') as f:
            for rec in subset:
                f.write(json.dumps(rec) + '\n')

train = FeatureManifest(str(train_manifest), include_audio_lip=False)
val = FeatureManifest(str(val_manifest), include_audio_lip=False)
print('train_len', len(train), 'val_len', len(val), 'pos_weight', compute_pos_weight(str(train_manifest), False))
train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=lambda b: collate(b, include_audio_lip=False))
val_loader = DataLoader(val, batch_size=2, shuffle=False, collate_fn=lambda b: collate(b, include_audio_lip=False))
model = build_biovision_visual_rppg_model()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([compute_pos_weight(str(train_manifest), False)]))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for e in range(1):
    model.train(); total = 0.0; num = 0
    for batch in train_loader:
        logits = model(batch['visual'], batch['rppg'], batch['lengths'])
        loss = criterion(logits, batch['labels'])
        opt.zero_grad(); loss.backward(); opt.step();
        total += loss.item() * batch['labels'].numel(); num += batch['labels'].numel();
    print('epoch', e, 'loss', total/num)
print('forward_ok', model(next(iter(val_loader))['visual'], next(iter(val_loader))['rppg'], next(iter(val_loader))['lengths']).shape)
