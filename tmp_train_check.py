import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.app.multimodal_model import build_biovision_visual_rppg_model
from backend.scripts.train_multimodal import FeatureManifest, collate, compute_pos_weight

root = Path(r'C:\BioVision')
feature_root = root / 'features' / 'celebdf'
train_manifest = feature_root / 'train_valid_subset.jsonl'
val_manifest = feature_root / 'val_valid_subset.jsonl'

if not train_manifest.exists() or not val_manifest.exists():
    manifest_path = root / 'manifests' / 'celebdf' / 'train.jsonl'
    rows = [json.loads(line) for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    usable = []
    for rec in rows:
        rel = rec.get('relative_video') or rec.get('video')
        if not rel:
            continue
        stem = Path(str(rel).replace('\\', '/')).with_suffix('').as_posix().replace('/', '_')
        visual_path = feature_root / f'{stem}_visual.npy'
        rppg_path = feature_root / f'{stem}_rppg.npy'
        if visual_path.exists() and rppg_path.exists():
            usable.append(rec)
    train_rows = usable[:1000]
    val_rows = usable[1000:1200]
    for name, subset in [('train_valid_subset.jsonl', train_rows), ('val_valid_subset.jsonl', val_rows)]:
        out_path = feature_root / name
        with out_path.open('w', encoding='utf-8') as f:
            for rec in subset:
                f.write(json.dumps(rec) + '\n')

train_ds = FeatureManifest(str(train_manifest), include_audio_lip=False)
val_ds = FeatureManifest(str(val_manifest), include_audio_lip=False)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate(b, include_audio_lip=False))
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=lambda b: collate(b, include_audio_lip=False))
model = build_biovision_visual_rppg_model()
pos_weight = torch.tensor([compute_pos_weight(str(train_manifest), False)], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for batch_idx, batch in enumerate(train_loader):
    logits = model(batch['visual'], batch['rppg'], batch['lengths'])
    loss = criterion(logits, batch['labels'])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if batch_idx == 0:
        print(f'BATCH0_loss={loss.item():.6f} labels={batch["labels"].tolist()} logits_shape={tuple(logits.shape)}')
        break

model.eval()
with torch.no_grad():
    first_batch = next(iter(val_loader))
    logits = model(first_batch['visual'], first_batch['rppg'], first_batch['lengths'])
    print(f'VAL_FORWARD_shape={tuple(logits.shape)}')
    print(f'VAL_LABELS={first_batch["labels"].tolist()}')
print(f'TRAIN_LEN={len(train_ds)} VAL_LEN={len(val_ds)}')
