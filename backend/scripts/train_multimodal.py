"""Train the paper-specified BioVision multimodal classifier.

The manifest is JSONL with one record per video:
{"visual": "...npy", "rppg": "...npy", "mfcc": "...npy",
 "mouth": "...npy", "label": 0}

Arrays are visual [T,3,224,224], rPPG [N], MFCC [T,40], and mouth [T,40].
The script intentionally requires precomputed, validated modalities so a
missing audio or landmark stream cannot be silently converted into zeros.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.app.multimodal_model import build_biovision_multimodal_model, build_biovision_visual_rppg_model


class FeatureManifest(Dataset):
    def __init__(self, manifest: str, include_audio_lip: bool = True):
        self.include_audio_lip = include_audio_lip
        self.path = Path(manifest).resolve()
        self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f'Manifest is empty: {self.path}')
        required = {'visual', 'rppg', 'label'} | ({'mfcc', 'mouth'} if include_audio_lip else set())
        for index, record in enumerate(self.records):
            missing = required - record.keys()
            if missing:
                raise ValueError(f'Record {index} missing fields: {sorted(missing)}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        arrays = {}
        names = ('visual', 'rppg', 'mfcc', 'mouth') if self.include_audio_lip else ('visual', 'rppg')
        for name in names:
            path = (self.path.parent / record[name]).resolve()
            if not path.exists():
                raise FileNotFoundError(f'{name} features missing for record {index}: {path}')
            arrays[name] = torch.from_numpy(np.load(path)).float()
        steps = arrays['visual'].shape[0]
        if self.include_audio_lip and (arrays['mfcc'].shape[0] != steps or arrays['mouth'].shape[0] != steps):
            raise ValueError(f'Record {index} has inconsistent temporal lengths')
        if steps == 0:
            raise ValueError(f'Record {index} has no visual observations')
        arrays['length'] = torch.tensor(steps, dtype=torch.long)
        arrays['label'] = torch.tensor(float(record['label']), dtype=torch.float32)
        return arrays


def collate(batch: List[Dict[str, torch.Tensor]], include_audio_lip: bool = True) -> Dict[str, torch.Tensor]:
    max_steps = max(item['length'].item() for item in batch)
    visual = torch.zeros(len(batch), max_steps, 3, 224, 224)
    mfcc = torch.zeros(len(batch), max_steps, 40) if include_audio_lip else None
    mouth = torch.zeros(len(batch), max_steps, 40) if include_audio_lip else None
    rppg = torch.stack([item['rppg'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    for row, item in enumerate(batch):
        steps = item['length'].item()
        visual[row, :steps] = item['visual']
        if include_audio_lip:
            mfcc[row, :steps] = item['mfcc']
            mouth[row, :steps] = item['mouth']
    result = {'visual': visual, 'rppg': rppg, 'lengths': lengths, 'labels': labels}
    if include_audio_lip:
        result.update({'mfcc': mfcc, 'mouth': mouth})
    return result


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        with torch.set_grad_enabled(training):
            logits = model(**inputs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * labels.numel()
        correct += ((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item()
        total += labels.numel()
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train BioVision multimodal fusion model')
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--val-manifest', required=True)
    parser.add_argument('--output', default='backend/models/biovision_multimodal.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=('full', 'visual-rppg'), default='full')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    include_audio_lip = args.mode == 'full'
    train_loader = DataLoader(FeatureManifest(args.train_manifest, include_audio_lip), batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, include_audio_lip))
    val_loader = DataLoader(FeatureManifest(args.val_manifest, include_audio_lip), batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate(batch, include_audio_lip))
    model = (build_biovision_multimodal_model() if include_audio_lip else build_biovision_visual_rppg_model()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float('inf')
    patience = 5
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, None, device)
        print(f'epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, output)
        else:
            stale += 1
            if stale >= patience:
                print('early stopping')
                break


if __name__ == '__main__':
    main()