"""Train the BioVision cached visual+rPPG fusion head with balanced sampling.

This experiment intentionally keeps the verified 320-feature architecture and
changes only the training sampler/loss protocol. The official test set is not
loaded by this script.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from backend.app.multimodal_model import build_cached_biovision_visual_rppg_model


class CachedFusionDataset(Dataset):
    def __init__(self, path: Path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.records = payload['records'] if isinstance(payload, dict) else payload
        if not self.records:
            raise ValueError(f'No records found in {path}')
        self.labels = np.asarray([int(record['label']) for record in self.records], dtype=np.int64)
        if set(self.labels.tolist()) != {0, 1}:
            raise ValueError(f'{path} must contain both REAL and FAKE labels')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        visual = torch.as_tensor(record['visual_features'], dtype=torch.float32)
        rppg = torch.as_tensor(record['rppg'], dtype=torch.float32).flatten()
        if tuple(visual.shape) != (32, 1792):
            raise ValueError(f'Unexpected visual shape at record {index}: {tuple(visual.shape)}')
        if not 1 <= rppg.numel() <= 240:
            raise ValueError(f'Unexpected rPPG length at record {index}: {rppg.numel()}')
        padded = torch.zeros(240, dtype=torch.float32)
        padded[:rppg.numel()] = rppg
        return visual, padded, torch.tensor(float(record['label']), dtype=torch.float32)


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    probabilities: List[float] = []
    labels: List[float] = []
    for visual, rppg, target in loader:
        visual, rppg, target = visual.to(device), rppg.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            logits = model(visual, rppg)
            loss = criterion(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * target.numel()
        probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(target.detach().cpu().numpy().tolist())
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    return total_loss / len(labels_array), labels_array, probabilities_array


def main():
    parser = argparse.ArgumentParser(description='Balanced BioVision cached fusion experiment')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = CachedFusionDataset(args.train)
    val_dataset = CachedFusionDataset(args.val)
    class_counts = np.bincount(train_dataset.labels, minlength=2)
    sample_weights = np.asarray([1.0 / class_counts[label] for label in train_dataset.labels], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_cached_biovision_visual_rppg_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history_path = args.history or args.output.with_suffix('.history.json')
    history = {
        'experiment': 'balanced_cached_fusion',
        'train_manifest': str(args.train.resolve()),
        'val_manifest': str(args.val.resolve()),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_real': int(class_counts[0]),
        'train_fake': int(class_counts[1]),
        'val_real': int((val_dataset.labels == 0).sum()),
        'val_fake': int((val_dataset.labels == 1).sum()),
        'epochs': [],
    }
    best_auc = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_labels, train_probabilities = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_labels, val_probabilities = run_epoch(model, val_loader, device)
        train_metrics = metrics(train_labels, train_probabilities)
        val_metrics = metrics(val_labels, val_probabilities)
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        history['epochs'].append(record)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_auc={train_metrics["auc"]} val_auc={val_metrics["auc"]} val_f1={val_metrics["f1"]:.4f}')
        if val_metrics['auc'] is not None and val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_auc,
                'experiment': history['experiment'],
            }, args.output)
        scheduler.step()
    history['epochs_completed'] = len(history['epochs'])
    history['training_seconds'] = round(time.time() - start_time, 3)
    history['best_val_auc'] = best_auc
    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(json.dumps({'checkpoint': str(args.output.resolve()), 'history': str(history_path.resolve()), 'best_val_auc': best_auc}, indent=2))


if __name__ == '__main__':
    main()