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
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.app.multimodal_model import build_biovision_multimodal_model, build_biovision_visual_rppg_model


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _normalize_cached_record(record: Dict[str, Any], include_audio_lip: bool = True) -> Dict[str, torch.Tensor]:
    if not isinstance(record, dict):
        raise TypeError(f'Expected a record dictionary, got {type(record).__name__}')
    visual = next((record.get(key) for key in ('visual', 'visual_features', 'frames', 'frame_sequence') if key in record and record.get(key) is not None), None)
    rppg = next((record.get(key) for key in ('rppg', 'rppg_signal', 'pulse', 'signal') if key in record and record.get(key) is not None), None)
    if visual is None or rppg is None:
        raise ValueError(f'Record missing a visual/rPPG payload: keys={sorted(record.keys())[:10]}')
    normalized = {
        'visual': torch.as_tensor(_to_numpy(visual), dtype=torch.float32),
        'rppg': torch.as_tensor(_to_numpy(rppg), dtype=torch.float32),
    }
    if include_audio_lip:
        mfcc = next((record.get(key) for key in ('mfcc', 'mfcc_features', 'audio_features') if key in record and record.get(key) is not None), None)
        mouth = next((record.get(key) for key in ('mouth', 'mouth_features', 'lip_features') if key in record and record.get(key) is not None), None)
        if mfcc is None or mouth is None:
            raise ValueError('Audio-lip branch requested but mfcc/mouth features are missing from this cached record')
        normalized['mfcc'] = torch.as_tensor(_to_numpy(mfcc), dtype=torch.float32)
        normalized['mouth'] = torch.as_tensor(_to_numpy(mouth), dtype=torch.float32)
    label = record.get('label')
    if label is None:
        label = record.get('y')
    if label is None and 'labels' in record:
        label = record['labels']
    if label is None:
        raise ValueError(f'Record missing label information: keys={sorted(record.keys())[:10]}')
    normalized['label'] = torch.tensor(float(_to_numpy(label).reshape(())), dtype=torch.float32)
    return normalized


def _load_cached_records(path: Path, include_audio_lip: bool = True) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    records: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if 'records' in payload and isinstance(payload['records'], list):
            records = payload['records']
        elif 'data' in payload and isinstance(payload['data'], list):
            records = payload['data']
        elif 'examples' in payload and isinstance(payload['examples'], list):
            records = payload['examples']
        elif all(key in payload for key in ('visual', 'rppg', 'label')) or all(key in payload for key in ('visual', 'rppg', 'labels')):
            batch = {}
            for key, value in payload.items():
                if key in ('visual', 'rppg', 'mfcc', 'mouth', 'label', 'labels'):
                    batch[key] = value
            batch_size = 1
            if 'labels' in batch and isinstance(batch['labels'], (list, tuple, np.ndarray, torch.Tensor)):
                batch_size = len(batch['labels'])
            elif 'label' in batch and isinstance(batch['label'], (list, tuple, np.ndarray, torch.Tensor)):
                batch_size = len(batch['label'])
            for index in range(batch_size):
                item = {}
                for key in ('visual', 'rppg', 'mfcc', 'mouth', 'label', 'labels'):
                    if key in batch:
                        value = batch[key]
                        if isinstance(value, torch.Tensor):
                            item[key] = value[index]
                        elif isinstance(value, np.ndarray):
                            item[key] = value[index]
                        elif isinstance(value, list):
                            item[key] = value[index]
                if 'labels' in item and 'label' not in item:
                    item['label'] = item['labels']
                if 'label' in item:
                    records.append(item)
        elif all(isinstance(value, dict) for value in payload.values()):
            records = list(payload.values())
        else:
            raise ValueError(f'Unsupported cached record structure in {path}: keys={sorted(payload.keys())[:20]}')
    else:
        raise TypeError(f'Unsupported cached record type in {path}: {type(payload).__name__}')
    normalized = []
    for index, record in enumerate(records):
        normalized.append(_normalize_cached_record(record, include_audio_lip=include_audio_lip))
    if not normalized:
        raise ValueError(f'No valid feature records were loaded from {path}')
    return normalized


def _resolve_feature_record_from_video(record: Dict[str, Any], manifest_path: Path, include_audio_lip: bool = True) -> Dict[str, Any]:
    relative_video = record.get('relative_video') or record.get('video')
    if relative_video is None:
        raise ValueError(f'Could not resolve a feature record because the manifest entry has no video path: {record}')
    relative_path = Path(str(relative_video).replace('\\', '/'))
    stem = relative_path.with_suffix('').as_posix().replace('/', '_')
    feature_roots = [
        (manifest_path.parent.parent / 'features' / manifest_path.parent.name).resolve(),
        (manifest_path.parent.parent.parent / 'features' / manifest_path.parent.name).resolve(),
        (manifest_path.parents[1] / 'features' / manifest_path.parent.name).resolve(),
    ]
    feature_root = next((root for root in feature_roots if root.exists()), feature_roots[0])
    candidate = {
        'visual': feature_root / f'{stem}_visual.npy',
        'rppg': feature_root / f'{stem}_rppg.npy',
    }
    if include_audio_lip:
        candidate['mfcc'] = feature_root / f'{stem}_mfcc.npy'
        candidate['mouth'] = feature_root / f'{stem}_mouth.npy'
    for name, path in candidate.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing extracted feature array for {relative_video}: {path}')
    resolved = {'visual': str(candidate['visual']), 'rppg': str(candidate['rppg']), 'label': int(record['label'])}
    if include_audio_lip:
        resolved.update({'mfcc': str(candidate['mfcc']), 'mouth': str(candidate['mouth'])})
    return resolved


class FeatureManifest(Dataset):
    def __init__(self, manifest: str, include_audio_lip: bool = True):
        self.include_audio_lip = include_audio_lip
        self.path = Path(manifest).resolve()
        if self.path.suffix.lower() in {'.pt', '.pth'}:
            self.records = _load_cached_records(self.path, include_audio_lip=include_audio_lip)
        else:
            raw_records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
            self.records = []
            for index, record in enumerate(raw_records):
                if all(key in record for key in ('visual', 'rppg', 'label')):
                    self.records.append(record)
                elif 'relative_video' in record or 'video' in record:
                    try:
                        patched = _resolve_feature_record_from_video(record, self.path, include_audio_lip=include_audio_lip)
                    except FileNotFoundError:
                        print(f'Skipping manifest record {index}: missing extracted feature array for {record.get("relative_video") or record.get("video")}')
                        continue
                    self.records.append(patched)
                else:
                    raise ValueError(f'Record {index} missing feature arrays or a source video path: {record}')
        if not self.records:
            raise ValueError(f'Manifest is empty: {self.path}')
        required = {'visual', 'rppg', 'label'} | ({'mfcc', 'mouth'} if include_audio_lip else set())
        for index, record in enumerate(self.records):
            if isinstance(record, dict):
                missing = required - record.keys()
            else:
                missing = required - set(record.keys())
            if missing:
                raise ValueError(f'Record {index} missing fields: {sorted(missing)}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        arrays = {}
        names = ('visual', 'rppg', 'mfcc', 'mouth') if self.include_audio_lip else ('visual', 'rppg')
        for name in names:
            value = record.get(name)
            if isinstance(value, str):
                path = Path(value)
                if not path.is_absolute():
                    path = (self.path.parent / path).resolve()
                if not path.exists():
                    raise FileNotFoundError(f'{name} features missing for record {index}: {path}')
                value = np.load(path, allow_pickle=False)
            elif value is None:
                raise KeyError(f'Record {index} is missing {name}')
            arrays[name] = torch.as_tensor(_to_numpy(value), dtype=torch.float32)
        steps = arrays['visual'].shape[0]
        if self.include_audio_lip and (arrays['mfcc'].shape[0] != steps or arrays['mouth'].shape[0] != steps):
            raise ValueError(f'Record {index} has inconsistent temporal lengths')
        if steps == 0:
            raise ValueError(f'Record {index} has no visual observations')
        arrays['length'] = torch.tensor(steps, dtype=torch.long)
        arrays['label'] = torch.tensor(float(_to_numpy(record['label']).reshape(())), dtype=torch.float32)
        return arrays


def collate(batch: List[Dict[str, torch.Tensor]], include_audio_lip: bool = True) -> Dict[str, torch.Tensor]:
    max_steps = max(item['length'].item() for item in batch)
    max_rppg = max(item['rppg'].shape[0] for item in batch)
    visual = torch.zeros(len(batch), max_steps, 3, 224, 224)
    mfcc = torch.zeros(len(batch), max_steps, 40) if include_audio_lip else None
    mouth = torch.zeros(len(batch), max_steps, 40) if include_audio_lip else None
    rppg = torch.zeros(len(batch), max_rppg)
    lengths = torch.stack([item['length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    for row, item in enumerate(batch):
        steps = item['length'].item()
        visual[row, :steps] = item['visual']
        rppg[row, :item['rppg'].shape[0]] = item['rppg']
        if include_audio_lip:
            mfcc[row, :steps] = item['mfcc']
            mouth[row, :steps] = item['mouth']
    result = {'visual': visual, 'rppg': rppg, 'lengths': lengths, 'labels': labels}
    if include_audio_lip:
        result.update({'mfcc': mfcc, 'mouth': mouth})
    return result


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    return (alpha_t * (1.0 - p_t) ** gamma * ce).mean()


def compute_pos_weight(train_manifest: str, include_audio_lip: bool) -> float:
    path = Path(train_manifest).resolve()
    if path.suffix.lower() in {'.pt', '.pth'}:
        records = _load_cached_records(path, include_audio_lip=include_audio_lip)
    else:
        records = FeatureManifest(str(path), include_audio_lip=include_audio_lip).records
    positives = sum(int(float(record['label'])) for record in records)
    negatives = len(records) - positives
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def augment_visual_batch(visual: torch.Tensor):
    batch_size, steps, channels, height, width = visual.shape
    augmented = visual.clone()
    for index in range(batch_size):
        if random.random() < 0.5:
            augmented[index] = torch.flip(augmented[index], dims=[-1])
        if random.random() < 0.2:
            noise = torch.randn_like(augmented[index]) * 0.02
            augmented[index] = augmented[index] + noise
    return augmented


def evaluate_threshold(labels, probabilities, metric='balanced_accuracy'):
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        else:
            score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def run_epoch(model, loader, optimizer, device, pos_weight=None, use_focal=False):
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0
    total = 0
    correct = 0
    probabilities = []
    labels_all = []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        if training and 'visual' in inputs:
            inputs['visual'] = augment_visual_batch(inputs['visual'])
        with torch.set_grad_enabled(training):
            logits = model(**inputs)
            if use_focal:
                loss = focal_loss(logits, labels)
            else:
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * labels.numel()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        total += labels.numel()
    return total_loss / total, correct / total, np.asarray(probabilities), np.asarray(labels_all)


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
    pos_weight = torch.tensor([compute_pos_weight(args.train_manifest, include_audio_lip)], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_score = -1.0
    patience = 6
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, use_focal=True)
        val_loss, val_accuracy, val_probs, val_labels = run_epoch(model, val_loader, None, device, pos_weight=pos_weight, use_focal=True)
        threshold, val_balanced_score = evaluate_threshold(val_labels, val_probs, metric='balanced_accuracy')
        print(f'epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} val_balanced_acc={val_balanced_score:.4f} threshold={threshold:.3f}')
        if val_balanced_score > best_val_score:
            best_val_score = val_balanced_score
            stale = 0
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'best_threshold': threshold}, output)
        else:
            stale += 1
            if stale >= patience:
                print('early stopping')
                break
        scheduler.step()


if __name__ == '__main__':
    main()