"""Evaluate a trained BioVision checkpoint on a held-out feature manifest.

Supports both the legacy JSONL manifests and Kaggle-style cached .pt record files.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.scripts.train_multimodal import FeatureManifest, collate
from backend.app.multimodal_model import build_biovision_multimodal_model, build_biovision_visual_rppg_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate BioVision multimodal checkpoint')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=('full', 'visual-rppg'), default='visual-rppg')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    include_audio_lip = args.mode == 'full'
    dataset = FeatureManifest(args.manifest, include_audio_lip)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, include_audio_lip),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = (build_biovision_multimodal_model() if include_audio_lip else build_biovision_visual_rppg_model()).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get('model_state_dict', state), strict=True)
    model.eval()

    probabilities = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            probabilities.extend(torch.sigmoid(outputs).cpu().tolist())
            labels.extend(batch['labels'].tolist())

    predictions = [int(probability >= args.threshold) for probability in probabilities]
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'manifest': str(Path(args.manifest).resolve()),
        'mode': args.mode,
        'threshold': args.threshold,
        'samples': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': round(accuracy, 6),
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
        'specificity': round(specificity, 6),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()