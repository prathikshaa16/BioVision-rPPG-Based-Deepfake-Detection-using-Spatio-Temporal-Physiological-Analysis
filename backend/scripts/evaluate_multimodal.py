"""Evaluate a trained BioVision checkpoint on a held-out feature manifest.

Supports both the legacy JSONL manifests and Kaggle-style cached .pt record files.
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
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
    parser.add_argument('--output-dir', default=None, help='Directory for metrics, ROC, confusion matrix, and predictions')
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
    balanced_accuracy = balanced_accuracy_score(labels, predictions) if len(set(labels)) > 1 else 0.0
    roc_auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else None
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'predictions.csv').open('w', encoding='utf-8') as stream:
        stream.write('index,label,probability,prediction\n')
        for index, (label, probability, prediction) in enumerate(zip(labels, probabilities, predictions)):
            stream.write(f'{index},{int(label)},{probability:.8f},{prediction}\n')
    if roc_auc is not None:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
        with (output_dir / 'roc.csv').open('w', encoding='utf-8') as stream:
            stream.write('false_positive_rate,true_positive_rate,threshold\n')
            for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds):
                stream.write(f'{fpr:.8f},{tpr:.8f},{threshold:.8f}\n')
    with (output_dir / 'confusion_matrix.csv').open('w', encoding='utf-8') as stream:
        stream.write('actual,predicted,count\n0,0,%d\n0,1,%d\n1,0,%d\n1,1,%d\n' % (tn, fp, fn, tp))
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
        'balanced_accuracy': round(balanced_accuracy, 6),
        'roc_auc': round(roc_auc, 6) if roc_auc is not None else None,
        'predictions_file': str((output_dir / 'predictions.csv').resolve()),
        'roc_file': str((output_dir / 'roc.csv').resolve()) if roc_auc is not None else None,
        'confusion_matrix_file': str((output_dir / 'confusion_matrix.csv').resolve()),
    }
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()