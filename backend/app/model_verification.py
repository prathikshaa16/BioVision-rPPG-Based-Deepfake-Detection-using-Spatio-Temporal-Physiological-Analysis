"""Internal model verification.

Establishes the mapping  model output -> probability -> REAL/FAKE label for the
existing EfficientNet-B4 checkpoint, using the checkpoint's own training
metadata where available.

The checkpoint is a plain state_dict (no embedded label map), so the label
mapping is taken from the published model card that ships with this exact
checkpoint (HuggingFace `honi05/deepfake-detection`), which documents:

    Sigmoid -> probability [0, 1]   (>= 0.5 = Fake)

and whose reference inference code is:

    prob = torch.sigmoid(model(x)).item()
    verdict = "FAKE" if prob >= 0.5 else "REAL"

This module never fabricates accuracy.  If no labelled evaluation video is
provided it explicitly reports that fake-class accuracy cannot be proven.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import MODEL_PATH
from .model import build_efficientnet_b4, load_checkpoint_into_model

# Documented class mapping for this checkpoint (from the model card metadata).
DOCUMENTED_MAPPING = {
    'output': 'single logit -> sigmoid -> probability in [0, 1]',
    'probability_meaning': 'sigmoid output is the FAKE probability',
    'label_rule': 'FAKE if probability >= 0.5 else REAL (per checkpoint source model card)',
    'source': 'https://huggingface.co/honi05/deepfake-detection',
}


def probe_model(device: str = 'cpu') -> Dict[str, Any]:
    """Load the checkpoint strict and probe its raw output range."""
    model = build_efficientnet_b4(num_classes=1)
    model.to(device)
    model, info = load_checkpoint_into_model(model, str(MODEL_PATH), device=device)
    model.eval()

    x = torch.rand(1, 3, 224, 224).to(device)
    with torch.inference_mode():
        logit = model(x).squeeze()
        prob = float(torch.sigmoid(logit).item())
        logit_val = float(logit.item())

    return {
        'checkpoint': os.path.basename(MODEL_PATH),
        'architecture': 'efficientnet-b4',
        'classifier_head': 'Dropout(0.4) -> Linear(1792,256) -> ReLU -> Dropout(0.2) -> Linear(256,1)',
        'strict_load': info.get('missing_keys', []) == [] and info.get('unexpected_keys', []) == [],
        'missing_keys': info.get('missing_keys', []),
        'unexpected_keys': info.get('unexpected_keys', []),
        'raw_output': 'sigmoid(logit)',
        'probe': {'logit': round(logit_val, 4), 'sigmoid_probability': round(prob, 4), 'in_unit_interval': 0.0 <= prob <= 1.0},
        'documented_mapping': DOCUMENTED_MAPPING,
        'biovision_classification_rule': {
            'mean_fake_probability <= 0.40': 'REAL',
            '0.40 < mean_fake_probability < 0.60': 'UNCERTAIN',
            'mean_fake_probability >= 0.60': 'FAKE',
        },
        'note': (
            'Class mapping is directionally consistent with the checkpoint model card. '
            'No labelled fake evaluation video is present in this repository, so fake-class '
            'accuracy cannot be independently proven here.'
        ),
    }


def run_labelled_validation(labelled_videos: List[Dict[str, Any]], device: str = 'cpu') -> Dict[str, Any]:
    """Run a list of labelled videos through the exact /upload inference path.

    labelled_videos: list of {"path": str, "ground_truth": "REAL"|"FAKE"}.
    Returns per-sample results and aggregate metrics computed ONLY from the
    actual labelled samples (never fabricated).
    """
    from .inference import analyze_video

    rows = []
    for sample in labelled_videos:
        try:
            result = analyze_video(sample['path'], device=device)
        except Exception as exc:  # NO_FACE / invalid video etc.
            rows.append({
                'filename': os.path.basename(sample['path']),
                'ground_truth': sample.get('ground_truth'),
                'prediction': 'ERROR',
                'confidence': None,
                'correct': None,
                'detail': str(exc),
            })
            continue

        rows.append({
            'filename': os.path.basename(sample['path']),
            'ground_truth': sample.get('ground_truth'),
            'prediction': result.get('result'),
            'confidence': result.get('confidence'),
            'fake_probability': result.get('fake_probability'),
            'correct': sample.get('ground_truth') is not None and result.get('result') == sample.get('ground_truth'),
            'detail': None,
        })

    evaluated = [r for r in rows if r['correct'] is not None]
    n = len(evaluated)
    correct = sum(1 for r in evaluated if r['correct'])
    accuracy = (correct / n) if n > 0 else None

    # Precision / recall / F1 for the FAKE class.
    tp = sum(1 for r in evaluated if r['ground_truth'] == 'FAKE' and r['prediction'] == 'FAKE')
    fp = sum(1 for r in evaluated if r['ground_truth'] == 'REAL' and r['prediction'] == 'FAKE')
    fn = sum(1 for r in evaluated if r['ground_truth'] == 'FAKE' and r['prediction'] != 'FAKE')

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else None

    return {
        'samples': rows,
        'evaluated': n,
        'accuracy': round(accuracy, 4) if accuracy is not None else None,
        'fake_class': {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4) if precision is not None else None,
            'recall': round(recall, 4) if recall is not None else None,
            'f1': round(f1, 4) if f1 is not None else None,
        },
        'note': (
            'Metrics above are computed only from the labelled videos actually present. '
            'If no labelled fake sample was provided, precision/recall/F1 for the fake class '
            'are reported as null.'
        ),
    }
