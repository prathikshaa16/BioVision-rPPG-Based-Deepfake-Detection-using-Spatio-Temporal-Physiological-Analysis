"""Transparent late fusion for visual and physiological evidence.

The available checkpoint is trained only on face images, so this module keeps
the visual classifier intact and adds a small, quality-gated rPPG evidence
channel. The fusion is deliberately configurable and reported in the API;
it is not a replacement for training a multimodal classifier on paired data.
"""

from typing import Any, Dict


VISUAL_WEIGHT = 0.80
RPPG_WEIGHT = 0.20


def fuse_probabilities(visual_probability: float, rppg: Dict[str, Any]) -> Dict[str, Any]:
    """Combine visual fake probability with an rPPG reliability signal.

    Low-quality or unavailable rPPG contributes no evidence. When available,
    its anomaly score is the complement of signal quality: a coherent pulse
    is treated as authentic physiological support, while a weak/absent pulse
    is treated as a cautious synthetic-signal indicator. The 20% cap prevents
    an unreliable physiological estimate from overruling the trained model.
    """
    visual = max(0.0, min(1.0, float(visual_probability)))
    quality = rppg.get('signal_quality') if rppg.get('status') == 'AVAILABLE' else None

    if quality is None:
        return {
            'probability': visual,
            'visual_probability': visual,
            'rppg_anomaly_score': None,
            'rppg_quality': None,
            'visual_weight': 1.0,
            'rppg_weight': 0.0,
            'method': 'visual-only (rPPG unavailable or below quality requirements)',
        }

    quality = max(0.0, min(1.0, float(quality)))
    rppg_anomaly = 1.0 - quality
    fused = VISUAL_WEIGHT * visual + RPPG_WEIGHT * rppg_anomaly
    return {
        'probability': round(fused, 6),
        'visual_probability': visual,
        'rppg_anomaly_score': round(rppg_anomaly, 6),
        'rppg_quality': quality,
        'visual_weight': VISUAL_WEIGHT,
        'rppg_weight': RPPG_WEIGHT,
        'method': 'quality-gated late fusion (EfficientNet-B4 + CHROM rPPG)',
    }