"""Transparent multimodal fusion for Visual, Physiological (rPPG), and Audio-Lip evidence.

Preserves the tri-modal architecture specification:
1. Visual-Temporal Branch: Spatio-temporal facial representation (256-d).
2. Physiological Branch: CHROM rPPG remote photoplethysmography (64-d).
3. Audio-Lip Branch: Paired MFCC speech and mouth landmark correspondence (128-d).

Fuses all 3 channels (256 + 64 + 128 = 448 features) with quality-gating and
graceful fallbacks when a modality is absent (e.g. silent videos or occluded pulse).
"""

from typing import Any, Dict, Optional


VISUAL_WEIGHT_TRIMODAL = 0.60
RPPG_WEIGHT_TRIMODAL = 0.20
AUDIO_LIP_WEIGHT_TRIMODAL = 0.20

VISUAL_WEIGHT_DUAL = 0.80
RPPG_WEIGHT_DUAL = 0.20


def fuse_probabilities(
    visual_probability: float,
    rppg: Dict[str, Any],
    audio_lip: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine visual fake probability with physiological (rPPG) and audio-lip evidence."""
    visual = max(0.0, min(1.0, float(visual_probability)))

    # Parse rPPG quality and physiological signals
    rppg_status = rppg.get('status', 'UNAVAILABLE')
    quality = rppg.get('signal_quality') if rppg_status == 'AVAILABLE' else None
    has_rppg = quality is not None
    quality = max(0.0, min(1.0, float(quality))) if has_rppg else 0.0

    hr = rppg.get('heart_rate_bpm') if has_rppg else None
    dfreq = rppg.get('dominant_frequency') if has_rppg else None
    metrics = rppg.get('quality_metrics') or {}
    snr = metrics.get('snr_db') or 0.0

    is_authentic_cardiac = (
        has_rppg and hr is not None and 50.0 <= hr <= 108.0 and
        (dfreq is None or (0.80 <= dfreq <= 1.80)) and
        (quality >= 0.35 or snr >= 3.5)
    )

    is_synthetic_jitter = (
        has_rppg and (
            (hr is not None and hr > 120.0) or
            (dfreq is not None and dfreq > 2.0)
        ) and (quality < 0.45)
    )

    if is_authentic_cardiac:
        rppg_anomaly = max(0.02, min(0.15, 0.20 * (1.0 - quality)))
    elif is_synthetic_jitter:
        rppg_anomaly = 0.82
    elif has_rppg:
        rppg_anomaly = 1.0 - quality
    else:
        rppg_anomaly = None

    # Parse Audio-Lip correspondence evidence
    has_audio_lip = False
    audio_anomaly = None
    audio_sync = None
    is_audio_sync = False
    is_audio_desync = False

    if audio_lip is not None and isinstance(audio_lip, dict):
        a_status = audio_lip.get('status', 'UNAVAILABLE')
        if a_status == 'AVAILABLE' and audio_lip.get('has_audio', False):
            has_audio_lip = True
            audio_anomaly = float(audio_lip.get('audio_lip_anomaly_score', 0.20))
            audio_sync = float(audio_lip.get('speech_lip_sync_score', 0.50))
            is_audio_sync = bool(audio_lip.get('is_synchronized', False))
            is_audio_desync = bool(audio_lip.get('is_desynchronized', False))

    # Decision logic & dynamic weight assignment
    # Case 1: Full Tri-Modal Fusion (Visual + rPPG + Audio-Lip)
    if has_rppg and has_audio_lip:
        if is_authentic_cardiac and is_audio_sync:
            # Both independent modalities confirm biological authenticity
            w_vis = 0.30
            w_rppg = 0.45
            w_audio = 0.25
            fused = w_vis * visual + w_rppg * rppg_anomaly + w_audio * audio_anomaly
            method = 'tri-modal fusion (verified authentic cardiac pulse + synchronized phoneme-viseme alignment)'
        elif is_synthetic_jitter or is_audio_desync:
            # Active synthetic manipulation signature in rPPG jitter or audio-visual desync
            w_vis = 0.35
            w_rppg = 0.35 if is_synthetic_jitter else 0.20
            w_audio = 0.45 if is_audio_desync else 0.20
            total_w = w_vis + w_rppg + w_audio
            w_vis, w_rppg, w_audio = w_vis / total_w, w_rppg / total_w, w_audio / total_w
            fused = w_vis * visual + w_rppg * rppg_anomaly + w_audio * audio_anomaly
            method = 'tri-modal fusion (synthetic generative jitter / audiovisual desynchronization flagged)'
        else:
            w_vis = VISUAL_WEIGHT_TRIMODAL
            w_rppg = RPPG_WEIGHT_TRIMODAL
            w_audio = AUDIO_LIP_WEIGHT_TRIMODAL
            fused = w_vis * visual + w_rppg * rppg_anomaly + w_audio * audio_anomaly
            method = 'tri-modal multimodal fusion (Visual-Temporal + CHROM rPPG + Audio-Lip)'

        return {
            'probability': round(float(fused), 6),
            'visual_probability': visual,
            'rppg_anomaly_score': round(float(rppg_anomaly), 6) if rppg_anomaly is not None else None,
            'rppg_quality': quality,
            'audio_lip_anomaly_score': round(float(audio_anomaly), 6) if audio_anomaly is not None else None,
            'audio_lip_sync_score': round(float(audio_sync), 4) if audio_sync is not None else None,
            'visual_weight': round(float(w_vis), 2),
            'rppg_weight': round(float(w_rppg), 2),
            'audio_lip_weight': round(float(w_audio), 2),
            'method': method,
            'is_authentic_cardiac': is_authentic_cardiac,
            'is_synthetic_jitter': is_synthetic_jitter,
            'is_audio_lip_synchronized': is_audio_sync,
            'is_audio_lip_desynchronized': is_audio_desync,
            'modalities_used': ['visual', 'rppg', 'audio_lip'],
        }

    # Case 2: Visual + Audio-Lip (rPPG unavailable or silent)
    if has_audio_lip and not has_rppg:
        if is_audio_sync:
            w_vis = 0.50
            w_audio = 0.50
            method = 'dual-modal fusion (Visual-Temporal + Synchronized Audio-Lip; rPPG unavailable)'
        elif is_audio_desync:
            w_vis = 0.40
            w_audio = 0.60
            method = 'dual-modal fusion (Audiovisual desynchronization detected; rPPG unavailable)'
        else:
            w_vis = 0.70
            w_audio = 0.30
            method = 'dual-modal fusion (Visual-Temporal + Audio-Lip)'
        fused = w_vis * visual + w_audio * audio_anomaly
        return {
            'probability': round(float(fused), 6),
            'visual_probability': visual,
            'rppg_anomaly_score': None,
            'rppg_quality': None,
            'audio_lip_anomaly_score': round(float(audio_anomaly), 6),
            'audio_lip_sync_score': round(float(audio_sync), 4),
            'visual_weight': round(float(w_vis), 2),
            'rppg_weight': 0.0,
            'audio_lip_weight': round(float(w_audio), 2),
            'method': method,
            'is_authentic_cardiac': False,
            'is_synthetic_jitter': False,
            'is_audio_lip_synchronized': is_audio_sync,
            'is_audio_lip_desynchronized': is_audio_desync,
            'modalities_used': ['visual', 'audio_lip'],
        }

    # Case 3: Visual + rPPG Dual-Modal (Audio track absent, silent, or unvoiced)
    if has_rppg:
        if is_authentic_cardiac:
            w_vis = 0.35 if visual > 0.50 else 0.65
            w_rppg = 0.65 if visual > 0.50 else 0.35
            method = 'dual-modal fusion (verified authentic biological cardiac pulse; audio absent)'
        elif is_synthetic_jitter:
            w_vis = 0.40
            w_rppg = 0.60
            method = 'dual-modal fusion (generative synthetic pixel jitter detected; audio absent)'
        else:
            w_vis = VISUAL_WEIGHT_DUAL
            w_rppg = RPPG_WEIGHT_DUAL
            method = 'dual-modal fusion (EfficientNet-B4 + CHROM rPPG; audio absent)'
        fused = w_vis * visual + w_rppg * rppg_anomaly
        return {
            'probability': round(float(fused), 6),
            'visual_probability': visual,
            'rppg_anomaly_score': round(float(rppg_anomaly), 6),
            'rppg_quality': quality,
            'audio_lip_anomaly_score': None,
            'audio_lip_sync_score': None,
            'visual_weight': round(float(w_vis), 2),
            'rppg_weight': round(float(w_rppg), 2),
            'audio_lip_weight': 0.0,
            'method': method,
            'is_authentic_cardiac': is_authentic_cardiac,
            'is_synthetic_jitter': is_synthetic_jitter,
            'is_audio_lip_synchronized': False,
            'is_audio_lip_desynchronized': False,
            'modalities_used': ['visual', 'rppg'],
        }

    # Case 4: Unimodal Visual Fallback
    return {
        'probability': visual,
        'visual_probability': visual,
        'rppg_anomaly_score': None,
        'rppg_quality': None,
        'audio_lip_anomaly_score': None,
        'audio_lip_sync_score': None,
        'visual_weight': 1.0,
        'rppg_weight': 0.0,
        'audio_lip_weight': 0.0,
        'method': 'visual-only (physiological and audio channels unavailable)',
        'is_authentic_cardiac': False,
        'is_synthetic_jitter': False,
        'is_audio_lip_synchronized': False,
        'is_audio_lip_desynchronized': False,
        'modalities_used': ['visual'],
    }