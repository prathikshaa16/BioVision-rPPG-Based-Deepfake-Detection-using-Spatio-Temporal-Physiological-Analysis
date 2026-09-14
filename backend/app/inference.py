import os
import statistics
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights

from .config import DEFAULT_SAMPLE_FRAMES, resolve_model_paths
from .face_processor import FaceProcessor
from .fusion import fuse_probabilities
from .model import build_efficientnet_b4, load_checkpoint_into_model
from .multimodal_model import build_cached_biovision_visual_rppg_model, build_biovision_model_from_checkpoint
from .rppg import run_rppg_analysis
from .video_processor import validate_video, sample_frame_indices, read_frames_by_indices

_MODEL = None
_MODEL_META = {}


def _get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(device: str = None, model_type: str = None, checkpoint_path: str = None):
    global _MODEL, _MODEL_META
    chosen_type, resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)
    if _MODEL is not None and _MODEL_META.get('model_type') == chosen_type and _MODEL_META.get('checkpoint_path') == str(resolved_path):
        return _MODEL, _MODEL_META

    device = device or _get_device()
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Model checkpoint not found at {resolved_path}")

    if chosen_type == 'cached':
        model = build_biovision_model_from_checkpoint(str(resolved_path))
        model.to(device)
        model, info = load_checkpoint_into_model(model, str(resolved_path), device=device)
        info['model_type'] = chosen_type
        info['checkpoint_path'] = str(resolved_path)
        model.eval()
        _MODEL = model
        _MODEL_META = info
        return model, info

    model = build_efficientnet_b4(num_classes=1)
    model.to(device)
    model, info = load_checkpoint_into_model(model, str(resolved_path), device=device)
    info['model_type'] = chosen_type
    info['checkpoint_path'] = str(resolved_path)
    model.eval()
    _MODEL = model
    _MODEL_META = info
    return model, info


def _prepare_cached_visual_features(video_path: str, device: str = 'cpu', n_frames: int = 32) -> torch.Tensor:
    """Create the notebook-compatible visual feature tensor [32, 1792]."""
    meta = validate_video(video_path)
    indices = sample_frame_indices(meta['frame_count'], n_samples=n_frames)
    frames = read_frames_by_indices(video_path, indices)
    fp = FaceProcessor(device=device)
    face_tensors: List[torch.Tensor] = []
    for frame in frames:
        if frame is None:
            continue
        boxes, _ = fp.detect_faces(frame)
        if not boxes:
            continue
        largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        crop = fp.detect_and_crop(frame)
        if not crop:
            continue
        largest_crop = max(crop, key=lambda item: item.size[0] * item.size[1])
        face_tensors.append(largest_crop)

    if not face_tensors:
        raise ValueError('NO_FACE_DETECTED')

    weights = EfficientNet_B4_Weights.DEFAULT
    transform = weights.transforms()
    batch = torch.stack([transform(image) for image in face_tensors[:n_frames]]).to(device)
    if batch.shape[0] < n_frames:
        pad = n_frames - batch.shape[0]
        batch = torch.cat([batch, batch[-1:].repeat(pad, 1, 1, 1)], dim=0)
    if batch.shape[0] > n_frames:
        batch = batch[:n_frames]

    backbone = models.efficientnet_b4(weights=weights)
    backbone = backbone.to(device)
    backbone.eval()
    with torch.inference_mode():
        features = backbone.features(batch)
        features = backbone.avgpool(features)
        features = torch.flatten(features, 1)
    return features.to(device)


def _prepare_cached_rppg_vector(video_path: str, meta: Dict[str, Any], device: str = 'cpu', size: int = 240):
    """Return a notebook-compatible [240] pulse vector padded/truncated to length 240 and the rppg signal dict."""
    signal = run_rppg_analysis(video_path, meta, device=device)
    values = signal.get('filtered_signal', {}).get('amplitude') if signal.get('status') == 'AVAILABLE' else None
    if values is None:
        values = np.zeros(size, dtype=np.float32)
    else:
        values = np.asarray(values, dtype=np.float32)
        if values.size < size:
            padded = np.zeros(size, dtype=np.float32)
            padded[:values.size] = values
            values = padded
        elif values.size > size:
            values = values[:size]
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return tensor, signal


def analyze_video(video_path: str, device: str = None, model_type: str = None, checkpoint_path: str = None) -> Dict[str, Any]:
    """Run video sampling, face detection, model inference, aggregation and rPPG.

    Dual-branch multimodal inference:
    1. Spatial & Temporal: 32 ordered facial crops -> EfficientNet-B4 (1792-d) -> 2-layer LSTM (256-d).
    2. Physiological: CHROM rPPG signal (240-d) -> 1D-CNN (64-d).
    3. Multimodal Late Fusion: Combined 320-d features classified by trained fusion head.
    """
    start_time = time.time()
    device = device or _get_device()
    chosen_type, resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)
    if chosen_type == 'cached':
        try:
            meta = validate_video(video_path)
        except Exception as exc:
            raise ValueError(f"Invalid video file: {exc}") from exc
        try:
            visual_features = _prepare_cached_visual_features(video_path, device=device, n_frames=32)
        except ValueError as exc:
            if 'NO_FACE_DETECTED' in str(exc):
                processing_time = time.time() - start_time
                return {
                    'analysis_id': f"{os.path.basename(video_path)}-{int(start_time * 1000)}",
                    'filename': os.path.basename(video_path),
                    'status': 'completed',
                    'result': 'NO_FACE',
                    'confidence': 0.0,
                    'fake_probability': 0.0,
                    'real_probability': 0.0,
                    'visual_fake_probability': None,
                    'frames_sampled': 32,
                    'frames_with_faces': 0,
                    'frames_without_faces': 32,
                    'faces_detected': 0,
                    'frame_predictions': [],
                    'processing_time': round(processing_time, 3),
                    'model_name': 'BioVision Spatio-Temporal + Physiological (EfficientNet-B4 + LSTM + CHROM rPPG)',
                    'model_version': os.path.basename(str(resolved_path)),
                    'model_kind': 'cached',
                    'device': device,
                    'meta': meta,
                    'explanation': (
                        'No usable face was detected across the sampled sequence, so deepfake '
                        'classification could not be performed. No REAL or FAKE probability was computed.'
                    ),
                    'rppg': {
                        'status': 'SKIPPED',
                        'explanation': 'No usable face was detected in the sampled frames, so no physiological signal was extracted.',
                        'frames_used': 0,
                        'heart_rate_bpm': None,
                        'dominant_frequency': None,
                        'signal_quality': None,
                    },
                }
            raise
        rppg_vector, rppg_signal = _prepare_cached_rppg_vector(video_path, meta, device=device, size=240)
        model, model_info = load_model(device=device, model_type=chosen_type, checkpoint_path=str(resolved_path))
        with torch.inference_mode():
            logits = model(visual_features.unsqueeze(0), rppg_vector.unsqueeze(0)).view(-1)
            probability = torch.sigmoid(logits[0]).detach().cpu().item()

            # Sequence trajectory across observations
            step_preds = []
            if hasattr(model, 'rppg_conv'):
                lstm_out, _ = model.visual_lstm(visual_features.unsqueeze(0))
                phys_feat = model.rppg_norm(model.rppg_conv(rppg_vector.unsqueeze(0).unsqueeze(1)).squeeze(-1))
                for t in range(visual_features.shape[0]):
                    step_vis = model.visual_norm(lstm_out[:, t, :])
                    step_fused = torch.cat((step_vis, phys_feat), dim=1)
                    step_p = torch.sigmoid(model.classifier(step_fused).view(-1)[0]).item()
                    step_preds.append(round(step_p, 4))
            elif hasattr(model, 'visual_lstm') or hasattr(model, 'bilstm'):
                for t in range(visual_features.shape[0]):
                    sub_vis = visual_features[:t+1].unsqueeze(0)
                    if sub_vis.shape[1] < 2:
                        sub_vis = torch.cat([sub_vis, sub_vis], dim=1)
                    sub_log = model(sub_vis, rppg_vector.unsqueeze(0)).view(-1)
                    step_p = torch.sigmoid(sub_log[0]).item()
                    step_preds.append(round(step_p, 4))
            else:
                step_preds = [round(probability, 4)] * visual_features.shape[0]

        opt_thresh = float(model_info.get('optimal_threshold', 0.50))
        margin = 0.05
        if probability >= opt_thresh + margin:
            result = 'FAKE'
        elif probability <= opt_thresh - margin:
            result = 'REAL'
        else:
            result = 'UNCERTAIN'
        processing_time = time.time() - start_time

        # Ensure rppg payload includes vector and length
        if isinstance(rppg_signal, dict):
            rppg_payload = dict(rppg_signal)
            rppg_payload['input_length'] = int(rppg_vector.shape[0])
            rppg_payload['vector'] = rppg_vector.detach().cpu().tolist()
        else:
            rppg_payload = {
                'status': 'AVAILABLE' if len(rppg_vector) else 'UNAVAILABLE',
                'input_length': int(rppg_vector.shape[0]),
                'vector': rppg_vector.detach().cpu().tolist(),
            }

        arch = model_info.get('architecture', model_info.get('protocol', 'BioVisionMultiHarmonic'))
        if arch == 'BioVisionMultiHarmonic':
            model_display_name = 'BioVision Multi-Harmonic Cardiac Physio-Spectral (32-Bin FFT + Attentive BiLSTM)'
        elif arch == 'BioVisionEnsemble':
            model_display_name = 'BioVision Multi-Harmonic 3-Seed Diversity Ensemble'
        elif arch == 'BioVisionCardiacSpectral':
            model_display_name = 'BioVision Cardiac-Bandpass Spectral (0.8-2.5 Hz + PNR + BiLSTM + Attention)'
        elif arch == 'BioVisionPhysioSpectral':
            model_display_name = 'BioVision Physio-Spectral Multi-Domain (BiLSTM + Multi-Scale FFT)'
        else:
            model_display_name = 'BioVision Spatio-Temporal + Physiological (EfficientNet-B4 + LSTM + CHROM rPPG)'

        denom = max(opt_thresh, 1.0 - opt_thresh)
        confidence = max(0.0, min(1.0, abs(probability - opt_thresh) / (denom if denom > 0 else 0.5)))

        return {
            'analysis_id': f"{os.path.basename(video_path)}-{int(start_time * 1000)}",
            'filename': os.path.basename(video_path),
            'status': 'completed',
            'result': result,
            'confidence': round(confidence, 4),
            'fake_probability': float(probability),
            'real_probability': 1.0 - float(probability),
            'visual_fake_probability': float(probability),
            'optimal_threshold': opt_thresh,
            'frames_sampled': 32,
            'frames_with_faces': 32,
            'frames_without_faces': 0,
            'faces_detected': 32,
            'frame_predictions': step_preds,
            'mean_probability': float(probability),
            'processing_time': round(processing_time, 3),
            'model_name': model_display_name,
            'model_version': os.path.basename(str(resolved_path)),
            'model_kind': 'cached',
            'device': device,
            'model_load_info': model_info,
            'meta': meta,
            'rppg': rppg_payload,
            'cached_features': {
                'visual_shape': [int(visual_features.shape[0]), int(visual_features.shape[1])],
                'rppg_shape': [int(rppg_vector.shape[0])],
            },
        }

    try:
        meta = validate_video(video_path)
    except Exception as exc:
        raise ValueError(f"Invalid video file: {exc}") from exc

    indices = sample_frame_indices(meta['frame_count'], n_samples=DEFAULT_SAMPLE_FRAMES)
    frames = read_frames_by_indices(video_path, indices)

    fp = FaceProcessor(device=device)
    processed_tensors = []
    frame_results: List[Dict[str, Any]] = []

    for idx, frame in zip(indices, frames):
        if frame is None:
            frame_results.append({
                'index': idx,
                'faces': 0,
                'prediction': None,
                'error': 'frame_read_failed',
                'boxes': [],
            })
            continue

        boxes, _ = fp.detect_faces(frame)
        if not boxes:
            frame_results.append({
                'index': idx,
                'faces': 0,
                'prediction': None,
                'error': 'NO_FACE_DETECTED',
                'boxes': [],
            })
            continue

        crops = fp.detect_and_crop(frame)
        if not crops:
            frame_results.append({
                'index': idx,
                'faces': len(boxes),
                'prediction': None,
                'error': 'NO_FACE_DETECTED',
                'boxes': [],
            })
            continue

        largest_crop = max(crops, key=lambda c: c.size[0] * c.size[1])
        tensor = fp.preprocess_pil(largest_crop)
        processed_tensors.append(tensor.to(device))
        frame_results.append({
            'index': idx,
            'faces': len(boxes),
            'prediction': None,
            'error': None,
            'boxes': [
                {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1)}
                for x1, y1, x2, y2 in boxes
            ],
        })

    frames_without_faces = sum(1 for item in frame_results if item['faces'] == 0)
    faces_detected = sum(item['faces'] for item in frame_results)

    if not processed_tensors:
        processing_time = time.time() - start_time
        resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)[1]
        return {
            'analysis_id': f"{os.path.basename(video_path)}-{int(start_time * 1000)}",
            'filename': os.path.basename(video_path),
            'status': 'completed',
            'result': 'NO_FACE',
            'frames_sampled': len(indices),
            'frames_with_faces': 0,
            'frames_without_faces': frames_without_faces,
            'faces_detected': faces_detected,
            'processing_time': round(processing_time, 3),
            'model_name': 'efficientnet-b4 + chrom-rppg fusion',
            'model_version': os.path.basename(resolved_path),
            'device': device,
            'explanation': (
                'No usable face was detected in the sampled frames, so deepfake classification '
                'was not performed. No REAL or FAKE probability was computed.'
            ),
            'meta': meta,
            'sampled_indices': indices,
            'frame_results': frame_results,
            'rppg': {
                'status': 'SKIPPED',
                'explanation': 'No usable face was detected in the sampled frames, so no physiological signal was extracted.',
                'frames_used': 0,
                'heart_rate_bpm': None,
                'dominant_frequency': None,
                'signal_quality': None,
                'quality_metrics': None,
                'signal': None,
                'filtered_signal': None,
                'frequency': None,
                'roi': None,
                'window': None,
            },
        }

    model, model_info = load_model(
        device=device,
        model_type=chosen_type,
        checkpoint_path=str(resolved_path),
    )
    batch = torch.stack(processed_tensors)
    with torch.inference_mode():
        logits = model(batch).squeeze(1)
        fake_probabilities = torch.sigmoid(logits).detach().cpu().tolist()

    p_idx = 0
    for item in frame_results:
        if item['faces'] > 0:
            item['prediction'] = float(fake_probabilities[p_idx])
            p_idx += 1

    valid_preds = [float(item['prediction']) for item in frame_results if item['prediction'] is not None]
    if not valid_preds:
        raise ValueError('NO_FACE_DETECTED')

    visual_mean_probability = float(statistics.fmean(valid_preds))
    median_probability = float(statistics.median(valid_preds))
    variance = float(statistics.pvariance(valid_preds)) if len(valid_preds) > 1 else 0.0
    std_probability = float(variance ** 0.5)
    min_probability = float(min(valid_preds))
    max_probability = float(max(valid_preds))
    # Consistency: 1 - std, clamped to [0, 1]. A low spread means consistent evidence.
    consistency = max(0.0, min(1.0, 1.0 - std_probability))

    # Run both evidence channels before deciding so the verdict reflects the
    # quality-gated late fusion rather than a diagnostic-only side channel.
    rppg = run_rppg_analysis(video_path, meta, device=device)
    fusion = fuse_probabilities(visual_mean_probability, rppg)
    mean_probability = float(fusion['probability'])

    if mean_probability >= 0.60:
        result = 'FAKE'
    elif mean_probability <= 0.40:
        result = 'REAL'
    else:
        result = 'UNCERTAIN'

    real_probability = 1.0 - mean_probability
    confidence = abs(mean_probability - 0.50) * 2.0
    confidence = max(0.0, min(1.0, confidence))

    processing_time = time.time() - start_time

    resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)[1]
    return {
        'analysis_id': f"{os.path.basename(video_path)}-{int(start_time * 1000)}",
        'filename': os.path.basename(video_path),
        'status': 'completed',
        'result': result,
        'confidence': confidence,
        'fake_probability': mean_probability,
        'real_probability': real_probability,
        'visual_fake_probability': visual_mean_probability,
        'fusion': fusion,
        'frames_sampled': len(indices),
        'frames_with_faces': len(valid_preds),
        'frames_without_faces': frames_without_faces,
        'faces_detected': faces_detected,
        'frame_predictions': [float(item['prediction']) for item in frame_results if item['prediction'] is not None],
        'mean_probability': mean_probability,
        'median_probability': median_probability,
        'std_probability': std_probability,
        'variance': variance,
        'min_probability': min_probability,
        'max_probability': max_probability,
        'consistency': consistency,
        'processing_time': round(processing_time, 3),
        'model_name': 'efficientnet-b4 + chrom-rppg fusion',
        'model_version': os.path.basename(resolved_path),
        'device': device,
        'model_load_info': model_info,
        'meta': meta,
        'sampled_indices': indices,
        'frame_results': frame_results,
        'batch_tensor_shape': [batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]],
        'rppg': rppg,
    }
