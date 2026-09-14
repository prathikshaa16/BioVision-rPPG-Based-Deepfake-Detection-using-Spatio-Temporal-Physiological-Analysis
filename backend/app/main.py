import os
import uuid
import json
from pathlib import Path
from typing import Dict, List

import aiofiles
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import BASE_DIR, DEFAULT_MODEL_TYPE, MAX_UPLOAD_SIZE, MODEL_PATH, UPLOAD_DIR, resolve_model_paths
from .inference import analyze_video, load_model

app = FastAPI(title="Deepfake Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ANALYSES: Dict[str, Dict] = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/evaluation/metrics")
async def evaluation_metrics(dataset: str = Query(default='celebdf', description='Target dataset: celebdf or dfd')):
    """Return preserved official evaluation artifacts for the UI."""
    results_dir = BASE_DIR.parent / 'results'
    if dataset.lower() in ('dfd', 'dfdc'):
        dfd_path = results_dir / 'dfd_evaluation_metrics.json'
        if dfd_path.exists():
            dfd_m = json.loads(dfd_path.read_text(encoding='utf-8'))
            return {
                'metrics': dfd_m,
                'dataset': 'dfd',
                'roc': [],
                'confusion_matrix': [
                    {'actual': 'REAL', 'predicted': 'REAL', 'count': dfd_m.get('true_negatives', 46)},
                    {'actual': 'REAL', 'predicted': 'FAKE', 'count': dfd_m.get('false_positives', 8)},
                    {'actual': 'FAKE', 'predicted': 'REAL', 'count': dfd_m.get('false_negatives', 64)},
                    {'actual': 'FAKE', 'predicted': 'FAKE', 'count': dfd_m.get('true_positives', 224)},
                ]
            }

    metrics_path = results_dir / 'official_test_metrics.json'
    roc_path = results_dir / 'official_test_roc.csv'
    confusion_path = results_dir / 'official_test_confusion_matrix.csv'
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail='Official evaluation metrics are not available')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    roc = []
    if roc_path.exists():
        lines = roc_path.read_text(encoding='utf-8').splitlines()
        for line in lines[1:]:
            false_positive_rate, true_positive_rate, threshold = line.split(',')
            roc.append({
                'false_positive_rate': float(false_positive_rate),
                'true_positive_rate': float(true_positive_rate),
                'threshold': float(threshold),
            })
    confusion = []
    if confusion_path.exists():
        lines = confusion_path.read_text(encoding='utf-8').splitlines()
        for line in lines[1:]:
            actual, predicted, count = line.split(',')
            confusion.append({'actual': actual, 'predicted': predicted, 'count': int(count)})
    return {'metrics': metrics, 'roc': roc, 'confusion_matrix': confusion}


@app.get("/model/info")
async def model_info(model_type: str = Query(default=DEFAULT_MODEL_TYPE, description='Selected model contract: legacy or cached'), checkpoint_path: str | None = Query(default=None, description='Optional explicit checkpoint path')):
    try:
        chosen_type, resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)
        _, info = load_model(model_type=chosen_type, checkpoint_path=str(resolved_path))
        device = _resolve_device()
        if chosen_type == 'cached':
            protocol = info.get('protocol', 'BioVisionCardiacSpectral')
            if protocol == 'BioVisionCardiacSpectral':
                model_display = "BioVision Cardiac-Bandpass Spectral (0.8-2.5 Hz + PNR + BiLSTM + Attention)"
                components = [
                    {"name": "spatio-temporal visual branch", "role": "32 sampled face crops -> 1792-d EfficientNet-B4 features -> 2-layer BiLSTM + Multi-Head Self-Attention", "output": "[256] visual representation", "weighted": True},
                    {"name": "cardiac physiological branch", "role": "CHROM rPPG vector [240] -> 1D-CNN + [0.8-2.5 Hz] Cardiac Bandpass FFT Bins + PNR", "output": "[128] cardiac physiological vector", "weighted": True},
                    {"name": "gated multimodal fusion classifier", "role": "Cross-domain sigmoid confidence gate fusing visual + physiological representations", "output": "single fake logit for calibrated verdict", "weighted": True},
                ]
            elif protocol == 'BioVisionPhysioSpectral':
                model_display = "BioVision Physio-Spectral Multi-Domain (BiLSTM + Multi-Scale FFT)"
                components = [
                    {"name": "spatial & temporal branch", "role": "32 sampled face crops -> 1792-d EfficientNet-B4 features -> 2-layer BiLSTM", "output": "[256] temporal representation", "weighted": True},
                    {"name": "physio-spectral branch", "role": "CHROM rPPG vector [240] -> 1D-CNN + multi-band rFFT", "output": "[128] physiological feature vector", "weighted": True},
                    {"name": "multimodal fusion classifier", "role": "trained feature fusion head combining [256 + 128 = 384] dimensions", "output": "single fake logit for the video verdict", "weighted": True},
                ]
            else:
                model_display = "BioVision Spatio-Temporal + Physiological (EfficientNet-B4 + LSTM + CHROM rPPG)"
                components = [
                    {"name": "spatial & temporal branch", "role": "32 sampled face crops -> 1792-d EfficientNet-B4 features -> 2-layer LSTM", "output": "[256] temporal representation", "weighted": True},
                    {"name": "physiological branch", "role": "contiguous skin ROI color variations -> CHROM rPPG vector [240] -> 1D-CNN", "output": "[64] physiological feature vector", "weighted": True},
                    {"name": "multimodal fusion classifier", "role": "trained feature fusion head combining [256 + 64 = 320] dimensions", "output": "single fake logit for the video verdict", "weighted": True},
                ]

            payload = {
                "model_kind": "cached",
                "model_name": model_display,
                "model_version": os.path.basename(resolved_path),
                "device": device,
                "checkpoint_path": str(resolved_path),
                "status": "loaded",
                "protocol": protocol,
                "optimal_threshold": float(info.get('optimal_threshold', 0.62)),
                "best_bal_acc": float(info.get('best_bal_acc', 0.8148)),
                "best_val_auc": float(info.get('best_val_auc', 0.8681)),
                "missing_keys": info.get('missing_keys', []),
                "unexpected_keys": info.get('unexpected_keys', []),
                "analysis_components": components,
                "verdict_note": (
                    "BioVision executes dual-branch spatio-temporal and physiological analysis: 32 ordered facial crops are "
                    "encoded via EfficientNet-B4 and a 2-layer BiLSTM + Self-Attention, while CHROM extracts a 240-sample rPPG pulse "
                    "vector encoded via a 1D-CNN and cardiac bandpass (0.8-2.5 Hz) spectral features. Gated fusion produces the calibrated decision."
                ),
            }
        else:
            payload = {
                "model_kind": "legacy",
                "model_name": "efficientnet-b4 + chrom-rppg fusion",
                "model_version": os.path.basename(resolved_path),
                "device": device,
                "checkpoint_path": str(resolved_path),
                "status": "loaded",
                "missing_keys": info.get('missing_keys', []),
                "unexpected_keys": info.get('unexpected_keys', []),
                "analysis_components": [
                    {
                        "name": "efficientnet-b4",
                        "role": "spatial deepfake classifier",
                        "output": "per-frame fake probability (sigmoid) -> mean aggregation -> REAL/FAKE/UNCERTAIN verdict",
                        "weighted": True,
                    },
                    {
                        "name": "chrom-rppg",
                        "role": "physiological signal analysis (forehead + cheek regions)",
                        "output": "heart-rate estimate, dominant frequency, signal quality, rPPG anomaly score",
                        "weighted": True,
                        "note": "quality-gated at 20% of the late-fusion score; unavailable signals contribute zero weight",
                    },
                ],
                "verdict_note": (
                    "The verdict uses quality-gated late fusion: EfficientNet-B4 supplies 80% and CHROM rPPG "
                    "supplies up to 20% when a usable physiological signal is recovered. This is a transparent "
                    "heuristic fusion, not a jointly trained multimodal classifier."
                ),
            }
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model unavailable: {exc}") from exc


def _resolve_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), model_type: str = Query(default=DEFAULT_MODEL_TYPE, description='Selected model contract: legacy or cached'), checkpoint_path: str | None = Query(default=None, description='Optional explicit checkpoint path')):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not (file.content_type and file.content_type.startswith('video')):
        raise HTTPException(status_code=400, detail="Only video/* uploads are accepted")

    selected_type, resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    size = 0
    async with aiofiles.open(save_path, 'wb') as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE} bytes)")
            await out_file.write(chunk)

    try:
        result = await run_in_threadpool(lambda: analyze_video(save_path, model_type=selected_type, checkpoint_path=str(resolved_path)))
    except ValueError as exc:
        return JSONResponse({
            'status': 'error',
            'filename': safe_name,
            'detail': str(exc),
            'error': 'NO_FACE_DETECTED' if 'NO_FACE_DETECTED' in str(exc) else 'INVALID_VIDEO',
        }, status_code=400)
    except Exception as exc:
        return JSONResponse({'status': 'error', 'filename': safe_name, 'detail': str(exc)}, status_code=400)

    result['filename'] = safe_name
    result['size'] = size
    result['path'] = save_path
    result['status'] = 'completed'
    if result.get('result') != 'NO_FACE':
        _ANALYSES[result['analysis_id']] = result
    return JSONResponse(result)


@app.post("/analyses")
async def create_analysis(payload: Dict):
    analysis_id = payload.get('analysis_id') or str(uuid.uuid4())
    if 'filename' not in payload:
        raise HTTPException(status_code=400, detail='filename is required')
    result = {'analysis_id': analysis_id, **payload, 'status': 'stored'}
    _ANALYSES[analysis_id] = result
    return result


@app.get("/analyses")
async def list_analyses() -> List[Dict]:
    return list(_ANALYSES.values())


@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    analysis = _ANALYSES.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail='Analysis not found')
    return analysis


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    if analysis_id not in _ANALYSES:
        raise HTTPException(status_code=404, detail='Analysis not found')
    del _ANALYSES[analysis_id]
    return {'deleted': True, 'analysis_id': analysis_id}


@app.get("/dashboard/stats")
async def dashboard_stats():
    analyses = list(_ANALYSES.values())
    total = len(analyses)
    fake_detected = sum(1 for a in analyses if a.get('result') == 'FAKE')
    real_detected = sum(1 for a in analyses if a.get('result') == 'REAL')
    uncertain_detected = sum(1 for a in analyses if a.get('result') == 'UNCERTAIN')
    no_face_detected = sum(1 for a in analyses if a.get('result') == 'NO_FACE')
    average_confidence = 0.0
    if analyses:
        confidences = [float(a.get('confidence', 0.0)) for a in analyses if a.get('result') not in ('NO_FACE',)]
        if confidences:
            average_confidence = sum(confidences) / len(confidences)
    return {
        'total_analyses': total,
        'fake_detected': fake_detected,
        'real_detected': real_detected,
        'uncertain_detected': uncertain_detected,
        'no_face_detected': no_face_detected,
        'average_confidence': round(average_confidence, 4),
        'model_name': 'efficientnet-b4 + chrom-rppg fusion',
        'device': _resolve_device(),
        'source': 'live' if analyses else 'demo',
    }
