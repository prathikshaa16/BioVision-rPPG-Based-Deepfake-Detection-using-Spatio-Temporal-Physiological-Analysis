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
async def evaluation_metrics():
    """Return the preserved official evaluation artifacts for the UI."""
    results_dir = BASE_DIR.parent / 'results'
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
            payload = {
                "model_kind": "cached",
                "model_name": "cached visual+rppg baseline",
                "model_version": os.path.basename(resolved_path),
                "device": device,
                "checkpoint_path": str(resolved_path),
                "status": "loaded",
                "missing_keys": info.get('missing_keys', []),
                "unexpected_keys": info.get('unexpected_keys', []),
                "analysis_components": [
                    {"name": "visual-temporal embeddings", "role": "32 sampled face crops -> 1792-d EfficientNet features", "output": "[32,1792] ordered embedding sequence", "weighted": True},
                    {"name": "chrom-rppg", "role": "contiguous physiological signal from the face ROI", "output": "[240] pulse vector aggregated by the notebook contract", "weighted": True},
                    {"name": "cached classifier", "role": "trained visual+rppg fusion head", "output": "single fake logit for the video verdict", "weighted": True},
                ],
                "verdict_note": (
                    "This is the Colab cached-feature BioVision baseline. It expects precomputed visual features and a "
                    "240-sample rPPG vector and runs a trained fusion head rather than the raw-frame legacy path."
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
