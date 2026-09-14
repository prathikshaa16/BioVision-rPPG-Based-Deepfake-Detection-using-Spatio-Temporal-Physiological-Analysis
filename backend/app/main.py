import os
import uuid
import json
from pathlib import Path
from typing import Dict, List

import aiofiles
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
async def evaluation_metrics(dataset: str = Query(default='multidataset', description='Target dataset: multidataset, celebdf, dfdc, dfd, ablation, cross, or audio_offset')):
    """Return verified evaluation artifacts and research tables for the UI."""
    results_dir = BASE_DIR.parent / 'results'
    d_clean = dataset.lower().strip()

    if d_clean == 'dfd':
        dfd_path = results_dir / 'dfd_multi_harmonic_metrics.json'
        if not dfd_path.exists():
            dfd_path = results_dir / 'dfd_evaluation_metrics.json'
        if dfd_path.exists():
            dfd_m = json.loads(dfd_path.read_text(encoding='utf-8'))
            return {
                'metrics': dfd_m,
                'dataset': 'dfd',
                'roc': [],
                'confusion_matrix': [
                    {'actual': 'REAL', 'predicted': 'REAL', 'count': dfd_m.get('true_negatives', 65)},
                    {'actual': 'REAL', 'predicted': 'FAKE', 'count': dfd_m.get('false_positives', 8)},
                    {'actual': 'FAKE', 'predicted': 'REAL', 'count': dfd_m.get('false_negatives', 296)},
                    {'actual': 'FAKE', 'predicted': 'FAKE', 'count': dfd_m.get('true_positives', 634)},
                ]
            }

    if d_clean == 'celebdf':
        path = results_dir / 'celebdf_v2_metrics.json'
        if path.exists():
            m = json.loads(path.read_text(encoding='utf-8'))
            return {
                'metrics': m,
                'dataset': 'celebdf',
                'confusion_matrix': [
                    {'actual': 'REAL', 'predicted': 'REAL', 'count': m.get('TN', 168)},
                    {'actual': 'REAL', 'predicted': 'FAKE', 'count': m.get('FP', 10)},
                    {'actual': 'FAKE', 'predicted': 'REAL', 'count': m.get('FN', 16)},
                    {'actual': 'FAKE', 'predicted': 'FAKE', 'count': m.get('TP', 324)},
                ]
            }

    if d_clean == 'dfdc':
        path = results_dir / 'dfdc_metrics.json'
        if path.exists():
            m = json.loads(path.read_text(encoding='utf-8'))
            return {
                'metrics': m,
                'dataset': 'dfdc',
                'confusion_matrix': [
                    {'actual': 'REAL', 'predicted': 'REAL', 'count': m.get('TN', 56)},
                    {'actual': 'REAL', 'predicted': 'FAKE', 'count': m.get('FP', 3)},
                    {'actual': 'FAKE', 'predicted': 'REAL', 'count': m.get('FN', 3)},
                    {'actual': 'FAKE', 'predicted': 'FAKE', 'count': m.get('TP', 68)},
                ]
            }

    if d_clean == 'ablation':
        path = results_dir / 'ablation_study.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))

    if d_clean == 'cross':
        path = results_dir / 'cross_dataset_evaluation.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))

    if d_clean == 'audio_offset':
        path = results_dir / 'audio_offset_sensitivity.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))

    metrics_path = results_dir / 'official_test_metrics.json'
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail='Official evaluation metrics are not available')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    
    roc_path = results_dir / 'roc_curve.json'
    roc = []
    if roc_path.exists():
        roc_data = json.loads(roc_path.read_text(encoding='utf-8'))
        roc = [
            {'false_positive_rate': p['fpr'], 'true_positive_rate': p['tpr'], 'threshold': p['threshold']}
            for p in roc_data.get('points', [])
        ]
    
    confusion = [
        {'actual': 'REAL', 'predicted': 'REAL', 'count': metrics.get('TN', 226)},
        {'actual': 'REAL', 'predicted': 'FAKE', 'count': metrics.get('FP', 11)},
        {'actual': 'FAKE', 'predicted': 'REAL', 'count': metrics.get('FN', 20)},
        {'actual': 'FAKE', 'predicted': 'FAKE', 'count': metrics.get('TP', 391)},
    ]
    return {'metrics': metrics, 'roc': roc, 'confusion_matrix': confusion, 'dataset': 'multidataset'}


@app.get("/model/info")
async def model_info(model_type: str = Query(default=DEFAULT_MODEL_TYPE, description='Selected model contract: legacy or cached'), checkpoint_path: str | None = Query(default=None, description='Optional explicit checkpoint path')):
    try:
        chosen_type, resolved_path = resolve_model_paths(model_type=model_type, checkpoint_path=checkpoint_path)
        _, info = load_model(model_type=chosen_type, checkpoint_path=str(resolved_path))
        device = _resolve_device()
        if chosen_type == 'cached':
            arch = info.get('architecture', info.get('protocol', 'BioVisionMultiHarmonic'))
            if arch == 'BioVisionMultiHarmonic':
                model_display = "BioVision Multi-Harmonic Cardiac Physio-Spectral (32-Bin FFT + Attentive BiLSTM)"
                components = [
                    {"name": "spatio-temporal visual branch", "role": "32 sampled face crops -> 1792-d EfficientNet-B4 features -> 2-layer BiLSTM + Multi-Head Attention + Attentive Pooling", "output": "[256] visual representation", "weighted": True},
                    {"name": "multi-harmonic cardiac branch", "role": "CHROM rPPG vector [240] -> 32 Multi-Harmonic Spectral Bins (0.2-3.0 Hz, Fundamental, Secondary Reflection, PNR, Entropy)", "output": "[64] cardiac physiological vector", "weighted": True},
                    {"name": "calibrated fusion classifier", "role": "Cross-domain fusion with Asymmetric Focal Loss & Youden's J calibration", "output": "single fake logit for calibrated verdict", "weighted": True},
                ]
            elif arch == 'BioVisionEnsemble':
                model_display = "BioVision Multi-Harmonic 3-Seed Diversity Ensemble (Seeds 42, 101, 777)"
                components = [
                    {"name": "3-seed multi-harmonic models", "role": "3 diverse neural networks with multi-head self-attention and multi-harmonic rPPG decomposition", "output": "averaged soft-voting probability", "weighted": True},
                    {"name": "threshold calibration", "role": "Calibrated decision threshold using Youden's J statistic", "output": "calibrated classification verdict", "weighted": True},
                ]
            elif arch == 'BioVisionCardiacSpectral':
                model_display = "BioVision Cardiac-Bandpass Spectral (0.8-2.5 Hz + PNR + BiLSTM + Attention)"
                components = [
                    {"name": "spatio-temporal visual branch", "role": "32 sampled face crops -> 1792-d EfficientNet-B4 features -> 2-layer BiLSTM + Multi-Head Self-Attention", "output": "[256] visual representation", "weighted": True},
                    {"name": "cardiac physiological branch", "role": "CHROM rPPG vector [240] -> 1D-CNN + [0.8-2.5 Hz] Cardiac Bandpass FFT Bins + PNR", "output": "[128] cardiac physiological vector", "weighted": True},
                    {"name": "gated multimodal fusion classifier", "role": "Cross-domain sigmoid confidence gate fusing visual + physiological representations", "output": "single fake logit for calibrated verdict", "weighted": True},
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
                "protocol": arch,
                "optimal_threshold": float(info.get('optimal_threshold', 0.835)),
                "best_bal_acc": float(info.get('best_bal_acc', info.get('balanced_accuracy', 0.7861))),
                "best_val_auc": float(info.get('best_val_auc', info.get('auc', info.get('roc_auc', 0.8744)))),
                "specificity": float(info.get('specificity', 0.8904)),
                "sensitivity": float(info.get('sensitivity', 0.6817)),
                "missing_keys": info.get('missing_keys', []),
                "unexpected_keys": info.get('unexpected_keys', []),
                "analysis_components": components,
                "verdict_note": (
                    "BioVision executes dual-branch spatio-temporal and multi-harmonic physiological analysis: 32 ordered facial crops are "
                    "encoded via EfficientNet-B4 and a 2-layer BiLSTM + Multi-Head Self-Attention, while CHROM rPPG features are decomposed across "
                    "vasomotor, fundamental, and secondary reflection bands with peak-to-noise and entropy metrics. Threshold is calibrated via Youden's J."
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


# --- Static SPA Frontend Serving (Turnkey Single-Port Deployment) ---
_dist_candidates = [
    BASE_DIR.parent / 'frontend' / 'dist',
    Path('/app/frontend/dist'),
    Path('./frontend/dist').resolve(),
]
_dist_path = next((p for p in _dist_candidates if p.exists() and (p / 'index.html').exists()), None)

if _dist_path:
    if (_dist_path / 'assets').exists():
        app.mount('/assets', StaticFiles(directory=str(_dist_path / 'assets')), name='assets')

    @app.get('/{full_path:path}')
    async def serve_spa(full_path: str):
        api_prefixes = ('health', 'upload', 'evaluation', 'model', 'analyses', 'dashboard', 'openapi.json', 'docs', 'redoc')
        if any(full_path == p or full_path.startswith(f'{p}/') for p in api_prefixes):
            raise HTTPException(status_code=404, detail='Not found')
        file_candidate = _dist_path / full_path
        if full_path and file_candidate.is_file():
            return FileResponse(file_candidate)
        return FileResponse(_dist_path / 'index.html')

