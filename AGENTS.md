# AGENTS.md

Deepfake detector monorepo: `frontend/` (React + Vite + TypeScript + Tailwind v4) and `backend/` (FastAPI + PyTorch). Runs on Windows; no git repo.

## Commands

- Backend (from `backend/`): `python -m uvicorn app.main:app --reload --port 8000`
  - Alternatively from repo root: `python -m uvicorn backend.app.main:app --port 8000` — the frontend's error hint and `backend/scripts/*.py` use this form and import `backend.app.*`, so they must be run from the repo root.
- Frontend: `npm run dev` (port 5173). No test/typecheck/lint scripts exist; typecheck is `npx tsc --noEmit`.
- Tests: only `backend/app/tests/test_video_processing.py` exists (OpenCV frame sampling). It uses package-relative imports (`from ..video_processor import ...`) but `tests/` has no `__init__.py` and there is no pytest config; pytest is **not** installed in `.venv`. Expect friction running it.
- `test_upload.py` (repo root) is an ad-hoc upload smoke test that requires a video at `uploads/real_test.mp4` and a running server.

## Backend gotchas

- Two requirements files, both needed for `/upload` and `/model/info`: `requirements.txt` (FastAPI core) + `requirements_extra.txt` (opencv-python, torch, torchvision, facenet-pytorch, Pillow, numpy). The root `.venv` contains **only** the core deps — running it will crash on inference.
- Model checkpoint `backend/models/best_model.pt` (~69 MB) is required; fetch with `python backend/scripts/download_checkpoint.py` (from HF `honi05/deepfake-detection`). Path is resolved relative to `app/config.py` (`MODEL_PATH`), so it is stable regardless of CWD. Checkpoint loads with `strict=True`; the classifier head must stay `Dropout(0.4) → Linear(1792,256) → ReLU → Dropout(0.2) → Linear(256,1)` or load fails.
- Uploaded videos are saved to `<cwd>/uploads` via `os.getcwd()` (`app/main.py`), **not** the `UPLOAD_DIR` constant in `config.py`. Where files land depends on where uvicorn was started; `os.getcwd()`-relative saves break if the server runs from `backend/`.
- All analyses live in the in-memory `_ANALYSES` dict — restarting the server wipes history and `/dashboard/stats` falls back to `source: "demo"`.
- Pipeline: uniform 15-frame sample → MTCNN detects faces (largest crop, 20px margin, 224×224) → EfficientNet-B4 sigmoid (per-frame fake probability) → aggregate mean/median/std. Classification: mean ≤ 0.40 REAL, ≥ 0.60 FAKE, else UNCERTAIN. No faces found → HTTP 400 `NO_FACE_DETECTED`; internally `analyze_video` still returns a `result: "NO_FACE"` payload with `rppg.status: "SKIPPED"`.
- rPPG (CHROM, de Haan & Jeanne 2013) is implemented in `backend/app/rppg.py` (pure numpy: RBJ Butterworth bandpass 0.8–3.0 Hz, polyfit detrend, FFT heart-rate). Runs on a contiguous ~10 s window (max 300 frames, min 60 usable face frames) using MTCNN boxes directly (not crops); forehead + cheek ROI. Returns `status: AVAILABLE|UNAVAILABLE|SKIPPED` with honest explanations, `heart_rate_bpm`, `signal_quality`, and downsampled signal/spectrum arrays. It is **diagnostic evidence only** — never blended into the verdict. Adds ~30–90 s CPU time; `/upload` `processing_time` includes it.
- Class mapping is verified by `backend/scripts/verify_mapping.py` (`python -m backend.scripts.verify_mapping --validate`): sigmoid = fake probability, ≥ 0.5 = FAKE per the checkpoint's model card; strict load with 0 missing/unexpected keys. No labelled FAKE video exists in the repo, so fake-class accuracy cannot be proven — the script reports nulls honestly.
- `/model/info` returns `analysis_components` (efficientnet-b4 = weighted, chrom-rppg = diagnostic-only) and `verdict_note`; `/dashboard/stats` now breaks out `uncertain_detected` and `no_face_detected` alongside real/fake.
- Device: `_get_device()` uses CUDA if available, else CPU (inference runs in a threadpool via `run_in_threadpool`).

## Frontend

- Backend URL is hardcoded `http://127.0.0.1:8000` in `Analysis.tsx`, `Dashboard.tsx`, `ModelInfo.tsx` — no Vite proxy or env config. Backend must run on port 8000.
- `/results` receives the upload payload via react-router state (`navigate('/results', { state: { upload } })`); a direct reload of `/results` shows the "no analysis data" empty state.
- 10 routes render under `DefaultLayout` in `App.tsx`; Tailwind v4 via `@tailwindcss/postcss`.
