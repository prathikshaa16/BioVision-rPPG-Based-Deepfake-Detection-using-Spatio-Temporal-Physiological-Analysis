# BioVision: rPPG-Based Deepfake Detection using Spatio-Temporal Deep Learning and Physiological Signal Analysis

BioVision combines spatial forensic cues from an EfficientNet-B4 face classifier
with temporal physiological evidence from CHROM remote photoplethysmography
(rPPG). The existing visual checkpoint is preserved, and a transparent,
quality-gated late-fusion layer combines its probability with the rPPG anomaly
score when a reliable pulse signal is available.

## Features
- quality-gated EfficientNet-B4 + CHROM rPPG late fusion
- spatio-temporal face and physiological signal analysis
- Real-time video analysis
- Dashboard for visualization
- Model verification tools

## Tech Stack
- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **ML Models**: PyTorch EfficientNet-B4, CHROM rPPG signal processing

## Why the title is accurate

- **Spatio-temporal deep learning**: MTCNN isolates faces across sampled video
	frames and EfficientNet-B4 learns spatial forensic features per frame; frame
	probabilities are aggregated over the video.
- **Physiological signal analysis**: CHROM extracts a pulse signal from forehead
	and cheek regions, filters it in the physiological band, and estimates heart
	rate and signal quality with FFT analysis.
- **Fusion**: available rPPG evidence contributes up to 20% of the final score,
	while the trained visual model contributes 80%. If rPPG is unavailable, the
	system falls back to the visual probability and reports that fact.

The fusion is intentionally heuristic because the supplied checkpoint was not
trained with physiological features. A learned multimodal fusion head would
require paired labeled video/rPPG training data and a new checkpoint.

## Training the full architecture

The trainable implementation is in `backend/app/multimodal_model.py`, with a
manifest-based trainer in `backend/scripts/train_multimodal.py`. Each JSONL
record must reference precomputed arrays for all required modalities:
`visual [T,3,224,224]`, `rppg [N]`, `mfcc [T,40]`, and `mouth [T,40]`, plus a
binary `label`. Train only after producing video-disjoint Celeb-DF v2 and
separate DFDC manifests:

```bash
python backend/scripts/train_multimodal.py --train-manifest data/celebdf_train.jsonl --val-manifest data/celebdf_val.jsonl
```

The script saves the best validation-loss checkpoint and reports accuracy for
each epoch. It does not claim the paper's 95.5% metrics; those require the
specified datasets, split manifest, and an independent evaluation run.

For the extracted archive at `C:\BioVision\Celeb-DF-v2`, create raw-video
manifests with:

```bash
python backend/scripts/make_celebdf_manifest.py C:\BioVision\Celeb-DF-v2 --output C:\BioVision\manifests\celebdf
```

This preserves the official 518-video test list and creates deterministic
train/validation files from the remaining videos. Extract the required
multimodal arrays with the feature extractor, which needs `librosa`,
`soundfile`, `mediapipe`, a downloaded MediaPipe Face Landmarker task model,
and either `ffmpeg` on `PATH` or the bundled
`imageio-ffmpeg` fallback:

```bash
python backend/scripts/extract_multimodal_features.py --manifest-dir C:\BioVision\manifests\celebdf --output-dir C:\BioVision\features\celebdf --face-landmarker-model C:\BioVision\face_landmarker.task --split train --limit 20
```

Remove `--limit 20` only after checking a small run. Videos missing a usable
face sequence, audio track, mouth landmarks, or CHROM-rPPG signal are rejected
and counted rather than zero-filled. The resulting feature manifests can be
passed to `train_multimodal.py`.

Extraction writes each accepted record immediately and supports interruption:
rerun the same command with `--resume` to skip completed videos. Rejected
videos and reasons are recorded in `<split>.rejected.jsonl`.

Celeb-DF-v2 has no audio stream in the inspected sample, so use the explicit
audio-free fallback for this dataset:

```bash
python backend/scripts/extract_multimodal_features.py --manifest-dir C:\BioVision\manifests\celebdf --output-dir C:\BioVision\features\celebdf --face-landmarker-model C:\BioVision\face_landmarker.task --split train --mode visual-rppg
python backend/scripts/train_multimodal.py --train-manifest C:\BioVision\features\celebdf\train.jsonl --val-manifest C:\BioVision\features\celebdf\val.jsonl --mode visual-rppg --output C:\BioVision\models\biovision_visual_rppg.pt
```

This fallback is a 320-feature model (256 visual-temporal + 64 rPPG). The
paper-faithful 448-feature model remains available for an audio-bearing
dataset and requires `--mode full`.

Evaluate the held-out Celeb-DF-v2 test features after training:

```bash
python backend/scripts/evaluate_multimodal.py --manifest C:\BioVision\features\celebdf\test.jsonl --checkpoint C:\BioVision\models\biovision_visual_rppg.pt --mode visual-rppg
```

The Colab cached-feature baseline has a separate checkpoint contract. Before
copying its `biovision_best.pt` into a deployment environment, verify it with:

```bash
python -m backend.scripts.verify_notebook_checkpoint C:\path\to\biovision_best.pt
```

This checks the `[32,1792]` visual embedding sequence, `[240]` rPPG input,
320-feature fusion head, and strict state-dict compatibility.

## Installation
```bash
# Clone the repository
git clone https://github.com/suba-6/BioVision-rPPG-Based-deepfake-detection-system.git

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
