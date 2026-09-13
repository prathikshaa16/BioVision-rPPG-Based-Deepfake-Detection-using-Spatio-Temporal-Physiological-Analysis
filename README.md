# BioVision- A rPPG Based Deepfake Detection using Spatio Temporal and Physiological Signal Analysis

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

## Verified Kaggle GPU baseline

The completed Kaggle run is preserved in [kaggle_biovision_gpu.ipynb](kaggle_biovision_gpu.ipynb).
It used the `BioVision Celeb-DF v2` dataset (Kaggle dataset version `19598487`)
on one NVIDIA Tesla T4 with PyTorch `2.10.0+cu128`. The dataset inventory was
590 `Celeb-real`, 5,639 `Celeb-synthesis`, and 300 `YouTube-real` videos.

The official 518-video test list was excluded before feature extraction and
model selection. The development cache contained 5,941 usable records after
the documented repair-only pass. The final identity-disjoint split used 4,933
training records and 1,008 validation records. An earlier split that produced
93 validation records with no REAL samples was rejected and must not be used.

The cached baseline contract is:

- 32 EfficientNet-B4 visual embeddings of shape `[32, 1792]`
- CHROM-rPPG vector padded/truncated to `[240]`
- two-layer LSTM with hidden size 256
- Conv1D rPPG branch projected to 64 dimensions
- 320-dimensional fusion classifier: `320 -> 256 -> 64 -> 1`
- batch size 16, learning rate `1e-4`, weight decay `1e-4`, seed 42
- maximum 30 epochs, patience 6, gradient clipping 1.0

The best validation checkpoint was epoch 14 with validation ROC-AUC
`0.735875`; early stopping ended training at epoch 20. The untouched official
test evaluation reported ROC-AUC `0.717184`, accuracy `0.741313`, precision
`0.729911`, recall `0.961765`, specificity `0.320225`, balanced accuracy
`0.640995`, F1 `0.829949`, and TP/TN/FP/FN `327/57/121/13`.

These values are historical artifacts from the Kaggle run, not targets to
optimize against. Preserve the notebook, checkpoint, training log, raw
predictions, and evaluation files together. The second dataset currently being
uploaded to Kaggle must receive its own manifest, provenance, split audit, and
experiment directory; do not merge it into this baseline or change a result
until the new data has passed the same checks.

Before connecting the trained model to the webpage, run
`backend/scripts/prepare_kaggle_release.py` inside the Kaggle notebook after
the training and official evaluation cells. It collects the checkpoint,
training history, split/validation reports, raw test predictions, ROC points,
confusion matrix, and final metrics under `/kaggle/working/BioVision_Release`.
Publish that directory as notebook output and keep it paired with the notebook
version that produced it.

### Next experiment: balanced cached fusion

The baseline training split is strongly fake-heavy, while validation has a
different class ratio. The isolated experiment trainer
`backend/scripts/train_cached_fusion.py` uses the same cached `[32,1792]` plus
`[240]` architecture with a weighted sampler, gradient clipping, validation
model selection, and epoch-by-epoch metrics. Run it in the Kaggle GPU runtime
against the existing identity-disjoint development files:

```bash
python backend/scripts/train_cached_fusion.py \
	--train /kaggle/working/BioVision_Final_Cache/identity_aware_split/train_records.pt \
	--val /kaggle/working/BioVision_Final_Cache/identity_aware_split/val_records.pt \
	--output /kaggle/working/BioVision_Balanced_Experiment/biovision_balanced.pt \
	--epochs 30 --batch-size 16 --lr 1e-4 --seed 42
```

This is a new experiment, not a replacement for the baseline. Do not evaluate
the official test set until the validation history has been inspected. A 94%
result is accepted only if it comes from the actual held-out run and its raw
predictions and metrics are preserved.

## Training the full architecture

The trainable implementation is in `backend/app/multimodal_model.py`, with a
manifest-based trainer in `backend/scripts/train_multimodal.py`. Keep the
official test manifest out of training and model selection. Each JSONL record
must reference precomputed arrays for all required modalities:
`visual [T,3,224,224]`, `rppg [N]`, `mfcc [T,40]`, and `mouth [T,40]`, plus a
binary `label`. Train only after producing video-disjoint Celeb-DF v2 and
separate DFDC manifests:

```bash
python backend/scripts/train_multimodal.py --train-manifest data/celebdf_train.jsonl --val-manifest data/celebdf_val.jsonl --epochs 3 --output results/pilot/biovision_visual_rppg.pt
```

The pilot writes `results/pilot/biovision_visual_rppg.history.json` with the
device/GPU, split sizes and class counts, seed, batch size, learning rate,
elapsed time, and epoch-by-epoch train/validation loss, accuracy, precision,
recall, F1, balanced accuracy, and ROC-AUC. It saves the best validation
checkpoint. The script does not claim the paper's 95.5% metrics; those require
the specified datasets, split manifest, and an independent evaluation run.

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

Pass `--output-dir results/final_evaluation` to save raw predictions, ROC
points, a confusion matrix, and metrics including accuracy, precision, recall,
F1, specificity, balanced accuracy, ROC-AUC, and TP/TN/FP/FN. Select thresholds
using validation data only; do not tune against the official test manifest.

The reproducibility order for p-v16 is: run a small extraction, run the GPU
pilot, inspect the saved train/validation curves and split class counts, check
for video-disjoint manifests and duplicate features, then run the held-out
test evaluation. Preserve every checkpoint, configuration, and raw artifact;
never replace a result because a later run looks better.

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
