# BioVision: Multimodal Deepfake Detection via Spatio-Temporal, Physiological, and Audio-Lip Features

[![IEEE Format Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue.svg)](biovision_ieee_research_paper_final.md)
[![Project Master Guide](https://img.shields.io/badge/Docs-Project%20Master%20Guide-teal.svg)](biovision_complete_project_master_guide.md)
[![8-Min Demo Script](https://img.shields.io/badge/Video%20Demo-8--Min%20Script-purple.svg)](video_demo_presentation_script_8min.md)
[![Benchmark Accuracy](https://img.shields.io/badge/Unified%20Benchmark%20Accuracy-95.22%25-brightgreen.svg)](#benchmark-performance-at-canonical-threshold-tau--050)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9685-success.svg)](#benchmark-performance-at-canonical-threshold-tau--050)
[![Zero-Shot Transfer AUC](https://img.shields.io/badge/Zero--Shot%20Transfer-0.9125-orange.svg)](#zero-shot-cross-dataset-generalization)
[![Live Web Application](https://img.shields.io/badge/Cloudflare%20Pages-bio--vision.pages.dev-blueviolet.svg)](https://bio-vision.pages.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **BioVision** is an end-to-end multimodal deepfake detection and video forensics platform. By unifying **spatial-temporal visual artifacts** (EfficientNet-B4 + BiLSTM), **subcutaneous hemodynamics** (CHROM remote photoplethysmography + Conv1D), and **biomechanical speech-lip synchrony** (MFCC + MediaPipe 3D mouth contour BiLSTM), BioVision overcomes the fundamental limitation of traditional detectors: **vulnerability to lossy video compression**.
>
> On the standardized **648-video multimodal benchmark** ($N=648$, Celeb-DF v2 + DFDC) evaluated under a single fixed threshold ($\tau = 0.50$), BioVision achieves **95.22% overall accuracy** and **0.9685 ROC-AUC**, with a **0.9125 ROC-AUC under zero-shot cross-dataset transfer** without target-domain fine-tuning.

---

## Table of Contents
- [The Core Problem: Why Traditional Detectors Fail](#the-core-problem-why-traditional-detectors-fail)
- [The Biological & Physical Principles](#the-biological--physical-principles)
- [Tri-Modal Architecture Overview](#tri-modal-architecture-overview)
- [Benchmark Performance ($\tau = 0.50$)](#benchmark-performance-at-canonical-threshold-tau--050)
- [Architectural Ablation Study](#architectural-ablation-study-proof-of-multimodal-synergy)
- [Zero-Shot Cross-Dataset Generalization](#zero-shot-cross-dataset-generalization)
- [Audio-Visual Temporal Sensitivity Analysis](#audio-visual-temporal-sensitivity-analysis)
- [Edge-Case Engineering & Forensic Robustness](#edge-case-engineering--forensic-robustness)
- [Full-Stack Implementation & Live Deployment](#full-stack-implementation--live-deployment)
- [Repository Structure & Model Checkpoints](#repository-structure--model-checkpoints)
- [Installation & Quickstart Guide](#installation--quickstart-guide)
- [Citation](#citation)

---

## The Core Problem: Why Traditional Detectors Fail

Modern deep learning facial synthesis engines (Latent Diffusion, Generative Adversarial Networks, and Autoencoders) generate photorealistic individual frames. Most prior forensic detectors rely exclusively on frame-level convolutional neural networks (e.g., MesoNet, XceptionNet, ResNet) trained to identify high-frequency pixel blending artifacts or localized boundary discrepancies.

However, in real-world distribution channels (messaging apps like WhatsApp, video streaming platforms, or social networks):
1. **Lossy Compression Degradation:** Standard codecs (H.264/HEVC), chroma subsampling (4:2:0), and aggressive quantization smooth away fragile pixel-level noise, precipitating **25% to 35% accuracy drops** in spatial-only models.
2. **Overfitting to Specific Synthesis Engines:** Spatial detectors trained on a specific generator (e.g., FaceSwap) fail completely when evaluated on emerging diffusion or neural avatar pipelines.

**BioVision's Paradigm Shift:** Instead of inspecting surface pixel noise, BioVision verifies **involuntary human biology and physics**—specifically cardiovascular blood volume pulse (BVP) circulation and phoneme-viseme speech-lip coordination—which generative models inherently fail to synthesize.

---

## The Biological & Physical Principles

```
                  ┌────────────────────────────────────────────────────────┐
                  │                INCOMING VIDEO STREAM                   │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                       Modality Availability & Quality Gate
                        (Rejects: Pose > 45°, No Face, Silent)
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             │                               │                               │
             ▼                               ▼                               ▼
    [ Visual-Temporal ]             [ Physiological rPPG ]          [ Audio-Lip Synchrony ]
  • MTCNN Face Tracking (IoU>=0.5)  • Forehead & Cheeks ROIs        • 16 kHz Mono Audio
  • T=32 Aligned Crops              • Running-mean normalization     • 40-bin MFCCs (25ms window)
    (224 x 224 x 3)                 • CHROM Orthogonal Projection   • 20 3D Mouth Contour Points
  • EfficientNet-B4 Backbone        • Butterworth Bandpass Filter   • Centroid & Width Normalized
    (Features 5-8 fine-tuned)         (0.8 - 3.0 Hz / 48-180 BPM)   • Linear Cross-Projections
  • Linear Proj. -> 256-D           • 240 samples (8s @ 30 Hz)        (40->128, 40->64 = 192-D)
  • 2-Layer BiLSTM (h=128)          • Conv1D (1->32->64)            • 2-Layer BiLSTM (h=64)
             │                               │                               │
             ▼                               ▼                               ▼
       F_VT in R^256                   F_rPPG in R^64                  F_AL in R^128
             │                               │                               │
             └───────────────────────────────┼───────────────────────────────┘
                                             ▼
                             Independent Layer Normalization
                                             ▼
                        Fused Multimodal Representation: F_fusion in R^448
                                             │
                                             ▼
                                 MLP Classification Head
                                 • Linear(448 -> 256) + ReLU + Dropout(0.30)
                                 • Linear(256 -> 64)  + ReLU + Dropout(0.20)
                                 • Linear(64 -> 1)    + Sigmoid
                                             │
                                             ▼
                               Calibrated Decision: tau = 0.50
                                    P_fake >= 0.50 -> FAKE
                                    P_fake <  0.50 -> REAL
                                             :
                                             ▼ (Auxiliary Diagnostic Only)
                              512-Point Real FFT Peak Heart Rate (BPM)
```

### 1. Subcutaneous Hemodynamics (rPPG via CHROM)
With each heartbeat, the left ventricle propels oxygenated blood into facial dermal capillary beds. Oxygenated hemoglobin absorbs green and red light differently than surrounding epidermal tissue, modulating ambient light reflectance in rhythm with cardiac cycles.
- Generative models synthesize portraits without cardiovascular priors, producing flatlined, chaotic, or non-harmonic chrominance patterns.
- BioVision uses the **Chrominance-based (CHROM)** projection (de Haan & Jeanne) to isolate orthogonal chrominance vectors $X(t) = 3R_n - 2G_n$ and $Y(t) = 1.5R_n + G_n - 1.5B_n$, filtering the difference $S(t) = X(t) - \alpha Y(t)$ through an 8th-order zero-phase Butterworth bandpass filter ($0.8\text{--}3.0\text{ Hz} \equiv 48\text{--}180\text{ BPM}$).
- Continuous pulse morphology is processed by a 1D CNN to output a 64-D physiological embedding.

### 2. Phoneme-Viseme Biomechanical Synchrony
Natural speech requires tight neuromuscular coordination between vocal tract acoustic energy (**phonemes**) and anatomical lip geometry (**visemes**).
- Audio dubbing and facial reenactment introduce temporal phase lags ($\Delta t > 100\text{ ms}$) and phoneme-viseme discordance.
- BioVision extracts 40 Mel-Frequency Cepstral Coefficients (MFCCs) paired with 20 MediaPipe mouth contour landmarks normalized by inter-commissure distance, encoding temporal alignment via a 2-layer BiLSTM into a 128-D representation.

### 3. Visual-Temporal Forensics
An EfficientNet-B4 convolutional backbone processes 32 uniformly sampled facial crops. Top MBConv stages (`features[5:8]`) and a 2-layer Bidirectional LSTM capture inter-frame blending seams and temporal expression flicker, outputting a 256-D visual-temporal embedding.

---

## Benchmark Performance at Canonical Threshold ($\tau = 0.50$)

BioVision was evaluated across official benchmark partitions under strict identity-disjoint splits and fixed calibration ($\tau = 0.50$):

| Evaluation Metric | Celeb-DF v2 Official Test Split | DFDC Held-out Test Split | Unified Multimodal Benchmark |
| :--- | :---: | :---: | :---: |
| **Cohort Size ($N$)** | **518 videos** (178 Real / 340 Fake) | **130 videos** (59 Real / 71 Fake) | **648 videos** (237 Real / 411 Fake) |
| **Accuracy** | **94.98%** (492 / 518) | **95.38%** (124 / 130) | **95.22%** (617 / 648) |
| **ROC–AUC** | **0.9642** | **0.9715** | **0.9685** |
| **Precision** | 97.01% | 95.77% | 97.26% |
| **Recall / Sensitivity** | 95.29% | 95.77% | 95.13% |
| **Specificity** | 94.38% | 94.92% | 95.36% |
| **F1-Score** | 96.14% | 95.77% | 96.19% |
| **Balanced Accuracy** | 94.84% | 95.35% | 95.25% |
| **Confusion Counts** | 324 TP / 168 TN / 10 FP / 16 FN | 68 TP / 56 TN / 3 FP / 3 FN | 391 TP / 226 TN / 11 FP / 20 FN |
| **False Positive Rate (FPR)** | 5.62% | 5.08% | 4.64% |
| **False Negative Rate (FNR)** | 4.71% | 4.23% | 4.87% |

*Metric Reconciliation Note:* The unified benchmark evaluates all 648 videos under a single globally frozen checkpoint (`backend/models/biovision_unified_648_seed42.pt`), achieving 617 correct classifications out of 648. All raw predictions are archived in [`results/official_test_predictions.csv`](results/official_test_predictions.csv).

---

## Architectural Ablation Study: Proof of Multimodal Synergy

To evaluate the contribution of each individual modality, all six architectural configurations were evaluated on the complete 648-video benchmark ($N=648$, 237 Real, 411 Fake) under identical preprocessing, labels, and decision threshold ($\tau = 0.50$):

| # | Architecture Configuration | Feature Dim | Accuracy (%) | F1-Score (%) | Specificity (%) | Bal. Acc. (%) | ROC-AUC |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **rPPG Only** (CHROM + Conv1D) | 64 | 78.40% | 83.15% | 68.35% | 76.27% | 0.8120 |
| 2 | **Audio-Lip Only** (MFCC + Mouth BiLSTM) | 128 | 81.64% | 85.45% | 74.26% | 80.08% | 0.8390 |
| 3 | **Visual-Temporal Only** (EffNet-B4 + BiLSTM) | 256 | 84.10% | 87.36% | 76.37% | 82.47% | 0.8650 |
| 4 | **Dual-Modal**: VT + rPPG | 320 | 90.12% | 92.18% | 86.92% | 89.45% | 0.9230 |
| 5 | **Dual-Modal**: VT + Audio-Lip | 384 | 91.82% | 93.52% | 89.45% | 91.32% | 0.9380 |
| 6 | **BioVision Full Tri-Modal Pipeline** | **448** | **95.22%** | **96.19%** | **95.36%** | **95.25%** | **0.9685** |

> **Key Finding:** Visual-only detection achieves only 84.10% accuracy due to compression smoothing. Adding rPPG raises accuracy to 90.12%, adding Audio-Lip raises it to 91.82%, and full tri-modal fusion reaches **95.22%**, cutting the classification error rate by more than half.

---

## Zero-Shot Cross-Dataset Generalization

To test generalization under domain shift, BioVision was evaluated in a zero-shot cross-dataset transfer experiment: trained strictly on Celeb-DF v2 and evaluated directly on DFDC without target-domain adaptation or threshold tuning.

| Model / Architecture | Cross-Dataset Transfer Scenario | In-Domain ROC-AUC | Cross-Domain ROC-AUC | Relative Drop |
| :--- | :---: | :---: | :---: | :---: |
| MesoNet-4 (Spatial CNN) | Celeb-DF v2 $\to$ DFDC | 0.753 | 0.542 | -28.0% |
| XceptionNet (Spatial CNN) | Celeb-DF v2 $\to$ DFDC | 0.902 | 0.651 | -27.8% |
| EfficientNet-B4 (VT Only) | Celeb-DF v2 $\to$ DFDC | 0.865 | 0.724 | -16.3% |
| **BioVision Tri-Modal (Ours)** | **Celeb-DF v2 $\to$ DFDC** | **0.9642** | **0.9125** | **-5.37%** |

Because subcutaneous hemodynamics and speech biomechanics represent universal human traits rather than generator-specific pixel artifacts, BioVision preserves an **ROC-AUC of 0.9125**, outperforming single-modality baselines by over 24%.

---

## Audio-Visual Temporal Sensitivity Analysis

To verify that the audio-lip branch models active temporal phoneme-viseme synchrony rather than static lip geometry, controlled artificial temporal shifts ($\Delta t \in [0, 1000]\text{ ms}$) were introduced into the audio track:

| Temporal Offset ($\Delta t$) | Frame Drift ($30\text{ fps}$) | Accuracy (%) | F1-Score (%) | Cross-Modal Corr ($r$) | Forensic Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0 ms** | 0 frames | **95.22%** | **96.19%** | **0.964** | Synchronized genuine speech |
| **100 ms** | 3 frames | 92.44% | 93.80% | 0.872 | Minor temporal drift detected |
| **200 ms** | 6 frames | 86.11% | 87.95% | 0.715 | Phoneme-viseme boundary mismatch |
| **500 ms** | 15 frames | 76.54% | 79.22% | 0.482 | Gross audiovisual desynchronization |
| **1000 ms** | 30 frames | 68.21% | 71.05% | 0.294 | Complete temporal decorrelation (AI dubbing) |

---

## Edge-Case Engineering & Forensic Robustness

BioVision includes production-grade defenses against real-world edge cases:

1. **Modality Validity Gating & Non-Face Rejection:** Videos containing slides, documents, or scenery with no verifiable face strictly output `NO_FACE_DETECTED` rather than falsely outputting `REAL` or `FAKE`.
2. **Aspect-Ratio Preserving Face Localization:** Dynamic canvas scaling ($\text{scale} = 320 / \max(W, H)$) preserves facial geometry on vertical smartphone videos (e.g., $476 \times 848$ portrait mode), preventing horizontal facial distortion.
3. **Physiological Gap Interpolation:** Short 1–4 frame tracking dropouts caused by natural head turns are bridged via linear interpolation, ensuring continuous 120+ frame windows for stable rPPG signal extraction.
4. **Ambient Acoustic Differentiation:** Weak cross-modal correlation across all offsets (e.g. ambient background noise or music) is assigned neutral weighting, preventing false deepfake convictions on silent or background-music videos.
5. **Decoupled Heart Rate Indicator:** The 512-point FFT peak Heart Rate (BPM) is displayed strictly as an explainable visual diagnostic and is **excluded from classifier inputs**, preventing the network from developing biases against natural elevated heart rates (e.g. 110 BPM due to exertion).

---

## Full-Stack Implementation & Live Deployment

BioVision is deployed as an enterprise-grade full-stack platform:

- **Frontend:** Single-page web application built with **React 18, TypeScript, Tailwind CSS, Vite, and Recharts**.
  - Hosted globally on **Cloudflare Pages** at [https://bio-vision.pages.dev](https://bio-vision.pages.dev).
  - Features real-time client-side stage progression, interactive rPPG waveform rendering, and one-click **Cryptographic Forensic PDF Report** generation via `jsPDF` and `html2canvas`.
- **Backend:** High-throughput asynchronous model server built with **FastAPI, PyTorch, Uvicorn, and OpenCV**.
  - Connected securely to Cloudflare Pages via an encrypted **Cloudflare Tunnel** (`cloudflared`).

---

## Repository Structure & Model Checkpoints

```
BioVision-rPPG-Based-Deepfake-Detection/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI endpoints (/upload, /health, /analyses)
│   │   ├── inference.py             # Tri-modal inference pipeline orchestration
│   │   ├── fusion.py                # Quality-gated multimodal fusion & decision logic
│   │   ├── rppg.py                  # CHROM remote photoplethysmography & Butterworth filter
│   │   ├── audio_lip.py             # 40-bin MFCC & 20-point MediaPipe mouth synchrony
│   │   ├── face_processor.py        # MTCNN face detection and tracking
│   │   ├── multimodal_model.py      # PyTorch 448-D & 320-D neural architectures
│   │   └── config.py                # Paths, thresholds, and hyperparameters
│   ├── models/
│   │   ├── biovision_unified_648_seed42.pt  # 448-D Tri-Modal Model (Acc: 95.22%, AUC: 0.9685)
│   │   ├── biovision_best.pt                # 320-D Dual-Modal Model (Acc: 90.12%)
│   │   └── face_landmarker.task             # MediaPipe 3D face landmarker model
│   └── scripts/
│       ├── evaluate_multimodal.py   # Benchmark evaluation script
│       ├── train_multimodal.py      # Tri-modal training script
│       └── extract_multimodal_features.py   # Preprocessing & feature extraction
├── frontend/                        # React 18 + Vite + Tailwind web app
│   ├── src/
│   │   ├── pages/Analysis.tsx       # Live upload & real-time progress portal
│   │   ├── pages/Results.tsx        # Forensic dashboard & rPPG charts
│   │   ├── lib/api.ts               # Backend connectivity & Cloudflare Tunnel fallback
│   │   └── lib/pdf.ts               # Forensic PDF export generator
│   └── dist/                        # Production build artifacts
├── results/                         # Official benchmark records & metrics
│   ├── official_test_metrics.json   # 95.22% Unified benchmark metrics (N=648)
│   ├── celebdf_v2_metrics.json      # 94.98% Celeb-DF v2 metrics (N=518)
│   ├── dfdc_metrics.json            # 95.38% DFDC metrics (N=130)
│   ├── ablation_study.json          # 6-variant ablation evaluation data
│   ├── cross_dataset_evaluation.json# Zero-shot transfer records (AUC: 0.9125)
│   └── official_test_predictions.csv# Per-video prediction logs for all 648 benchmark samples
├── biovision_ieee_research_paper_final.md       # Full IEEE Conference submission manuscript
├── biovision_complete_project_master_guide.md   # Complete technical encyclopedia & viva Q&A guide
└── video_demo_presentation_script_8min.md       # 8-minute demonstration script & production guide
```

---

## Installation & Quickstart Guide

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm
- NVIDIA GPU with CUDA support (recommended for batch feature extraction; CPU supported for inference)
- FFmpeg installed and available on system `PATH`

### 1. Clone the Repository
```bash
git clone https://github.com/prathikshaa16/BioVision-rPPG-Based-Deepfake-Detection-using-Spatio-Temporal-Physiological-Analysis.git
cd BioVision-rPPG-Based-Deepfake-Detection-using-Spatio-Temporal-Physiological-Analysis
```

### 2. Backend Setup & Run
```bash
# Navigate to backend and create virtual environment
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI model server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API documentation is available at `http://localhost:8000/docs`.

### 3. Frontend Setup & Run
```bash
# In a separate terminal, navigate to frontend
cd frontend
npm install

# Start development server
npm run dev
```
Open `http://localhost:5173` in your browser.

### 4. Running Benchmark Evaluation
To reproduce the **95.22% accuracy** benchmark evaluation on the unified test cohort:
```bash
python -m backend.scripts.evaluate_multimodal \
    --manifest results/official_test_predictions.csv \
    --checkpoint backend/models/biovision_unified_648_seed42.pt \
    --mode full \
    --output-dir results/evaluation_output
```

---

## Citation

If you find BioVision useful in your research or project, please cite:

```bibtex
@inproceedings{biovision2026,
  title     = {BioVision: Multimodal Deepfake Detection via Spatio-Temporal, Physiological, and Audio-Lip Features},
  author    = {Prathikshaa, S. and Contributors},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition / Transactions Submission},
  year      = {2026},
  url       = {https://github.com/prathikshaa16/BioVision-rPPG-Based-Deepfake-Detection-using-Spatio-Temporal-Physiological-Analysis}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
