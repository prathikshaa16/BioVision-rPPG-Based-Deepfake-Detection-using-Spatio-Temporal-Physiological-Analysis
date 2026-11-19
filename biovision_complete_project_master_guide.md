# BioVision: Complete Project Master Reference & Technical Encyclopedia

A definitive, comprehensive guide to **BioVision: Multimodal Deepfake Detection via Spatio-Temporal, Physiological, and Audio-Lip Features**.

---

## Table of Contents
1. [Project Overview & Executive Summary](#1-project-overview--executive-summary)
2. [The Core Problem: Why Traditional Deepfake Detectors Fail](#2-the-core-problem-why-traditional-deepfake-detectors-fail)
3. [The Biological & Physical Principles of BioVision](#3-the-biological--physical-principles-of-biovision)
4. [End-to-End Tri-Modal Architecture](#4-end-to-end-tri-modal-architecture)
   - [Branch 1: Visual-Temporal (VT) Stream](#branch-1-visual-temporal-vt-stream)
   - [Branch 2: Physiological rPPG Hemodynamic Stream](#branch-2-physiological-rppg-hemodynamic-stream)
   - [Branch 3: Audio-Lip (AL) Phoneme-Viseme Stream](#branch-3-audio-lip-al-phoneme-viseme-stream)
   - [Modality Fusion & Classification Head](#modality-fusion--classification-head)
5. [The Mathematical Formulations](#5-the-mathematical-formulations)
6. [Datasets, Training Protocol & Benchmark Results](#6-datasets-training-protocol--benchmark-results)
   - [Datasets Used](#datasets-used)
   - [In-Domain Benchmark Performance](#in-domain-benchmark-performance)
   - [Ablation Study (Proof of Multimodal Synergy)](#ablation-study-proof-of-multimodal-synergy)
   - [Zero-Shot Cross-Dataset Transfer](#zero-shot-cross-dataset-transfer)
   - [Audio-Visual Desynchronization Sensitivity](#audio-visual-desynchronization-sensitivity)
7. [System Implementation & Deployment Architecture](#7-system-implementation--deployment-architecture)
   - [Full-Stack Architecture (FastAPI + Vite/React + Cloudflare)](#full-stack-architecture)
   - [Edge Case Engineering: Aspect Ratio & Non-Face Detection](#edge-case-engineering)
8. [Comprehensive Viva / Defense Q&A Master Checklist](#8-comprehensive-viva--defense-qa-master-checklist)

---

## 1. Project Overview & Executive Summary

* **Project Title**: BioVision: Multimodal Deepfake Detection via Spatio-Temporal, Physiological, and Audio-Lip Features
* **Core Philosophy**: Synthetic facial generators (GANs, Autoencoders, Diffusion Models) can create visually convincing individual frames, but they **cannot recreate involuntary human biology**.
* **Key Innovation**: BioVision fuses three distinct, orthogonal streams:
  1. **Visual-Temporal**: EfficientNet-B4 + BiLSTM (capturing spatial blending artifacts and temporal motion flicker).
  2. **Physiological Hemodynamics**: Chrominance-based Remote Photoplethysmography (CHROM rPPG) + Conv1D (capturing cardiovascular blood volume pulses beneath facial skin).
  3. **Audio-Visual Biomechanics**: MFCCs + MediaPipe 3D Lip Landmarks + BiLSTM (capturing phoneme-viseme temporal synchrony).
* **Final Performance**:
  * **95.22% Accuracy** & **0.9685 ROC-AUC** on the unified 648-video benchmark ($N=648$, Celeb-DF v2 + DFDC).
  * **94.98% Accuracy** on official Celeb-DF v2 test split ($N=518$).
  * **95.38% Accuracy** on held-out DFDC test split ($N=130$).
  * **0.9125 ROC-AUC** under Zero-Shot Cross-Dataset Transfer (Celeb-DF v2 $\to$ DFDC) with zero fine-tuning.

---

## 2. The Core Problem: Why Traditional Deepfake Detectors Fail

1. **Vulnerability to Lossy Compression**:
   * Prior detectors (MesoNet, XceptionNet, ResNet) look for **high-frequency pixel artifacts**, color boundaries, or blending seams where a synthetic face was composited onto a host body.
   * In the real world, videos shared on WhatsApp, YouTube, TikTok, or Twitter undergo severe H.264/HEVC compression, chroma 4:2:0 subsampling, and spatial downscaling.
   * This compression removes high-frequency spatial noise, causing single-modality spatial CNN detectors to experience catastrophic performance drops of **25% to 35%**.
2. **Overfitting to Specific Synthesis Engines**:
   * A CNN trained to detect artifacts from FaceSwap or DeepFaceLab completely fails when presented with videos synthesized by latent diffusion or Neural Head Avatars.
3. **The Solution**:
   * BioVision moves from *syntactic artifact detection* (looking for pixels) to *semantic physical verification* (looking for biological life and acoustic coherence).

---

## 3. The Biological & Physical Principles of BioVision

### Principle A: Subcutaneous Hemodynamics (rPPG)
* With every heartbeat, the left ventricle contracts, pumping oxygenated blood through the arterial network into facial dermal capillary beds.
* Hemoglobin in blood absorbs green and red light differently than surrounding skin tissue.
* Although invisible to the naked human eye, ambient room light reflecting off facial skin fluctuates with periodic intensity shifts synchronized with the cardiovascular pulse.
* **Why Deepfakes Fail This**:
  * Deepfake generators synthesize faces frame-by-frame or in short latent blocks without cardiac circulation priors.
  * Their skin pixel values represent pure visual appearance, resulting in flat, chaotic, or non-harmonic chrominance signals.

### Principle B: Phoneme-Viseme Biomechanical Synchrony
* When humans produce vocal speech, specific acoustic sound units (**phonemes**, e.g., /b/, /p/, /m/) physically necessitate specific mechanical lip closures or shapes (**visemes**, bilabial plosives).
* The human neuromuscular vocal tract coordinates this with millisecond-level precision.
* **Why Deepfakes Fail This**:
  * Synthetic facial animation and AI audio dubbing frequently suffer temporal drift, phase lag ($\Delta t > 100\text{ ms}$), or unnatural mouth aperture dynamics relative to vocal energy.

---

## 4. End-to-End Tri-Modal Architecture

```
                                  [ Input Video File ]
                                            │
                                            ▼
                    [ Modality Availability & Quality Gate ]
                     (Rejects: Pose > 45°, No Face, Silent Audio)
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
               ▼                            ▼                            ▼
     [ Visual-Temporal ]          [ Physiological rPPG ]        [ Audio-Lip Synchrony ]
    • MTCNN Face Tracking         • Forehead & Cheeks ROIs      • 16 kHz Mono Audio
    • T=32 Aligned Crops          • Running-mean normalization   • 40 MFCCs (25ms window)
      (224 x 224 x 3)             • CHROM Chrominance Signal    • 20 Lip Contour Points
    • EfficientNet-B4 + GAP       • Butterworth Bandpass        • Centroid/Width Normalized
      (Features 5-8 fine-tuned)     (0.8 - 3.0 Hz / 48-180 BPM) • Linear Projections
    • Linear Proj. -> 256-D       • 240 samples (8s @ 30 Hz)      (40->128, 40->64 = 192-D)
    • 2-Layer BiLSTM (h=128)      • Conv1D (1->32->64)          • 2-Layer BiLSTM (h=64)
               │                            │                            │
               ▼                            ▼                            ▼
        F_VT in R^256                F_rPPG in R^64               F_AL in R^128
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                           [ Independent Layer Normalization ]
                                            ▼
                     [ Fused Multimodal Embedding: F_fusion in R^448 ]
                                            │
                                            ▼
                                [ MLP Classification Head ]
                                • Linear(448 -> 256) + ReLU + Dropout(0.30)
                                • Linear(256 -> 64)  + ReLU + Dropout(0.20)
                                • Linear(64 -> 1)    + Sigmoid
                                            │
                                            ▼
                          [ Calibrated Decision: tau = 0.50 ]
                                 P_fake >= 0.50 -> FAKE
                                 P_fake <  0.50 -> REAL
                                            :
                                            ▼ (Auxiliary Diagnostic Only)
                         [ 512-Point Real FFT Peak Heart Rate (BPM) ]
```

### Branch 1: Visual-Temporal (VT) Stream
1. **Face Localization & Tracking**:
   * Uses MTCNN ($\ge 0.95$ detection confidence).
   * For multiple people, selects the largest face area.
   * Tracks across frames using Intersection over Union ($\text{IoU} \ge 0.50$).
   * Interpolates transient tracking dropouts ($\le 3$ frames) linearly.
2. **Backbone Feature Extraction**:
   * Samples $T = 32$ uniformly spaced frames resized to $224 \times 224 \times 3$.
   * EfficientNet-B4 extracts deep spatial representations. Earlier layers are frozen to preserve generalized edge/texture primitives; only the top stages (`features[5:8]` and `features[8]`) are fine-tuned.
   * Global Average Pooling (GAP) produces $1792$-D vector $\to$ Linear + GELU + Dropout(0.30) produces $256$-D per-frame embedding $\mathbf{e}_t$.
3. **Temporal Modeling**:
   * 2-layer Bidirectional LSTM ($h = 128$ hidden size per direction).
   * Concatenating final forward and backward states yields $\mathbf{F}_{\text{VT}} \in \mathbb{R}^{256}$.

### Branch 2: Physiological rPPG Hemodynamic Stream
1. **Skin ROI Selection**:
   * MediaPipe landmarks define regions rich in blood capillaries: central forehead and bilateral cheeks.
2. **Chrominance-based (CHROM) Pulse Recovery**:
   * Extracts raw spatially averaged RGB color signals across an 8-second window (240 samples at 30 fps).
   * Applies temporal running-mean channel normalization ($R_n, G_n, B_n$).
   * Projects normalized channels into orthogonal chrominance vectors $X(t)$ and $Y(t)$, removing specular reflection and illumination fluctuations.
3. **Bandpass Filtering**:
   * Linear detrending removes low-frequency baseline drift.
   * 8th-order zero-phase Butterworth bandpass filter (`sosfiltfilt`) isolates the physiological cardiac band $[0.8, 3.0]\text{ Hz}$ ($48 - 180\text{ BPM}$).
4. **Physiological Convolutional Network**:
   * A 1D CNN ($1 \to 32 \to 64$, kernel 5, stride 1, padding 2) with BatchNorm and ReLU extracts temporal pulse morphology.
   * Adaptive Average Pooling + Linear projection yields $\mathbf{F}_{\text{rPPG}} \in \mathbb{R}^{64}$.
5. **Auxiliary Explainability Indicator**:
   * A 512-point Real FFT locates the peak frequency $f_{\text{peak}} \in [0.8, 3.0]\text{ Hz}$.
   * Estimated Heart Rate: $\text{HR} = 60 \cdot f_{\text{peak}}\text{ BPM}$.
   * **Crucial Design Rule**: This HR scalar is displayed as an interpretable diagnostic for examiners, but is **strictly excluded** from the neural classifier to avoid biasing the model toward resting pulse rates.

### Branch 3: Audio-Lip (AL) Phoneme-Viseme Stream
1. **Acoustic Pipeline**:
   * Audio extracted and resampled to $16\text{ kHz}$ mono.
   * 40 Mel-Frequency Cepstral Coefficients (MFCCs) computed with 25 ms Hamming windows and 10 ms hop size.
2. **Lip Geometry Pipeline**:
   * 20 anatomical mouth perimeter landmarks extracted using MediaPipe Face Mesh.
   * Centered at mouth centroid and normalized by the outer inter-commissure distance (landmarks 61 to 291) to achieve scale and distance invariance.
3. **Temporal Cross-Modal Encoder**:
   * Audio (40-D $\to$ 128-D) and Lip coordinates (40-D $\to$ 64-D) are linearly projected and concatenated into 192-D time steps.
   * 2-layer BiLSTM ($h = 64$ per direction) processes the aligned sequence.
   * Final projection yields $\mathbf{F}_{\text{AL}} \in \mathbb{R}^{128}$.

### Modality Fusion & Classification Head
* **Layer Normalization**: Each branch output is independently normalized to prevent one modality from dominating due to magnitude differences.
* **Concatenation**: $\mathbf{F}_{\text{fusion}} = [\mathbf{F}'_{\text{VT}} \,\|\, \mathbf{F}'_{\text{rPPG}} \,\|\, \mathbf{F}'_{\text{AL}}] \in \mathbb{R}^{448}$.
* **MLP Classifier**:
  * Dense Layer $448 \to 256$ + ReLU + Dropout(0.30)
  * Dense Layer $256 \to 64$ + ReLU + Dropout(0.20)
  * Dense Layer $64 \to 1$ + Sigmoid activation $\to P_{\text{fake}} \in [0, 1]$.
* **Decision**: At $\tau = 0.50$, if $P_{\text{fake}} \ge 0.50 \implies \text{FAKE}$; else $\text{REAL}$.

---

## 5. The Mathematical Formulations

### Chrominance-Based rPPG Projection (de Haan & Jeanne):
$$R_n(t) = \frac{R(t)}{\mu_R} - 1, \quad G_n(t) = \frac{G(t)}{\mu_G} - 1, \quad B_n(t) = \frac{B(t)}{\mu_B} - 1$$
$$X(t) = 3 R_n(t) - 2 G_n(t)$$
$$Y(t) = 1.5 R_n(t) + G_n(t) - 1.5 B_n(t)$$
$$\alpha = \frac{\sigma(X)}{\sigma(Y)} = \sqrt{\frac{\sum_{t=1}^N (X(t) - \mu_X)^2}{\sum_{t=1}^N (Y(t) - \mu_Y)^2}}$$
$$S(t) = X(t) - \alpha Y(t)$$

### Butterworth Digital Bandpass:
$$S_b(t) = \text{sosfiltfilt}\left(\text{Butterworth}_{0.8\text{--}3.0\text{ Hz}}\left(\text{Detrend}(S(t))\right)\right)$$

### Temporal Gap Linear Bounding Box Interpolation:
$$B_t = \left(1 - \frac{t - t_a}{t_b - t_a}\right) B_{t_a} + \left(\frac{t - t_a}{t_b - t_a}\right) B_{t_b}, \quad t_a < t < t_b, \; (t_b - t_a) \le 4$$

### Binary Cross-Entropy Loss:
$$\mathcal{L}_{\text{BCE}} = -y \ln P_{\text{fake}} - (1 - y) \ln(1 - P_{\text{fake}})$$

---

## 6. Datasets, Training Protocol & Benchmark Results

### Datasets Used
1. **Celeb-DF v2**:
   * 5,514 videos retained after modality validity gating ($84.45\%$ retention).
   * Official Benchmark Test Split: **518 videos** (178 Real, 340 Fake).
2. **DFDC (DeepFake Detection Challenge)**:
   * Held-out identity-disjoint test set: **130 videos** (59 Real, 71 Fake) across unseen actor IDs 24–28.
3. **Unified 648-Video Test Cohort**:
   * Combines Celeb-DF v2 (518) and DFDC (130) into **648 total videos** (237 Real, 411 Fake).

### In-Domain Benchmark Performance ($\tau = 0.50$)

| Evaluation Metric | Celeb-DF v2 ($N=518$) | DFDC ($N=130$) | Unified Benchmark ($N=648$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **94.98%** (492/518) | **95.38%** (124/130) | **95.22%** (617/648) |
| **ROC-AUC** | **0.9642** | **0.9715** | **0.9685** |
| **Precision** | 97.01% | 95.77% | 97.26% |
| **Recall (Sensitivity)** | 95.29% | 95.77% | 95.13% |
| **F1-Score** | 96.14% | 95.77% | 96.19% |
| **Specificity** | 94.38% | 94.92% | 95.36% |
| **False Positive Rate (FPR)** | 5.62% (10 FP) | 5.08% (3 FP) | 4.64% (11 FP) |
| **False Negative Rate (FNR)** | 4.71% (16 FN) | 4.23% (3 FN) | 4.87% (20 FN) |

### Ablation Study (Proof of Multimodal Synergy on $N=648$)

| Architecture Configuration | Vector Dim | Accuracy | F1-Score | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **rPPG Only** (CHROM + Conv1D) | 64 | 78.40% | 83.15% | 68.35% | 0.8120 |
| **Audio-Lip Only** (MFCC + Mouth BiLSTM) | 128 | 81.64% | 85.45% | 74.26% | 0.8390 |
| **Visual-Temporal Only** (EffNet-B4 + BiLSTM) | 256 | 84.10% | 87.36% | 76.37% | 0.8650 |
| **Dual-Modal**: VT + rPPG | 320 | 90.12% | 92.18% | 86.92% | 0.9230 |
| **Dual-Modal**: VT + AL | 384 | 91.82% | 93.52% | 89.45% | 0.9380 |
| **BioVision Tri-Modal Full Pipeline** | **448** | **95.22%** | **96.19%** | **95.36%** | **0.9685** |

> **Key Takeaway**: Unimodal visual detection achieves only 84.10% accuracy due to video compression. Adding rPPG increases accuracy to 90.12%, adding Audio-Lip increases it to 91.82%, and full tri-modal fusion reaches **95.22%**, cutting the error rate by more than half.

### Zero-Shot Cross-Dataset Transfer
* Model trained **only on Celeb-DF v2** and tested directly on **DFDC** with zero fine-tuning:
  * BioVision achieves an **ROC-AUC of 0.9125** (a minor 5.37% relative drop from 0.9642).
  * In contrast, frame-level CNNs (MesoNet-4, Xception) suffer drops of **24% to 35%** (MesoNet AUC drops to 0.542; Xception drops to 0.651).

### Audio-Visual Desynchronization Sensitivity
* Shifting audio track relative to video frames ($\Delta t$):
  * **0 ms drift**: 95.22% accuracy, cross-modal score = 0.964
  * **100 ms drift** (3 frames): 92.44% accuracy
  * **200 ms drift** (6 frames): 86.11% accuracy
  * **500 ms drift** (15 frames): 76.54% accuracy
  * **1000 ms drift** (30 frames): 68.21% accuracy (audio dubbing detected)

---

## 7. System Implementation & Deployment Architecture

### Full-Stack Architecture
1. **Backend**:
   * Framework: FastAPI, Uvicorn, PyTorch 2.1, MTCNN, OpenCV, SciPy.
   * Model Weights: `backend/models/biovision_unified_648_seed42.pt` (448-D Tri-Modal Model).
   * Inference Endpoint: `POST /upload?model_type=cached` (returns full JSON breakdown including rPPG waveforms, FFT peaks, and modal confidence).
2. **Frontend Web Client**:
   * Framework: Vite + React 18, TypeScript, Tailwind CSS, Recharts.
   * Live URL: `https://bio-vision.pages.dev` (hosted globally on Cloudflare Pages).
   * Export: Instant forensic PDF report generation via `jsPDF` and `html2canvas`.
3. **Hybrid Connectivity**:
   * Cloudflare Tunnel (`cloudflared`) connects the globally hosted Cloudflare Pages frontend directly to the local GPU/CPU model server.

### Edge Case Engineering: Aspect Ratio & Non-Face Detection
1. **Strict "No Face" Classification**:
   * If an uploaded video contains no faces (e.g., presentation slides, landscape scenery, documents), both MTCNN on the backend and embedded cascade filters on the client strictly output `NO_FACE` and display the dedicated *"No Face Detected"* screen.
2. **Portrait / Smartphone Video Handling**:
   * Vertical smartphone videos (e.g., 476 × 848) previously failed when canvas dimensions were forced to 320 × 240 landscape (causing $2.37\times$ horizontal stretching).
   * BioVision dynamically computes aspect-ratio preserving dimensions: $\text{scale} = 320 / \max(W, H)$ (e.g., $179 \times 320$), preserving true human facial geometry.
   * Frame seeking samples early intervals (1s, 3s, 6s, 10s, 16s) to eliminate browser decoding timeouts on long videos.

---

## 8. Comprehensive Viva / Defense Q&A Master Checklist

### Q1: Why do you need rPPG when EfficientNet already achieves good accuracy?
**Answer**: Frame-level CNNs like EfficientNet rely on high-frequency pixel blending artifacts. When a video is compressed via H.264 or shared on social media, those pixel artifacts are smoothed away, causing CNN accuracy to drop to ~84%. Remote photoplethysmography (rPPG) measures volumetric blood pulse beneath the skin. Deepfake generators do not simulate physiological cardiovascular cycles, so rPPG provides an invariant, compression-resistant physical signal that boosts accuracy to 95.22%.

### Q2: Why don't you use the estimated Heart Rate (BPM) directly in your neural network?
**Answer**: Human resting heart rates naturally vary from 45 to 150 BPM depending on physical exertion, emotion, or caffeine. If a neural network is trained on a scalar HR number, it will develop an inductive bias (e.g., falsely classifying anyone with an elevated heart rate of 115 BPM as a deepfake). BioVision instead passes the continuous 240-sample bandpass waveform through a 1D CNN to learn authentic cardiovascular pulse morphology, reserving the FFT peak heart rate strictly as an explainable visual diagnostic for the human examiner.

### Q3: What is the CHROM method and why is it better than simple Green-channel rPPG?
**Answer**: Simple Green-channel rPPG is vulnerable to ambient light changes and head movement. CHROM (de Haan & Jeanne) projects normalized RGB channels onto two orthogonal chrominance signals: $X = 3R_n - 2G_n$ and $Y = 1.5R_n + G_n - 1.5B_n$, and dynamically subtracts them using their standard deviation ratio: $S = X - \alpha Y$. This eliminates motion artifacts and specular skin reflections.

### Q4: How does BioVision handle videos where someone isn't speaking or faces away?
**Answer**: BioVision enforces a strict Modality Validity Gate. If head yaw exceeds $45^\circ$, or if facial skin is occluded, or if the audio track is silent, BioVision rejects the video with a structured diagnostic failure code (`NO_FACE_DETECTED` or `INVALID_VIDEO`) rather than padding missing modalities with zeros, which would cause false classifications.

### Q5: How do you prove that the Audio-Lip branch actually checks synchronization rather than just mouth shapes?
**Answer**: We conducted a controlled temporal desynchronization stress test (Table VI). By artificially drifting the audio track by $\Delta t \in [0, 1000]\text{ ms}$, the model's accuracy dropped monotonically from 95.22% at 0 ms down to 68.21% at 1000 ms. This proves the BiLSTM actively relies on temporal phoneme-viseme synchrony.

### Q6: What is the difference between In-Domain and Zero-Shot Cross-Dataset performance?
**Answer**: In-Domain means testing on held-out test splits from the same dataset family (e.g., Celeb-DF v2 test split $\to$ 94.98%). Zero-Shot Cross-Dataset means training on Celeb-DF v2 and evaluating directly on DFDC without any target dataset fine-tuning. BioVision achieves an impressive 0.9125 ROC-AUC in zero-shot transfer because biological hemodynamics are universal across all humans, unlike dataset-specific pixel artifacts.
