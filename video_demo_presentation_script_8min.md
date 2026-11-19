# BioVision — 8-Minute Video Demonstration Script & Production Guide

A complete, production-ready 8-minute (480-second) video demonstration guide for **BioVision: Multimodal Deepfake Detection via Spatio-Temporal, Physiological, and Audio-Lip Features**.

---

## Video Specifications & Setup Checklist

| Parameter | Recommended Setting |
| :--- | :--- |
| **Total Duration** | **8 minutes 00 seconds** (480 seconds) |
| **Target Audience** | Academic evaluators, external examiners, defense committees, conference reviewers |
| **Screen Resolution** | 1080p Full HD (1920 × 1080) at 60 fps |
| **Browser Setup** | Chrome / Edge in Fullscreen (`F11`), clean bookmarks bar hidden |
| **Live Web App** | [https://bio-vision.pages.dev](https://bio-vision.pages.dev) |
| **Local Model Server** | FastAPI running on `http://127.0.0.1:8000` via Cloudflare Tunnel |
| **Sample Test Assets** | 1. Authentic Real Video (`real_sample.mp4` / `WhatsApp Video...9.00.24 PM.mp4`)<br>2. Deepfake Video (`fake_sample.mp4`)<br>3. Non-face video (`WhatsApp Video...18.36.05.mp4` or presentation slide) |

---

## 8-Minute Timeline Overview

```
[0:00 - 1:00] Scene 1: Introduction, Problem Statement & Motivation (60s)
[1:00 - 2:15] Scene 2: The Core Scientific Innovation — Tri-Modal Architecture (75s)
[2:15 - 3:30] Scene 3: Deep Dive into rPPG Hemodynamics & Audio-Lip Biomechanics (75s)
[3:30 - 4:45] Scene 4: System Architecture & Web Application Tour (75s)
[4:45 - 6:00] Scene 5: Live Video Ingestion & Forensic Inference Demo (Real vs. Fake) (75s)
[6:00 - 7:00] Scene 6: Forensic Robustness & Edge Cases (Non-Face & Vertical Videos) (60s)
[7:00 - 7:45] Scene 7: Quantitative Benchmark Evaluation & Ablation Validation (45s)
[7:45 - 8:00] Scene 8: Real-World Impact & Conclusion (15s)
```

---

## Scene-by-Scene Script with On-Screen Actions & Spoken Narration

### Scene 1: Introduction, Problem Statement & Motivation
* **Timestamp:** `0:00 - 1:00` (60 seconds)
* **On Screen:** Start at the homepage of **[https://bio-vision.pages.dev](https://bio-vision.pages.dev)** in fullscreen. Show the dark-themed hero header, title, and live backend connection badge. Keep the cursor positioned naturally over the title.

#### 🎙️ Spoken Narration:
> *"Hello everyone. Welcome to the comprehensive demonstration of **BioVision** — an advanced multimodal deepfake detection platform leveraging spatio-temporal facial features, remote photoplethysmography, and audio-lip temporal representations.*
>
> *Today’s generative artificial intelligence models—such as Generative Adversarial Networks, Autoencoders, and Latent Diffusion models—can synthesize remarkably photorealistic facial imagery. These synthetic media, commonly known as deepfakes, present severe threats to digital identity, cybersecurity, judicial evidence, and democratic processes.*
>
> *Most existing deepfake detectors rely solely on frame-level convolutional networks that look for pixel-level synthesis artifacts or boundary blending seams. However, in real-world scenarios—such as videos shared on messaging apps or social media—lossy video compression, spatial downscaling, and color quantization strip away these high-frequency visual cues, causing traditional detectors to fail.*
>
> *BioVision introduces a paradigm shift. Instead of solely inspecting surface pixels, we analyze **involuntary human biology**—specifically the subcutaneous cardiovascular blood volume pulse and the biomechanical synchrony between speech phonemes and lip articulation."*

---

### Scene 2: The Core Scientific Innovation — Tri-Modal Architecture
* **Timestamp:** `1:00 - 2:15` (75 seconds)
* **On Screen:** Smoothly scroll down the homepage to the **Architecture & Core Methodology** section. Hover consecutively over the three modality cards: (1) Visual-Temporal, (2) Physiological rPPG, and (3) Audio-Lip Synchrony.

#### 🎙️ Spoken Narration:
> *"BioVision is engineered around a mathematically grounded **tri-modal forensic architecture**:*
>
> *First, the **Visual-Temporal Branch**: We utilize Multi-Task Cascaded Convolutional Networks (MTCNN) to track facial regions across 32 uniformly sampled frames. These frames are processed through an EfficientNet-B4 convolutional backbone combined with a 2-layer Bidirectional LSTM. This stream captures spatial facial blending artifacts and temporal expression flickering across consecutive frames, yielding a 256-dimensional feature vector.*
>
> *Second, the **Physiological rPPG Branch**: Using Remote Photoplethysmography, BioVision extracts cardiovascular blood volume pulse signals directly from facial skin capillary beds. Deep learning generative models synthesize faces frame-by-frame without understanding biological hemodynamics. Our chrominance-based method isolates the cardiac waveform and processes it through a 1D Convolutional Neural Network, producing a 64-dimensional physiological representation.*
>
> *Third, the **Audio-Lip Biomechanical Branch**: Natural human speech requires precise biomechanical coordination between the vocal tract and anatomical lip articulation. We extract 40 Mel-Frequency Cepstral Coefficients from the audio track and 20 MediaPipe 3D mouth contour landmarks from the video, encoding their temporal alignment with a BiLSTM into a 128-dimensional representation.*
>
> *All three representations undergo independent Layer Normalization and are concatenated into a unified **448-dimensional multimodal vector**, classified by a calibrated Multi-Layer Perceptron."*

---

### Scene 3: Deep Dive into rPPG Hemodynamics & Audio-Lip Biomechanics
* **Timestamp:** `2:15 - 3:30` (75 seconds)
* **On Screen:** Navigate to the **Model Info** or **Learn** tab in the top navigation bar. Show the technical diagrams detailing the CHROM algorithm, the Butterworth filter frequency band (0.8 to 3.0 Hz), and the MediaPipe lip landmark topology.

#### 🎙️ Spoken Narration:
> *"Let us examine the underlying physical and mathematical principles of our two biological branches.*
>
> *In the physiological branch, we target the forehead and bilateral cheek regions of interest, which exhibit the highest concentration of facial capillaries. As oxygenated blood is pumped by the heart, hemoglobin absorption alters ambient light reflectance.*
>
> *To eliminate motion drift and specular lighting reflections, we implement the **Chrominance-based (CHROM) projection algorithm** formulated by de Haan and Jeanne. We normalize the RGB color channels by their temporal running means and construct two orthogonal chrominance signals: $X = 3R_n - 2G_n$, and $Y = 1.5R_n + G_n - 1.5B_n$. By dynamically subtracting them using their standard deviation ratio, we extract the raw blood volume pulse.*
>
> *This signal is detrended and passed through an 8th-order zero-phase digital Butterworth bandpass filter tuned strictly to the physiological human heart rate band between 0.8 and 3.0 Hertz, corresponding to 48 to 180 beats per minute.*
>
> *Concurrently, our audio-lip branch pairs acoustic phonetic energy with geometric mouth aperture dynamics normalized by inter-commissure distance. This allows BioVision to detect sub-second temporal phase drift and AI dubbing desynchronizations."*

---

### Scene 4: System Architecture & Web Application Tour
* **Timestamp:** `3:30 - 4:45` (75 seconds)
* **On Screen:** Click through the header navigation tabs:
  1. Show the **Dashboard** showing historical statistics.
  2. Click on the **Backend Status** pill in the top right to reveal the active Cloudflare Tunnel connection (`status: healthy`, latency, active model checkpoint `biovision_unified_648_seed42.pt`).
  3. Navigate to the **Analysis** portal to show the upload interface.

#### 🎙️ Spoken Narration:
> *"Now, let us examine the system deployment architecture.*
>
> *BioVision is deployed as a production-grade full-stack platform. The backend is built on **FastAPI and PyTorch**, running high-throughput inference on GPU and multi-core CPU architectures. The frontend is a modern, responsive single-page application built using **React 18, TypeScript, Tailwind CSS, and Vite**.*
>
> *(Action: Click on Backend Status indicator)*
> *Notice our live connectivity indicator: The globally distributed Cloudflare Pages frontend communicates securely with our local deep learning model server via an encrypted Cloudflare Tunnel. This architecture provides zero-trust security and allows full forensic processing without exposing local ports.*
>
> *(Action: Navigate to the Analysis page)*
> *In the Analysis portal, users can submit video files in MP4, AVI, or WebM format up to 500 megabytes. The interface provides real-time client-side validation, progress tracking, and dynamic stage progression as the video undergoes ingestion, face tracking, rPPG extraction, and multimodal fusion."*

---

### Scene 5: Live Video Ingestion & Forensic Inference Demo (Real vs. Fake)
* **Timestamp:** `4:45 - 6:00` (75 seconds)
* **On Screen:** 
  1. Drag and drop an authentic video (`real_sample.mp4`). Click **"Analyze Video"**.
  2. Watch the progress bar advance through the 8 pipeline stages.
  3. Arrive at the **Results Dashboard**:
     - Point out the **AUTHENTIC / REAL** badge and high confidence score.
     - Scroll to the **rPPG Cardiac Waveform** showing a periodic sine-like pulse.
     - Point out the **Power Spectrum** showing a clear dominant peak around 72–75 BPM.
     - Show the **Audio-Lip Synchrony** alignment.
  4. Click the **"Export Forensic PDF Report"** button to download the audit report.
  5. Go back to Analysis and repeat with a synthetic deepfake video (`fake_sample.mp4`). Show the resulting **DEEPFAKE / FAKE** red badge, flatlined rPPG spectrum, and multimodal breakdown.

#### 🎙️ Spoken Narration:
> *"Let us conduct live forensic testing.*
>
> *(Action: Upload authentic video and click Analyze)*
> *First, we submit an authentic test video. As the video processes, the backend extracts the 32 facial crops, filters the chrominance signals, extracts 40 MFCC acoustic vectors, and fuses them through our 448-dimensional classifier.*
>
> *(Action: Arrive at Results screen)*
> *Here are the comprehensive forensic results:*
> *At the top, we receive a definitive verdict: **AUTHENTIC (REAL)** with high confidence.*
>
> *(Action: Point cursor at the rPPG waveform chart)*
> *Observe the rPPG physiological card: Look at the periodic, rhythmic blood volume pulse waveform and its corresponding Fast Fourier Transform power spectrum. Notice the sharp, well-defined peak at approximately 74 beats per minute. This is the physiological signature of authentic human cardiovascular circulation.*
>
> *(Action: Point at the PDF export button)*
> *Examiners can click 'Export Forensic PDF Report' to generate a tamper-evident cryptographic forensic report.*
>
> *(Action: Quickly show deepfake test result)*
> *In contrast, when analyzing a deepfake video, observe how the rPPG waveform exhibits chaotic, non-harmonic fluctuations with a suppressed power spectrum, and the audio-lip synchrony flags significant phonetic phase lag, triggering an immediate **SYNTHETIC / DEEPFAKE** classification."*

---

### Scene 6: Forensic Robustness & Edge Cases (Non-Face & Vertical Videos)
* **Timestamp:** `6:00 - 7:00` (60 seconds)
* **On Screen:** 
  1. Navigate to Analysis and upload a non-face video (`WhatsApp Video 2026-08-05...` / presentation slide).
  2. Click Analyze. Show the instant transition to the dedicated **"NO FACE DETECTED"** screen.
  3. Upload a vertical smartphone video with a real face (`WhatsApp Video 2026-05-20...`).
  4. Show how BioVision preserves the aspect ratio without distortion and accurately classifies the real face.

#### 🎙️ Spoken Narration:
> *"A critical requirement in real-world digital forensics is **robustness to edge cases and input validation**.*
>
> *(Action: Upload non-face video)*
> *Here, we upload a video that contains presentation slides with no human face. Many commercial detectors make the dangerous mistake of guessing or defaulting to 'Real' with high confidence.*
>
> *(Action: Point to the NO FACE DETECTED screen)*
> *BioVision enforces a strict **Modality Validity Gate**. If our MTCNN and cascade trackers confirm zero verifiable facial tracks across the sampled frames, the system strictly outputs **'NO FACE DETECTED'**, preventing false positives and ensuring forensic integrity.*
>
> *(Action: Upload vertical video and show result)*
> *Furthermore, smartphone videos are frequently recorded in vertical portrait format. While legacy systems distort portrait videos by stretching them into landscape canvases—destroying facial geometry—BioVision features dynamic aspect-ratio preservation and early-interval seek sampling, ensuring reliable detection across any orientation and resolution."*

---

### Scene 7: Quantitative Benchmark Evaluation & Ablation Validation
* **Timestamp:** `7:00 - 7:45` (45 seconds)
* **On Screen:** Navigate to the **Metrics** page. Show the benchmark comparison table and the ablation study bar charts.

#### 🎙️ Spoken Narration:
> *"BioVision has been rigorously evaluated on leading forensic benchmarks:*
>
> *On the official **Celeb-DF v2** test split of 518 videos, BioVision achieves **94.98% accuracy** and a **0.9642 ROC-AUC**.*
> *On the held-out **DeepFake Detection Challenge (DFDC)** test split of 130 videos, it achieves **95.38% accuracy** and a **0.9715 ROC-AUC**.*
> *Across our unified 648-video benchmark evaluated under a single fixed threshold of $\tau = 0.50$, BioVision attains **95.22% overall accuracy**.*
>
> *Crucially, our **architectural ablation study** proves the necessity of multimodal fusion: Visual-temporal analysis alone achieves 84.10% accuracy due to video compression artifacts. Adding physiological rPPG raises accuracy to 90.12%, adding audio-lip raises it to 91.82%, and full tri-modal fusion reaches **95.22%**, cutting the classification error rate by more than half.*
>
> *Moreover, under **Zero-Shot Cross-Dataset Transfer** from Celeb-DF v2 to DFDC, BioVision preserves an exceptional **ROC-AUC of 0.9125**, outperforming single-modality baselines by over 24%."*

---

### Scene 8: Real-World Impact & Conclusion
* **Timestamp:** `7:45 - 8:00` (15 seconds)
* **On Screen:** Return to the Homepage hero screen. Display the GitHub repository link, project credits, and live deployment URL.

#### 🎙️ Spoken Narration:
> *"In conclusion, BioVision bridges the gap between deep learning and biological reality. By uniting visual appearance, cardiovascular hemodynamics, and speech-lip biomechanics, we deliver a robust, explainable defense against synthetic media manipulation.*
>
> *Thank you for your time. The code, trained model weights, and live web application are fully open and accessible for independent verification."*

---

## Recording & Delivery Tips for the Presenter

1. **Pacing**: Speak at a steady, deliberate pace (~130 words per minute). Do not rush through mathematical names or metric numbers.
2. **Cursor Motion**: Move your mouse deliberately. Point the cursor at the specific chart, badge, or table cell you are speaking about, and leave it still for 2–3 seconds so viewers can easily read the numbers.
3. **Browser Audio**: When demonstrating the audio-lip branch, ensure your recording software captures system audio at a low background volume so the video's speech can be subtly heard.
4. **Transition Smoothness**: Use smooth scrolling (`Page Down` or a calibrated mouse wheel) rather than rapid jerking.
