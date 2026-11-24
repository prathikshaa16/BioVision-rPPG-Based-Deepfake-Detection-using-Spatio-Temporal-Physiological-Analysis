# BioVision Phase II Report: Final Guide Review Corrections Package

This document contains the exact, copy-pasteable LaTeX replacement snippets addressing all items from the latest guide review:

---

## Summary of Latest Corrections

| # | Item & Location | Guide Feedback | Action in `main.tex` |
|---|---|---|---|
| **1** | **Chapter 6, Section 6.7** (p. 40 / PDF p. 51) | Replace "confidence" with "prediction score" | Replace `with confidence and analysis details` with `with the model prediction score and analysis details.` |
| **2** | **Section 7.2.1, TC01 & TC02 Logs** (p. 51 / PDF p. 62) | Remove "verified authentic biological cardiac pulse" and "confidence and fused probability" | Replace with neutral, scientifically defensible execution log descriptions. |
| **3** | **Figure 7.4 Caption & Context** (p. 56 / PDF p. 67) | Terminology inconsistency with "Confidence" on UI | Update Fig. 7.4 caption to specify "model prediction score (labeled as confidence in prototype UI)". |
| **4** | **Equation (5.7) LSTM Notation** (p. 31 / PDF p. 42) | Ensure candidate cell state $\tilde{\mathbf{c}}_t$ has tilde and updated $\mathbf{c}_t$ does not | Verified standard LSTM notation with explicit candidate vs. updated state definitions. |
| **5** | **Section 8.6 Heading** (p. 60 / PDF p. 71) | Soften "Legal Evidence Verification" | Rename to `\section{Support for Legal and Administrative Media Review}`. |
| **6** | **Preliminary Page ii Acknowledgement** (PDF p. 5) | Paragraphs 5 & 6 are still duplicate/repetitive | Merge the two Dr. A. Rosline Mary paragraphs into a single cohesive paragraph. |
| **7** | **Table 5.4 Footer Overlap** (PDF p. 45) | Table 5.4 slightly collides with page footer | Add `\small` or adjust row separation to fit within margins cleanly. |

---

## 1. Chapter 6, Section 6.7: Prediction Score Terminology

**Location:** Chapter 6, Section 6.7, Page 40 (PDF Page 51), Line 4 of Section 6.7.

### 📝 Replacement Text:
```latex
From a user’s perspective, an analysis begins by selecting a video and following the application as it prepares frames and facial regions. The progress view makes each processing stage visible, and the result dashboard brings the final prediction together with the model prediction score and analysis details. Functional checks also show that invalid uploads and videos without a detected face receive a clear response, helping users understand the application’s input-handling workflow.
```

*(Changed: "confidence and analysis details" $\to$ "the model prediction score and analysis details".)*

---

## 2. Section 7.2.1: Execution Logs (TC01 & TC02)

**Location:** Chapter 7, Section 7.2.1, Page 51 (PDF Page 62), Paragraph under heading.

### 📝 Replacement Text:
```latex
\noindent\textbf{TC01 and TC02: Successful real and fake video ingestion.} For TC01, the authentic interview asset \texttt{15242.mp4} ($1080 \times 1920$ pixels, 284 frames) returned HTTP 200 OK with result REAL, model score 0.8078, estimated heart rate 85.56 BPM, SNR 4.98 dB, and a tri-modal fusion result with an available rPPG signal. For TC02, the latent-diffusion face-swap asset \texttt{ai.mp4} returned HTTP 200 OK with result FAKE, model score of 0.5831, and rPPG status UNAVAILABLE (pulse suppressed).
```

*(Changed: Removed "verified authentic biological cardiac pulse" and simplified TC02 from "confidence and fused probability 0.5831" to "model score of 0.5831".)*

---

## 3. Figure 7.4 & Dashboard Terminology

**Location:** Chapter 7, Section 7.4, Page 56 (PDF Page 67).

### 📝 Replacement Caption for Figure 7.4:
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{figures/dashboard_result.png}
\caption{BioVision interface: result dashboard. The screen presents the video-level classification verdict, model prediction score (labeled as confidence in the prototype web UI), detected-face count, and inference processing details.}
\label{fig:dashboard}
\end{figure}
```

---

## 4. Equation (5.7): Standard LSTM Cell-State Notation

**Location:** Chapter 5, Section 5.1.7, Page 31 (PDF Page 42).

### 📝 Replacement LaTeX for Equation (5.7) and Surrounding Text:
```latex
A standard LSTM cell updates its internal memory from the current input vector $\mathbf{x}_t$ and the preceding hidden state $\mathbf{h}_{t-1}$. To distinguish the intermediate candidate cell state from the accumulated memory cell state, the recurrence is formulated as:
\begin{align}
\mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i), \nonumber \\
\mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f), \nonumber \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o), \nonumber \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c), \label{eq:lstm_equations} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t, \nonumber \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t). \nonumber
\end{align}
Here, $\mathbf{i}_t$, $\mathbf{f}_t$, and $\mathbf{o}_t$ denote the input, forget, and output gate activations, respectively; $\tilde{\mathbf{c}}_t$ represents the candidate cell state vector; $\mathbf{c}_t$ is the updated cell memory state; $\mathbf{h}_t$ is the recurrent hidden state vector; and $\odot$ denotes the Hadamard (element-wise) product.
```

*(Verification: $\tilde{\mathbf{c}}_t$ carries the tilde in the $\tanh$ candidate activation, and $\mathbf{c}_t$ without tilde is the updated cell state.)*

---

## 5. Section 8.6 Heading: Professional Scope Alignment

**Location:** Chapter 8, Section 8.6, Page 60 (PDF Page 71).

### 📝 Replacement LaTeX:
```latex
\section{Support for Legal and Administrative Media Review}
\label{sec:legal_support}
```

*(Changed from `\section{Legal Evidence Verification}`. The body text of Section 8.6 is already responsible and is retained as-is.)*

---

## 6. Preliminary Page ii: Deduped Acknowledgement

**Location:** Preliminary Page ii (PDF Page 5).

### 📝 Replacement LaTeX (combines paragraphs 5 & 6):
```latex
We extend our sincere gratitude to our Project Coordinator and Project Guide, \textbf{Dr. A. Rosline Mary}, Associate Professor, Department of Computer Science and Engineering (Data Science), for her invaluable guidance, thoughtful technical suggestions, and consistent support throughout the conceptualization, implementation, and evaluation of this Project Phase II work.
```

---

## 7. Table 5.4 Formatting Fix (Prevent Page Footer Collision)

**Location:** Chapter 5, Page 34 (PDF Page 45).

### 📝 Enclose Table 5.4 in `\small` or reduce `\tabcolsep`:
```latex
\begin{table}[htbp]
\centering
\small
\caption{End-to-End Training and Optimization Hyperparameters}
\label{tab:training_hyperparameters}
\begin{tabular}{|p{0.32\linewidth}|p{0.62\linewidth}|}
\hline
\textbf{Parameter or component} & \textbf{Setting} \\
\hline
Visual backbone initialization & EfficientNet-B4 initialized from torchvision's default ImageNet-1k weights. \\
\hline
Trainable visual layers & Final three Mobile Inverted Bottleneck (MBConv) stages \texttt{features[5:8]} and final $1\times 1$ expansion layer \texttt{features[8]}. \\
\hline
Frozen visual layers & Stem and early stages \texttt{features[0:5]}. \\
\hline
Physiological and audio-lip encoders & CHROM Conv1D and audio-lip BiLSTM trained from scratch with Xavier-uniform initialization. \\
\hline
Loss function & Binary cross-entropy on sigmoid probabilities; real = 0, fake = 1. \\
\hline
Optimizer & AdamW; $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$. \\
\hline
Learning rate & Initial rate $1.0 \times 10^{-4}$; cosine annealing to $1.0 \times 10^{-6}$ over 30 epochs. \\
\hline
Weight decay & $1.0 \times 10^{-4}$. \\
\hline
Gradient clipping & Maximum gradient norm of 1.0. \\
\hline
Batch size & 8 videos (256 face crops, 1,920 rPPG samples, and 256 aligned MFCC time steps per batch). \\
\hline
Dropout & $p=0.30$ between BiLSTM layers and after first MLP layer ($448 \to 256$); $p=0.20$ after 64-D physiological, 128-D audio-lip projection, and second MLP layer ($256 \to 64$). \\
\hline
Validation and checkpoint selection & Validation loss checked after each epoch; early stopping after 5 epochs without improvement; retain checkpoint with lowest validation loss. \\
\hline
Training duration & Up to 30 epochs. \\
\hline
Data partitions & Celeb-DF v2: 4,496 training, 500 validation, and 518 test videos; DFDC: 70 validation and 130 test videos. \\
\hline
Random seed & 42 for PyTorch, NumPy, and CUDA. \\
\hline
Final model selection & Lowest validation loss on the validation cohort. \\
\hline
Hardware & NVIDIA Tesla T4 GPU (16 GB GDDR6), Intel Xeon CPU at 2.20 GHz, and 32 GB RAM. \\
\hline
\end{tabular}
\end{table}
```
