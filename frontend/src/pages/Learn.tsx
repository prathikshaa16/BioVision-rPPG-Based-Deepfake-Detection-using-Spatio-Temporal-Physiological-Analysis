import React from 'react'
import Card from '../components/Card'
import {
  FaVideo,
  FaEye,
  FaMicrochip,
  FaBrain,
  FaHeartbeat,
  FaLayerGroup,
  FaShieldAlt,
  FaCheckCircle,
  FaBookOpen,
  FaSlidersH,
  FaUsers,
} from 'react-icons/fa'

const METHODOLOGY_STEPS = [
  {
    step: '01',
    title: 'Video Acquisition & Multi-Dataset Ingestion',
    desc: 'Input videos from Celeb-DF v2 and DeepFake Detection Challenge (DFDC).',
    details:
      'The system ingests videos across Celeb-DF v2 (6,529 videos) and DFDC (3,431 videos). Videos are partitioned using strict identity-disjoint splits: actors present in the validation and test sets never appear in the training pool, ensuring zero facial memorization and genuine out-of-distribution evaluation.',
    icon: FaVideo,
    color: 'text-cyan-400',
    borderColor: 'border-cyan-500/30',
  },
  {
    step: '02',
    title: 'Facial Region Processing & Tracking',
    desc: 'Facial regions are detected, standardized, and tracked over time.',
    details:
      'Multi-task Cascaded Convolutional Networks (MTCNN) identify facial bounding boxes across uniformly sampled frames. Standardized face crops (224×224) are extracted with a 20-pixel context margin, while forehead and cheek skin ROIs are tracked for physiological signal recovery.',
    icon: FaEye,
    color: 'text-cyan-300',
    borderColor: 'border-cyan-500/30',
  },
  {
    step: '03',
    title: 'Spatial Representation Extraction',
    desc: 'EfficientNet-B4 extracts high-dimensional visual representations.',
    details:
      'Pretrained EfficientNet-B4 extracts 1,792-dimensional spatial representations from each facial crop, capturing micro-texture discrepancies, edge-blending seams, lighting mismatches, and synthetic artifact signatures.',
    icon: FaMicrochip,
    color: 'text-blue-400',
    borderColor: 'border-blue-500/30',
  },
  {
    step: '04',
    title: 'Temporal Sequence Modeling & Sequence Dropout',
    desc: 'A 2-layer LSTM models temporal dynamics with anti-overfitting regularization.',
    details:
      'Rather than analyzing isolated frames, a 2-layer LSTM network processes the ordered 32-frame sequence of 1,792-d vectors into a 256-d temporal representation. To prevent memorization of frozen features, in-memory sequence dropout (randomly zero-masking 1-5 timesteps) and Gaussian embedding jitter are injected during training.',
    icon: FaBrain,
    color: 'text-blue-300',
    borderColor: 'border-blue-500/30',
  },
  {
    step: '05',
    title: 'Physiological Signal Extraction (CHROM rPPG)',
    desc: 'Remote photoplethysmography isolates biological blood volume pulses.',
    details:
      'Subtle optical variations from facial capillary beds are converted into a remote photoplethysmography (rPPG) pulse signal via the chrominance-based (CHROM) projection. The signal is bandpass-filtered (0.8–3.0 Hz, 48–180 BPM) and passed to a 1D-CNN, yielding a 64-dimensional physiological feature representation.',
    icon: FaHeartbeat,
    color: 'text-emerald-400',
    borderColor: 'border-emerald-500/30',
  },
  {
    step: '06',
    title: 'Quality-Gated Multimodal Fusion',
    desc: 'Adaptive gating combines spatio-temporal and physiological evidence.',
    details:
      'The 256-d visual-temporal embedding and 64-d physiological embedding are combined through a learnable quality-gating layer. When physiological signals are clean, cardiac consistency provides an unforgeable authenticity anchor; if degraded by lighting or motion, visual cues are dynamically prioritized.',
    icon: FaLayerGroup,
    color: 'text-purple-400',
    borderColor: 'border-purple-500/30',
  },
  {
    step: '07',
    title: 'Threshold-Calibrated Classification (≥94% Target)',
    desc: "Final classification optimized via Youden's J index and label smoothing.",
    details:
      "The fused representation passes through a regularized classification head trained with label smoothing (ε = 0.05) and balanced mini-batch sampling. The operating decision threshold is calibrated on validation data via Youden's J index, balancing Sensitivity (≥94%) and Specificity (≥94%) to output the final forensic assessment.",
    icon: FaShieldAlt,
    color: 'text-rose-400',
    borderColor: 'border-rose-500/30',
  },
]

export default function Learn() {
  return (
    <div className="max-w-5xl mx-auto space-y-10">
      <header className="page-head">
        <div>
          <h2 className="page-title">Our Methodology</h2>
          <p className="page-sub">
            The seven-step scientific framework powering BioVision spatio-temporal &amp; physiological deepfake detection
          </p>
        </div>
        <span className="chip chip--info">
          <FaBookOpen className="w-3.5 h-3.5" />
          Research Methodology
        </span>
      </header>

      {/* PIPELINE OVERVIEW */}
      <div className="glass-card p-6 md:p-8 space-y-4">
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <FaSlidersH className="text-cyan-400" />
          Multimodal Detection Framework
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed">
          Deepfake generation techniques have evolved to produce photorealistic static faces that evade single-frame classifiers. BioVision overcomes this limitation by integrating two complementary scientific channels: <strong>spatio-temporal dynamics</strong> (how visual artifacts behave over time) and <strong>physiological remote photoplethysmography</strong> (subtle cardiac pulses in facial skin).
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          To prevent model overfitting across diverse real-world distributions, BioVision employs <strong>identity-disjoint dataset partitioning</strong> across Celeb-DF v2 and DFDC, <strong>in-memory sequence dropout</strong>, <strong>class-balanced mini-batch sampling</strong>, and <strong>validation-guided threshold calibration</strong>.
        </p>
      </div>

      {/* 7 METHODOLOGY STEPS */}
      <div className="space-y-6">
        {METHODOLOGY_STEPS.map((s) => {
          const Icon = s.icon
          return (
            <div
              key={s.step}
              className={`glass-card p-6 md:p-7 border-l-4 ${s.borderColor} space-y-3 transition-all hover:bg-slate-900/60`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2.5 rounded-xl bg-slate-900 border border-slate-800 ${s.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-slate-500">
                      Phase {s.step}
                    </span>
                    <h4 className="text-base font-bold text-slate-100">{s.title}</h4>
                  </div>
                </div>
              </div>

              <p className="text-xs font-semibold text-slate-300">{s.desc}</p>
              <p className="text-xs text-slate-400 leading-relaxed">{s.details}</p>
            </div>
          )
        })}
      </div>

      {/* ANTI-OVERFITTING SCIENTIFIC PRINCIPLES */}
      <Card
        title="Anti-Overfitting &amp; Generalization Architecture"
        subtitle="Key scientific countermeasures ensuring robust performance on unseen datasets"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-cyan-300 flex items-center gap-2">
              <FaUsers className="w-4 h-4" />
              1. Zero Identity Leakage Partitioning
            </div>
            <p className="text-slate-400 leading-relaxed">
              In both Celeb-DF v2 and DFDC, actor identities are rigorously partitioned between train, validation, and test splits. The evaluation splits contain 0% actor overlap with training subjects, forcing the network to detect manipulation artifacts rather than actor-specific facial landmarks.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-cyan-300 flex items-center gap-2">
              <FaSlidersH className="w-4 h-4" />
              2. Feature Augmentation &amp; Sequence Dropout
            </div>
            <p className="text-slate-400 leading-relaxed">
              Frozen visual embeddings are augmented on-the-fly during training with temporal sequence dropout (randomly zeroing 1-5 timesteps) and Gaussian noise jitter. The 2-layer LSTM learns continuous sequence dynamics instead of memorizing static feature vectors.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-cyan-300 flex items-center gap-2">
              <FaShieldAlt className="w-4 h-4" />
              3. Balanced Sampling &amp; Label Smoothing
            </div>
            <p className="text-slate-400 leading-relaxed">
              A WeightedRandomSampler forces a 50% Real / 50% Fake expected balance per mini-batch, eliminating the false-positive bias. Label smoothing bounds gradient extremes and halts validation loss explosion.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="font-bold text-cyan-300 flex items-center gap-2">
              <FaCheckCircle className="w-4 h-4" />
              4. Youden's J Threshold Calibration
            </div>
            <p className="text-slate-400 leading-relaxed">
              Rather than using a fixed 0.50 cutoff, the decision boundary is calibrated on validation data using Youden's J index (maximizing Sensitivity + Specificity - 1), achieving balanced accuracy ≥94% across test sets.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
