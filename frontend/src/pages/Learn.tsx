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
} from 'react-icons/fa'

const METHODOLOGY_STEPS = [
  {
    step: '01',
    title: 'Video Acquisition',
    desc: 'Video is provided as the input to BioVision.',
    details:
      'The system accepts standard video formats (MP4, AVI, MOV, WebM). Uniform temporal sequence sampling extracts representative facial observations across the full duration of the video, ensuring temporal relationships are preserved.',
    icon: FaVideo,
    color: 'text-cyan-400',
    borderColor: 'border-cyan-500/30',
  },
  {
    step: '02',
    title: 'Facial Region Processing',
    desc: 'Facial regions are detected and standardized for analysis.',
    details:
      'Multi-task Cascaded Convolutional Networks (MTCNN) identify facial bounding boxes in sampled frames. Face crops are extracted with a 20-pixel margin and resized to 224×224 pixels. Forehead and cheek regions of interest (ROIs) are tracked for physiological signal extraction.',
    icon: FaEye,
    color: 'text-cyan-300',
    borderColor: 'border-cyan-500/30',
  },
  {
    step: '03',
    title: 'Spatial Representation',
    desc: 'EfficientNet-B4 extracts visual representations from facial regions.',
    details:
      'A pretrained EfficientNet-B4 convolutional neural network extracts 1,792-dimensional feature vectors from each standardized face crop, encoding micro-textures, blending artifacts, and boundary inconsistencies.',
    icon: FaMicrochip,
    color: 'text-blue-400',
    borderColor: 'border-blue-500/30',
  },
  {
    step: '04',
    title: 'Temporal Representation',
    desc: 'An LSTM models temporal dependencies across the sequence.',
    details:
      'Rather than treating frames as independent static images, a 2-layer Long Short-Term Memory (LSTM) network analyzes the ordered sequence of 1,792-d spatial embeddings, capturing cross-frame dynamics and motion consistency into a compact 256-d temporal representation.',
    icon: FaBrain,
    color: 'text-blue-300',
    borderColor: 'border-blue-500/30',
  },
  {
    step: '05',
    title: 'Physiological Signal Extraction',
    desc: 'CHROM-based rPPG processing extracts physiological information from facial color variations.',
    details:
      'Remote photoplethysmography (rPPG) monitors subtle color fluctuations in facial capillary beds caused by cardiac pulse cycles. The chrominance-based (CHROM) method isolates hemoglobin absorption from lighting and motion artifacts, outputting a 240-sample pulse vector encoded via a 1D-CNN into 64 physiological features.',
    icon: FaHeartbeat,
    color: 'text-emerald-400',
    borderColor: 'border-emerald-500/30',
  },
  {
    step: '06',
    title: 'Multimodal Fusion',
    desc: 'Visual/temporal and physiological evidence are integrated.',
    details:
      'The 256-d visual-temporal representation and 64-d physiological representation are concatenated into a 320-d multimodal feature vector. A quality-gated late fusion strategy weights the visual branch at 80% and physiological branch at 20%, adjusting dynamically if signal quality is compromised.',
    icon: FaLayerGroup,
    color: 'text-purple-400',
    borderColor: 'border-purple-500/30',
  },
  {
    step: '07',
    title: 'Deepfake Classification',
    desc: 'The fused representation is used to generate the final assessment.',
    details:
      'A multilayer perceptron classification head processes the fused representation, applying sigmoid activation to produce a continuous manipulation probability. Operating thresholds optimized via Youden J on validation benchmarks output the final verdict: REAL, FAKE, or UNCERTAIN.',
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

      {/* 7-STEP METHODOLOGY FLOW */}
      <div className="space-y-4">
        {METHODOLOGY_STEPS.map((s) => (
          <div key={s.step} className={`glass-card p-6 border ${s.borderColor} hover:border-cyan-400/50 transition-colors`}>
            <div className="flex flex-col md:flex-row md:items-start gap-5">
              <div className="flex-shrink-0 flex items-center md:flex-col gap-3">
                <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center font-mono font-extrabold text-cyan-400 text-base shadow-inner">
                  {s.step}
                </div>
                <s.icon className={`w-5 h-5 ${s.color} hidden md:block`} />
              </div>

              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-slate-100">{s.title}</h3>
                  <span className="text-[11px] font-mono text-slate-500 uppercase tracking-wider">Step {s.step}</span>
                </div>
                <p className="text-sm font-semibold text-cyan-300/90">{s.desc}</p>
                <p className="text-xs text-slate-400 leading-relaxed pt-1">{s.details}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* CORE SCIENTIFIC INSIGHTS */}
      <Card title="Why Single-Frame Classification Is Insufficient">
        <div className="space-y-3 text-sm text-slate-400 leading-relaxed">
          <p>
            Traditional deepfake detectors rely on frame-by-frame convolutional classification, averaging individual predictions to reach a verdict. This approach suffers from two critical vulnerabilities:
          </p>
          <ul className="space-y-2 pl-4 border-l-2 border-cyan-500/30 text-xs">
            <li>
              <strong className="text-slate-200">Loss of Temporal Coherence:</strong> Manipulation artifacts frequently manifest as flickering boundaries, unnatural motion velocity, or inconsistent eye-blink patterns across time. Isolated frame analysis cannot detect these temporal disruptions.
            </li>
            <li>
              <strong className="text-slate-200">Absence of Physiological Grounding:</strong> Generative models and face-swap autoencoders synthesize visual textures but do not simulate the subtle subcutaneous blood circulation that produces authentic cardiac pulses (rPPG).
            </li>
          </ul>
          <p className="pt-2 text-xs text-slate-300">
            BioVision unites spatio-temporal deep learning with remote photoplethysmography to evaluate both how a face looks and how it behaves biologically over time.
          </p>
        </div>
      </Card>
    </div>
  )
}
