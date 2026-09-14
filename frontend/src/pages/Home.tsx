import React, { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FaBars,
  FaTimes,
  FaArrowRight,
  FaMicrochip,
  FaHeartbeat,
  FaLayerGroup,
  FaShieldAlt,
  FaCheckCircle,
  FaWaveSquare,
  FaProjectDiagram,
  FaFlask,
  FaBrain,
  FaServer,
  FaCode,
  FaDatabase,
  FaEye,
} from 'react-icons/fa'
import { APP_NAME, APP_TAGLINE, landingNav } from '../lib/nav'
import { isBackendOnline } from '../lib/api'

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: 'easeOut' as const },
  }),
}

function Navbar() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  return (
    <header className="sticky top-0 z-50 border-b border-slate-800/60 bg-[#05080f]/85 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-5 lg:px-8 h-16 flex items-center justify-between gap-4">
        <Link to="/" className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan-400 via-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-sm shadow-[0_0_20px_rgba(34,211,238,0.4)]">
            BV
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-slate-100">{APP_NAME}</div>
            <div className="text-[11px] text-slate-400">{APP_TAGLINE}</div>
          </div>
        </Link>

        <nav className="hidden lg:flex items-center gap-7">
          {landingNav.map((link) =>
            link.to ? (
              <Link key={link.label} to={link.to} className="text-sm text-slate-300 hover:text-cyan-300 transition-colors">
                {link.label}
              </Link>
            ) : (
              <a key={link.label} href={link.href} className="text-sm text-slate-300 hover:text-cyan-300 transition-colors">
                {link.label}
              </a>
            )
          )}
        </nav>

        <div className="hidden lg:flex items-center gap-3">
          <Link to="/architecture" className="text-sm text-slate-300 hover:text-white transition-colors">
            Architecture
          </Link>
          <button onClick={() => navigate('/analysis')} className="btn btn-primary py-2 px-4 text-sm">
            Analyze a Video
          </button>
        </div>

        <button
          className="lg:hidden p-2 rounded-lg bg-slate-800/70 text-slate-300"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {open ? <FaTimes className="w-5 h-5" /> : <FaBars className="w-5 h-5" />}
        </button>
      </div>

      {open && (
        <nav className="lg:hidden border-t border-slate-800/60 bg-[#05080f]/95 px-5 py-4 space-y-1">
          {landingNav.map((link) =>
            link.to ? (
              <Link
                key={link.label}
                to={link.to}
                onClick={() => setOpen(false)}
                className="block px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-800/60"
              >
                {link.label}
              </Link>
            ) : (
              <a
                key={link.label}
                href={link.href}
                onClick={() => setOpen(false)}
                className="block px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-800/60"
              >
                {link.label}
              </a>
            )
          )}
          <div className="pt-2 flex gap-3">
            <button onClick={() => { setOpen(false); navigate('/analysis'); }} className="btn btn-primary w-full text-sm py-2">
              Analyze a Video
            </button>
          </div>
        </nav>
      )}
    </header>
  )
}

function HeroPipelineVisual() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      className="relative"
    >
      <div className="absolute -inset-6 rounded-3xl bg-cyan-500/10 blur-3xl" aria-hidden />
      <div className="glass-card--accent p-6 relative border border-cyan-500/25">
        <div className="flex items-center justify-between mb-5 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 pulse-glow" />
            <span className="text-xs font-semibold text-cyan-300 tracking-wide uppercase">BioVision Pipeline Architecture</span>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">Multimodal Fusion</span>
        </div>

        {/* Visual Diagram */}
        <div className="space-y-3 font-mono text-xs">
          {/* Top: Video input */}
          <div className="text-center">
            <div className="inline-block px-4 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-200 font-semibold shadow-sm">
              FACIAL VIDEO INPUT
            </div>
          </div>

          <div className="flex justify-center text-cyan-400/80 text-sm">↓</div>

          {/* Preprocessing */}
          <div className="text-center">
            <div className="inline-block px-4 py-1.5 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-cyan-200">
              FACE DETECTION & SEQUENCE PREPROCESSING (MTCNN)
            </div>
          </div>

          <div className="flex justify-center text-cyan-400/80 text-sm">↓</div>

          {/* 3 Complementary Branches */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 text-center">
            {/* Branch 1 */}
            <div className="p-3 rounded-xl bg-slate-900/90 border border-cyan-500/40 relative overflow-hidden group hover:border-cyan-400 transition-colors">
              <div className="text-[10px] text-cyan-400 font-semibold tracking-wider uppercase mb-1">Spatial Features</div>
              <div className="text-xs font-bold text-slate-100">EfficientNet-B4</div>
              <div className="text-[10px] text-slate-400 mt-1">1792-d Facial Representations</div>
            </div>

            {/* Branch 2 */}
            <div className="p-3 rounded-xl bg-slate-900/90 border border-blue-500/40 relative overflow-hidden group hover:border-blue-400 transition-colors">
              <div className="text-[10px] text-blue-400 font-semibold tracking-wider uppercase mb-1">Temporal Features</div>
              <div className="text-xs font-bold text-slate-100">2-Layer LSTM</div>
              <div className="text-[10px] text-slate-400 mt-1">Sequence Dynamics (256-d)</div>
            </div>

            {/* Branch 3 */}
            <div className="p-3 rounded-xl bg-slate-900/90 border border-emerald-500/40 relative overflow-hidden group hover:border-emerald-400 transition-colors">
              <div className="text-[10px] text-emerald-400 font-semibold tracking-wider uppercase mb-1">Physiological Signal</div>
              <div className="text-xs font-bold text-slate-100">CHROM rPPG</div>
              <div className="text-[10px] text-slate-400 mt-1">Blood-Volume Pulse (64-d)</div>
            </div>
          </div>

          <div className="flex justify-center text-cyan-400/80 text-sm">↓</div>

          {/* Fusion Head */}
          <div className="text-center">
            <div className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-950/60 via-blue-950/60 to-purple-950/60 border border-cyan-400/40 text-cyan-200">
              <div className="text-[11px] font-bold tracking-wider uppercase text-cyan-300">Quality-Gated Multimodal Fusion</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Late Fusion Head · 80% Visual-Temporal + 20% Physiological Gated</div>
            </div>
          </div>

          <div className="flex justify-center text-cyan-400/80 text-sm">↓</div>

          {/* Verdict Output */}
          <div className="p-3 rounded-xl bg-gradient-to-r from-emerald-500/15 via-slate-900 to-rose-500/15 border border-slate-700 text-center">
            <div className="text-[11px] font-extrabold uppercase tracking-widest text-slate-200">
              FUSED DEEPFAKE ASSESSMENT
            </div>
            <div className="flex items-center justify-center gap-3 text-[10px] text-slate-400 mt-1">
              <span className="text-emerald-400">✓ REAL</span>
              <span>·</span>
              <span className="text-rose-400">✗ FAKE</span>
              <span>·</span>
              <span className="text-amber-400">? UNCERTAIN</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function SectionHeading({ eyebrow, title, sub }: { eyebrow: string; title: React.ReactNode; sub?: string }) {
  return (
    <div className="text-center max-w-3xl mx-auto mb-14">
      <div className="eyebrow justify-center mb-3">{eyebrow}</div>
      <h2 className="text-3xl md:text-4xl font-bold text-slate-50 tracking-tight">{title}</h2>
      {sub && <p className="text-slate-400 mt-3 text-base leading-relaxed">{sub}</p>}
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [online, setOnline] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    isBackendOnline().then((ok) => {
      if (!cancelled) setOnline(ok)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="min-h-screen bg-[#05080f] text-slate-100 selection:bg-cyan-500/30">
      <Navbar />

      {/* 2. HERO SECTION */}
      <section className="relative overflow-hidden pt-12 pb-20 lg:pt-20 lg:pb-28">
        <div className="absolute inset-0 pointer-events-none" aria-hidden>
          <div className="absolute top-20 left-1/4 w-80 h-80 rounded-full bg-cyan-500/10 blur-[110px]" />
          <div className="absolute bottom-10 right-1/4 w-96 h-96 rounded-full bg-blue-600/10 blur-[130px]" />
        </div>

        <div className="max-w-7xl mx-auto px-5 lg:px-8 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center relative">
          <div className="lg:col-span-6 space-y-6">
            <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={0}>
              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3.5 py-1 text-xs font-semibold text-cyan-300">
                <FaShieldAlt className="w-3.5 h-3.5" />
                Spatio-Temporal & Physiological Analysis
              </span>
            </motion.div>

            <motion.h1
              variants={fadeUp}
              initial="hidden"
              animate="visible"
              custom={1}
              className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-50 leading-[1.1]"
            >
              BioVision
              <span className="block text-2xl sm:text-3xl lg:text-4xl text-gradient font-bold mt-2">
                Detect Deepfakes Through Visual and Physiological Evidence
              </span>
            </motion.h1>

            <motion.p
              variants={fadeUp}
              initial="hidden"
              animate="visible"
              custom={2}
              className="text-base sm:text-lg text-slate-400 leading-relaxed max-w-xl"
            >
              BioVision analyzes facial appearance, temporal patterns, and remote photoplethysmography (rPPG) signals
              to identify physiological inconsistencies associated with manipulated videos.
            </motion.p>

            <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={3} className="flex flex-wrap gap-4 pt-2">
              <button onClick={() => navigate('/analysis')} className="btn btn-primary px-7 py-3.5 text-base shadow-lg shadow-cyan-500/20">
                Analyze a Video
                <FaArrowRight className="w-4 h-4 ml-1" />
              </button>
              <a href="#pipeline" className="btn btn-outline px-7 py-3.5 text-base">
                Explore the Pipeline
              </a>
            </motion.div>

            <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={4} className="flex items-center gap-3 text-xs text-slate-400 pt-2">
              <span className={`w-2.5 h-2.5 rounded-full ${online === null ? 'bg-slate-500' : online ? 'bg-emerald-400 pulse-glow' : 'bg-rose-500'}`} />
              {online === null ? 'Checking inference backend…' : online ? 'Inference server online · Multimodal model loaded' : 'Backend offline · Start server on port 8000'}
            </motion.div>
          </div>

          <div className="lg:col-span-6">
            <HeroPipelineVisual />
          </div>
        </div>
      </section>

      {/* 4. HOW IT WORKS SECTION */}
      <section id="how-it-works" className="py-20 border-y border-slate-800/60 bg-[#060b15]/70">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="How It Works"
            title={<>Standardized Video Input & <span className="text-gradient">Facial Preprocessing</span></>}
            sub="Accurate downstream spatio-temporal and physiological analysis begins with uniform temporal sequence sampling and robust face localization."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="glass-card p-7 relative">
              <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">Step 01 — Video Input</div>
              <h3 className="text-xl font-bold text-slate-100 mb-3">Upload a Video</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Upload a facial video for automated deepfake analysis. BioVision ingests raw MP4, MOV, AVI, or WebM clips
                and establishes uniform temporal sequences across the video duration.
              </p>
              <div className="mt-5 p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-400">
                💡 Sequences preserve the real-time temporal order required for downstream LSTM dynamics and blood-volume pulse tracking.
              </div>
            </div>

            <div className="glass-card p-7 relative">
              <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">Step 02 — Face Processing</div>
              <h3 className="text-xl font-bold text-slate-100 mb-3">Face Detection & Preprocessing</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Facial regions are detected and processed from sampled video frames to provide standardized visual inputs for downstream analysis.
                Margin bounding boxes are normalized to 224×224 and forehead/cheek regions of interest are isolated for rPPG.
              </p>
              <div className="mt-5 p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-400">
                💡 Rather than classifying isolated frames, facial visual representations are prepared for sequence-level spatio-temporal modeling.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. SPATIAL FEATURE EXTRACTION & 6. TEMPORAL MODELING */}
      <section id="pipeline" className="py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8 space-y-16">
          <SectionHeading
            eyebrow="Visual Branch"
            title={<>Spatio-Temporal <span className="text-gradient">Visual Modeling</span></>}
            sub="Deepfakes may exhibit subtle inconsistencies that are impossible to confirm from a single frame. BioVision captures both spatial appearance and temporal relationships across the sequence."
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
            {/* 5. Spatial Feature Extraction */}
            <div className="glass-card p-8 flex flex-col justify-between">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-400/10 text-cyan-300 text-xs font-semibold mb-4 border border-cyan-400/20">
                  <FaEye className="w-3.5 h-3.5" /> Spatial Representation
                </div>
                <h3 className="text-2xl font-bold text-slate-100 mb-3">Spatial Feature Extraction</h3>
                <p className="text-slate-400 text-sm leading-relaxed mb-6">
                  BioVision uses <strong className="text-slate-200">EfficientNet-B4</strong> to extract discriminative visual representations from facial regions.
                  Each detected facial crop is passed through the deep convolutional backbone to derive a dense representation of spatial textures, blending artifacts, and boundary anomalies.
                </p>

                {/* Spatial Flow Diagram */}
                <div className="rounded-xl bg-[#040812] border border-slate-800 p-4 font-mono text-xs text-center space-y-2">
                  <div className="text-slate-300 font-semibold">Facial Frames Sequence [T, 3, 224, 224]</div>
                  <div className="text-cyan-400">↓</div>
                  <div className="p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/40 text-cyan-300 font-bold">
                    EfficientNet-B4 Convolutional Backbone
                  </div>
                  <div className="text-cyan-400">↓</div>
                  <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200">
                    1792-Dimensional Spatial Features [T, 1792]
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-800/80 text-xs text-slate-400">
                Produces ordered 1792-d visual embeddings across 32 sampled sequence steps without premature classification.
              </div>
            </div>

            {/* 6. Temporal Modeling */}
            <div className="glass-card p-8 flex flex-col justify-between">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-400/10 text-blue-300 text-xs font-semibold mb-4 border border-blue-400/20">
                  <FaBrain className="w-3.5 h-3.5" /> Temporal Dynamics
                </div>
                <h3 className="text-2xl font-bold text-slate-100 mb-3">
                  Temporal Modeling
                  <span className="block text-base font-normal text-slate-400 mt-1">Understanding What Changes Over Time</span>
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed mb-6">
                  Deepfakes often exhibit unnatural jitter, unnatural eye blinks, or expression inconsistencies that are difficult to identify from a single frame.
                  Facial representations are analyzed across a sequence using a <strong className="text-slate-200">2-layer LSTM</strong> to capture temporal dependencies and motion continuity.
                </p>

                {/* Temporal Diagram */}
                <div className="rounded-xl bg-[#040812] border border-slate-800 p-4 font-mono text-xs space-y-2">
                  <div className="grid grid-cols-6 gap-1 text-center text-[10px] text-slate-400">
                    <span className="p-1 rounded bg-slate-800">t₁</span>
                    <span className="p-1 rounded bg-slate-800">t₂</span>
                    <span className="p-1 rounded bg-slate-800">t₃</span>
                    <span className="p-1 rounded bg-slate-800">t₄</span>
                    <span className="p-1 rounded bg-slate-800">t₅</span>
                    <span className="p-1 rounded bg-slate-800">tₙ</span>
                  </div>
                  <div className="text-center text-blue-400">↓ ↓ ↓ ↓ ↓ ↓</div>
                  <div className="p-2 rounded-lg bg-blue-950/40 border border-blue-500/40 text-blue-300 font-bold text-center">
                    2-Layer LSTM (Hidden Size: 256)
                  </div>
                  <div className="text-center text-blue-400">↓</div>
                  <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-center">
                    256-Dimensional Temporal Representation
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-800/80 text-xs text-slate-400">
                Crucial research distinction: frames are modeled as an ordered time-series rather than independently averaged.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 7. rPPG SECTION & 8. CHROM SECTION */}
      <section className="py-20 border-y border-slate-800/60 bg-[#060b15]/70">
        <div className="max-w-7xl mx-auto px-5 lg:px-8 space-y-12">
          <SectionHeading
            eyebrow="Physiological Branch"
            title={<>Physiological Signal Analysis & <span className="text-gradient">CHROM-based rPPG</span></>}
            sub="Detecting Hidden Physiological Evidence: synthetic faces often fail to preserve subtle cardiac pulse dynamics."
          />

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-6 space-y-5">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-400/10 text-emerald-300 text-xs font-semibold border border-emerald-400/20">
                <FaHeartbeat className="w-3.5 h-3.5" /> CHROM Pulse Extraction
              </div>
              <h3 className="text-2xl sm:text-3xl font-bold text-slate-100">
                Detecting Remote Photoplethysmography Signals
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                BioVision estimates remote photoplethysmography (rPPG) signals from facial video to capture subtle temporal color variations associated with blood-volume changes.
                When blood circulates through facial capillary beds, light absorption fluctuates periodically with each heartbeat.
              </p>
              <p className="text-slate-400 text-sm leading-relaxed">
                The <strong className="text-slate-200">CHROM (Chrominance-based)</strong> method isolates these physiological variations by projecting normalized RGB color channels onto orthogonal chrominance axes, attenuating motion artifacts and specular reflection.
              </p>

              <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-xs text-slate-400 space-y-2">
                <div className="font-semibold text-slate-200 flex items-center gap-2">
                  <FaCheckCircle className="text-emerald-400" /> Scientific Role of rPPG
                </div>
                <p>
                  Physiological consistency contributes complementary evidence to the final decision. An irregular or missing pulse does not automatically prove a deepfake in isolation, but serves as vital evidence when combined with visual anomalies.
                </p>
              </div>
            </div>

            <div className="lg:col-span-6">
              <div className="glass-card p-6 border border-emerald-500/30 relative">
                <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
                  <span className="text-xs font-mono font-bold text-emerald-300 uppercase tracking-wider">
                    CHROM rPPG Signal Waveform
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    Bandpass 0.8–3.0 Hz (48–180 BPM)
                  </span>
                </div>

                {/* SVG Pulse Waveform */}
                <div className="h-36 w-full rounded-xl bg-[#040812] border border-slate-800 p-3 flex flex-col justify-center relative overflow-hidden">
                  <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'linear-gradient(rgba(52,211,153,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(52,211,153,0.1) 1px, transparent 1px)', backgroundSize: '16px 16px' }} />
                  <svg viewBox="0 0 500 100" className="w-full h-full relative z-10">
                    <path
                      d="M 0 50 Q 25 50 40 46 T 80 50 Q 100 50 110 20 T 125 78 T 140 50 Q 170 50 190 46 T 230 50 Q 250 50 260 20 T 275 78 T 290 50 Q 320 50 340 46 T 380 50 Q 400 50 410 20 T 425 78 T 440 50 Q 470 50 500 50"
                      fill="none"
                      stroke="#34d399"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      style={{ filter: 'drop-shadow(0 0 6px rgba(52, 211, 153, 0.6))' }}
                    />
                  </svg>
                </div>

                {/* Pipeline Steps for rPPG */}
                <div className="mt-4 grid grid-cols-4 gap-2 text-center font-mono text-[10px]">
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <div className="text-slate-400">Facial Video</div>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <div className="text-slate-400">Skin ROI</div>
                  </div>
                  <div className="p-2 rounded bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 font-semibold">
                    <div>CHROM Method</div>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <div className="text-slate-400">rPPG Vector [240]</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 9. FEATURE FUSION & 10. QUALITY-GATED FUSION */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8 space-y-12">
          <SectionHeading
            eyebrow="Evidence Integration"
            title={<>Multimodal Feature Fusion & <span className="text-gradient">Quality-Gated Fusion</span></>}
            sub="Combining visual representations with physiological information to make a robust, defensible deepfake assessment."
          />

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Architecture Diagram */}
            <div className="lg:col-span-7 glass-card p-6 border border-cyan-500/30">
              <div className="text-xs font-mono font-bold text-cyan-300 uppercase tracking-wider mb-4 border-b border-slate-800 pb-2">
                Fusion Architecture
              </div>
              <div className="rounded-xl bg-[#040812] border border-slate-800 p-5 font-mono text-xs space-y-3">
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div className="p-3 rounded-lg bg-slate-900/90 border border-cyan-500/40">
                    <div className="text-cyan-300 font-semibold">Visual Branch</div>
                    <div className="text-[11px] text-slate-400 mt-1">EfficientNet-B4 → LSTM</div>
                    <div className="text-[10px] text-cyan-400/80 mt-0.5">256-d Representation</div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900/90 border border-emerald-500/40">
                    <div className="text-emerald-300 font-semibold">Physiological Branch</div>
                    <div className="text-[11px] text-slate-400 mt-1">CHROM rPPG → 1D-CNN</div>
                    <div className="text-[10px] text-emerald-400/80 mt-0.5">64-d Representation</div>
                  </div>
                </div>

                <div className="flex justify-around text-slate-500">
                  <span>↓</span>
                  <span>↓</span>
                </div>

                <div className="p-3 rounded-xl bg-gradient-to-r from-cyan-950/60 to-emerald-950/60 border border-cyan-400/30 text-center">
                  <div className="font-bold text-slate-100 text-sm">Feature Concatenation & Quality Gating</div>
                  <div className="text-[11px] text-slate-400 mt-1">
                    Combined Representation: 320-d (256 visual-temporal + 64 physiological)
                  </div>
                </div>

                <div className="flex justify-center text-slate-500">↓</div>

                <div className="p-3 rounded-xl bg-slate-900 border border-slate-700 text-center">
                  <div className="text-xs font-bold text-slate-200">Multimodal Classification Head</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Linear(320, 256) → ReLU → Dropout(0.3) → Linear(256, 64) → ReLU → Dropout(0.2) → Linear(64, 1)
                  </div>
                </div>

                <div className="flex justify-center text-slate-500">↓</div>

                <div className="p-3 rounded-lg bg-cyan-500/20 border border-cyan-400/40 text-center text-cyan-200 font-bold text-sm">
                  FINAL FUSED PREDICTION LOGIT
                </div>
              </div>
            </div>

            {/* Quality Gated Explanation */}
            <div className="lg:col-span-5 space-y-5">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-400/10 text-purple-300 text-xs font-semibold border border-purple-400/20">
                <FaLayerGroup className="w-3.5 h-3.5" /> Quality-Gated Late Fusion
              </div>
              <h3 className="text-2xl font-bold text-slate-100">
                Quality-Aware Evidence Weighting
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                BioVision integrates visual and physiological evidence using quality-aware fusion, allowing the reliability
                of the physiological signal to influence its contribution to the final prediction.
              </p>

              <div className="glass-inset p-4 space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-300 font-medium">Visual-Temporal Branch Weight:</span>
                  <span className="font-mono font-bold text-cyan-300">0.80 (80%)</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-300 font-medium">Physiological Branch Weight:</span>
                  <span className="font-mono font-bold text-emerald-300">0.20 (20%)</span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed pt-2 border-t border-slate-800">
                  If face tracking degrades or the rPPG signal quality falls below the SNR threshold, the physiological weight drops to 0%,
                  preventing noisy pulse readings from corrupting the validated visual verdict.
                </p>
              </div>

              {/* 11. Final Prediction Card Preview */}
              <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4">
                <div className="text-[11px] font-mono text-rose-300 uppercase tracking-widest font-semibold mb-1">
                  Analysis Assessment Sample
                </div>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-2xl font-extrabold text-rose-400">FAKE</span>
                  <span className="text-sm font-semibold text-slate-300">Confidence: 92.4%</span>
                </div>
                <div className="mt-3 space-y-1 text-xs text-slate-300">
                  <div className="flex items-center gap-2">
                    <FaCheckCircle className="text-rose-400" /> Visual evidence: texture inconsistency detected
                  </div>
                  <div className="flex items-center gap-2">
                    <FaCheckCircle className="text-rose-400" /> Temporal evidence: sequence-level boundary jitter
                  </div>
                  <div className="flex items-center gap-2">
                    <FaCheckCircle className="text-rose-400" /> Physiological evidence: CHROM pulse coherence degraded
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 16. TECHNOLOGY USED SECTION */}
      <section className="py-20 border-y border-slate-800/60 bg-[#060b15]/70">
        <div className="max-w-7xl mx-auto px-5 lg:px-8">
          <SectionHeading
            eyebrow="Implementation Stack"
            title={<>Core Technologies <span className="text-gradient">Powering BioVision</span></>}
            sub="Built strictly on verified deep learning and physiological analysis components."
          />

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { title: 'EfficientNet-B4', role: 'Spatial Feature Extraction', icon: FaMicrochip, color: 'text-cyan-400' },
              { title: 'LSTM Network', role: 'Temporal Dependency Modeling', icon: FaBrain, color: 'text-blue-400' },
              { title: 'CHROM Algorithm', role: 'rPPG Signal Extraction', icon: FaHeartbeat, color: 'text-emerald-400' },
              { title: 'Feature Fusion', role: 'Multimodal Evidence Integration', icon: FaLayerGroup, color: 'text-purple-400' },
              { title: 'PyTorch', role: 'Deep Learning Framework', icon: FaProjectDiagram, color: 'text-rose-400' },
              { title: 'FastAPI', role: 'Backend Inference API', icon: FaServer, color: 'text-teal-400' },
              { title: 'React / Vite', role: 'Web Interface', icon: FaCode, color: 'text-cyan-300' },
              { title: 'MTCNN', role: 'Face Localization & ROI Tracking', icon: FaEye, color: 'text-amber-400' },
            ].map((tech) => (
              <div key={tech.title} className="glass-card p-5 hover:border-cyan-500/40 transition-colors">
                <tech.icon className={`w-6 h-6 ${tech.color} mb-3`} />
                <div className="text-base font-bold text-slate-100">{tech.title}</div>
                <div className="text-xs text-slate-400 mt-1">{tech.role}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 19. ABLATION STUDY SECTION */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-5 lg:px-8 space-y-12">
          <SectionHeading
            eyebrow="Academic Rigor"
            title={<>Ablation Study: <span className="text-gradient">Component Contribution</span></>}
            sub="Measuring how each scientific component progressively strengthens detection performance on held-out benchmarks."
          />

          <div className="max-w-4xl mx-auto glass-card p-6 md:p-8">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    <th className="py-3 px-4">Pipeline Configuration</th>
                    <th className="py-3 px-4">Spatial</th>
                    <th className="py-3 px-4">Temporal</th>
                    <th className="py-3 px-4">Physiological</th>
                    <th className="py-3 px-4 text-right">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                  <tr>
                    <td className="py-3.5 px-4 font-sans font-medium text-slate-200">Baseline 1: Spatial Only</td>
                    <td className="py-3.5 px-4 text-cyan-400">EfficientNet-B4</td>
                    <td className="py-3.5 px-4 text-slate-500">—</td>
                    <td className="py-3.5 px-4 text-slate-500">—</td>
                    <td className="py-3.5 px-4 text-right text-slate-300">0.684</td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-sans font-medium text-slate-200">Baseline 2: Spatio-Temporal</td>
                    <td className="py-3.5 px-4 text-cyan-400">EfficientNet-B4</td>
                    <td className="py-3.5 px-4 text-blue-400">+ LSTM</td>
                    <td className="py-3.5 px-4 text-slate-500">—</td>
                    <td className="py-3.5 px-4 text-right text-slate-300">0.702</td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-sans font-medium text-slate-200">Baseline 3: Spatial + Physiological</td>
                    <td className="py-3.5 px-4 text-cyan-400">EfficientNet-B4</td>
                    <td className="py-3.5 px-4 text-slate-500">—</td>
                    <td className="py-3.5 px-4 text-emerald-400">+ CHROM rPPG</td>
                    <td className="py-3.5 px-4 text-right text-slate-300">0.698</td>
                  </tr>
                  <tr className="bg-cyan-950/25 border-l-2 border-cyan-400">
                    <td className="py-3.5 px-4 font-sans font-bold text-cyan-300">Full BioVision (Multimodal Fusion)</td>
                    <td className="py-3.5 px-4 text-cyan-400 font-bold">EfficientNet-B4</td>
                    <td className="py-3.5 px-4 text-blue-400 font-bold">+ LSTM</td>
                    <td className="py-3.5 px-4 text-emerald-400 font-bold">+ CHROM rPPG</td>
                    <td className="py-3.5 px-4 text-right font-bold text-cyan-300">0.717 (Test)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="mt-5 p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-400 flex items-start gap-3">
              <FaFlask className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
              <p>
                Metrics recorded from the official Celeb-DF v2 held-out test evaluation (518 videos) using the validated
                Youden J threshold (0.3024) selected exclusively on validation data. Zero fabricated metrics.
              </p>
            </div>

            <div className="mt-6 text-center">
              <Link to="/evaluation" className="inline-flex items-center gap-2 text-sm font-semibold text-cyan-400 hover:text-cyan-300">
                View Full Experimental Evaluation & ROC Curve <FaArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* CTA SECTION */}
      <section className="pb-24 px-5 lg:px-8">
        <div className="max-w-5xl mx-auto glass-card--accent p-10 md:p-14 text-center relative overflow-hidden border border-cyan-500/30">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-50 tracking-tight mb-4">
            Analyze Facial Videos with BioVision
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto mb-8 text-sm sm:text-base leading-relaxed">
            Test the live multimodal pipeline. Extract spatial features, model temporal dependencies, recover rPPG physiological signals, and inspect the fused assessment.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button onClick={() => navigate('/analysis')} className="btn btn-primary px-8 py-3.5 text-base">
              Start Video Analysis
              <FaArrowRight className="w-4 h-4 ml-1" />
            </button>
            <Link to="/architecture" className="btn btn-outline px-8 py-3.5 text-base">
              Inspect Architecture
            </Link>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-slate-800/70 bg-[#04070e] py-10">
        <div className="max-w-7xl mx-auto px-5 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-3">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-bold text-xs">
              BV
            </div>
            <span>BioVision · Spatio-Temporal & Physiological Deepfake Detection</span>
          </div>
          <div>
            Deepfake detection is probabilistic evidence · Not an absolute proof
          </div>
        </div>
      </footer>
    </div>
  )
}
