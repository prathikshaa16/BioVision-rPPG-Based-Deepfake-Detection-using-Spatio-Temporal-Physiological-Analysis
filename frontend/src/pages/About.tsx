import React from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/Card'
import { APP_NAME, APP_TAGLINE } from '../lib/nav'
import {
  FaEye,
  FaFingerprint,
  FaMicrochip,
  FaShieldAlt,
  FaArrowRight,
  FaHeartbeat,
  FaBrain,
  FaLayerGroup,
  FaFlask,
} from 'react-icons/fa'

export default function About() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <header className="page-head">
        <div>
          <h2 className="page-title">About {APP_NAME}</h2>
          <p className="page-sub">Spatio-temporal deep learning with remote physiological signal analysis</p>
        </div>
        <span className="chip chip--info">
          <FaShieldAlt className="w-3.5 h-3.5" />
          Research Rationale
        </span>
      </header>

      {/* 25. WHY BIOVISION? */}
      <div className="glass-card--accent p-8 space-y-5 border border-cyan-500/30">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-bold text-base flex-shrink-0 shadow-[0_0_20px_rgba(34,211,238,0.4)]">
            BV
          </div>
          <div>
            <h3 className="text-2xl font-bold text-slate-50">Why BioVision?</h3>
            <p className="text-sm text-cyan-300/90 font-medium">{APP_NAME} · {APP_TAGLINE}</p>
          </div>
        </div>

        <div className="space-y-4 text-sm text-slate-300 leading-relaxed pt-2">
          <p className="text-base font-medium text-slate-100">
            Conventional visual deepfake detection can focus heavily on spatial appearance. BioVision extends this analysis
            by incorporating temporal dynamics and physiological information extracted from facial video.
          </p>
          <p className="text-slate-400">
            The goal is to evaluate complementary evidence rather than relying solely on appearance.
            While spatial generators continue to achieve photorealistic image quality, synthetic videos struggle to reproduce
            the synchronized interplay of micro-temporal expressions and natural blood-volume cardiac pulses (rPPG) across facial skin.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <Card title="Our Scientific Rationale">
          <p className="text-xs text-slate-400 leading-relaxed">
            Move beyond isolated frame classification. Real-world manipulation is sequence-level and physiological.
            Evaluating spatio-temporal representations alongside chrominance pulse dynamics provides multiple layers of forensic verification.
          </p>
        </Card>

        <Card title="Multimodal Architecture">
          <p className="text-xs text-slate-400 leading-relaxed">
            EfficientNet-B4 extracts 1,792-d spatial embeddings, a 2-layer LSTM models 256-d temporal dynamics,
            and CHROM rPPG derives a 64-d physiological representation. A quality-gated late fusion layer integrates the evidence.
          </p>
        </Card>

        <Card title="Defensible Benchmarks">
          <p className="text-xs text-slate-400 leading-relaxed">
            We adhere strictly to research integrity. No metrics are fabricated.
            Performance numbers are reported directly from validated held-out test partitions (such as Celeb-DF v2 official test) with locked thresholds.
          </p>
        </Card>
      </div>

      <Card title="The Multimodal Pipeline at a Glance">
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/80 border border-cyan-500/30 space-y-2">
              <div className="text-xs font-mono font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-2">
                <FaEye /> Spatial Branch
              </div>
              <p className="text-xs text-slate-400">
                EfficientNet-B4 analyzes facial appearance, blending margins, and synthesis artifacts per crop.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/80 border border-blue-500/30 space-y-2">
              <div className="text-xs font-mono font-bold text-blue-300 uppercase tracking-wider flex items-center gap-2">
                <FaBrain /> Temporal Branch
              </div>
              <p className="text-xs text-slate-400">
                A 2-layer LSTM models sequence relationships across 32 ordered observations to catch temporal flickering.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/80 border border-emerald-500/30 space-y-2">
              <div className="text-xs font-mono font-bold text-emerald-300 uppercase tracking-wider flex items-center gap-2">
                <FaHeartbeat /> Physiological Branch
              </div>
              <p className="text-xs text-slate-400">
                CHROM rPPG isolates periodic color shifts caused by cardiac cycles, checking for authentic biological blood flow.
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
            <span className="text-xs text-slate-400">
              Explore the detailed layer dimensions, tensor shapes, and live status on the Architecture page.
            </span>
            <Link to="/architecture" className="btn btn-primary text-xs py-2 px-4 flex-shrink-0">
              View Architecture <FaArrowRight className="ml-1" />
            </Link>
          </div>
        </div>
      </Card>
    </div>
  )
}
