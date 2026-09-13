import React, { useState, useEffect } from 'react'
import Card from '../components/Card'
import { API_BASE } from '../lib/api'
import {
  FaMicrochip,
  FaExclamationTriangle,
  FaSyncAlt,
  FaHdd,
  FaLayerGroup,
  FaBrain,
  FaHeartbeat,
  FaEye,
  FaProjectDiagram,
  FaServer,
  FaCheckCircle,
} from 'react-icons/fa'

interface ModelData {
  model_name: string
  model_version: string
  device: string
  checkpoint_path: string
  status: string
  missing_keys: string[]
  unexpected_keys: string[]
  analysis_components?: {
    name: string
    role: string
    output: string
    weighted?: boolean
    note?: string
  }[]
  verdict_note?: string
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-1 font-mono">{label}</div>
      <div className="text-base font-semibold text-slate-100">{value}</div>
    </div>
  )
}

export default function ModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/model/info?model_type=cached`)
      if (res.ok) {
        const data = await res.json()
        setModelInfo(data)
      } else {
        setError('Backend not available')
      }
    } catch {
      setError('Could not connect to backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <header className="page-head">
        <div>
          <h2 className="page-title">BioVision Architecture</h2>
          <p className="page-sub">
            Dual-branch spatio-temporal and physiological deepfake detection network
          </p>
        </div>
        <span className="chip chip--info">
          <FaProjectDiagram className="w-3.5 h-3.5" />
          Multimodal Network
        </span>
      </header>

      {error && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <FaExclamationTriangle className="w-6 h-6 text-rose-300 flex-shrink-0" />
            <div>
              <p className="text-rose-200 font-semibold">{error}</p>
              <p className="text-rose-300/80 text-xs mt-0.5">
                Start the backend server on port 8000: <code className="font-mono">python -m uvicorn backend.app.main:app --port 8000</code>
              </p>
            </div>
          </div>
          <button onClick={load} className="btn btn-outline text-sm flex-shrink-0">
            <FaSyncAlt className="w-3.5 h-3.5" />
            Retry
          </button>
        </div>
      )}

      {/* 15. DEDICATED ARCHITECTURE DIAGRAM */}
      <Card
        title="Complete BioVision System Architecture"
        subtitle="End-to-end tensor dataflow across visual, temporal, and physiological branches"
      >
        <div className="rounded-2xl bg-[#040812] border border-slate-800 p-6 font-mono text-xs text-slate-300 space-y-4">
          {/* Input video */}
          <div className="text-center">
            <div className="inline-block px-5 py-2 rounded-xl bg-slate-800/90 border border-slate-700 text-slate-100 font-bold shadow-md">
              INPUT VIDEO
            </div>
          </div>

          <div className="flex justify-center text-cyan-400">│</div>
          <div className="flex justify-center text-cyan-400">↓</div>

          {/* Face Detection */}
          <div className="text-center">
            <div className="inline-block px-5 py-2 rounded-xl bg-cyan-950/40 border border-cyan-500/40 text-cyan-200 font-bold">
              FACE DETECTION &amp; ROI CROPPING (MTCNN)
            </div>
          </div>

          <div className="flex justify-center text-cyan-400">│</div>
          <div className="flex justify-center text-slate-500">┌────────────────────────┴────────────────────────┐</div>

          {/* Dual Branch Flow */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-center pt-1">
            {/* Visual Branch */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-900/60 border border-cyan-500/30">
              <div className="text-cyan-300 font-extrabold text-sm uppercase tracking-wider flex items-center justify-center gap-2">
                <FaEye className="text-cyan-400" /> VISUAL BRANCH
              </div>
              <div className="text-cyan-400">↓</div>
              <div className="p-2.5 rounded-lg bg-slate-800 border border-cyan-500/40 text-slate-100 font-semibold">
                EfficientNet-B4 Backbone
                <div className="text-[10px] text-slate-400 mt-0.5">ImageNet Pretrained · 32 Sampled Crops</div>
              </div>
              <div className="text-cyan-400">↓</div>
              <div className="p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-cyan-200">
                Spatial Features: [32, 1792]
              </div>
              <div className="text-blue-400">↓</div>
              <div className="p-2.5 rounded-lg bg-slate-800 border border-blue-500/40 text-slate-100 font-semibold">
                2-Layer LSTM Network
                <div className="text-[10px] text-slate-400 mt-0.5">Hidden Size: 256 · Dropout: 0.20</div>
              </div>
              <div className="text-blue-400">↓</div>
              <div className="p-2 rounded-lg bg-blue-950/40 border border-blue-500/30 text-blue-200">
                Temporal Representation: [256]
              </div>
            </div>

            {/* Physiological Branch */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-900/60 border border-emerald-500/30">
              <div className="text-emerald-300 font-extrabold text-sm uppercase tracking-wider flex items-center justify-center gap-2">
                <FaHeartbeat className="text-emerald-400" /> PHYSIOLOGICAL BRANCH
              </div>
              <div className="text-emerald-400">↓</div>
              <div className="p-2.5 rounded-lg bg-slate-800 border border-emerald-500/40 text-slate-100 font-semibold">
                CHROM rPPG Algorithm
                <div className="text-[10px] text-slate-400 mt-0.5">Forehead &amp; Cheek Skin Color Variations</div>
              </div>
              <div className="text-emerald-400">↓</div>
              <div className="p-2 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-200">
                rPPG Pulse Vector: [240]
              </div>
              <div className="text-emerald-400">↓</div>
              <div className="p-2.5 rounded-lg bg-slate-800 border border-emerald-500/40 text-slate-100 font-semibold">
                1D-CNN Physiological Encoder
                <div className="text-[10px] text-slate-400 mt-0.5">Conv1d(1→32) → Conv1d(32→64) → AdaptiveAvgPool</div>
              </div>
              <div className="text-emerald-400">↓</div>
              <div className="p-2 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-200">
                Physiological Features: [64]
              </div>
            </div>
          </div>

          <div className="flex justify-center text-slate-500">└────────────────────────┬────────────────────────┘</div>
          <div className="flex justify-center text-cyan-400">↓</div>

          {/* Fusion */}
          <div className="text-center">
            <div className="p-3.5 rounded-xl bg-gradient-to-r from-cyan-950/80 via-blue-950/80 to-purple-950/80 border border-cyan-400/50 text-cyan-200 max-w-xl mx-auto shadow-lg">
              <div className="font-extrabold text-sm text-cyan-300 uppercase tracking-wider">
                MULTIMODAL FEATURE FUSION HEAD
              </div>
              <div className="text-[11px] text-slate-300 mt-1">
                Concatenation: [320] (256 visual-temporal + 64 physiological)
              </div>
              <div className="text-[10px] text-slate-400 mt-1">
                Linear(320→256) → ReLU → Dropout(0.3) → Linear(256→64) → ReLU → Dropout(0.2) → Linear(64→1)
              </div>
            </div>
          </div>

          <div className="flex justify-center text-cyan-400">↓</div>

          {/* Output */}
          <div className="text-center">
            <div className="inline-block px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500/20 via-cyan-500/20 to-rose-500/20 border border-cyan-400/40 font-bold text-slate-100 shadow-md">
              FINAL FUSED DEEPFAKE PREDICTION (Logit → Sigmoid Probability)
            </div>
          </div>
        </div>
      </Card>

      {/* MODEL STATUS & METADATA */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Live Checkpoint Status">
          {modelInfo ? (
            <div className="space-y-4">
              <Row label="Architecture Contract" value={modelInfo.model_name} />
              <Row label="Loaded Checkpoint" value={modelInfo.model_version} />
              <Row
                label="Device Environment"
                value={
                  <span className="inline-flex items-center gap-2">
                    <FaMicrochip className="w-4 h-4 text-cyan-400" />
                    {modelInfo.device.toUpperCase()}
                  </span>
                }
              />
              <Row
                label="Weight Loading Verification"
                value={
                  <span className="inline-flex items-center gap-2 text-emerald-400">
                    <FaCheckCircle className="w-4 h-4" />
                    {modelInfo.missing_keys.length === 0 && modelInfo.unexpected_keys.length === 0
                      ? 'Strict load passed (0 missing, 0 unexpected)'
                      : `${modelInfo.missing_keys.length} missing, ${modelInfo.unexpected_keys.length} unexpected`}
                  </span>
                }
              />
            </div>
          ) : (
            <p className="text-sm text-slate-500">Connecting to model metadata service…</p>
          )}
        </Card>

        <Card title="Tensor Specifications">
          <div className="space-y-3 font-mono text-xs">
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center">
              <span className="text-slate-400">Input Sequence Shape</span>
              <span className="text-cyan-300 font-bold">[B, 32, 3, 224, 224]</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center">
              <span className="text-slate-400">EfficientNet-B4 Spatial Output</span>
              <span className="text-cyan-300 font-bold">[B, 32, 1792]</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center">
              <span className="text-slate-400">LSTM Hidden Dynamics</span>
              <span className="text-blue-300 font-bold">[B, 256]</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center">
              <span className="text-slate-400">rPPG 1D-CNN Output</span>
              <span className="text-emerald-300 font-bold">[B, 64]</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center">
              <span className="text-slate-400">Multimodal Fusion Vector</span>
              <span className="text-purple-300 font-bold">[B, 320]</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center">
              <span className="text-slate-400">Classification Output</span>
              <span className="text-slate-100 font-bold">[B, 1]</span>
            </div>
          </div>
        </Card>
      </div>

      {/* ANALYSIS COMPONENTS */}
      <Card title="Integrated Multimodal Subsystems">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-cyan-500/30 space-y-2">
            <div className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
              <FaMicrochip /> Spatial Subsystem
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Extracts high-order deep convolutional features from facial crops, focusing on synthesis artifacts, boundaries, and skin textures.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-blue-500/30 space-y-2">
            <div className="flex items-center gap-2 text-blue-300 font-bold text-sm">
              <FaBrain /> Temporal Subsystem
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Models cross-frame dependencies using stacked recurrent LSTM units to detect unnatural motion, expression flicker, or discontinuity.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-emerald-500/30 space-y-2">
            <div className="flex items-center gap-2 text-emerald-300 font-bold text-sm">
              <FaHeartbeat /> Physiological Subsystem
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Recovers the blood-volume pulse via chrominance-based rPPG, checking whether authentic biological cardiac rhythms exist across the face.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
