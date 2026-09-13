import React, { useState, useEffect } from 'react'
import Card from '../components/Card'
import { API_BASE } from '../lib/api'
import { FaMicrochip, FaExclamationTriangle, FaSyncAlt, FaHdd, FaLayerGroup } from 'react-icons/fa'

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
      <div className="text-sm text-slate-500 mb-1">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
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
      const res = await fetch(`${API_BASE}/model/info`)
      if (res.ok) {
        const data = await res.json()
        setModelInfo(data)
      } else {
        setError('Backend not available')
      }
    } catch (err) {
      setError('Could not connect to backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Model Information</h2>
          <p className="page-sub">Spatio-temporal deep learning with physiological signal analysis</p>
        </div>
        <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border border-slate-700 bg-slate-900/60 text-slate-300">
          <FaMicrochip className="w-3.5 h-3.5 text-cyan-300" />
          multimodal fusion
        </span>
      </header>

      {error && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <FaExclamationTriangle className="w-6 h-6 text-rose-300 flex-shrink-0" />
            <div>
              <p className="text-rose-200 font-semibold">{error}</p>
              <p className="text-rose-300/80 text-sm mt-0.5">
                Start the backend on port 8000 (python -m uvicorn backend.app.main:app --port 8000), then retry.
              </p>
            </div>
          </div>
          <button onClick={load} className="btn btn-outline text-sm flex-shrink-0">
            <FaSyncAlt className="w-3.5 h-3.5" />
            Retry
          </button>
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-3 text-slate-500">
          <div className="h-6 w-6 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
          <span>Loading model information…</span>
        </div>
      )}

      {modelInfo && (
        <>
          <Card title="Model Status">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Row label="Model Name" value={modelInfo.model_name} />
              <div>
                <div className="text-sm text-slate-500 mb-1">Status</div>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 bg-emerald-400 rounded-full pulse-glow" />
                  <span className="text-lg font-semibold text-emerald-400 capitalize">{modelInfo.status}</span>
                </div>
              </div>
              <Row label="Device" value={<span className="inline-flex items-center gap-2"><FaMicrochip className="w-4 h-4 text-cyan-400" />{modelInfo.device}</span>} />
              <Row label="Checkpoint" value={<span className="inline-flex items-center gap-2 text-base"><FaHdd className="w-4 h-4 text-slate-500" />{modelInfo.model_version}</span>} />
            </div>
          </Card>

          <Card title="Architecture">
            <div className="space-y-4">
              <div>
                <p className="font-medium text-slate-100 mb-2">Base Model</p>
                <p className="text-slate-500">Cached BioVision fusion: 32 EfficientNet-B4 embeddings + CHROM-rPPG + learned temporal classifier</p>
              </div>
              <div>
                <p className="font-medium text-slate-100 mb-2">Classification Head</p>
                <div className="rounded-xl bg-black/40 border border-slate-800 p-4 text-sm font-mono text-cyan-200/90 space-y-1">
                  <div>Visual sequence [32, 1792] → 2-layer LSTM → 256-D</div>
                  <div>rPPG vector [240] → Conv1D → 64-D</div>
                  <div>LayerNorm(256) + LayerNorm(64) → 320-D fusion</div>
                  <div>Linear(320, 256) → ReLU → Dropout(0.3)</div>
                  <div>Linear(256, 64) → ReLU → Dropout(0.2) → Linear(64, 1)</div>
                </div>
              </div>
              <div>
                <p className="font-medium text-slate-100 mb-2">Output</p>
                <p className="text-slate-500">Sigmoid activation → probability [0, 1]</p>
              </div>
            </div>
          </Card>

          <Card title="Model Load Status">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="glass-inset p-4 flex justify-between items-center">
                <span className="text-slate-500">Missing keys</span>
                <span className={`font-semibold ${modelInfo.missing_keys.length === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {modelInfo.missing_keys.length}
                </span>
              </div>
              <div className="glass-inset p-4 flex justify-between items-center">
                <span className="text-slate-500">Unexpected keys</span>
                <span className={`font-semibold ${modelInfo.unexpected_keys.length === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {modelInfo.unexpected_keys.length}
                </span>
              </div>
            </div>
          </Card>

          <Card title="Analysis Components" subtitle="What actually runs in this pipeline — and what does not">
            {modelInfo.analysis_components && modelInfo.analysis_components.length > 0 ? (
              <div className="space-y-4">
                {modelInfo.analysis_components.map((component) => (
                  <div key={component.name} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono text-sm font-semibold text-cyan-200">{component.name}</span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                          component.weighted
                            ? 'bg-cyan-400/10 text-cyan-300 border-cyan-400/30'
                            : 'bg-violet-400/10 text-violet-300 border-violet-400/30'
                        }`}
                      >
                        {component.weighted ? 'Determines verdict' : 'Diagnostic only'}
                      </span>
                    </div>
                    <p className="text-sm text-slate-400 mt-1.5">{component.role}</p>
                    <p className="text-xs text-slate-500 mt-0.5">{component.output}</p>
                    {component.note && <p className="text-xs text-slate-500 mt-1 italic">{component.note}</p>}
                  </div>
                ))}
                {modelInfo.verdict_note && (
                  <p className="text-xs text-slate-500 bg-slate-800/40 border border-slate-700/60 rounded-xl p-3">
                    {modelInfo.verdict_note}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                Analysis components are reported by the backend. If empty, the server may be running an older build.
              </p>
            )}
          </Card>

          <Card title="Inference Pipeline">
            <div className="space-y-4 text-sm text-slate-500">
              <div>
                <p className="font-semibold text-slate-200 mb-1 flex items-center gap-2"><FaLayerGroup className="w-4 h-4 text-cyan-400" />Input Processing</p>
                <p>Video → uniform 32-observation sampling → largest-face crop → ImageNet preprocessing → EfficientNet-B4 embeddings [32,1792]</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-1">Model Inference</p>
                <p>Embedding sequence → 2-layer LSTM temporal summary → 256-D visual representation</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-1">Physiological Signal (rPPG)</p>
                <p>Forehead + cheek ROIs → CHROM projection → detrend → 0.8–3.0 Hz bandpass → fixed [240] physiological vector → 64-D representation.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-1">Aggregation</p>
                <p>Concatenate 256-D visual-temporal and 64-D rPPG representations → trained 320-D fusion classifier → fake logit → sigmoid probability.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-1">Classification</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>REAL: fake probability ≤ 0.40</li>
                  <li>FAKE: fake probability ≥ 0.60</li>
                  <li>UNCERTAIN: 0.40 &lt; fake probability &lt; 0.60</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Performance Characteristics">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="font-semibold text-slate-200 mb-2">Accuracy Metrics</p>
                <p className="text-slate-500">Official Celeb-DF v2 test metrics are available on the Metrics page. Results still depend on video quality, resolution, and subject matter.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-2">Processing Speed</p>
                <p className="text-slate-500">Typically 30–90 seconds per video on CPU; the rPPG physiological extraction is the slowest step.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-2">Limitations</p>
                <p className="text-slate-500">Model bias, failure modes on unusual content, requires visible faces for analysis.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-2">Recommendations</p>
                <p className="text-slate-500">For critical applications, combine with manual review and other forensic methods.</p>
              </div>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
