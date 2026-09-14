import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import { ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { API_BASE } from '../lib/api'
import { getHistory } from '../lib/api'
import type { AnalysisResult } from '../lib/types'
import { formatPercent, formatDateTime } from '../lib/format'
import {
  FaChartPie,
  FaMicrochip,
  FaVideo,
  FaArrowRight,
  FaExclamationTriangle,
  FaPlusCircle,
  FaCheckCircle,
  FaTimesCircle,
} from 'react-icons/fa'

interface DashboardStats {
  total_analyses: number
  fake_detected: number
  real_detected: number
  uncertain_detected: number
  no_face_detected: number
  average_confidence: number
  model_name: string
  device: string
  source: string
}

interface ModelInfo {
  model_name: string
  model_version: string
  device: string
  status: string
}

const COLORS: Record<string, string> = { Real: '#34d399', Fake: '#fb7185', Uncertain: '#fbbf24', 'No Face': '#94a3b8' }

function PieTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null
  const p = payload[0]
  return (
    <div className="rounded-lg border border-slate-700 bg-[#0a1120]/95 px-3 py-2 shadow-xl">
      <div className="flex items-center gap-2 text-xs text-slate-300">
        <span className="w-2 h-2 rounded-full" style={{ background: p.payload.color }} />
        {p.name}: <span className="font-semibold">{p.value}</span>
      </div>
    </div>
  )
}

function StatCard({ title, value, sub, accent, icon }: { title: string; value: string; sub: string; accent: string; icon: React.ReactNode }) {
  return (
    <Card title={title}>
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className={`text-4xl font-bold ${accent}`}>{value}</div>
          <p className="text-sm text-slate-500 mt-1">{sub}</p>
        </div>
        <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center text-cyan-300">
          {icon}
        </div>
      </div>
    </Card>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)
  const history = useMemo(() => getHistory(), [])
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false

    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/dashboard/stats`)
        if (!cancelled) {
          if (res.ok) {
            const data = await res.json()
            setStats(data)
            setBackendUp(true)
          } else {
            setBackendUp(false)
          }
        }
      } catch {
        if (!cancelled) setBackendUp(false)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    const fetchModel = async () => {
      try {
        const res = await fetch(`${API_BASE}/model/info`)
        if (res.ok && !cancelled) setModelInfo(await res.json())
      } catch {
        // model info is non-critical
      }
    }

    fetchStats()
    fetchModel()
    const interval = setInterval(fetchStats, 5000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  const isLive = !!stats && stats.source === 'live'
  const total = stats?.total_analyses || 0
  const fake = stats?.fake_detected || 0
  const real = stats?.real_detected || 0
  const uncertain = stats?.uncertain_detected ?? Math.max(0, total - real - fake)
  const noFace = stats?.no_face_detected ?? 0
  const avgConfidence = (stats?.average_confidence || 0) * 100

  const distribution = [
    { name: 'Real', value: real, color: COLORS.Real },
    { name: 'Fake', value: fake, color: COLORS.Fake },
    { name: 'Uncertain', value: uncertain, color: COLORS.Uncertain },
    { name: 'No Face', value: noFace, color: COLORS['No Face'] },
  ].filter((d) => d.value > 0)

  const recentActivity = history.slice(0, 5)
  const totalAnalyses = total > 0 ? total : history.length

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-sub">AI-powered video forensics and analysis</p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-full text-xs font-medium border ${
              backendUp === null
                ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                : backendUp === false
                  ? 'border-rose-500/30 bg-rose-500/10 text-rose-300'
                  : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                backendUp === null ? 'bg-amber-400' : backendUp === false ? 'bg-rose-400' : 'bg-emerald-400 pulse-glow'
              }`}
            />
            {backendUp === null ? 'Checking…' : backendUp === false ? 'Backend Offline' : isLive ? 'Live Data' : 'Ready'}
          </span>
          <button onClick={() => navigate('/analysis')} className="btn btn-primary text-sm px-4 py-2">
            <FaVideo className="w-3.5 h-3.5" />
            Analyze
          </button>
        </div>
      </header>

      {backendUp === false && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
          <FaExclamationTriangle className="w-5 h-5 text-amber-300 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <p className="text-amber-200 font-medium">Backend not reachable</p>
            <p className="text-amber-300/80 mt-1">
              Stats below reflect analyses stored in this browser. Start the server from the project root with{' '}
              <code className="text-amber-200 bg-amber-500/15 px-1.5 py-0.5 rounded font-mono text-xs">
                python -m uvicorn backend.app.main:app --port 8000
              </code>{' '}
              for live server statistics.
            </p>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-3 text-slate-500">
          <div className="h-6 w-6 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
          <span>Loading statistics…</span>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard title="Total Analyses" value={String(totalAnalyses)} sub={isLive ? 'Server + browser records' : 'Stored in this browser'} accent="text-cyan-300" icon={<FaChartPie className="w-5 h-5" />} />
            <StatCard title="Real Videos" value={String(real)} sub={total > 0 ? `${((real / total) * 100).toFixed(0)}% of total` : 'No real videos yet'} accent="text-emerald-400" icon={<FaCheckCircle className="w-5 h-5" />} />
            <StatCard title="Fake Detected" value={String(fake)} sub={total > 0 ? `${((fake / total) * 100).toFixed(0)}% of total` : 'No fake videos yet'} accent="text-rose-400" icon={<FaTimesCircle className="w-5 h-5" />} />
            <StatCard title="Avg Confidence" value={total > 0 ? `${avgConfidence.toFixed(0)}%` : '—'} sub="Analysis certainty" accent="text-blue-400" icon={<FaMicrochip className="w-5 h-5" />} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card title="Result Distribution" subtitle={isLive ? 'From live backend statistics' : 'From this browser'}>
              {total > 0 ? (
                <div className="flex flex-col items-center py-2">
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={distribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={44}
                        outerRadius={78}
                        dataKey="value"
                        paddingAngle={3}
                        stroke="none"
                      >
                        {distribution.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex flex-wrap justify-center gap-4 mt-2">
                    {distribution.map((d) => (
                      <span key={d.name} className="flex items-center gap-1.5 text-xs text-slate-400">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ background: d.color }} />
                        {d.name}: <span className="font-semibold text-slate-200">{d.value}</span>
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="h-32 flex flex-col items-center justify-center text-slate-500 text-sm gap-2">
                  <FaChartPie className="w-6 h-6 text-slate-600" />
                  No analyses yet
                </div>
              )}
            </Card>

            <Card title="Model Status" subtitle="EfficientNet-B4 forensic classifier">
              {modelInfo ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Model</span>
                    <span className="font-semibold text-slate-100">{modelInfo.model_name}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Status</span>
                    <span className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${backendUp === false ? 'bg-slate-600' : 'bg-emerald-400 pulse-glow'}`} />
                      <span className={`font-semibold capitalize ${backendUp === false ? 'text-slate-500' : 'text-emerald-400'}`}>
                        {backendUp === false ? 'unavailable' : modelInfo.status}
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Device</span>
                    <span className="font-semibold text-slate-100">{modelInfo.device}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Checkpoint</span>
                    <span className="font-mono text-xs text-slate-500 truncate max-w-[50%]">{modelInfo.model_version}</span>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Model</span>
                    <span className="font-semibold text-slate-100">EfficientNet-B4</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Status</span>
                    <span className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${backendUp === false ? 'bg-slate-600' : 'bg-emerald-400 pulse-glow'}`} />
                      <span className={`font-semibold ${backendUp === false ? 'text-slate-500' : 'text-emerald-400'}`}>
                        {backendUp === false ? 'Unavailable' : 'Ready'}
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">Device</span>
                    <span className="font-semibold text-slate-100">{stats?.device || 'CPU'}</span>
                  </div>
                </div>
              )}
            </Card>

            <Card title="Quick Actions">
              <div className="space-y-2.5">
                <button onClick={() => navigate('/analysis')} className="btn btn-primary w-full justify-between">
                  Analyze Video
                  <FaArrowRight className="w-4 h-4" />
                </button>
                <button onClick={() => navigate('/history')} className="btn btn-outline w-full justify-between">
                  View History
                  <FaArrowRight className="w-4 h-4" />
                </button>
                <button onClick={() => navigate('/model')} className="btn btn-outline w-full justify-between">
                  Model Details
                  <FaArrowRight className="w-4 h-4" />
                </button>
                <button onClick={() => navigate('/metrics')} className="btn btn-ghost w-full justify-between">
                  Metrics
                  <FaArrowRight className="w-4 h-4" />
                </button>
              </div>
            </Card>
          </div>

          {total === 0 && history.length === 0 && (
            <div className="glass-card p-10 text-center">
              <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center mx-auto mb-5">
                <FaVideo className="w-7 h-7 text-cyan-300" />
              </div>
              <p className="text-slate-200 text-lg mb-2">No analyses yet</p>
              <p className="text-slate-500 text-sm mb-6">Upload a video to begin analyzing for deepfakes with the real model.</p>
              <button onClick={() => navigate('/analysis')} className="btn btn-primary px-7 py-3">
                <FaPlusCircle className="w-4 h-4" />
                Upload Your First Video
              </button>
            </div>
          )}

          {recentActivity.length > 0 && (
            <Card title="Recent Activity" subtitle="Most recent analyses from this browser">
              <div className="space-y-2">
                {recentActivity.map((a: AnalysisResult) => (
                  <button
                    key={a.analysis_id}
                    onClick={() => navigate('/results', { state: { upload: a } })}
                    className="w-full flex items-center justify-between gap-3 p-3 rounded-xl glass-inset hover:bg-slate-800/40 transition-colors text-left"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span
                        className={`w-2 h-2 rounded-full flex-shrink-0 ${
                          a.result === 'REAL'
                            ? 'bg-emerald-400'
                            : a.result === 'FAKE'
                              ? 'bg-rose-400'
                              : a.result === 'NO_FACE'
                                ? 'bg-slate-400'
                                : 'bg-amber-400'
                        }`}
                      />
                      <span className="text-sm font-medium text-slate-200 truncate">{a.filename}</span>
                    </div>
                    <div className="flex items-center gap-4 flex-shrink-0">
                      <span className="text-xs text-slate-500">{formatDateTime(a.analyzed_at)}</span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${
                          a.result === 'REAL'
                            ? 'bg-emerald-400/10 text-emerald-300 border-emerald-400/30'
                            : a.result === 'FAKE'
                              ? 'bg-rose-400/10 text-rose-300 border-rose-400/30'
                              : a.result === 'NO_FACE'
                                ? 'bg-slate-400/10 text-slate-300 border-slate-400/30'
                                : 'bg-amber-400/10 text-amber-300 border-amber-400/30'
                        }`}
                      >
                        {a.result === 'NO_FACE' ? 'NO FACE DETECTED' : `${a.result} · ${formatPercent(a.confidence, 0)}`}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </Card>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card title="About BioVision">
              <div className="space-y-2 text-sm text-slate-500 leading-relaxed">
                <p>
                  BioVision combines spatial facial representations, temporal sequence dynamics, and remote photoplethysmography (rPPG)
                  physiological signals to detect manipulated videos.
                </p>
                <p>
                  EfficientNet-B4 extracts 1,792-d spatial features, a 2-layer LSTM captures sequence-level temporal dependencies,
                  and CHROM rPPG extracts subcutaneous blood-volume pulse signals.
                </p>
                <p>
                  Quality-gated late fusion integrates visual-temporal evidence (80%) with physiological evidence (20%)
                  to produce a defensible multimodal assessment.
                </p>
              </div>
            </Card>
            <Card title="Important Notes">
              <ul className="space-y-2 text-sm text-slate-500 list-disc list-inside">
                <li>No analysis is 100% accurate</li>
                <li>For critical applications, combine with manual review</li>
                <li>Videos require visible faces for analysis</li>
                <li>Results may vary based on video quality</li>
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
