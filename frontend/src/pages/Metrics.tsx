import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import { API_BASE, getHistory } from '../lib/api'
import { formatDuration, formatPercent } from '../lib/format'
import { FaChartLine, FaFlask, FaExclamationTriangle, FaServer, FaDatabase } from 'react-icons/fa'

interface LiveStats {
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

export default function Metrics() {
  const history = useMemo(() => getHistory(), [])
  const [live, setLive] = useState<LiveStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/dashboard/stats`)
        if (res.ok && !cancelled) {
          setLive(await res.json())
          setBackendUp(true)
        } else if (!cancelled) {
          setBackendUp(false)
        }
      } catch {
        if (!cancelled) setBackendUp(false)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  const total = history.length
  const real = history.filter((h) => h.result === 'REAL').length
  const fake = history.filter((h) => h.result === 'FAKE').length
  const noFace = history.filter((h) => h.result === 'NO_FACE').length
  const uncertain = total - real - fake - noFace
  const withConfidence = history.filter((h) => h.result !== 'NO_FACE' && typeof h.confidence === 'number' && !Number.isNaN(h.confidence))
  const avgConfidence = withConfidence.length > 0
    ? withConfidence.reduce((sum, h) => sum + h.confidence, 0) / withConfidence.length
    : 0
  const avgProcessing = total > 0 ? history.reduce((sum, h) => sum + h.processing_time, 0) / total : 0
  const avgFrames = total > 0 ? history.reduce((sum, h) => sum + h.frames_sampled, 0) / total : 0

  const liveTotal = live?.total_analyses || 0

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Metrics</h2>
          <p className="page-sub">Real statistics from the analysis pipeline — nothing fabricated</p>
        </div>
        <span
          className={`chip ${
            loading ? 'chip--idle' : backendUp === false ? 'chip--err' : 'chip--ok'
          }`}
        >
          <span className={`status-dot ${loading ? 'bg-slate-500' : backendUp === false ? 'bg-rose-400' : 'bg-emerald-400 pulse-glow'}`} />
          {loading ? 'Checking…' : backendUp === false ? 'Backend offline' : `Live server · ${liveTotal} analysis${liveTotal === 1 ? '' : 's'}`}
        </span>
      </header>

      {loading ? (
        <div className="glass-card p-10 flex items-center justify-center gap-3 text-slate-500">
          <div className="h-6 w-6 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
          <span>Loading metrics…</span>
        </div>
      ) : backendUp === false ? (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-6 flex items-start gap-4">
          <div className="h-11 w-11 rounded-xl bg-rose-500/20 border border-rose-400/40 flex items-center justify-center text-rose-300 flex-shrink-0">
            <FaServer className="w-5 h-5" />
          </div>
          <div>
            <p className="text-rose-200 font-semibold">Metrics unavailable</p>
            <p className="text-rose-300/80 text-sm mt-1 leading-relaxed max-w-2xl">
              The analysis backend is not reachable on <code className="text-rose-200 bg-rose-500/15 px-1.5 py-0.5 rounded font-mono text-xs">http://127.0.0.1:8000</code>,
              so live server metrics cannot be displayed. Start it from the project root with{' '}
              <code className="text-rose-200 bg-rose-500/15 px-1.5 py-0.5 rounded font-mono text-xs">
                python -m uvicorn backend.app.main:app --port 8000
              </code>{' '}
              and refresh this page. No accuracy or detection statistics are invented.
            </p>
          </div>
        </div>
      ) : (
        <div className="glass-card--accent p-6">
          <div className="flex items-center gap-2 text-xs font-medium text-cyan-200 mb-4">
            <FaChartLine className="w-4 h-4" />
            Live backend metrics · source: {live?.source || 'unknown'}
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <div className="text-4xl font-bold text-cyan-300">{liveTotal}</div>
              <div className="text-xs text-slate-500 mt-1">Total analyses (server)</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-emerald-400">{live?.real_detected || 0}</div>
              <div className="text-xs text-slate-500 mt-1">Real detected</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-rose-400">{live?.fake_detected || 0}</div>
              <div className="text-xs text-slate-500 mt-1">Fake detected</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-blue-400">
                {liveTotal > 0 ? `${((live?.average_confidence || 0) * 100).toFixed(0)}%` : '—'}
              </div>
              <div className="text-xs text-slate-500 mt-1">Average confidence</div>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-800/70">
            <div>
              <div className="text-2xl font-bold text-amber-400">{live?.uncertain_detected || 0}</div>
              <div className="text-xs text-slate-500 mt-1">Uncertain (server)</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-300">{live?.no_face_detected || 0}</div>
              <div className="text-xs text-slate-500 mt-1">No face (server)</div>
            </div>
          </div>
          <p className="text-[11px] text-slate-600 mt-4">
            These values come directly from <code className="font-mono">GET /dashboard/stats</code>.
          </p>
        </div>
      )}

      {total > 0 && (
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-4">
            <FaDatabase className="w-4 h-4 text-cyan-300" />
            Local analysis history (this browser) — real stored results
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-3xl font-bold text-slate-100">{total}</div>
              <div className="text-xs text-slate-500 mt-1">Analyses</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-emerald-400">{real}</div>
              <div className="text-xs text-slate-500 mt-1">Real</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-rose-400">{fake}</div>
              <div className="text-xs text-slate-500 mt-1">Fake</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-amber-400">{uncertain}</div>
              <div className="text-xs text-slate-500 mt-1">Uncertain</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-slate-300">{noFace}</div>
              <div className="text-xs text-slate-500 mt-1">No face detected</div>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6 pt-4 border-t border-slate-800/70">
            <div>
              <div className="text-2xl font-bold text-blue-400">{formatPercent(avgConfidence)}</div>
              <div className="text-xs text-slate-500 mt-1">Avg confidence</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-100">{formatDuration(avgProcessing)}</div>
              <div className="text-xs text-slate-500 mt-1">Avg processing time</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-100">{avgFrames.toFixed(1)}</div>
              <div className="text-xs text-slate-500 mt-1">Avg frames sampled</div>
            </div>
          </div>
        </div>
      )}

      <Card title="Model Benchmarks" subtitle="Not fabricated — awaiting evaluation on a public labeled test set">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'False Pos. Rate'].map((m) => (
            <div key={m} className="glass-inset p-4">
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <FaFlask className="w-4 h-4 text-slate-500" />
                {m}
              </div>
              <div className="text-sm text-slate-600 mt-2">Awaiting benchmark</div>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-600 mt-4">
          Confusion against ground truth requires a labeled evaluation dataset and is intentionally left blank rather
          than invented.
        </p>
      </Card>

      <div className="glass-card p-5 flex items-start gap-3">
        <FaExclamationTriangle className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
        <p className="text-sm text-slate-500 leading-relaxed">
          All metrics on this page come from the real <code className="text-slate-400 font-mono text-xs">/dashboard/stats</code>{' '}
          response and stored analyses. No synthetic or placeholder numbers are presented as results.
        </p>
      </div>
    </div>
  )
}
