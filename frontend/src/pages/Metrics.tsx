import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import { API_BASE, getHistory } from '../lib/api'
import { formatDuration, formatPercent } from '../lib/format'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import type { EvaluationResponse } from '../lib/types'
import { FaChartLine, FaExclamationTriangle, FaServer, FaDatabase } from 'react-icons/fa'

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
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [res, evaluationRes] = await Promise.all([
          fetch(`${API_BASE}/dashboard/stats`),
          fetch(`${API_BASE}/evaluation/metrics`),
        ])
        if (res.ok && !cancelled) {
          setLive(await res.json())
          if (evaluationRes.ok) setEvaluation(await evaluationRes.json())
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

      {evaluation && (
        <div className="space-y-6">
          <Card title="Official Evaluation" subtitle={`${evaluation.metrics.dataset} · preserved held-out test result`}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                ['Accuracy', evaluation.metrics.accuracy, 'text-cyan-300'],
                ['Precision', evaluation.metrics.precision, 'text-emerald-300'],
                ['Recall / Sensitivity', evaluation.metrics.recall_sensitivity, 'text-rose-300'],
                ['Specificity', evaluation.metrics.specificity, 'text-amber-300'],
                ['F1 score', evaluation.metrics.f1, 'text-blue-300'],
                ['Balanced accuracy', evaluation.metrics.balanced_accuracy, 'text-violet-300'],
                ['ROC-AUC', evaluation.metrics.roc_auc, 'text-cyan-300'],
                ['Validation AUC', evaluation.metrics.validation_auc, 'text-slate-200'],
              ].map(([label, value, color]) => (
                <div key={String(label)} className="glass-inset p-4">
                  <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
                  <div className={`text-3xl font-bold mt-2 ${color}`}>{formatPercent(Number(value))}</div>
                </div>
              ))}
            </div>
            <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              {[
                ['True positives', evaluation.metrics.TP],
                ['True negatives', evaluation.metrics.TN],
                ['False positives', evaluation.metrics.FP],
                ['False negatives', evaluation.metrics.FN],
              ].map(([label, value]) => (
                <div key={String(label)} className="flex items-center justify-between border-t border-slate-800/70 pt-3">
                  <span className="text-slate-500">{label}</span>
                  <span className="font-semibold text-slate-200">{value}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-5">Official test threshold: {evaluation.metrics.selected_threshold.toFixed(3)} · {evaluation.metrics.test_samples} videos · no test-time tuning.</p>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="ROC Curve" subtitle="Saved official test predictions">
              <div className="h-64">
                <ResponsiveContainer>
                  <LineChart data={evaluation.roc} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                    <XAxis dataKey="false_positive_rate" type="number" domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                    <YAxis dataKey="true_positive_rate" type="number" domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                    <Tooltip formatter={(value) => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'} />
                    <Line type="monotone" dataKey="true_positive_rate" stroke="#22d3ee" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card title="Confusion Matrix" subtitle="Official test counts">
              <div className="grid grid-cols-2 gap-3 h-64">
                {[
                  ['REAL → REAL', evaluation.metrics.TN, 'bg-emerald-400/15 border-emerald-400/25 text-emerald-200'],
                  ['REAL → FAKE', evaluation.metrics.FP, 'bg-rose-400/15 border-rose-400/25 text-rose-200'],
                  ['FAKE → REAL', evaluation.metrics.FN, 'bg-amber-400/15 border-amber-400/25 text-amber-200'],
                  ['FAKE → FAKE', evaluation.metrics.TP, 'bg-cyan-400/15 border-cyan-400/25 text-cyan-200'],
                ].map(([label, value, tone]) => (
                  <div key={String(label)} className={`rounded-xl border flex flex-col items-center justify-center ${tone}`}>
                    <span className="text-xs uppercase tracking-wider opacity-80">{label}</span>
                    <span className="text-4xl font-bold mt-2">{value}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}

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
