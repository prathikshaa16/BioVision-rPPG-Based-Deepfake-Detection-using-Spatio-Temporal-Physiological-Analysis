import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import { API_BASE, mergeHistory, saveLastResult } from '../lib/api'
import type { AnalysisResult } from '../lib/types'
import { isAnalysisResult } from '../lib/types'
import { formatDateTime, formatDuration, formatPercent } from '../lib/format'
import { FaHistory, FaExclamationTriangle, FaArrowRight, FaPlusCircle, FaSearch } from 'react-icons/fa'

const RESULT_BADGE: Record<AnalysisResult['result'], string> = {
  REAL: 'bg-emerald-400/10 text-emerald-300 border border-emerald-400/30',
  FAKE: 'bg-rose-400/10 text-rose-300 border border-rose-400/30',
  UNCERTAIN: 'bg-amber-400/10 text-amber-300 border border-amber-400/30',
  NO_FACE: 'bg-slate-400/10 text-slate-300 border border-slate-400/30',
}

const DOT: Record<AnalysisResult['result'], string> = {
  REAL: 'bg-emerald-400',
  FAKE: 'bg-rose-400',
  UNCERTAIN: 'bg-amber-400',
  NO_FACE: 'bg-slate-400',
}

export default function History() {
  const [entries, setEntries] = useState<AnalysisResult[]>([])
  const [loading, setLoading] = useState(true)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      setEntries(mergeHistory([]))
      try {
        const res = await fetch(`${API_BASE}/analyses`)
        if (res.ok && !cancelled) {
          const data: unknown = await res.json()
          if (Array.isArray(data)) {
            setEntries(mergeHistory(data.filter(isAnalysisResult)))
            setBackendUp(true)
          }
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

  const openResult = (entry: AnalysisResult) => {
    saveLastResult(entry)
    navigate('/results', { state: { upload: entry } })
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">History</h2>
          <p className="page-sub">Real analyses from the EfficientNet-B4 pipeline</p>
        </div>
        {entries.length > 0 && (
          <button onClick={() => navigate('/analysis')} className="btn btn-primary text-sm px-4 py-2">
            <FaPlusCircle className="w-4 h-4" />
            New Analysis
          </button>
        )}
      </header>

      {backendUp === false && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
          <FaExclamationTriangle className="w-5 h-5 text-amber-300 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <p className="text-amber-200 font-medium">Backend not reachable</p>
            <p className="text-amber-300/80 mt-1">
              Showing analyses stored in this browser. Start the server on port 8000 to include server-side history.
            </p>
          </div>
        </div>
      )}

      <Card title="Recent Analyses" subtitle={`${entries.length} result${entries.length === 1 ? '' : 's'}`}>
        {loading ? (
          <div className="flex items-center gap-3 py-8 justify-center text-slate-500">
            <div className="h-6 w-6 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
            <span>Loading history…</span>
          </div>
        ) : entries.length === 0 ? (
          <div className="text-center py-14">
            <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center mx-auto mb-4">
              <FaHistory className="w-6 h-6 text-cyan-300" />
            </div>
            <p className="text-slate-200 text-lg mb-2">No analyses yet</p>
            <p className="text-slate-500 text-sm mb-6">Upload a video to generate your first deepfake analysis.</p>
            <button onClick={() => navigate('/analysis')} className="btn btn-primary px-6 py-2.5">
              Analyze a Video
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-slate-500 border-b border-slate-800">
                <tr>
                  <th className="px-2 py-2 font-medium">Filename</th>
                  <th className="px-2 py-2 font-medium">Result</th>
                  <th className="px-2 py-2 font-medium">Confidence</th>
                  <th className="px-2 py-2 font-medium">Date</th>
                  <th className="px-2 py-2 font-medium">Processing Time</th>
                  <th className="px-2 py-2" />
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={entry.analysis_id} className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors">
                    <td className="px-2 py-3 max-w-xs">
                      <div className="flex items-center gap-2 font-medium text-slate-200 truncate">
                        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${DOT[entry.result]}`} />
                        {entry.filename}
                      </div>
                    </td>
                    <td className="px-2 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${RESULT_BADGE[entry.result]}`}>
                        {entry.result}
                      </span>
                    </td>
                    <td className="px-2 py-3 text-slate-300">
                      {entry.result === 'NO_FACE' ? '—' : formatPercent(entry.confidence)}
                    </td>
                    <td className="px-2 py-3 text-slate-500 whitespace-nowrap">{formatDateTime(entry.analyzed_at)}</td>
                    <td className="px-2 py-3 text-slate-500 whitespace-nowrap">{formatDuration(entry.processing_time)}</td>
                    <td className="px-2 py-3 text-right">
                      <button
                        onClick={() => openResult(entry)}
                        className="inline-flex items-center gap-1.5 text-cyan-400 hover:text-cyan-300 font-medium"
                      >
                        View <FaSearch className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="text-center">
        <button onClick={() => navigate('/analysis')} className="btn btn-outline">
          <FaArrowRight className="w-4 h-4" />
          Analyze Another Video
        </button>
      </div>
    </div>
  )
}
