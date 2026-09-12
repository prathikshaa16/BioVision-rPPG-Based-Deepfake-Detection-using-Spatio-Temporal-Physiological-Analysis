import React, { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import SimpleLineChart from '../components/charts/SimpleLineChart'
import RppgCharts from '../components/charts/RppgCharts'
import type { AnalysisResult, FrameOverlay, RppgData } from '../lib/types'
import { isAnalysisResult } from '../lib/types'
import { getLastResult, saveLastResult, addToHistory } from '../lib/api'
import { formatFileSize, formatDuration, formatDateTime, formatPercent, buildExplanation } from '../lib/format'
import { downloadReportPdf } from '../lib/pdf'
import {
  FaCheckCircle,
  FaTimesCircle,
  FaQuestionCircle,
  FaArrowRight,
  FaHistory,
  FaExclamationTriangle,
  FaChartLine,
  FaUserSlash,
  FaFilePdf,
  FaHeartbeat,
  FaWaveSquare,
  FaInfoCircle,
} from 'react-icons/fa'

interface ResultTheme {
  label: string
  title: string
  description: string
  ring: string
  ringSoft: string
  solid: string
  textColor: string
  border: string
  chip: string
  icon: React.ReactNode
}

const THEMES: Record<AnalysisResult['result'], ResultTheme> = {
  REAL: {
    label: 'REAL',
    title: 'Authentic Video',
    description: 'No deepfake indicators were detected across the sampled frames.',
    ring: '#34d399',
    ringSoft: 'rgba(52, 211, 153, 0.18)',
    solid: 'from-emerald-400 to-emerald-600',
    textColor: 'text-emerald-400',
    border: 'border-emerald-400/30',
    chip: 'bg-emerald-400/10 text-emerald-300 border border-emerald-400/30',
    icon: <FaCheckCircle className="w-7 h-7" />,
  },
  FAKE: {
    label: 'FAKE',
    title: 'Deepfake Detected',
    description: 'The model detected manipulation artifacts with elevated confidence.',
    ring: '#fb7185',
    ringSoft: 'rgba(251, 113, 133, 0.18)',
    solid: 'from-rose-400 to-rose-600',
    textColor: 'text-rose-400',
    border: 'border-rose-400/30',
    chip: 'bg-rose-400/10 text-rose-300 border border-rose-400/30',
    icon: <FaTimesCircle className="w-7 h-7" />,
  },
  UNCERTAIN: {
    label: 'UNCERTAIN',
    title: 'Inconclusive Result',
    description: 'Frame predictions were too close to the decision boundary to be conclusive.',
    ring: '#fbbf24',
    ringSoft: 'rgba(251, 191, 36, 0.18)',
    solid: 'from-amber-400 to-amber-600',
    textColor: 'text-amber-400',
    border: 'border-amber-400/30',
    chip: 'bg-amber-400/10 text-amber-300 border border-amber-400/30',
    icon: <FaQuestionCircle className="w-7 h-7" />,
  },
  NO_FACE: {
    label: 'NO FACE DETECTED',
    title: 'No Verifiable Face Found',
    description: 'No usable face was detected across the sampled frames, so no deepfake verdict could be produced.',
    ring: '#94a3b8',
    ringSoft: 'rgba(148, 163, 184, 0.18)',
    solid: 'from-slate-400 to-slate-600',
    textColor: 'text-slate-300',
    border: 'border-slate-400/30',
    chip: 'bg-slate-400/10 text-slate-300 border border-slate-400/30',
    icon: <FaUserSlash className="w-7 h-7" />,
  },
}

function ConfidenceRing({ confidence, color, glow }: { confidence: number; color: string; glow: string }) {
  const size = 200
  const radius = 84
  const stroke = 12
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(1, confidence))
  const offset = circumference * (1 - clamped)

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <div className="absolute inset-3 rounded-full blur-2xl" style={{ background: glow }} />
      <svg width={size} height={size} className="relative transform -rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(148,163,184,0.15)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.22,1,0.36,1)', filter: `drop-shadow(0 0 8px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-5xl font-extrabold text-slate-50">{Math.round(clamped * 100)}%</div>
        <div className="text-[11px] text-slate-500 mt-1 uppercase tracking-widest">Confidence</div>
      </div>
    </div>
  )
}

function StatTile({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <Card title={label}>
      <div className={`text-3xl font-bold ${accent || 'text-slate-100'}`}>{value}</div>
      {sub && <p className="text-sm text-slate-500 mt-1">{sub}</p>}
    </Card>
  )
}

function ProbabilityBalance({ fakeProbability, realProbability }: { fakeProbability: number; realProbability: number }) {
  const fake = Math.max(0, Math.min(100, fakeProbability * 100))
  const real = Math.max(0, Math.min(100, realProbability * 100))

  return (
    <div className="mt-5 space-y-2.5">
      <div className="flex items-center justify-between text-[11px] uppercase tracking-wider">
        <span className="text-emerald-300">Real signal</span>
        <span className="text-rose-300">Manipulation signal</span>
      </div>
      <div className="h-2.5 rounded-full bg-slate-800 overflow-hidden flex">
        <div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500" style={{ width: `${real}%` }} />
        <div className="h-full bg-gradient-to-r from-rose-400 to-rose-500" style={{ width: `${fake}%` }} />
      </div>
      <div className="flex items-center justify-between text-xs font-semibold">
        <span className="text-slate-300">{formatPercent(realProbability)}</span>
        <span className="text-slate-300">{formatPercent(fakeProbability)}</span>
      </div>
    </div>
  )
}

function probabilityColor(p: number | null): string {
  if (p === null) return 'bg-slate-700'
  if (p >= 0.6) return 'bg-rose-400'
  if (p <= 0.4) return 'bg-emerald-400'
  return 'bg-amber-400'
}

function classifyFrame(p: number | null): string {
  if (p === null) return '—'
  if (p >= 0.6) return 'FAKE'
  if (p <= 0.4) return 'REAL'
  return 'UNCERTAIN'
}

function classifyChip(p: number | null): string {
  if (p === null) return 'chip chip--idle'
  if (p >= 0.6) return 'chip chip--err'
  if (p <= 0.4) return 'chip chip--ok'
  return 'chip chip--warn'
}

function FaceVisualization({ result }: { result: AnalysisResult }) {
  const overlays = result.frame_overlays || []
  const hasImages = overlays.some((o) => o.image_url)

  if (hasImages) {
    const boxColor: Record<string, string> = {
      'emerald-400': '#34d399',
      'rose-400': '#fb7185',
      'amber-400': '#fbbf24',
    }
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {overlays.map((o: FrameOverlay) => {
          const p = o.prediction ?? null
          const colorName = p === null ? 'amber-400' : p >= 0.6 ? 'rose-400' : p <= 0.4 ? 'emerald-400' : 'amber-400'
          const color = boxColor[colorName]
          return (
            <div key={o.index} className="glass-inset overflow-hidden relative aspect-video">
              <img src={o.image_url} alt={`Frame ${o.index}`} className="w-full h-full object-cover" />
              {(o.boxes || []).map((b, i) => (
                <div
                  key={i}
                  className="absolute border-2"
                  style={{ borderColor: color, left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` }}
                />
              ))}
              <span className="absolute top-1.5 left-1.5 text-[10px] font-semibold px-1.5 py-0.5 rounded bg-black/70" style={{ color }}>
                Frame {o.index} · {(o.boxes || []).length} face(s)
              </span>
            </div>
          )
        })}
      </div>
    )
  }

  const rows = result.frame_results?.length
    ? result.frame_results
    : (result.frame_predictions || []).map((p, i) => ({ index: i, faces: result.frames_with_faces > 0 ? 1 : 0, prediction: p, error: null }))

  if (rows.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500 text-sm">
        No per-frame breakdown was returned for this analysis.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {rows.map((r) => {
          const p = r.prediction
          const hasFace = r.faces > 0
          return (
            <div key={r.index} className="glass-inset p-3 relative overflow-hidden">
              <div className={`absolute inset-x-0 top-0 h-1 ${probabilityColor(p)}`} />
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono text-slate-500">F{r.index + 1}</span>
                {r.error ? (
                  <FaExclamationTriangle className="w-3.5 h-3.5 text-slate-500" />
                ) : hasFace ? (
                  <span className="w-2 h-2 rounded-full bg-cyan-400" title="Face detected" />
                ) : (
                  <span className="w-2 h-2 rounded-full bg-slate-600" title="No face" />
                )}
              </div>
              <div className="text-sm font-semibold text-slate-200">
                {p === null ? '—' : formatPercent(p, 1)}
              </div>
              <div className="text-[10px] text-slate-500">
                {r.error ? r.error : hasFace ? `${r.faces} face${r.faces > 1 ? 's' : ''}` : 'no face'}
              </div>
              <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${probabilityColor(p)}`}
                  style={{ width: `${p === null ? 0 : Math.max(2, Math.min(100, p * 100))}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
      <p className="text-[11px] text-slate-600">
        Real per-frame output from the pipeline. Face bounding-box overlays will render here automatically if the
        backend ever includes frame previews ({'frame_overlays'}).
      </p>
    </div>
  )
}

function NoFaceView({ result }: { result: AnalysisResult }) {
  const navigate = useNavigate()
  const facesDetected = result.faces_detected ?? result.frames_with_faces ?? 0
  const notProvided = <span className="italic text-slate-600 text-xs">Not provided by analysis</span>

  const frameRows = result.frame_results && result.frame_results.length
    ? result.frame_results
    : (result.sampled_indices || []).map((index, i) => ({
        index,
        faces: 0,
        prediction: null,
        error: i >= (result.frames_with_faces || 0) ? 'NO_FACE_DETECTED' : 'frame_read_failed',
      }))

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Analysis Results</h2>
          <p className="page-sub">
            <span className="font-medium text-slate-300">{result.filename}</span>
            {result.analyzed_at && <span className="text-slate-600"> · {formatDateTime(result.analyzed_at)}</span>}
          </p>
        </div>
        <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${THEMES.NO_FACE.chip}`}>
          <FaUserSlash className="w-4 h-4" />
          NO FACE DETECTED
        </span>
      </header>

      <div className={`glass-card--accent p-6 md:p-8 ${THEMES.NO_FACE.border}`}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-center">
          <div className="flex justify-center">
            <div className="relative" style={{ width: 200, height: 200 }}>
              <div className="absolute inset-3 rounded-full blur-2xl" style={{ background: THEMES.NO_FACE.ringSoft }} />
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-slate-400 to-slate-600 text-white flex items-center justify-center">
                  <FaUserSlash className="w-12 h-12" />
                </div>
                <div className="text-2xl font-extrabold text-slate-200">NO FACE</div>
              </div>
            </div>
          </div>

          <div className="text-center md:text-left">
            <div className="flex items-center justify-center md:justify-start gap-3 mb-3">
              <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${THEMES.NO_FACE.solid} text-white flex items-center justify-center`}>
                <FaUserSlash className="w-7 h-7" />
              </div>
              <div>
                <div className={`text-4xl font-extrabold tracking-tight ${THEMES.NO_FACE.textColor}`}>NO FACE DETECTED</div>
                <div className="text-sm font-medium text-slate-400 mt-1">{THEMES.NO_FACE.title}</div>
              </div>
            </div>
            <p className="text-sm text-slate-500 max-w-sm mx-auto md:mx-0 leading-relaxed">
              No detectable face was found in the sampled frames. A reliable deepfake verdict cannot be determined from this video.
            </p>
          </div>

          <div className="space-y-2.5 text-sm">
            <div className="flex justify-between border-b border-slate-700/70 pb-2">
              <span className="text-slate-500">Faces Detected</span>
              <span className="font-semibold text-slate-100">0</span>
            </div>
            <div className="flex justify-between border-b border-slate-700/70 pb-2">
              <span className="text-slate-500">Frames Sampled</span>
              <span className="font-semibold text-slate-100">{result.frames_sampled}</span>
            </div>
            <div className="flex justify-between border-b border-slate-700/70 pb-2">
              <span className="text-slate-500">Processing Time</span>
              <span className="font-semibold text-slate-100">{formatDuration(result.processing_time)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Model</span>
              <span className="font-semibold text-slate-100">{result.model_name || 'EfficientNet-B4'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatTile label="Frames Sampled" value={String(result.frames_sampled)} sub="Uniform sampling from video" accent="text-slate-200" />
        <StatTile label="Faces Detected" value={String(facesDetected)} sub="Across all sampled frames" accent="text-slate-200" />
        <StatTile label="Processing Time" value={formatDuration(result.processing_time)} sub="End-to-end analysis" />
        <StatTile label="File Size" value={result.size ? formatFileSize(result.size) : '—'} sub="Uploaded video" />
      </div>

      <Card
        title="Frame Analysis"
        subtitle="Actual per-frame output — face detection status for every sampled frame"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500 border-b border-slate-800">
              <tr>
                <th className="px-2 py-2 font-medium">Frame</th>
                <th className="px-2 py-2 font-medium">Faces</th>
                <th className="px-2 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {frameRows.map((r) => (
                <tr key={r.index} className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors">
                  <td className="px-2 py-2.5 font-mono text-slate-400">#{r.index + 1}</td>
                  <td className="px-2 py-2.5 text-slate-300">{r.faces}</td>
                  <td className="px-2 py-2.5">
                    <span className="inline-flex items-center gap-1.5 text-slate-400">
                      <FaUserSlash className="w-3.5 h-3.5 text-slate-500" />
                      no face
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-[11px] text-slate-600 mt-3">
            The face detector (MTCNN) reported no usable face in any sampled frame, so no deepfake inference was run.
          </p>
        </div>
      </Card>

      <Card title="Sampled Frames" subtitle="Real per-frame pipeline output — no faces detected">
        <FaceVisualization result={result} />
      </Card>

      <Card title="Why No Verdict Was Produced">
        <p className="text-sm text-slate-400 leading-relaxed">
          The EfficientNet-B4 deepfake classifier requires face crops to produce a prediction. Because no usable face
          was detected across all {result.frames_sampled} sampled frame{result.frames_sampled === 1 ? '' : 's'}, no
          inference was run and no REAL or FAKE probability, confidence value, or verdict was computed. The video is not
          classified as FAKE or REAL, and no probability values are invented.
        </p>
      </Card>

      <FusionCard fusion={result.fusion} visualProbability={result.visual_fake_probability} />
      <RppgCard rppg={result.rppg} />

      <Card title="Report">
        <p className="text-sm text-slate-500 leading-relaxed mb-4">
          Download a PDF report stating NO FACE DETECTED. The report includes the filename, timestamp, processing time,
          frames sampled, faces detected (0), and this explanation — without any fabricated probabilities.
        </p>
        <button onClick={() => downloadReportPdf(result)} className="btn btn-primary">
          <FaFilePdf className="w-4 h-4" />
          Download Report (PDF)
        </button>
      </Card>

      <Card title="Video Information">
        <div className="space-y-3 text-sm">
          <div className="flex justify-between gap-3">
            <span className="text-slate-500 flex-shrink-0">Filename:</span>
            <span className="font-medium text-slate-200 truncate">{result.filename}</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="text-slate-500 flex-shrink-0">Analysis ID:</span>
            <span className="font-mono text-xs text-slate-500 truncate">{result.analysis_id}</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="text-slate-500 flex-shrink-0">Analyzed:</span>
            <span className="font-medium text-slate-200">{formatDateTime(result.analyzed_at)}</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="text-slate-500 flex-shrink-0">File Size:</span>
            <span className="font-medium text-slate-200">{result.size ? formatFileSize(result.size) : notProvided}</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="text-slate-500 flex-shrink-0">Status:</span>
            <span className="font-medium text-slate-300">✓ Completed — no face found</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="text-slate-500 flex-shrink-0">Model:</span>
            <span className="font-medium text-slate-200">{result.model_name || 'EfficientNet-B4'}</span>
          </div>
          {result.model_version ? (
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Checkpoint:</span>
              <span className="font-medium text-slate-200 truncate">{result.model_version}</span>
            </div>
          ) : (
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Checkpoint:</span>
              {notProvided}
            </div>
          )}
          {result.device ? (
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Device:</span>
              <span className="font-medium text-slate-200">{result.device.toUpperCase()}</span>
            </div>
          ) : (
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Device:</span>
              {notProvided}
            </div>
          )}
          {result.meta ? (
            <>
              <div className="pt-1 border-t border-slate-800 flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Resolution:</span>
                <span className="font-medium text-slate-200">{result.meta.width} × {result.meta.height}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">FPS:</span>
                <span className="font-medium text-slate-200">{result.meta.fps}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Duration:</span>
                <span className="font-medium text-slate-200">{formatDuration(result.meta.duration)}</span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Frame Count:</span>
                <span className="font-medium text-slate-200">{result.meta.frame_count}</span>
              </div>
            </>
          ) : (
            <>
              <div className="pt-1 border-t border-slate-800 flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Resolution:</span>
                {notProvided}
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">FPS:</span>
                {notProvided}
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Duration:</span>
                {notProvided}
              </div>
            </>
          )}
        </div>
      </Card>

      <div className="flex flex-col sm:flex-row gap-3">
        <button onClick={() => navigate('/analysis')} className="btn btn-primary flex-1 py-3.5">
          Analyze Another Video
        </button>
        <button onClick={() => navigate('/history')} className="btn btn-outline flex-1 py-3.5">
          <FaHistory className="w-4 h-4" />
          View History
        </button>
      </div>
    </div>
  )
}

function RppgCard({ rppg }: { rppg?: RppgData }) {
  if (!rppg) {
    return (
      <Card title="Physiological Signal Analysis (rPPG)">
        <p className="text-sm text-slate-500 leading-relaxed">
          No rPPG physiological analysis was returned for this video.
        </p>
      </Card>
    )
  }

  if (rppg.status === 'SKIPPED') {
    return (
      <Card
        title="Physiological Signal Analysis (rPPG)"
        subtitle="Not applicable for this video"
        action={
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400">
            <FaInfoCircle className="w-3.5 h-3.5" /> Skipped
          </span>
        }
      >
        <p className="text-sm text-slate-500 leading-relaxed">
          rPPG requires visible faces in the sampled frames. Because no usable face was detected, no
          physiological signal was extracted and no heart-rate estimate is reported.
        </p>
      </Card>
    )
  }

  if (rppg.status === 'UNAVAILABLE' || rppg.heart_rate_bpm == null) {
    return (
      <Card
        title="Physiological Signal Analysis (rPPG)"
        subtitle="Signal could not be recovered"
        action={
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-300">
            <FaExclamationTriangle className="w-3.5 h-3.5" /> Unavailable
          </span>
        }
      >
        <p className="text-sm text-slate-500 leading-relaxed">
          A physiological signal could not be reliably recovered from this video — common for very short clips,
          heavy motion, or faces too small to sample. No heart-rate estimate is fabricated here.
        </p>
        {rppg.explanation && (
          <p className="text-sm text-slate-400 mt-3 bg-slate-800/40 border border-slate-700/60 rounded-xl p-3.5">
            {rppg.explanation}
          </p>
        )}
      </Card>
    )
  }

  const hasCharts = !!rppg.signal || !!rppg.filtered_signal || !!rppg.frequency
  const quality = rppg.signal_quality
  return (
    <Card
      title="Physiological Signal Analysis (rPPG)"
      subtitle="CHROM pulse signal from forehead + cheek regions — quality-gated evidence in the final fusion"
      action={
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-300">
          <FaWaveSquare className="w-3.5 h-3.5" /> Signal recovered
        </span>
      }
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="glass-inset p-4">
          <div className="flex items-center gap-2 text-[11px] text-slate-500 uppercase tracking-wider mb-1">
            <FaHeartbeat className="w-3.5 h-3.5 text-rose-400" /> Heart Rate
          </div>
          <div className="text-2xl font-bold text-slate-100">
            {rppg.heart_rate_bpm != null ? `${Math.round(rppg.heart_rate_bpm)} BPM` : '—'}
          </div>
          <div className="text-xs text-slate-500 mt-1">Estimated from dominant frequency</div>
        </div>
        <div className="glass-inset p-4">
          <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">Dominant Frequency</div>
          <div className="text-2xl font-bold text-slate-100">
            {rppg.dominant_frequency != null ? `${rppg.dominant_frequency} Hz` : '—'}
          </div>
          <div className="text-xs text-slate-500 mt-1">Peak in 0.8–3.0 Hz band</div>
        </div>
        <div className="glass-inset p-4">
          <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">Signal Quality</div>
          <div className="text-2xl font-bold text-slate-100">
            {quality != null ? formatPercent(quality, 0) : '—'}
          </div>
          <div className="text-xs text-slate-500 mt-1">SNR, peak prominence, spectral concentration</div>
        </div>
        <div className="glass-inset p-4">
          <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">Frames Used</div>
          <div className="text-2xl font-bold text-slate-100">{rppg.frames_used}</div>
          <div className="text-xs text-slate-500 mt-1">
            {rppg.window ? `of ${rppg.window.frames_read} read @ ${rppg.window.fps} FPS` : 'in analysis window'}
          </div>
        </div>
      </div>

      {hasCharts && <RppgCharts rppg={rppg} />}

      <p className="text-xs text-slate-500 leading-relaxed mt-4 border-t border-slate-800 pt-3">
        {rppg.explanation}
      </p>
    </Card>
  )
}

function FusionCard({ fusion, visualProbability }: { fusion?: AnalysisResult['fusion']; visualProbability?: number }) {
  if (!fusion) return null
  const hasRppg = fusion.rppg_weight > 0 && fusion.rppg_quality != null
  return (
    <Card title="Spatio-Temporal Evidence Fusion" subtitle="EfficientNet-B4 spatial cues combined with CHROM rPPG temporal physiology">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-inset p-4">
          <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">Visual Evidence</div>
          <div className="text-2xl font-bold text-slate-100">{formatPercent(visualProbability ?? fusion.visual_probability)}</div>
          <div className="text-xs text-slate-500 mt-1">EfficientNet-B4 mean</div>
        </div>
        <div className="glass-inset p-4">
          <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">rPPG Contribution</div>
          <div className="text-2xl font-bold text-slate-100">{hasRppg ? `${Math.round(fusion.rppg_weight * 100)}%` : '0%'}</div>
          <div className="text-xs text-slate-500 mt-1">{hasRppg ? `Quality ${formatPercent(fusion.rppg_quality!, 0)}` : 'Signal unavailable'}</div>
        </div>
        <div className="glass-inset p-4">
          <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">Fused Score</div>
          <div className="text-2xl font-bold text-cyan-300">{formatPercent(fusion.probability)}</div>
          <div className="text-xs text-slate-500 mt-1">{hasRppg ? '80% visual + 20% rPPG' : 'Visual-only fallback'}</div>
        </div>
      </div>
    </Card>
  )
}

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  useEffect(() => {
    const fromState = (location.state as { upload?: unknown } | null)?.upload
    if (isAnalysisResult(fromState)) {
      const normalized: AnalysisResult = {
        ...fromState,
        analyzed_at: fromState.analyzed_at || new Date().toISOString(),
      }
      saveLastResult(normalized)
      addToHistory(normalized)
      setResult(normalized)
      setLoading(false)
      return
    }
    const cached = getLastResult()
    if (cached) {
      setResult(cached)
    }
    setLoading(false)
  }, [location.state])

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto flex flex-col items-center justify-center py-24">
        <div className="h-10 w-10 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
        <p className="text-slate-500 mt-4">Loading results…</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="page-head">
          <div>
            <h2 className="page-title">Analysis Results</h2>
            <p className="page-sub">No analysis data available</p>
          </div>
        </header>
        <div className="glass-card p-10 text-center">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-cyan-400/15 to-blue-500/15 border border-cyan-400/25 flex items-center justify-center mx-auto mb-5">
            <FaQuestionCircle className="w-7 h-7 text-cyan-300" />
          </div>
          <p className="text-slate-200 text-lg mb-2">No analysis data available</p>
          <p className="text-slate-500 text-sm mb-6 max-w-sm mx-auto">
            Upload a video to get a real deepfake analysis from the EfficientNet-B4 pipeline.
          </p>
          <button onClick={() => navigate('/analysis')} className="btn btn-primary px-7 py-3">
            Analyze a Video
            <FaArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    )
  }

  if (result.result === 'NO_FACE') {
    return <NoFaceView result={result} />
  }

  const theme = THEMES[result.result]
  const detectionRate = result.frames_sampled > 0 ? Math.round((result.frames_with_faces / result.frames_sampled) * 100) : 0
  const chartData = (result.frame_predictions || []).map((p, i) => ({ name: String(i + 1), value: p }))
  const hasVideoMeta = !!result.meta
  const notProvided = <span className="italic text-slate-600 text-xs">Not provided by analysis</span>

  const frameRows = result.frame_results && result.frame_results.length
    ? result.frame_results
    : (result.frame_predictions || []).map((p, i) => ({
        index: i,
        faces: result.frames_with_faces > 0 ? 1 : 0,
        prediction: p,
        error: null,
      }))

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Analysis Results</h2>
          <p className="page-sub">
            <span className="font-medium text-slate-300">{result.filename}</span>
            {result.analyzed_at && <span className="text-slate-600"> · {formatDateTime(result.analyzed_at)}</span>}
          </p>
        </div>
        <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${theme.chip}`}>
          {theme.icon}
          {theme.label}
        </span>
      </header>

      <div className={`glass-card--accent p-6 md:p-8 ${theme.border}`}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-center">
          <div className="flex justify-center">
            <ConfidenceRing confidence={result.confidence} color={theme.ring} glow={theme.ringSoft} />
          </div>

          <div className="text-center md:text-left">
            <div className="flex items-center justify-center md:justify-start gap-3 mb-3">
              <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${theme.solid} text-white flex items-center justify-center`}>
                {theme.icon}
              </div>
              <div>
                <div className={`text-5xl font-extrabold tracking-tight ${theme.textColor}`}>{theme.label}</div>
                <div className="text-sm font-medium text-slate-400 mt-1">{theme.title}</div>
              </div>
            </div>
            <p className="text-sm text-slate-500 max-w-sm mx-auto md:mx-0 leading-relaxed">{theme.description}</p>
          </div>

          <div className="space-y-2.5 text-sm">
            <div className="flex justify-between border-b border-slate-700/70 pb-2">
              <span className="text-slate-500">Fake Probability</span>
              <span className="font-semibold text-slate-100">{formatPercent(result.fake_probability)}</span>
            </div>
            <div className="flex justify-between border-b border-slate-700/70 pb-2">
              <span className="text-slate-500">Real Probability</span>
              <span className="font-semibold text-slate-100">{formatPercent(result.real_probability)}</span>
            </div>
            <div className="flex justify-between border-b border-slate-700/70 pb-2">
              <span className="text-slate-500">Confidence</span>
              <span className="font-semibold text-slate-100">{formatPercent(result.confidence)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Processing Time</span>
              <span className="font-semibold text-slate-100">{formatDuration(result.processing_time)}</span>
            </div>
            <ProbabilityBalance fakeProbability={result.fake_probability} realProbability={result.real_probability} />
          </div>
        </div>
      </div>

      <Card title="Report">
        <p className="text-sm text-slate-500 leading-relaxed mb-4">
          Download a PDF report of this analysis. The report includes the verdict, confidence, real/fake probabilities,
          per-frame analysis, the rPPG physiological findings, and the model explanation.
        </p>
        <button onClick={() => downloadReportPdf(result)} className="btn btn-primary">
          <FaFilePdf className="w-4 h-4" />
          Download PDF Report
        </button>
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatTile label="Frames Sampled" value={String(result.frames_sampled)} sub="Uniform sampling from video" accent="text-cyan-300" />
        <StatTile
          label="Faces Detected"
          value={String(result.frames_with_faces)}
          sub={`${detectionRate}% of sampled frames`}
          accent="text-cyan-300"
        />
        <StatTile label="Mean Probability" value={formatPercent(result.mean_probability)} sub="Average fake likelihood" />
        <StatTile label="Median Probability" value={formatPercent(result.median_probability)} sub="Robust to outliers" />
        <StatTile
          label="Consistency (StdDev)"
          value={formatPercent(result.std_probability)}
          sub={result.std_probability < 0.1 ? 'Low variation across frames' : 'Some frame-to-frame variation'}
        />
        <StatTile label="Processing Time" value={formatDuration(result.processing_time)} sub="End-to-end analysis" />
        <StatTile label="File Size" value={result.size ? formatFileSize(result.size) : '—'} sub="Uploaded video" />
        <StatTile label="Model" value={result.model_name || 'EfficientNet-B4'} sub={result.device ? `Running on ${result.device.toUpperCase()}` : 'Forensic classifier'} />
      </div>

      <Card
        title="Frame-by-Frame Analysis"
        subtitle={`Fake probability per sampled frame (${chartData.length} frames) · dashed lines mark REAL (≤40%) and FAKE (≥60%) thresholds`}
      >
        <SimpleLineChart
          data={chartData}
          color={theme.ring}
          height={300}
          thresholds={[
            { value: 0.6, color: '#fb7185', label: 'FAKE' },
            { value: 0.4, color: '#34d399', label: 'REAL' },
          ]}
        />
      </Card>

      <RppgCard rppg={result.rppg} />

      <FusionCard fusion={result.fusion} visualProbability={result.visual_fake_probability} />

      <Card
        title="Frame Analysis"
        subtitle="Index, face count, probability, and classification for each sampled frame — from the actual backend response"
      >
        {frameRows.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <FaChartLine className="w-6 h-6 text-cyan-300" />
            </div>
            <div className="empty-state-title">No frame-level data</div>
            <div className="empty-state-sub">Frame-level probability data was not provided by this analysis.</div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-slate-500 border-b border-slate-800">
                <tr>
                  <th className="px-2 py-2 font-medium">Frame</th>
                  <th className="px-2 py-2 font-medium">Faces</th>
                  <th className="px-2 py-2 font-medium">Fake Probability</th>
                  <th className="px-2 py-2 font-medium">Classification</th>
                </tr>
              </thead>
              <tbody>
                {frameRows.map((r) => {
                  const p = r.prediction
                  return (
                    <tr key={r.index} className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors">
                      <td className="px-2 py-2.5 font-mono text-slate-400">#{r.index + 1}</td>
                      <td className="px-2 py-2.5 text-slate-300">{r.error ? '—' : r.faces}</td>
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-2.5">
                          <div className="w-28 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${probabilityColor(p)}`}
                              style={{ width: `${p === null ? 0 : Math.max(2, Math.min(100, p * 100))}%` }}
                            />
                          </div>
                          <span className="text-slate-200 font-medium w-14 text-right">{p === null ? '—' : formatPercent(p, 1)}</span>
                        </div>
                      </td>
                      <td className="px-2 py-2.5">
                        <span className={classifyChip(p)}>{classifyFrame(p)}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <p className="text-[11px] text-slate-600 mt-3">
              Per-frame classification uses the same thresholds as the backend verdict: ≤40% REAL, ≥60% FAKE.
            </p>
          </div>
        )}
      </Card>

      <Card title="Sampled Frames" subtitle="Real per-frame pipeline output — face detection and fake probability per frame">
        <FaceVisualization result={result} />
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Statistical Summary">
          <div className="space-y-4">
            <div>
              <div className="text-sm font-medium text-slate-300 mb-1">Mean (Average)</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-slate-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded"
                    style={{ width: `${Math.max(0, Math.min(100, result.mean_probability * 100))}%` }}
                  />
                </div>
                <span className="text-sm font-semibold text-slate-100 w-16 text-right">{formatPercent(result.mean_probability)}</span>
              </div>
              <p className="text-xs text-slate-600 mt-1">Used for the final classification decision</p>
            </div>

            <div>
              <div className="text-sm font-medium text-slate-300 mb-1">Median</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-slate-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded"
                    style={{ width: `${Math.max(0, Math.min(100, result.median_probability * 100))}%` }}
                  />
                </div>
                <span className="text-sm font-semibold text-slate-100 w-16 text-right">{formatPercent(result.median_probability)}</span>
              </div>
              <p className="text-xs text-slate-600 mt-1">Middle value — robust to outlier frames</p>
            </div>

            <div className="pt-2 border-t border-slate-800 space-y-1 text-xs text-slate-500">
              <p><strong className="text-slate-300">Classification Thresholds:</strong></p>
              <p>• FAKE: mean ≥ 60%</p>
              <p>• REAL: mean ≤ 40%</p>
              <p>• UNCERTAIN: 40% &lt; mean &lt; 60%</p>
            </div>
          </div>
        </Card>

        <Card title="Video Information">
          <div className="space-y-3 text-sm">
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Filename:</span>
              <span className="font-medium text-slate-200 truncate">{result.filename}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Analysis ID:</span>
              <span className="font-mono text-xs text-slate-500 truncate">{result.analysis_id}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Analyzed:</span>
              <span className="font-medium text-slate-200">{formatDateTime(result.analyzed_at)}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">File Size:</span>
              <span className="font-medium text-slate-200">{result.size ? formatFileSize(result.size) : notProvided}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Status:</span>
              <span className="font-medium text-emerald-400">✓ Completed</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-slate-500 flex-shrink-0">Model:</span>
              <span className="font-medium text-slate-200">{result.model_name || 'EfficientNet-B4'}</span>
            </div>
            {result.model_version ? (
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Checkpoint:</span>
                <span className="font-medium text-slate-200 truncate">{result.model_version}</span>
              </div>
            ) : (
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Checkpoint:</span>
                {notProvided}
              </div>
            )}
            {result.device ? (
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Device:</span>
                <span className="font-medium text-slate-200">{result.device.toUpperCase()}</span>
              </div>
            ) : (
              <div className="flex justify-between gap-3">
                <span className="text-slate-500 flex-shrink-0">Device:</span>
                {notProvided}
              </div>
            )}
            {hasVideoMeta && result.meta ? (
              <>
                <div className="pt-1 border-t border-slate-800 flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">Resolution:</span>
                  <span className="font-medium text-slate-200">{result.meta.width} × {result.meta.height}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">FPS:</span>
                  <span className="font-medium text-slate-200">{result.meta.fps}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">Duration:</span>
                  <span className="font-medium text-slate-200">{formatDuration(result.meta.duration)}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">Frame Count:</span>
                  <span className="font-medium text-slate-200">{result.meta.frame_count}</span>
                </div>
              </>
            ) : (
              <>
                <div className="pt-1 border-t border-slate-800 flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">Resolution:</span>
                  {notProvided}
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">FPS:</span>
                  {notProvided}
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-slate-500 flex-shrink-0">Duration:</span>
                  {notProvided}
                </div>
              </>
            )}
          </div>
        </Card>
      </div>

      <Card title="Explanation" subtitle="Generated from the model's per-frame predictions">
        <p className="text-sm text-slate-400 leading-relaxed">{buildExplanation(result)}</p>
      </Card>

      <Card title="How to Interpret Results">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border-l-4 border-emerald-500 pl-4">
            <p className="font-semibold text-emerald-400 mb-1">✓ REAL</p>
            <p className="text-sm text-slate-500">
              The model detected no significant deepfake artifacts. The video is likely authentic, though no analysis is 100% certain.
            </p>
          </div>
          <div className="border-l-4 border-rose-500 pl-4">
            <p className="font-semibold text-rose-400 mb-1">✕ FAKE</p>
            <p className="text-sm text-slate-500">
              The model detected deepfake characteristics with high confidence. Manual review is recommended for critical applications.
            </p>
          </div>
          <div className="border-l-4 border-amber-500 pl-4">
            <p className="font-semibold text-amber-400 mb-1">? UNCERTAIN</p>
            <p className="text-sm text-slate-500">
              Results are inconclusive. The video may benefit from additional analysis methods or expert review.
            </p>
          </div>
        </div>
      </Card>

      <div className="flex flex-col sm:flex-row gap-3">
        <button onClick={() => navigate('/analysis')} className="btn btn-primary flex-1 py-3.5">
          Analyze Another Video
        </button>
        <button onClick={() => navigate('/history')} className="btn btn-outline flex-1 py-3.5">
          <FaHistory className="w-4 h-4" />
          View History
        </button>
      </div>
    </div>
  )
}
