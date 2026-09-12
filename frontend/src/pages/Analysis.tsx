import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import UploadDropzone from '../components/UploadDropzone'
import { API_BASE, MAX_UPLOAD_BYTES, UPLOAD_TIMEOUT_MS, isBackendOnline } from '../lib/api'
import type { AnalysisResult } from '../lib/types'
import { isAnalysisResult } from '../lib/types'
import { saveLastResult, addToHistory } from '../lib/api'
import { formatFileSize } from '../lib/format'
import {
  FaCloudUploadAlt,
  FaFilm,
  FaFingerprint,
  FaRobot,
  FaLayerGroup,
  FaFileAlt,
  FaExclamationTriangle,
  FaCheckCircle,
  FaCircleNotch,
  FaHourglassHalf,
  FaServer,
  FaVideo,
  FaWaveSquare,
} from 'react-icons/fa'

type Phase = 'idle' | 'uploading' | 'analyzing'

const BACKEND_HINT = 'python -m uvicorn backend.app.main:app --port 8000'

const STAGES = [
  { icon: FaCloudUploadAlt, label: 'Uploading video' },
  { icon: FaFilm, label: 'Extracting frames' },
  { icon: FaFingerprint, label: 'Detecting faces' },
  { icon: FaRobot, label: 'Running AI analysis' },
  { icon: FaWaveSquare, label: 'Extracting physiological signal' },
  { icon: FaLayerGroup, label: 'Aggregating results' },
  { icon: FaFileAlt, label: 'Generating report' },
]

export default function Analysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState<number>(0)
  const [error, setError] = useState<string | null>(null)
  const [backendError, setBackendError] = useState(false)
  const [online, setOnline] = useState<boolean | null>(null)
  const xhrRef = useRef<XMLHttpRequest | null>(null)
  const navigate = useNavigate()

  const loading = phase !== 'idle'

  const checkBackend = useCallback(async () => {
    const ok = await isBackendOnline(4000)
    setOnline(ok)
    if (ok) setBackendError(false)
    return ok
  }, [])

  useEffect(() => {
    checkBackend()
  }, [checkBackend])

  useEffect(() => {
    return () => {
      if (xhrRef.current) xhrRef.current.abort()
    }
  }, [])

  const handleFiles = useCallback((files: FileList) => {
    setError(null)
    setBackendError(false)
    const file = files[0]
    if (!file) return

    if (!file.type.startsWith('video')) {
      setError('Only video files are allowed. Please select a video file.')
      return
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`File is too large (max 500 MB). "${file.name}" is ${formatFileSize(file.size)}.`)
      return
    }
    if (file.size === 0) {
      setError('The selected video file is empty and cannot be analyzed.')
      return
    }

    setSelectedFile(file)
  }, [])

  const handleClear = useCallback(() => {
    setSelectedFile(null)
    setError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select a video file first.')
      return
    }
    if (loading) return

    if (!(await checkBackend())) {
      setBackendError(true)
      setError('The analysis server is not running, so the video cannot be analyzed.')
      return
    }

    setError(null)
    setBackendError(false)
    setPhase('uploading')
    setProgress(0)

    const form = new FormData()
    form.append('file', selectedFile)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/upload`)
    xhr.timeout = UPLOAD_TIMEOUT_MS
    xhrRef.current = xhr

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress((e.loaded / e.total) * 100)
    }

    xhr.upload.onload = () => setPhase('analyzing')

    xhr.onload = () => {
      xhrRef.current = null
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data: unknown = JSON.parse(xhr.responseText)
          if (!isAnalysisResult(data)) {
            setPhase('idle')
            setProgress(0)
            setError('The server returned an unexpected response. Please try again.')
            return
          }
          const result: AnalysisResult = {
            ...data,
            analyzed_at: data.analyzed_at || new Date().toISOString(),
            size: data.size || selectedFile.size,
          }
          saveLastResult(result)
          addToHistory(result)
          setSelectedFile(null)
          setPhase('idle')
          setProgress(0)
          navigate('/results', { state: { upload: result } })
        } catch {
          setPhase('idle')
          setProgress(0)
          setError('Unexpected server response format. Please try again.')
        }
        return
      }

      setPhase('idle')
      setProgress(0)
      if (xhr.status === 0) {
        setBackendError(true)
        setError('Could not reach the analysis server. Make sure it is running on port 8000.')
        return
      }
      try {
        const resp = JSON.parse(xhr.responseText)
        if (resp.error === 'NO_FACE_DETECTED') {
          setError('No faces were detected in the video. Please try a video with visible faces.')
        } else if (resp.error === 'INVALID_VIDEO') {
          setError(resp.detail || 'The file could not be read as a valid video. Please check the file and try again.')
        } else if (resp.detail) {
          setError(resp.detail)
        } else {
          setError(`Upload failed (${xhr.status}). Please try again.`)
        }
      } catch {
        setError(`Upload failed (${xhr.status}). Please try again.`)
      }
    }

    xhr.onerror = () => {
      xhrRef.current = null
      setPhase('idle')
      setProgress(0)
      setBackendError(true)
      setError('Could not reach the analysis server. Make sure it is running on port 8000.')
    }

    xhr.ontimeout = () => {
      xhrRef.current = null
      setPhase('idle')
      setProgress(0)
      setError('The request timed out. This can happen with very large files — try a smaller video.')
    }

    xhr.send(form)
  }, [selectedFile, loading, navigate, checkBackend])

  const backendOffline = online === false

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h1 className="page-title">Video Analysis</h1>
          <p className="page-sub">
            Upload a video and the BioVision forensic pipeline will analyze it for deepfake indicators using
            face-level EfficientNet-B4 inference.
          </p>
        </div>
        <span className="chip chip--info">
          <span className="status-dot bg-cyan-400" />
          Forensic Analysis Workspace
        </span>
      </header>

      {backendOffline && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5 flex flex-col md:flex-row md:items-center gap-4">
          <div className="h-11 w-11 rounded-xl bg-rose-500/20 border border-rose-400/40 flex items-center justify-center text-rose-300 flex-shrink-0">
            <FaExclamationTriangle className="w-5 h-5" />
          </div>
          <div className="flex-1">
            <p className="text-rose-200 font-semibold">Analysis server is unavailable</p>
            <p className="text-rose-300/80 text-sm mt-1">
              The FastAPI backend is not responding on <code className="text-rose-200 bg-rose-500/15 px-1.5 py-0.5 rounded font-mono text-xs">http://127.0.0.1:8000</code>.
              Start it from the project root with:
            </p>
            <code className="inline-block mt-2 px-3 py-1.5 rounded-lg bg-black/40 border border-rose-400/30 font-mono text-xs text-rose-200">
              {BACKEND_HINT}
            </code>
          </div>
          <div className="flex flex-col gap-2 flex-shrink-0">
            <button onClick={() => checkBackend()} className="btn btn-outline text-sm">
              Retry Connection
            </button>
            <button onClick={() => navigate('/model')} className="btn btn-ghost text-sm">
              Model Status
            </button>
          </div>
        </div>
      )}

      {backendError && !backendOffline && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <FaServer className="text-rose-300 w-5 h-5" />
            <p className="text-rose-200 text-sm font-medium">{error}</p>
          </div>
          <button onClick={() => checkBackend()} className="btn btn-outline text-sm px-3 py-1.5">
            Retry
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card
            title="Upload Video"
            subtitle={
              selectedFile
                ? `Selected: ${selectedFile.name} (${formatFileSize(selectedFile.size)})`
                : 'MP4, MOV, MKV, AVI, WebM · max 500 MB'
            }
            action={backendOffline ? <span className="inline-flex items-center gap-1.5 text-xs font-medium text-rose-300"><FaExclamationTriangle className="w-3.5 h-3.5" /> Server offline</span> : undefined}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <span className="text-xs text-slate-500">Pipeline: 15 frames → MTCNN → EfficientNet-B4 + CHROM rPPG → quality-gated fusion</span>
                <span
                  className={`chip ${
                    backendOffline
                      ? 'chip--err'
                      : error
                        ? 'chip--err'
                        : phase === 'uploading' || phase === 'analyzing'
                          ? 'chip--info'
                          : 'chip--idle'
                  }`}
                >
                  {backendOffline ? (
                    <>
                      <span className="status-dot bg-rose-400" /> Server Offline
                    </>
                  ) : error ? (
                    <>
                      <span className="status-dot bg-rose-400" /> Error
                    </>
                  ) : phase === 'uploading' ? (
                    <>
                      <span className="status-dot bg-cyan-400 pulse-glow" /> Uploading
                    </>
                  ) : phase === 'analyzing' ? (
                    <>
                      <span className="status-dot bg-cyan-400 pulse-glow" /> Analyzing
                    </>
                  ) : (
                    <>
                      <span className="status-dot bg-slate-500" /> Ready
                    </>
                  )}
                </span>
              </div>

              <UploadDropzone
                onFiles={handleFiles}
                onClear={handleClear}
                disabled={loading}
                progress={loading && phase === 'uploading' ? progress : undefined}
                selectedFile={selectedFile ?? undefined}
              />

              {error && (
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 flex items-start gap-3">
                  <FaExclamationTriangle className="w-4 h-4 text-rose-300 mt-0.5 flex-shrink-0" />
                  <p className="text-rose-200 text-sm">{error}</p>
                </div>
              )}

              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!selectedFile || loading || backendOffline}
                className="btn btn-primary w-full py-3.5 text-base"
              >
                {loading ? (
                  <>
                    <FaCircleNotch className="w-5 h-5 animate-spin" />
                    {phase === 'uploading' ? 'Uploading…' : 'Analyzing…'}
                  </>
                ) : (
                  <>
                    <FaVideo className="w-4 h-4" />
                    Analyze Video
                  </>
                )}
              </button>

              {backendOffline && (
                <p className="text-xs text-slate-500 text-center">
                  Analyze is disabled while the analysis server is offline — no results are ever fabricated.
                </p>
              )}

              {loading && (
                <div className="rounded-2xl border border-cyan-400/25 bg-cyan-400/5 p-5 space-y-4">
                  {phase === 'uploading' && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2.5 text-sm font-medium text-slate-200">
                          <FaCircleNotch className="w-4 h-4 text-cyan-300 animate-spin" />
                          Uploading {formatFileSize(selectedFile?.size || 0)} — {Math.round(Math.min(100, progress))}%
                        </div>
                        <span className="text-sm font-semibold text-cyan-300">{Math.round(Math.min(100, progress))}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-200"
                          style={{ width: `${Math.max(3, Math.min(100, progress))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  <div className="space-y-2.5">
                    <div className="flex items-center gap-2.5 text-sm font-medium text-slate-200">
                      <FaHourglassHalf className="w-4 h-4 text-cyan-300 animate-pulse" />
                      {phase === 'uploading' ? 'Preparing analysis pipeline…' : 'Running the detection pipeline — this can take 30–90 seconds on CPU (includes physiological signal extraction).'}
                    </div>
                    {phase === 'analyzing' && (
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div className="h-full w-1/3 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full animate-[progress-slide_1.4s_ease-in-out_infinite]" />
                      </div>
                    )}
                  </div>

                  <div className="space-y-1.5 pt-1">
                    {STAGES.map((stage, i) => {
                      const done = phase === 'analyzing' && i === 0
                      const active = phase === 'analyzing' && i === 1
                      return (
                        <div key={stage.label} className={`flex items-center gap-3 text-sm ${done ? 'text-emerald-300' : active ? 'text-cyan-200' : 'text-slate-500'}`}>
                          {done ? (
                            <FaCheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                          ) : active ? (
                            <FaCircleNotch className="w-4 h-4 text-cyan-300 animate-spin flex-shrink-0" />
                          ) : (
                            <stage.icon className="w-4 h-4 flex-shrink-0 opacity-60" />
                          )}
                          <span>{stage.label}</span>
                          {active && <span className="text-[11px] text-cyan-400/70 ml-auto">in progress</span>}
                        </div>
                      )
                    })}
                    <p className="text-[11px] text-slate-500 pt-1">
                      The backend does not report per-stage completion — the upload stage reflects real progress, the rest run server-side.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </Card>

          <Card title="How It Works">
            <div className="space-y-3">
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 text-[#04121d] flex items-center justify-center font-semibold text-sm">1</div>
                <div>
                  <p className="font-medium text-slate-100">Frame Extraction</p>
                  <p className="text-slate-500 text-sm">Video is sampled at uniform intervals to select representative frames</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 text-[#04121d] flex items-center justify-center font-semibold text-sm">2</div>
                <div>
                  <p className="font-medium text-slate-100">Face Detection</p>
                  <p className="text-slate-500 text-sm">MTCNN detects faces in each frame and extracts the largest face region</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 text-[#04121d] flex items-center justify-center font-semibold text-sm">3</div>
                <div>
                  <p className="font-medium text-slate-100">AI Analysis</p>
                  <p className="text-slate-500 text-sm">EfficientNet-B4 model analyzes each face for deepfake indicators</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 text-[#04121d] flex items-center justify-center font-semibold text-sm">4</div>
                <div>
                  <p className="font-medium text-slate-100">Physiological Signal (rPPG)</p>
                  <p className="text-slate-500 text-sm">CHROM pulse extracted from forehead + cheek regions — quality-gated evidence contributes up to 20%</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 text-[#04121d] flex items-center justify-center font-semibold text-sm">5</div>
                <div>
                  <p className="font-medium text-slate-100">Results</p>
                  <p className="text-slate-500 text-sm">Per-frame predictions are aggregated to generate a final classification</p>
                </div>
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card title="Requirements">
            <ul className="space-y-2.5 text-sm text-slate-400">
              <li className="flex items-start gap-2">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span>Video formats: MP4, MOV, MKV, AVI, WebM</span>
              </li>
              <li className="flex items-start gap-2">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span>Visible faces in the video</span>
              </li>
              <li className="flex items-start gap-2">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span>Maximum file size: 500 MB</span>
              </li>
              <li className="flex items-start gap-2">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span>Processing time: typically 30–90 seconds (rPPG signal extraction is the slowest step)</span>
              </li>
            </ul>
          </Card>

          <Card title="What to Expect">
            <div className="space-y-2.5 text-sm text-slate-400">
              <p><strong className="text-emerald-400">REAL:</strong> no deepfake indicators detected</p>
              <p><strong className="text-rose-400">FAKE:</strong> manipulation detected with elevated confidence</p>
              <p><strong className="text-amber-400">UNCERTAIN:</strong> inconclusive — manual review advised</p>
              <p className="pt-2 text-xs text-slate-500">Verdicts come from the backend classification (mean ≤ 40% REAL, ≥ 60% FAKE).</p>
            </div>
          </Card>

          <Card title="Privacy & Data">
            <p className="text-sm text-slate-500 leading-relaxed">
              Videos are uploaded directly to your local analysis server and processed on your machine. Results are
              stored only in your browser and in the server's in-memory store.
            </p>
          </Card>
        </div>
      </div>
    </div>
  )
}
