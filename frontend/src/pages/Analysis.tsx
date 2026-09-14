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
  FaHeartbeat,
  FaBrain,
  FaEye,
} from 'react-icons/fa'

type Phase = 'idle' | 'uploading' | 'analyzing'

const BACKEND_HINT = 'python -m uvicorn backend.app.main:app --port 8000'

const STAGES = [
  { icon: FaCloudUploadAlt, label: 'Video uploaded' },
  { icon: FaEye, label: 'Facial regions detected' },
  { icon: FaFingerprint, label: 'Spatial features extracted' },
  { icon: FaBrain, label: 'Temporal patterns analyzed' },
  { icon: FaWaveSquare, label: 'rPPG signal extracted' },
  { icon: FaHeartbeat, label: 'Physiological features analyzed' },
  { icon: FaLayerGroup, label: 'Multimodal features fused' },
  { icon: FaFileAlt, label: 'Generating final assessment...' },
]

export default function Analysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [progress, setProgress] = useState<number>(0)
  const [activeStage, setActiveStage] = useState<number>(0)
  const [error, setError] = useState<string | null>(null)
  const [backendError, setBackendError] = useState(false)
  const [online, setOnline] = useState<boolean | null>(null)
  const xhrRef = useRef<XMLHttpRequest | null>(null)
  const stageTimerRef = useRef<number | null>(null)
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
      if (stageTimerRef.current) clearInterval(stageTimerRef.current)
    }
  }, [])

  // Simulate realistic stage progression while server runs inference
  useEffect(() => {
    if (phase === 'analyzing') {
      setActiveStage(1)
      const timer = window.setInterval(() => {
        setActiveStage((prev) => (prev < 6 ? prev + 1 : prev))
      }, 3500)
      stageTimerRef.current = timer
      return () => clearInterval(timer)
    } else if (phase === 'uploading') {
      setActiveStage(0)
    } else {
      setActiveStage(0)
      if (stageTimerRef.current) clearInterval(stageTimerRef.current)
    }
  }, [phase])

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
    xhr.open('POST', `${API_BASE}/upload?model_type=cached`)
    xhr.timeout = UPLOAD_TIMEOUT_MS
    xhrRef.current = xhr

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress((e.loaded / e.total) * 100)
    }

    xhr.upload.onload = () => {
      setPhase('analyzing')
      setActiveStage(1)
    }

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
            Upload a facial video to execute the full BioVision spatio-temporal and physiological analysis pipeline.
          </p>
        </div>
        <span className="chip chip--info">
          <span className="status-dot bg-cyan-400" />
          Multimodal Pipeline
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
            <button onClick={() => navigate('/architecture')} className="btn btn-ghost text-sm">
              Architecture Status
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
            title="Upload Facial Video"
            subtitle={
              selectedFile
                ? `Selected: ${selectedFile.name} (${formatFileSize(selectedFile.size)})`
                : 'MP4, MOV, MKV, AVI, WebM · max 500 MB'
            }
            action={backendOffline ? <span className="inline-flex items-center gap-1.5 text-xs font-medium text-rose-300"><FaExclamationTriangle className="w-3.5 h-3.5" /> Server offline</span> : undefined}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <span className="text-xs text-slate-400">
                  Target: EfficientNet-B4 (Spatial) + LSTM (Temporal) + CHROM (rPPG) + Multimodal Fusion
                </span>
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
                      <span className="status-dot bg-cyan-400 pulse-glow" /> Analyzing Pipeline
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
                className="btn btn-primary w-full py-3.5 text-base shadow-lg shadow-cyan-500/20"
              >
                {loading ? (
                  <>
                    <FaCircleNotch className="w-5 h-5 animate-spin" />
                    {phase === 'uploading' ? 'Uploading Video…' : 'Running BioVision Pipeline…'}
                  </>
                ) : (
                  <>
                    <FaVideo className="w-4 h-4" />
                    Start Analysis
                  </>
                )}
              </button>

              {backendOffline && (
                <p className="text-xs text-slate-500 text-center">
                  Analysis is disabled while the backend server is offline. No results are ever simulated.
                </p>
              )}

              {/* 14. PROCESSING STATUS PAGE */}
              {loading && (
                <div className="rounded-2xl border border-cyan-400/25 bg-cyan-950/20 p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-cyan-500/20 pb-3">
                    <span className="text-xs font-mono font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-2">
                      <FaRobot className="w-4 h-4" /> BIOVISION ANALYSIS
                    </span>
                    <span className="text-xs font-semibold text-cyan-400">
                      {phase === 'uploading' ? `${Math.round(Math.min(100, progress))}%` : 'In Progress'}
                    </span>
                  </div>

                  {phase === 'uploading' && (
                    <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-200"
                        style={{ width: `${Math.max(3, Math.min(100, progress))}%` }}
                      />
                    </div>
                  )}

                  {phase === 'analyzing' && (
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div className="h-full w-1/3 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full animate-[progress-slide_1.4s_ease-in-out_infinite]" />
                    </div>
                  )}

                  {/* 8-Stage Research Pipeline Progression */}
                  <div className="space-y-2 pt-1 font-mono text-xs">
                    {STAGES.map((stage, i) => {
                      const isDone = (phase === 'analyzing' && i < activeStage) || (phase === 'analyzing' && i === 0)
                      const isActive = phase === 'analyzing' && i === activeStage
                      const isPending = phase === 'uploading' || i > activeStage

                      return (
                        <div
                          key={stage.label}
                          className={`flex items-center gap-3 py-1 px-2 rounded-lg transition-colors ${
                            isDone
                              ? 'text-emerald-300 bg-emerald-950/20'
                              : isActive
                                ? 'text-cyan-200 bg-cyan-950/40 font-semibold border border-cyan-500/30'
                                : 'text-slate-500 opacity-60'
                          }`}
                        >
                          {isDone ? (
                            <FaCheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                          ) : isActive ? (
                            <FaCircleNotch className="w-4 h-4 text-cyan-300 animate-spin flex-shrink-0" />
                          ) : (
                            <span className="w-4 h-4 rounded-full border border-slate-700 inline-block flex-shrink-0 text-center text-[10px] leading-3 text-slate-600">
                              {i + 1}
                            </span>
                          )}
                          <span className="flex-1">{stage.label}</span>
                          {isActive && (
                            <span className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">
                              processing...
                            </span>
                          )}
                          {isDone && (
                            <span className="text-[10px] text-emerald-400">
                              ✓ complete
                            </span>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </Card>

          <Card title="How BioVision Analyzes the Video">
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-cyan-500/20 border border-cyan-400/30 text-cyan-300 flex items-center justify-center font-bold text-xs">1</div>
                <div>
                  <p className="font-semibold text-slate-100 text-sm">Sequence-Level Sampling</p>
                  <p className="text-slate-400 text-xs mt-0.5">Thirty-two facial observations are selected across the video to preserve temporal context.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-cyan-500/20 border border-cyan-400/30 text-cyan-300 flex items-center justify-center font-bold text-xs">2</div>
                <div>
                  <p className="font-semibold text-slate-100 text-sm">Face Region Standardization</p>
                  <p className="text-slate-400 text-xs mt-0.5">MTCNN isolates the primary facial bounding box and normalizes crops to 224×224.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-cyan-500/20 border border-cyan-400/30 text-cyan-300 flex items-center justify-center font-bold text-xs">3</div>
                <div>
                  <p className="font-semibold text-slate-100 text-sm">Spatial Feature Extraction</p>
                  <p className="text-slate-400 text-xs mt-0.5">EfficientNet-B4 generates 1792-dimensional embeddings capturing fine visual and edge artifacts.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-blue-500/20 border border-blue-400/30 text-blue-300 flex items-center justify-center font-bold text-xs">4</div>
                <div>
                  <p className="font-semibold text-slate-100 text-sm">Temporal LSTM Dependency Modeling</p>
                  <p className="text-slate-400 text-xs mt-0.5">A 2-layer LSTM analyzes the ordered visual embeddings to capture cross-frame dynamics (256-d representation).</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-emerald-500/20 border border-emerald-400/30 text-emerald-300 flex items-center justify-center font-bold text-xs">5</div>
                <div>
                  <p className="font-semibold text-slate-100 text-sm">CHROM Physiological Signal Recovery</p>
                  <p className="text-slate-400 text-xs mt-0.5">Extracts remote photoplethysmography pulses from facial skin color variations in the 0.8–3.0 Hz cardiac band.</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-purple-500/20 border border-purple-400/30 text-purple-300 flex items-center justify-center font-bold text-xs">6</div>
                <div>
                  <p className="font-semibold text-slate-100 text-sm">Quality-Gated Late Fusion</p>
                  <p className="text-slate-400 text-xs mt-0.5">The trained fusion head combines 80% visual-temporal evidence with 20% quality-gated physiological evidence.</p>
                </div>
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card title="Video Requirements">
            <ul className="space-y-3 text-sm text-slate-400">
              <li className="flex items-start gap-2.5">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span><strong className="text-slate-200">Video Formats:</strong> MP4, MOV, MKV, AVI, WebM</span>
              </li>
              <li className="flex items-start gap-2.5">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span><strong className="text-slate-200">Visible Subject:</strong> Requires clearly detectable facial regions</span>
              </li>
              <li className="flex items-start gap-2.5">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span><strong className="text-slate-200">Max File Size:</strong> 500 MB</span>
              </li>
              <li className="flex items-start gap-2.5">
                <FaCheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                <span><strong className="text-slate-200">Processing Time:</strong> Typically 30–75s on CPU; hardware acceleration available via CUDA GPU</span>
              </li>
            </ul>
          </Card>

          <Card title="Assessment Interpretation">
            <div className="space-y-3 text-sm text-slate-400">
              <div>
                <span className="text-emerald-400 font-bold">REAL:</span> Fused likelihood indicates coherent spatio-temporal dynamics and physiological pulse.
              </div>
              <div>
                <span className="text-rose-400 font-bold">FAKE:</span> Significant visual, temporal, or physiological inconsistency detected.
              </div>
              <div>
                <span className="text-amber-400 font-bold">UNCERTAIN:</span> Boundary case (40%–60%); manual review recommended.
              </div>
              <p className="pt-2 text-xs text-slate-500 border-t border-slate-800">
                Decision operating point: optimized via Youden's J on the validated benchmark.
              </p>
            </div>
          </Card>

          <Card title="Privacy & Research Integrity">
            <p className="text-xs text-slate-400 leading-relaxed">
              Videos are processed entirely within the local API pipeline. No data is stored externally or shared.
              All reported metrics reflect authentic model outputs from the validated checkpoint.
            </p>
          </Card>
        </div>
      </div>
    </div>
  )
}
