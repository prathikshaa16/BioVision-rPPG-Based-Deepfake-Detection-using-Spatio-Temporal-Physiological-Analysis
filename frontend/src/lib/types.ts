export type ResultLabel = 'REAL' | 'FAKE' | 'UNCERTAIN' | 'NO_FACE'

export interface FaceBox {
  x: number
  y: number
  w: number
  h: number
}

export interface FrameResult {
  index: number
  faces: number
  prediction: number | null
  error: string | null
  boxes?: FaceBox[]
}

export interface RppgWindow {
  fps: number
  frames_requested: number
  frames_read: number
  usable_frames: number
  start_index?: number
  end_index?: number
}

export interface RppgSignal {
  timestamps: number[]
  r: number[]
  g: number[]
  b: number[]
}

export interface RppgFilteredSignal {
  timestamps: number[]
  amplitude: number[]
}

export interface RppgFrequency {
  frequencies: number[]
  power: number[]
}

export interface RppgData {
  status: 'AVAILABLE' | 'UNAVAILABLE' | 'SKIPPED'
  explanation?: string
  frames_used: number
  heart_rate_bpm: number | null
  dominant_frequency: number | null
  signal_quality: number | null
  quality_metrics: {
    snr_db?: number | null
    peak_prominence?: number | null
    spectral_concentration?: number | null
    nfft?: number
  } | null
  signal: RppgSignal | null
  filtered_signal: RppgFilteredSignal | null
  frequency: RppgFrequency | null
  roi: { forehead: boolean; cheeks: boolean; method?: string } | null
  window: RppgWindow | null
}

export interface FusionData {
  probability: number
  visual_probability: number
  rppg_anomaly_score: number | null
  rppg_quality: number | null
  visual_weight: number
  rppg_weight: number
  method: string
}

export interface VideoMeta {
  fps: number
  frame_count: number
  duration: number
  width: number
  height: number
}

/**
 * Consumable face overlay data. The backend exposes detected face regions when
 * available; fields are optional so the UI degrades gracefully.
 */
export interface FrameOverlay {
  index: number
  image_url?: string
  faces?: number
  boxes?: FaceBox[]
  prediction?: number | null
}

export interface AnalysisResult {
  analysis_id: string
  filename: string
  status: string
  result: ResultLabel
  confidence: number
  fake_probability: number
  real_probability: number
  visual_fake_probability?: number
  frames_sampled: number
  frames_with_faces: number
  faces_detected?: number
  frame_predictions: number[]
  mean_probability: number
  median_probability: number
  std_probability: number
  processing_time: number
  explanation?: string
  size?: number
  path?: string
  model_name?: string
  model_version?: string
  device?: string
  analyzed_at?: string
  meta?: VideoMeta
  frame_results?: FrameResult[]
  sampled_indices?: number[]
  frame_overlays?: FrameOverlay[]
  rppg?: RppgData
  fusion?: FusionData
}

export function isAnalysisResult(value: unknown): value is AnalysisResult {
  if (!value || typeof value !== 'object') return false
  const v = value as Record<string, unknown>
  return (
    typeof v.filename === 'string' &&
    typeof v.result === 'string' &&
    (v.result === 'REAL' || v.result === 'FAKE' || v.result === 'UNCERTAIN' || v.result === 'NO_FACE')
  )
}

export function timestampOf(result: AnalysisResult): number {
  if (result.analyzed_at) {
    const ts = Date.parse(result.analyzed_at)
    if (!Number.isNaN(ts)) return ts
  }
  return 0
}
