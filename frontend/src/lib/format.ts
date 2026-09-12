import type { AnalysisResult } from './types'

export function formatFileSize(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1)
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

export function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m ${s}s`
}

export function formatDateTime(iso?: string): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatPercent(value: number, digits = 1): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function buildExplanation(result: AnalysisResult): string {
  const conf = Math.round(result.confidence * 100)
  const fake = formatPercent(result.fake_probability, 1)
  const real = formatPercent(result.real_probability, 1)
  const std = formatPercent(result.std_probability, 1)

  if (result.result === 'FAKE') {
    return `The model classified this video as FAKE with ${conf}% confidence (mean fake probability ${fake}). ` +
      `Frame-level predictions varied by ${std} around the mean, indicating ${result.std_probability < 0.1 ? 'strong, consistent evidence of manipulation across the sampled frames' : 'some frame-to-frame disagreement that the aggregation averaged out'}. ` +
      `Treat this video as potentially manipulated and verify with additional forensic checks.`
  }
  if (result.result === 'REAL') {
    return `The model classified this video as REAL with ${conf}% confidence (mean real probability ${real}). ` +
      `Frame-level predictions were ${result.std_probability < 0.1 ? 'consistently low and stable' : 'mostly low with minor variation'} ` +
      `(std ${std}), showing no meaningful deepfake artifacts. No analysis is 100% certain, but this video shows no strong indicators of manipulation.`
  }
  return `The model could not reach a conclusive decision. With ${conf}% confidence and a mean fake probability of ${fake}, ` +
    `predictions fell inside the uncertain band (40%–60%). This usually happens with lower-quality footage, unusual lighting, ` +
    `or subtle manipulation. We recommend manual review of the video.`
}
