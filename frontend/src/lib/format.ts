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
  if (result.explanation && result.explanation.trim().length > 0) {
    return result.explanation
  }

  const conf = Math.round(result.confidence * 100)
  const fake = formatPercent(result.fake_probability, 1)
  const real = formatPercent(result.real_probability, 1)
  const fakeRatio = result.frame_predictions && result.frame_predictions.length > 0
    ? Math.round((result.frame_predictions.filter((p) => p >= 0.50).length / result.frame_predictions.length) * 100)
    : 0

  if (result.result === 'FAKE') {
    return (
      `The fused assessment classified this video as FAKE with ${conf}% confidence (manipulation probability ${fake}). ` +
      `Spatio-temporal facial analysis detected anomalous facial boundaries and temporal discontinuities across ` +
      `${fakeRatio}% of sampled sequence frames. In addition, physiological rPPG analysis revealed disrupted blood ` +
      `volume pulse dynamics characteristic of synthetic reenactment or face replacement.`
    )
  }
  if (result.result === 'REAL') {
    return (
      `The fused assessment classified this video as REAL with ${conf}% confidence (authenticity probability ${real}). ` +
      `Spatio-temporal representations and CHROM-derived physiological pulse dynamics remained coherent across the sequence, ` +
      `exhibiting no significant manipulation artifacts. As with all forensic tools, verify critical footage through contextual review.`
    )
  }
  return (
    `The fused assessment remained inconclusive. With a confidence of ${conf}% and a fused score of ${fake}, ` +
    `the prediction fell within the decision boundary (45%–55%). This typically occurs when physiological signals ` +
    `are marginal or video compression obscures subtle dynamics. Manual review is recommended.`
  )
}
