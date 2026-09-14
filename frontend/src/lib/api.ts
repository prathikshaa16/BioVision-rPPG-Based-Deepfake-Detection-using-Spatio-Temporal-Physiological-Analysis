import type { AnalysisResult } from './types'
import { isAnalysisResult } from './types'
import { timestampOf } from './types'

export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')
export const MAX_UPLOAD_BYTES = 500 * 1024 * 1024
export const UPLOAD_TIMEOUT_MS = 10 * 60 * 1000

export async function isBackendOnline(timeoutMs = 5000): Promise<boolean> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal })
    return res.ok
  } catch {
    return false
  } finally {
    clearTimeout(timer)
  }
}

const LAST_RESULT_KEY = 'biovision:lastResult'
const HISTORY_KEY = 'biovision:history'
const MAX_HISTORY = 50

export function getLastResult(): AnalysisResult | null {
  try {
    const raw = localStorage.getItem(LAST_RESULT_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    return isAnalysisResult(parsed) ? parsed : null
  } catch {
    return null
  }
}

export function saveLastResult(result: AnalysisResult): void {
  try {
    localStorage.setItem(LAST_RESULT_KEY, JSON.stringify(result))
  } catch {
    // storage full or unavailable — non-fatal
  }
}

export function clearLastResult(): void {
  try {
    localStorage.removeItem(LAST_RESULT_KEY)
  } catch {
    // ignore
  }
}

export function getHistory(): AnalysisResult[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isAnalysisResult)
  } catch {
    return []
  }
}

export function addToHistory(result: AnalysisResult): AnalysisResult[] {
  const history = getHistory()
  const next = [result, ...history.filter((h) => h.analysis_id !== result.analysis_id)].slice(0, MAX_HISTORY)
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
  } catch {
    // ignore
  }
  return next
}

export function mergeHistory(entries: AnalysisResult[]): AnalysisResult[] {
  const seen = new Map<string, AnalysisResult>()
  for (const entry of [...entries, ...getHistory()]) {
    if (isAnalysisResult(entry) && entry.analysis_id) seen.set(entry.analysis_id, entry)
  }
  return Array.from(seen.values())
    .sort((a, b) => timestampOf(b) - timestampOf(a))
    .slice(0, MAX_HISTORY)
}
