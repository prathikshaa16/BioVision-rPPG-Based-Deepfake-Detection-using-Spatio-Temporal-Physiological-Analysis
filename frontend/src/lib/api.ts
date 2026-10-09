import type { AnalysisResult } from './types'
import { isAnalysisResult } from './types'
import { timestampOf } from './types'

export const API_URL_KEY = 'biovision:api_url'

export function getApiBase(): string {
  if (typeof window !== 'undefined') {
    const custom = localStorage.getItem(API_URL_KEY)
    if (custom && custom.trim()) {
      return custom.trim().replace(/\/+$/, '')
    }
  }
  return (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') || ''
}

export function setApiBase(url: string): void {
  try {
    if (!url || !url.trim()) {
      localStorage.removeItem(API_URL_KEY)
    } else {
      localStorage.setItem(API_URL_KEY, url.trim().replace(/\/+$/, ''))
    }
  } catch {}
}

export const API_BASE = {
  toString: () => getApiBase(),
  valueOf: () => getApiBase(),
}

export const MAX_UPLOAD_BYTES = 500 * 1024 * 1024
export const UPLOAD_TIMEOUT_MS = 10 * 60 * 1000

export async function isBackendOnline(timeoutMs = 5000, targetUrl?: string): Promise<boolean> {
  const base = targetUrl !== undefined ? targetUrl.replace(/\/+$/, '') : getApiBase()
  if (!base && typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    return false
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${base}/health`, { signal: controller.signal })
    if (!res.ok) return false
    const contentType = res.headers.get('content-type') || ''
    if (!contentType.includes('application/json')) return false
    const data = await res.json()
    return !!data && (data.status === 'healthy' || data.status === 'ok')
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
