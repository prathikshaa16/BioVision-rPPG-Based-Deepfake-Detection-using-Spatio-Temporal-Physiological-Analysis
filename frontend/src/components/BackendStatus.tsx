import React, { useEffect, useState } from 'react'
import { getApiBase, setApiBase, isBackendOnline } from '../lib/api'
import { FaPlug, FaTimes, FaCheckCircle, FaExclamationTriangle, FaSyncAlt } from 'react-icons/fa'

type Status = 'checking' | 'online' | 'offline'

const STATUS_STYLES: Record<Status, { dot: string; text: string; label: string }> = {
  checking: { dot: 'bg-amber-400', text: 'text-amber-300', label: 'Checking' },
  online: { dot: 'bg-emerald-400', text: 'text-emerald-300', label: 'Live API' },
  offline: { dot: 'bg-cyan-400', text: 'text-cyan-300', label: 'Ready' },
}

export default function BackendStatus({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<Status>('checking')
  const [showModal, setShowModal] = useState(false)
  const [apiUrl, setApiUrl] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null)

  const check = async () => {
    const online = await isBackendOnline(4000)
    setStatus(online ? 'online' : 'offline')
  }

  useEffect(() => {
    setApiUrl(getApiBase())
    check()
    const interval = setInterval(check, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleOpenModal = () => {
    setApiUrl(getApiBase())
    setTestResult(null)
    setShowModal(true)
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const ok = await isBackendOnline(5000, apiUrl.trim())
      if (ok) {
        setTestResult({ ok: true, msg: 'Connected successfully! FastAPI backend is active and responding.' })
      } else {
        setTestResult({
          ok: false,
          msg: 'Connection failed. Ensure the server or cloudflared tunnel is running with HTTPS.',
        })
      }
    } catch {
      setTestResult({ ok: false, msg: 'Network error while attempting to connect.' })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    setApiBase(apiUrl.trim())
    await check()
    setShowModal(false)
  }

  const handleReset = async () => {
    setApiBase('')
    setApiUrl('')
    await check()
    setTestResult(null)
  }

  const s = STATUS_STYLES[status]

  return (
    <>
      {compact ? (
        <button
          type="button"
          onClick={handleOpenModal}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-900/60 border border-slate-700/70 hover:border-cyan-500/50 transition-all cursor-pointer"
          title="Click to configure live backend connection"
        >
          <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${status === 'online' ? 'pulse-glow' : ''}`} />
          <span className={s.text}>{status === 'online' ? 'Live API Online' : 'System Ready'}</span>
          <FaPlug className="w-2.5 h-2.5 text-slate-400 ml-0.5" />
        </button>
      ) : (
        <button
          type="button"
          onClick={handleOpenModal}
          className={`rounded-xl px-4 py-2 text-sm font-medium border inline-flex items-center gap-2 transition-all cursor-pointer ${
            status === 'online'
              ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
              : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300'
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${s.dot} ${status === 'online' ? 'pulse-glow' : ''}`} />
          {status === 'online' ? 'Live backend connected' : 'BioVision System Ready'}
          <span className="text-xs text-slate-400 font-normal ml-1">· Configure API</span>
        </button>
      )}

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-in fade-in duration-150">
          <div className="bg-[#0b1222] border border-slate-700/80 rounded-2xl w-full max-w-lg p-6 shadow-2xl space-y-5 text-left">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-cyan-950/60 border border-cyan-500/30 text-cyan-400">
                  <FaPlug className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-100">Connect Live Backend API</h3>
                  <p className="text-xs text-slate-400">Wire this web interface to your real Python GPU/model server</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              >
                <FaTimes className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-300">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-slate-200">How to expose your local backend over HTTPS:</div>
                <p className="text-slate-400 leading-relaxed">
                  In PowerShell, run your Cloudflare tunnel:
                </p>
                <code className="block p-2 rounded bg-black/60 font-mono text-[11px] text-cyan-300 select-all border border-cyan-500/20">
                  &amp; "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
                </code>
                <p className="text-slate-400 mt-1">
                  Copy the generated <strong className="text-cyan-300">https://*.trycloudflare.com</strong> URL and paste it below.
                </p>
              </div>

              <div className="space-y-1.5">
                <label className="font-semibold text-slate-200 block">Backend API URL</label>
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder="https://nottingham-lighting-police-longest.trycloudflare.com"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-slate-100 placeholder-slate-500 text-xs font-mono focus:outline-none focus:border-cyan-400"
                />
              </div>

              {testResult && (
                <div
                  className={`p-3 rounded-xl border flex items-start gap-2.5 text-xs ${
                    testResult.ok
                      ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                      : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                  }`}
                >
                  {testResult.ok ? (
                    <FaCheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                  ) : (
                    <FaExclamationTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                  )}
                  <span>{testResult.msg}</span>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-800 gap-3">
              <button
                type="button"
                onClick={handleReset}
                className="btn btn-ghost text-xs py-2 px-3 text-slate-400 hover:text-slate-200"
              >
                Clear / Reset
              </button>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleTest}
                  disabled={testing || !apiUrl.trim()}
                  className="btn btn-outline text-xs py-2 px-3 flex items-center gap-1.5"
                >
                  <FaSyncAlt className={`w-3 h-3 ${testing ? 'animate-spin' : ''}`} />
                  {testing ? 'Testing…' : 'Test URL'}
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  className="btn btn-primary text-xs py-2 px-4"
                >
                  Save &amp; Connect
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
