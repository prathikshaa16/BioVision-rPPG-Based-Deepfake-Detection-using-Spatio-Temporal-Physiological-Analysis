import React, { useEffect, useState } from 'react'
import Card from '../components/Card'
import { API_BASE, clearLastResult } from '../lib/api'
import { FaServer, FaDatabase, FaTrashAlt, FaPalette, FaSlidersH, FaInfoCircle, FaMoon } from 'react-icons/fa'

const SETTINGS_KEY = 'biovision:settings'

interface SettingsState {
  persistHistory: boolean
  reducedMotion: boolean
}

function loadSettings(): SettingsState {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY)
    if (raw) return { persistHistory: true, reducedMotion: false, ...JSON.parse(raw) }
  } catch {
    // ignore
  }
  return { persistHistory: true, reducedMotion: false }
}

function Toggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={onToggle}
      className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${on ? 'bg-cyan-400' : 'bg-slate-700'}`}
    >
      <span
        className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${
          on ? 'translate-x-5' : ''
        }`}
      />
    </button>
  )
}

export default function Settings() {
  const [settings, setSettings] = useState<SettingsState>(loadSettings)

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
    } catch {
      // ignore
    }
  }, [settings])

  const handleClear = () => {
    if (!window.confirm('This will remove locally stored analysis history. This cannot be undone.')) return
    clearLastResult()
    try {
      localStorage.removeItem('biovision:history')
    } catch {
      // ignore
    }
    setSettings((s) => ({ ...s, persistHistory: true }))
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <header className="page-head">
        <div>
          <h2 className="page-title">Settings</h2>
          <p className="page-sub">Application preferences for BioVision</p>
        </div>
        <span className="chip chip--info">
          <span className="status-dot bg-cyan-400" />
          Dark Only
        </span>
      </header>

      <Card title="Appearance" action={<FaPalette className="w-4 h-4 text-slate-500" />}>
        <div className="space-y-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-medium text-slate-100 flex items-center gap-2">
                <FaMoon className="w-3.5 h-3.5 text-cyan-300" />
                BioVision Dark Theme
              </div>
              <div className="text-xs text-slate-500 mt-0.5">
                BioVision is a dark-first forensic product. A light theme switch is intentionally not provided.
              </div>
            </div>
            <span className="chip chip--idle">Always on</span>
          </div>
          <div className="flex items-center justify-between gap-4 border-t border-slate-800/70 pt-4">
            <div>
              <div className="text-sm font-medium text-slate-100">Reduced motion</div>
              <div className="text-xs text-slate-500 mt-0.5">Minimize animations across the interface.</div>
            </div>
            <Toggle on={settings.reducedMotion} onToggle={() => setSettings((s) => ({ ...s, reducedMotion: !s.reducedMotion }))} />
          </div>
        </div>
      </Card>

      <Card title="Analysis Preferences" action={<FaSlidersH className="w-4 h-4 text-slate-500" />}>
        <div className="space-y-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-medium text-slate-100">Persist results to history</div>
              <div className="text-xs text-slate-500 mt-0.5">
                Save every analysis result in this browser so it survives a page refresh.
              </div>
            </div>
            <Toggle on={settings.persistHistory} onToggle={() => setSettings((s) => ({ ...s, persistHistory: !s.persistHistory }))} />
          </div>
          <div className="glass-inset p-4 text-xs text-slate-500 leading-relaxed">
            Analysis always runs through the real backend <code className="text-cyan-300 font-mono">POST /upload</code> endpoint.
            These preferences only control how results are retained locally — nothing is ever fabricated.
          </div>
        </div>
      </Card>

      <Card title="Backend" action={<FaServer className="w-4 h-4 text-slate-500" />}>
        <div className="space-y-3 text-sm text-slate-400">
          <div className="flex justify-between items-center">
            <span className="data-row-label">API endpoint</span>
            <code className="font-mono text-xs text-cyan-300 bg-slate-900/60 border border-slate-800 px-2 py-1 rounded">{API_BASE}</code>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            The frontend talks to the FastAPI backend directly. Start it from the project root with{' '}
            <code className="text-slate-300 bg-slate-900/60 border border-slate-800 px-1.5 py-0.5 rounded font-mono text-xs">
              python -m uvicorn backend.app.main:app --port 8000
            </code>
            . When offline, the app clearly indicates it and disables analysis.
          </p>
        </div>
      </Card>

      <Card title="Data / Local History" action={<FaDatabase className="w-4 h-4 text-slate-500" />}>
        <div className="space-y-4">
          <div className="text-sm text-slate-500 leading-relaxed">
            Local browser storage holds your last result and up to 50 past analyses. Clearing removes{' '}
            <code className="text-slate-300 font-mono text-xs">biovision:lastResult</code> and{' '}
            <code className="text-slate-300 font-mono text-xs">biovision:history</code> from localStorage. It does not
            affect the server's in-memory history.
          </div>
          <button onClick={handleClear} className="btn btn-danger">
            <FaTrashAlt className="w-4 h-4" />
            Clear Local Analysis History
          </button>
        </div>
      </Card>

      <Card title="About BioVision" action={<FaInfoCircle className="w-4 h-4 text-slate-500" />}>
        <div className="space-y-3 text-sm">
          <div className="data-row">
            <span className="data-row-label">Product</span>
            <span className="data-row-value">BioVision — AI Deepfake Detection &amp; Video Forensics</span>
          </div>
          <div className="data-row">
            <span className="data-row-label">Detection model</span>
            <span className="data-row-value">BioVision cached EfficientNet-B4 sequence + rPPG LSTM fusion</span>
          </div>
          <div className="data-row">
            <span className="data-row-label">Pipeline</span>
            <span className="data-row-value">Face sequence → spatial features → temporal LSTM → rPPG → trained fusion</span>
          </div>
          <div className="data-row">
            <span className="data-row-label">Frontend</span>
            <span className="data-row-value">React · Vite · Tailwind v4</span>
          </div>
          <div className="data-row">
            <span className="data-row-label">Account / billing</span>
            <span className="data-row-value text-slate-500">None — pricing shown on the landing page is presentation-only</span>
          </div>
        </div>
      </Card>
    </div>
  )
}
