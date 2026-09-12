import React, { useEffect, useState } from 'react'
import { isBackendOnline } from '../lib/api'

type Status = 'checking' | 'online' | 'offline'

const STATUS_STYLES: Record<Status, { dot: string; text: string; label: string }> = {
  checking: { dot: 'bg-amber-400', text: 'text-amber-300', label: 'Checking' },
  online: { dot: 'bg-emerald-400', text: 'text-emerald-300', label: 'Online' },
  offline: { dot: 'bg-rose-500', text: 'text-rose-300', label: 'Offline' },
}

export default function BackendStatus({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<Status>('checking')

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      const online = await isBackendOnline(4000)
      if (!cancelled) setStatus(online ? 'online' : 'offline')
    }
    check()
    const interval = setInterval(check, 10000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  const s = STATUS_STYLES[status]

  if (compact) {
    return (
      <span
        className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-slate-900/60 border border-slate-700/70"
        title={status === 'offline' ? 'Analysis server unreachable — start the backend on port 8000' : 'Analysis server status'}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${status === 'online' ? 'pulse-glow' : ''}`} />
        <span className={s.text}>Server {s.label}</span>
      </span>
    )
  }

  return (
    <div
      className={`rounded-xl px-4 py-2 text-sm font-medium border inline-flex items-center gap-2 ${
        status === 'online'
          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
          : status === 'offline'
            ? 'border-rose-500/30 bg-rose-500/10 text-rose-300'
            : 'border-amber-500/30 bg-amber-500/10 text-amber-300'
      }`}
    >
      <span className={`w-2 h-2 rounded-full ${s.dot} ${status === 'online' ? 'pulse-glow' : ''}`} />
      {status === 'online' ? 'Live backend connected' : status === 'offline' ? 'Backend offline' : 'Checking backend…'}
    </div>
  )
}
