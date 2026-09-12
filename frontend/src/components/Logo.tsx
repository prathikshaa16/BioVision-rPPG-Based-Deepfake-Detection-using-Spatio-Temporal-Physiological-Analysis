import React from 'react'
import { Link } from 'react-router-dom'
import { APP_NAME, APP_TAGLINE } from '../lib/nav'

export default function Logo({ size = 'md', light = false }: { size?: 'sm' | 'md'; light?: boolean }) {
  const box = size === 'sm' ? 'h-8 w-8 text-[11px]' : 'h-10 w-10 text-sm'
  return (
    <Link to={light ? '/' : '/dashboard'} className="flex items-center gap-3 group">
      <div
        className={`${box} rounded-xl bg-gradient-to-br from-cyan-400 via-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold tracking-tight shrink-0 shadow-[0_0_20px_rgba(34,211,238,0.35)]`}
      >
        BV
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold text-slate-100 group-hover:text-cyan-300 transition-colors">
          {APP_NAME}
        </div>
        <div className="text-[11px] text-slate-500">{APP_TAGLINE}</div>
      </div>
    </Link>
  )
}
