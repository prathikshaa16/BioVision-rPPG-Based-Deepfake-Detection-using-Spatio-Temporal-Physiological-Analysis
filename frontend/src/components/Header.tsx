import React from 'react'
import { useNavigate } from 'react-router-dom'
import Logo from './Logo'
import BackendStatus from './BackendStatus'
import { FaBars, FaVideo } from 'react-icons/fa'

export default function Header({ onMenu }: { onMenu: () => void }) {
  const navigate = useNavigate()

  return (
    <header className="bg-[#070d1a]/80 backdrop-blur-xl border-b border-slate-800/70 sticky top-0 z-30">
      <div className="px-4 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenu}
            aria-label="Open navigation"
            className="lg:hidden p-2 rounded-lg bg-slate-800/70 hover:bg-slate-700/70 text-slate-300 transition-colors"
          >
            <FaBars className="w-4 h-4" />
          </button>
          <div className="hidden sm:block lg:hidden">
            <Logo size="sm" />
          </div>
          <div className="hidden lg:flex items-center gap-2 text-sm text-slate-500">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 pulse-glow" />
            Forensic Analysis Workspace
          </div>
        </div>

        <div className="flex items-center gap-3">
          <BackendStatus compact />
          <button
            onClick={() => navigate('/analysis')}
            className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-400 to-cyan-500 text-[#04121d] text-sm font-semibold hover:from-cyan-300 hover:to-cyan-400 transition-all shadow-[0_0_16px_rgba(34,211,238,0.3)]"
          >
            <FaVideo className="w-3.5 h-3.5" />
            Analyze
          </button>
          <div className="h-9 w-9 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-semibold text-sm shadow-[0_0_14px_rgba(34,211,238,0.35)]">
            BV
          </div>
        </div>
      </div>
    </header>
  )
}
