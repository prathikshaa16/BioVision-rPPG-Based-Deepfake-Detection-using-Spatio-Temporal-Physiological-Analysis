import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import Logo from './Logo'
import { navItems } from '../lib/nav'
import { FaChevronLeft, FaChevronRight, FaBolt, FaProjectDiagram, FaMicrochip } from 'react-icons/fa'

interface SidebarProps {
  collapsed: boolean
  onToggleCollapsed: () => void
  mobileOpen: boolean
  onCloseMobile: () => void
}

export default function Sidebar({ collapsed, onToggleCollapsed, mobileOpen, onCloseMobile }: SidebarProps) {
  const navigate = useNavigate()
  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden transition-opacity ${
          mobileOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onCloseMobile}
        aria-hidden
      />

      <aside
        className={`fixed lg:sticky lg:top-0 lg:h-screen inset-y-0 left-0 z-50 flex flex-col
          bg-[#070d1c]/95 lg:bg-[#080e1c]/80 backdrop-blur-xl border-r border-slate-800/70
          w-72 transition-all duration-300 ease-out
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          ${collapsed ? 'lg:w-[76px]' : 'lg:w-72'}`}
      >
        <button
          onClick={onToggleCollapsed}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="hidden lg:flex absolute -right-3 top-20 z-10 h-7 w-7 rounded-full items-center justify-center bg-slate-800/95 border border-slate-700 text-slate-400 hover:text-cyan-300 hover:border-cyan-400/40 transition-colors shadow-lg"
        >
          {collapsed ? <FaChevronRight className="w-3 h-3" /> : <FaChevronLeft className="w-3 h-3" />}
        </button>
        <div className={`flex items-center justify-between px-5 pt-5 pb-6 ${collapsed ? 'lg:px-4 lg:justify-center' : ''}`}>
          {collapsed ? (
            <div className="hidden lg:flex h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 items-center justify-center text-white font-bold text-sm shadow-[0_0_18px_rgba(34,211,238,0.35)]">
              BV
            </div>
          ) : (
            <Logo />
          )}
          <button
            onClick={onCloseMobile}
            className="lg:hidden p-2 rounded-lg bg-slate-800/70 text-slate-300 text-xs"
            aria-label="Close navigation"
          >
            ✕
          </button>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto hidden-scrollbar px-3">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={onCloseMobile}
                title={collapsed ? item.label : undefined}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150 ${
                    collapsed ? 'lg:justify-center' : ''
                  } ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500/15 to-blue-500/10 text-cyan-300 border border-cyan-400/20 shadow-[inset_0_0_0_1px_rgba(34,211,238,0.08)]'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border border-transparent'
                  }`
                }
              >
                <Icon className="text-sm shrink-0" />
                <span className={`truncate text-sm font-medium ${collapsed ? 'lg:hidden' : ''}`}>{item.label}</span>
              </NavLink>
            )
          })}
        </nav>

        <div className="mt-auto space-y-3 px-3.5 pb-3">
          {collapsed ? (
            <button
              onClick={() => {
                onCloseMobile()
                navigate('/architecture')
              }}
              title="BioVision Multimodal Engine (3-Branch)"
              className="hidden lg:flex mx-auto h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/10 border border-cyan-400/25 items-center justify-center hover:border-cyan-400/50 transition-colors"
            >
              <FaMicrochip className="w-4 h-4 text-cyan-300" />
            </button>
          ) : (
            <div className="rounded-xl p-3.5 bg-gradient-to-br from-cyan-950/40 via-slate-900/60 to-blue-950/30 border border-cyan-500/25 shadow-lg relative overflow-hidden">
              <div className="flex items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                  <span className="text-[11px] font-bold text-cyan-200 uppercase tracking-wider font-mono">
                    Multimodal Pipeline
                  </span>
                </div>
                <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  94.6% AUC
                </span>
              </div>

              <div className="space-y-1 text-[11px] font-mono text-slate-400 mb-3 bg-black/30 rounded-lg p-2 border border-slate-800/80">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Visual:</span>
                  <span className="text-slate-200">EfficientNet-B4 + BiLSTM</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Physiology:</span>
                  <span className="text-cyan-300">CHROM rPPG (32-Bin FFT)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Fusion:</span>
                  <span className="text-slate-200">448-d Late Fusion Head</span>
                </div>
              </div>

              <button
                onClick={() => {
                  onCloseMobile()
                  navigate('/architecture')
                }}
                className="w-full text-xs font-semibold py-2 px-3 rounded-lg bg-cyan-500/20 text-cyan-200 hover:bg-cyan-500/30 border border-cyan-400/40 hover:border-cyan-300 transition-all flex items-center justify-center gap-1.5 shadow-[0_0_12px_rgba(34,211,238,0.15)]"
              >
                <FaProjectDiagram className="w-3.5 h-3.5 text-cyan-300" />
                <span>Inspect Architecture</span>
              </button>
            </div>
          )}

          <div className={`pt-1 ${collapsed ? 'lg:hidden' : ''}`}>
            <div className="flex items-center justify-center gap-2 text-[11px] text-slate-400 font-mono">
              <span className="h-5 w-5 rounded bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-[9px] font-bold text-white shadow-[0_0_8px_rgba(34,211,238,0.4)]">
                BV
              </span>
              <span>BioVision Multimodal v2.4</span>
              <span className="text-slate-400">·</span>
              <span className="text-emerald-400 font-sans">Active</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
