import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import Logo from './Logo'
import { navItems } from '../lib/nav'
import { FaChevronLeft, FaChevronRight, FaBolt } from 'react-icons/fa'

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

        <div className="mt-6 space-y-4 px-4">
          {collapsed ? (
            <div className="hidden lg:block mx-auto h-11 w-11 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/10 border border-cyan-400/25 flex items-center justify-center">
              <FaBolt className="w-4 h-4 text-cyan-300" />
            </div>
          ) : (
            <div className="rounded-xl p-4 bg-gradient-to-br from-cyan-500/12 via-blue-500/8 to-transparent border border-cyan-400/20">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 pulse-glow" />
                <span className="text-xs font-semibold text-cyan-200 uppercase tracking-wide">BioVision Pro</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                Unlock higher throughput and priority inference on the full analysis pipeline.
              </p>
              <button
                onClick={() => {
                  onCloseMobile()
                  navigate('/')
                }}
                className="w-full text-xs font-semibold py-2 rounded-lg bg-cyan-400/90 text-[#04121d] hover:bg-cyan-300 transition-colors"
              >
                View Plans
              </button>
            </div>
          )}

          <div className={`pb-3 ${collapsed ? 'lg:hidden' : ''}`}>
            <div className="flex items-center justify-center gap-2 text-[11px] text-slate-600">
              <span className="h-6 w-6 rounded-md bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-[9px] font-bold text-white">
                BV
              </span>
              <span>BioVision · EfficientNet-B4</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
