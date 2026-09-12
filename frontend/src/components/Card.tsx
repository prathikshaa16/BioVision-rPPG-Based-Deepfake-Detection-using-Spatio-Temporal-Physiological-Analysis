import React from 'react'

interface CardProps {
  title?: string
  subtitle?: string
  action?: React.ReactNode
  className?: string
  glow?: boolean
  children: React.ReactNode
}

export default function Card({ title, subtitle, action, className = '', glow = false, children }: CardProps) {
  return (
    <div className={`${glow ? 'glass-card--accent' : 'glass-card'} p-5 ${className}`}>
      {(title || action) && (
        <div className="flex items-start justify-between gap-4 mb-4">
          <div className="min-w-0">
            {title && <div className="text-sm font-semibold text-slate-100">{title}</div>}
            {subtitle && <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>}
          </div>
          {action && <div className="flex-shrink-0">{action}</div>}
        </div>
      )}
      {children}
    </div>
  )
}
