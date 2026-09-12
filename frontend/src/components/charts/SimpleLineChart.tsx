import React from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts'

export interface ChartThreshold {
  value: number
  color: string
  label: string
}

interface SimpleLineChartProps {
  data: { name: string; value: number }[]
  color?: string
  thresholds?: ChartThreshold[]
  height?: number
  domain?: [number, number]
}

const GRID = 'rgba(148, 163, 184, 0.12)'
const TICK = '#64748b'

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null
  const value = payload[0].value
  return (
    <div className="rounded-lg border border-slate-700 bg-[#0a1120]/95 px-3 py-2 shadow-xl">
      <div className="text-xs text-slate-400">Frame {label}</div>
      <div className="text-sm font-semibold text-cyan-300">{(Number(value) * 100).toFixed(1)}% fake</div>
    </div>
  )
}

export default function SimpleLineChart({
  data,
  color = '#22d3ee',
  thresholds = [],
  height = 260,
  domain = [0, 1],
}: SimpleLineChartProps) {
  if (!data || data.length === 0) return <div className="text-slate-500 text-sm">No frame-level data available</div>
  const gradId = `grad-${color.replace('#', '')}`
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 24, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.4} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
          <XAxis dataKey="name" tick={{ fill: TICK, fontSize: 12 }} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis
            domain={domain}
            tick={{ fill: TICK, fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={42}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Tooltip content={<ChartTooltip />} />
          {thresholds.map((t) => (
            <ReferenceLine
              key={`${t.label}-${t.value}`}
              y={t.value}
              stroke={t.color}
              strokeDasharray="5 4"
              label={{ value: t.label, position: 'insideTopLeft', fill: t.color, fontSize: 11, dy: -6 }}
            />
          ))}
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2.5}
            fill={`url(#${gradId})`}
            dot={{ r: 3, fill: color, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
