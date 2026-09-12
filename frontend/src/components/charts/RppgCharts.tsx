import React from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Legend } from 'recharts'
import type { RppgData } from '../../lib/types'

const GRID = 'rgba(148, 163, 184, 0.12)'
const TICK = '#64748b'

function RppgTooltip({ active, payload, label, unit }: any) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="rounded-lg border border-slate-700 bg-[#0a1120]/95 px-3 py-2 shadow-xl text-xs">
      <div className="text-slate-400 mb-1">
        {unit === 's' ? `${label}s` : `${label} Hz`}
      </div>
      {payload.map((entry: any) => (
        <div key={entry.dataKey} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: entry.stroke || entry.color }} />
          <span className="text-slate-400 capitalize">{String(entry.dataKey)}:</span>
          <span className="font-semibold text-slate-100">{Number(entry.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

export default function RppgCharts({ rppg }: { rppg: RppgData }) {
  const signal = rppg.signal
  const filtered = rppg.filtered_signal
  const spectrum = rppg.frequency
  const dominantHz = rppg.dominant_frequency

  const signalData =
    signal && signal.timestamps.length
      ? signal.timestamps.map((t, i) => ({
          t,
          r: signal.r[i] ?? null,
          g: signal.g[i] ?? null,
          b: signal.b[i] ?? null,
        }))
      : []

  const filteredData =
    filtered && filtered.timestamps.length
      ? filtered.timestamps.map((t, i) => ({ t, amplitude: filtered.amplitude[i] ?? null }))
      : []

  const spectrumData =
    spectrum && spectrum.frequencies.length
      ? spectrum.frequencies.map((f, i) => ({ f, power: spectrum.power[i] ?? 0 }))
      : []

  return (
    <div className="space-y-6">
      {signalData.length > 0 && (
        <div>
          <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Forehead + cheek ROI — mean color intensity over time</p>
          <div style={{ width: '100%', height: 220 }}>
            <ResponsiveContainer>
              <LineChart data={signalData} margin={{ top: 8, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                <XAxis dataKey="t" tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={{ stroke: GRID }} unit="s" />
                <YAxis tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} width={46} />
                <Tooltip content={<RppgTooltip unit="s" />} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                <Line type="monotone" dataKey="r" stroke="#f87171" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="g" stroke="#4ade80" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="b" stroke="#60a5fa" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {filteredData.length > 0 && (
        <div>
          <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">CHROM pulse signal (0.8–3.0 Hz bandpassed)</p>
          <div style={{ width: '100%', height: 180 }}>
            <ResponsiveContainer>
              <LineChart data={filteredData} margin={{ top: 8, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                <XAxis dataKey="t" tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={{ stroke: GRID }} unit="s" />
                <YAxis tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} width={46} />
                <Tooltip content={<RppgTooltip unit="s" />} />
                <Line type="monotone" dataKey="amplitude" stroke="#22d3ee" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {spectrumData.length > 0 && (
        <div>
          <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">
            Frequency spectrum
            {dominantHz != null && (
              <span className="ml-2 normal-case text-cyan-300 font-medium">peak {dominantHz} Hz ({Math.round(dominantHz * 60)} BPM)</span>
            )}
          </p>
          <div style={{ width: '100%', height: 200 }}>
            <ResponsiveContainer>
              <LineChart data={spectrumData} margin={{ top: 8, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                <XAxis dataKey="f" tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={{ stroke: GRID }} unit=" Hz" />
                <YAxis tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} width={46} />
                <Tooltip content={<RppgTooltip unit="Hz" />} />
                {dominantHz != null && (
                  <ReferenceLine
                    x={dominantHz}
                    stroke="#22d3ee"
                    strokeDasharray="5 4"
                    label={{ value: `${dominantHz} Hz`, position: 'insideTopRight', fill: '#22d3ee', fontSize: 11 }}
                  />
                )}
                <Line type="monotone" dataKey="power" stroke="#a78bfa" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
