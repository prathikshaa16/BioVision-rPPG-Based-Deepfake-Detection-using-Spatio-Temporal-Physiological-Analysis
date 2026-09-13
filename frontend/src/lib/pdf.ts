import type { AnalysisResult } from './types'
import { buildExplanation, formatDateTime, formatDuration, formatFileSize, formatPercent } from './format'

const PAGE_WIDTH = 612
const PAGE_HEIGHT = 792
const MARGIN = 60
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2
const BOTTOM_MARGIN = 56

const HELVETICA_WIDTHS = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
  556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
  1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
  667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
  333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
  556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]

function pdfEscape(text: string): string {
  let out = ''
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0
    if (code === 0x28) out += '\\('
    else if (code === 0x29) out += '\\)'
    else if (code === 0x5c) out += '\\\\'
    else if (code >= 0x20 && code <= 0x7e) out += ch
    else if (code >= 0xa0 && code <= 0xff) out += ch
    else out += '?'
  }
  return out
}

function textWidth(text: string, fontSize: number): number {
  let total = 0
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0
    if (code >= 32 && code <= 126) total += HELVETICA_WIDTHS[code - 32]
    else total += 500
  }
  return (total * fontSize) / 1000
}

function wrapText(text: string, fontSize: number): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  if (words.length === 0) return ['']
  const lines: string[] = []
  let current = ''
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word
    if (textWidth(candidate, fontSize) <= CONTENT_WIDTH) {
      current = candidate
    } else {
      if (current) lines.push(current)
      current = word
    }
  }
  if (current) lines.push(current)
  return lines
}

function latin1Bytes(text: string): ArrayBuffer {
  const buffer = new ArrayBuffer(text.length)
  const bytes = new Uint8Array(buffer)
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i) & 0xff
  return buffer
}

function serializePdf(objects: string[]): ArrayBuffer {
  const header = '%PDF-1.4\n'
  const entries: string[] = []
  const offsets: number[] = []
  let offset = header.length
  for (let i = 0; i < objects.length; i++) {
    offsets.push(offset)
    const entry = `${i + 1} 0 obj\n${objects[i]}\nendobj\n`
    entries.push(entry)
    offset += entry.length
  }
  const xrefOffset = offset
  let xref = `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`
  for (const o of offsets) xref += `${String(o).padStart(10, '0')} 00000 n \n`
  const trailer = `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`
  return latin1Bytes(header + entries.join('') + xref + trailer)
}

interface ReportRow {
  label: string
  value: string
}

interface ReportSection {
  title: string
  rows?: ReportRow[]
  paragraphs?: string[]
}

interface ReportContent {
  title: string
  subtitle: string
  verdict: string
  sections: ReportSection[]
  footer: string
}

function buildReportPdf(content: ReportContent): Blob {
  const pageOps: string[][] = []
  let ops: string[] = []
  let y = PAGE_HEIGHT - MARGIN

  const ensureSpace = (needed: number) => {
    if (y - needed < BOTTOM_MARGIN) {
      pageOps.push(ops)
      ops = []
      y = PAGE_HEIGHT - MARGIN
    }
  }

  const addText = (text: string, font: string, size: number) => {
    const leading = size + 6
    for (const line of wrapText(text, size)) {
      ensureSpace(leading)
      ops.push(`${font} ${size} Tf`)
      ops.push(`1 0 0 1 ${MARGIN} ${y} Tm`)
      ops.push(`(${pdfEscape(line)}) Tj`)
      y -= leading
    }
  }

  const addBlank = (size: number) => {
    const leading = size + 6
    ensureSpace(leading)
    y -= leading
  }

  const addRule = () => {
    ensureSpace(14)
    y -= 6
    ops.push(`0.25 g`)
    ops.push(`${MARGIN} ${y} m ${PAGE_WIDTH - MARGIN} ${y} l S`)
    ops.push(`0 g`)
    y -= 8
  }

  addText(content.title, 'F1', 20)
  addText(content.subtitle, 'F2', 11)
  addBlank(8)
  addText(content.verdict, 'F1', 24)
  addBlank(10)
  addRule()

  for (const section of content.sections) {
    addRule()
    addText(section.title, 'F1', 12)
    addBlank(4)
    if (section.rows) {
      for (const row of section.rows) addText(`${row.label}: ${row.value}`, 'F2', 11)
    }
    if (section.paragraphs) {
      for (const paragraph of section.paragraphs) addText(paragraph, 'F2', 11)
    }
  }

  pageOps.push(ops)

  const numPages = pageOps.length
  const font1Obj = 3 + numPages
  const font2Obj = 4 + numPages
  const contentStart = 5 + numPages

  const objects: string[] = []
  objects.push('<< /Type /Catalog /Pages 2 0 R >>')
  objects.push(`<< /Type /Pages /Kids [${pageOps.map((_, i) => `${3 + i} 0 R`).join(' ')}] /Count ${numPages} >>`)
  for (let i = 0; i < numPages; i++) {
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}] ` +
        `/Resources << /Font << /F1 ${font1Obj} 0 R /F2 ${font2Obj} 0 R >> >> ` +
        `/Contents ${contentStart + i} 0 R >>`
    )
  }
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>')
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>')
  for (const pageOpsEntry of pageOps) {
    const footerOps = [
      `0.5 g`,
      `1 0 0 1 ${MARGIN} 40 Tm`,
      `(${pdfEscape(content.footer)}) Tj`,
      `0 g`,
    ]
    const contentStream = ['BT', ...pageOpsEntry, ...footerOps, 'ET'].join('\n')
    objects.push(`<< /Length ${contentStream.length} >>\nstream\n${contentStream}\nendstream`)
  }

  return new Blob([serializePdf(objects)], { type: 'application/pdf' })
}

function reportMetaRows(result: AnalysisResult): ReportRow[] {
  const rows: ReportRow[] = [
    { label: 'Filename', value: result.filename },
    { label: 'Analysis ID', value: result.analysis_id },
    { label: 'Analyzed At', value: formatDateTime(result.analyzed_at) },
    { label: 'Processing Time', value: formatDuration(result.processing_time) },
    { label: 'File Size', value: result.size ? formatFileSize(result.size) : '—' },
  ]
  if (result.device) rows.push({ label: 'Device', value: String(result.device).toUpperCase() })
  if (result.meta) {
    rows.push({ label: 'Resolution', value: `${result.meta.width} × ${result.meta.height}` })
    rows.push({ label: 'FPS', value: String(result.meta.fps) })
    rows.push({ label: 'Duration', value: formatDuration(result.meta.duration) })
    rows.push({ label: 'Frame Count', value: String(result.meta.frame_count) })
  }
  return rows
}

function observationLabel(p: number | null): string {
  if (p === null) return 'NO OBSERVATION'
  if (p >= 0.6) return 'FAKE EVIDENCE'
  if (p <= 0.4) return 'AUTHENTIC EVIDENCE'
  return 'UNCERTAIN'
}

function buildObservationLines(result: AnalysisResult): string[] {
  const frameRows = result.frame_results && result.frame_results.length
    ? result.frame_results
    : (result.frame_predictions || []).map((p, i) => ({
        index: i,
        faces: result.frames_with_faces > 0 ? 1 : 0,
        prediction: p,
        error: null,
      }))
  if (frameRows.length === 0) return ['No sequence observation breakdown was returned for this analysis.']
  return frameRows.map((r) => {
    const faces = r.error ? 'n/a' : String(r.faces)
    const prob = r.prediction === null ? '—' : formatPercent(r.prediction, 1)
    return `Sequence Step #${r.index + 1} | Faces: ${faces} | Representation Anomaly: ${prob} | ${observationLabel(r.prediction)}`
  })
}

export function generateNoFaceReportPdf(result: AnalysisResult): Blob {
  const explanation =
    result.explanation || 'No detectable face was found in the sampled frames. A reliable deepfake verdict cannot be determined from this video.'
  const note =
    'Because no usable face was detected, no REAL or FAKE probabilities or confidence values were computed and none are reported here.'

  return buildReportPdf({
    title: 'BIOVISION',
    subtitle: 'AI Deepfake Detection & Video Forensics',
    verdict: 'NO FACE DETECTED',
    sections: [
      {
        title: 'Analysis Summary',
        rows: [
          { label: 'Verdict', value: 'NO FACE DETECTED' },
          { label: 'Faces Detected', value: String(result.faces_detected ?? result.frames_with_faces ?? 0) },
          { label: 'Frames Sampled', value: String(result.frames_sampled) },
          ...reportMetaRows(result),
        ],
      },
      { title: 'Explanation', paragraphs: [explanation] },
      { title: 'Note', paragraphs: [note] },
    ],
    footer: 'Generated by BioVision deepfake detector - no fake/real probabilities were computed',
  })
}

function buildRppgSection(result: AnalysisResult): ReportSection {
  const rppg = result.rppg
  if (!rppg) {
    return {
      title: 'Physiological Signal Analysis (rPPG)',
      paragraphs: ['No rPPG physiological analysis was returned for this video.'],
    }
  }
  if (rppg.status === 'SKIPPED') {
    return {
      title: 'Physiological Signal Analysis (rPPG)',
      paragraphs: [
        'rPPG was skipped because no usable face was detected in the sampled frames.',
      ],
    }
  }
  if (rppg.status === 'UNAVAILABLE' || !rppg.heart_rate_bpm) {
    return {
      title: 'Physiological Signal Analysis (rPPG)',
      paragraphs: [
        'A physiological (rPPG) signal could not be reliably recovered from this video. This is expected for very short clips, heavy motion, or faces too small to sample.',
        rppg.explanation || 'The physiological signal was not recoverable.',
        'No heart-rate estimate is reported because none could be computed honestly.',
      ],
    }
  }
  const rows: ReportRow[] = [
    { label: 'Status', value: 'Signal recovered' },
    { label: 'Frames Used', value: String(rppg.frames_used) },
    { label: 'Heart Rate Estimate', value: `${rppg.heart_rate_bpm} BPM` },
    { label: 'Dominant Frequency', value: `${rppg.dominant_frequency} Hz` },
    { label: 'Signal Quality', value: rppg.signal_quality != null ? formatPercent(rppg.signal_quality, 0) : '—' },
    { label: 'Method', value: 'CHROM (de Haan & Jeanne 2013), forehead + cheek regions' },
    { label: 'Analysis Window', value: rppg.window ? `${rppg.window.frames_read} frames @ ${rppg.window.fps} FPS` : '—' },
  ]
  const metrics = rppg.quality_metrics
  if (metrics && metrics.snr_db != null) rows.push({ label: 'SNR', value: `${metrics.snr_db} dB` })
  return {
    title: 'Physiological Signal Analysis (rPPG)',
    rows,
    paragraphs: [
      'The CHROM rPPG signal is quality-gated and contributes up to 20% of the fused verdict when a reliable signal is available. EfficientNet-B4 contributes 80%.',
    ],
  }
}

export function generateVerdictReportPdf(result: AnalysisResult): Blob {
  const verdict = result.result
  const confidence = formatPercent(result.confidence)
  const explanation = result.explanation || buildExplanation(result)

  const sections: ReportSection[] = [
    {
      title: 'Analysis Summary',
      rows: [
        { label: 'Verdict', value: verdict },
        { label: 'Confidence', value: confidence },
        ...reportMetaRows(result),
      ],
    },
    {
      title: 'Detection Statistics',
      rows: [
        { label: 'Sequence Observations', value: String(result.frames_sampled) },
        { label: 'Faces Detected', value: String(result.faces_detected ?? result.frames_with_faces ?? 0) },
        { label: 'Fused Fake Probability', value: formatPercent(result.fake_probability) },
        { label: 'Authentic Probability', value: formatPercent(result.real_probability) },
        { label: 'Visual Anomaly Score', value: formatPercent(result.visual_fake_probability ?? result.mean_probability) },
        { label: 'Sequence Consistency (1-StdDev)', value: formatPercent(1.0 - result.std_probability) },
      ],
    },
    { title: 'Sequence Observation Analysis', paragraphs: buildObservationLines(result) },
    buildRppgSection(result),
    { title: 'Explanation', paragraphs: [explanation] },
    {
      title: 'Model',
      rows: [
        { label: 'Architecture', value: result.model_name || 'BioVision Spatio-Temporal + rPPG Fusion' },
        { label: 'Checkpoint', value: result.model_version || 'biovision_best.pt' },
        { label: 'Device', value: result.device ? String(result.device).toUpperCase() : '—' },
      ],
    },
  ]

  return buildReportPdf({
    title: 'BIOVISION',
    subtitle: 'Spatio-Temporal & Physiological Deepfake Detection',
    verdict,
    sections,
    footer: 'Generated by BioVision - EfficientNet-B4 + LSTM + CHROM rPPG late-fusion analysis',
  })
}

export function downloadReportPdf(result: AnalysisResult): void {
  const blob = result.result === 'NO_FACE' ? generateNoFaceReportPdf(result) : generateVerdictReportPdf(result)
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  const base = (result.filename || 'video').replace(/\.[^.]+$/, '') || 'video'
  const slug = result.result.toLowerCase().replace(/[^a-z0-9]/g, '-')
  anchor.href = url
  anchor.download = `${base}-${slug}-report.pdf`
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function downloadNoFaceReportPdf(result: AnalysisResult): void {
  downloadReportPdf(result)
}
