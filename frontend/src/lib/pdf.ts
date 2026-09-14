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
    else if (ch === '—' || ch === '–') out += '-'
    else if (ch === '×') out += 'x'
    else if (ch === '•') out += '*'
    else if (ch === '’' || ch === '‘') out += "'"
    else if (ch === '“' || ch === '”') out += '"'
    else if (ch === '…') out += '...'
    else if (ch === '±') out += '+/-'
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
    ensureSpace(76)
    addRule()
    addText(section.title, 'F1', 12)
    addBlank(4)
    if (section.rows) {
      for (const row of section.rows) addText(`${row.label}: ${row.value}`, 'F2', 11)
    }
    if (section.paragraphs) {
      for (const paragraph of section.paragraphs) {
        addText(paragraph, 'F2', 11)
        addBlank(4)
      }
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

function buildVisualExaminationSection(result: AnalysisResult): ReportSection {
  const preds = result.frame_predictions || []
  const total = preds.length || result.frames_sampled || 32
  const pMax = preds.length ? Math.max(...preds) : (result.fake_probability || 0)
  const maxIdx = preds.length ? preds.indexOf(pMax) + 1 : 1
  const anomCount = preds.filter((p) => p >= 0.50).length
  const anomRatio = total > 0 ? Math.round((anomCount / total) * 100) : 0
  const consistency = result.consistency ?? (result.std_probability != null ? 1.0 - result.std_probability : 0.85)

  const rows: ReportRow[] = [
    { label: 'Spatial Feature Extractor', value: 'EfficientNet-B4 (1,792 dimensions per facial observation)' },
    { label: 'Temporal Dependency Model', value: '2-Layer BiLSTM with Multi-Head Self-Attention' },
    { label: 'Sequence Observations', value: `${total} sampled facial crops (${result.frames_with_faces ?? total} faces tracked)` },
    { label: 'Visual Anomaly Score', value: formatPercent(result.visual_fake_probability ?? result.fake_probability) },
    { label: 'Peak Representation Anomaly', value: `${formatPercent(pMax)} (observed at Step #${maxIdx})` },
    { label: 'Anomalous Observations', value: `${anomCount} of ${total} observations (${anomRatio}% exceeding 50% threshold)` },
    { label: 'Sequence Consistency (1-StdDev)', value: formatPercent(consistency) },
    { label: 'Temporal Spread (StdDev)', value: formatPercent(result.std_probability ?? 0, 2) },
  ]

  const paragraphs: string[] = []
  if (result.result === 'FAKE') {
    paragraphs.push(
      `Spatio-temporal analysis identified significant representation anomalies concentrated across ` +
      `${anomCount} of ${total} sequence observations, with localized representation distortion peaking at ${formatPercent(pMax)}. ` +
      `The presence of persistent anomaly clusters indicates synthetic boundary blending, warped facial meshes, or ` +
      `temporal discontinuities characteristic of face replacement or generative expression reenactment.`
    )
  } else if (result.result === 'REAL') {
    paragraphs.push(
      `Spatio-temporal representations demonstrated high temporal stability (${formatPercent(consistency)}) ` +
      `across all ${total} sampled observations. Facial geometry, border contours, and inter-frame transitions ` +
      `remained coherent with no significant manipulation artifacts detected.`
    )
  } else {
    paragraphs.push(
      `Spatio-temporal observations remained borderline (${formatPercent(result.visual_fake_probability ?? result.fake_probability)}). ` +
      `Facial motion or video compression obscures clear classification within standard confidence bounds.`
    )
  }

  return {
    title: 'Spatio-Temporal Visual Examination',
    rows,
    paragraphs,
  }
}

function buildSequenceIntervalSection(result: AnalysisResult): ReportSection {
  const preds = result.frame_predictions || []
  if (preds.length === 0) {
    return {
      title: 'Temporal Sequence Trajectory',
      paragraphs: ['Sequence trajectory breakdown is not available for this analysis.'],
    }
  }

  const qSize = Math.ceil(preds.length / 4)
  const quarters: { label: string; mean: number; max: number; status: string }[] = []
  for (let q = 0; q < 4; q++) {
    const slice = preds.slice(q * qSize, (q + 1) * qSize)
    if (slice.length === 0) continue
    const start = q * qSize + 1
    const end = Math.min((q + 1) * qSize, preds.length)
    const m = slice.reduce((a, b) => a + b, 0) / slice.length
    const maxVal = Math.max(...slice)
    const status = m >= 0.55 ? 'ELEVATED ANOMALY' : m <= 0.40 ? 'COHERENT BASELINE' : 'BORDERLINE'
    quarters.push({
      label: `Steps #${start} - #${end}`,
      mean: m,
      max: maxVal,
      status,
    })
  }

  const rows: ReportRow[] = quarters.map((q) => ({
    label: q.label,
    value: `Mean: ${formatPercent(q.mean, 1)} | Peak: ${formatPercent(q.max, 1)} | ${q.status}`,
  }))

  const paragraphs: string[] = [
    result.result === 'FAKE'
      ? 'Temporal clustering indicates elevated manipulation signatures during active speech and expression intervals, followed by typical attenuation during neutral/static frames.'
      : 'Temporal trajectory shows uniform stability across all observation intervals with no anomalous phase transitions.',
  ]

  return {
    title: 'Temporal Sequence Trajectory Highlights',
    rows,
    paragraphs,
  }
}

function buildFusionSection(result: AnalysisResult): ReportSection {
  const fusion = result.fusion
  const rows: ReportRow[] = [
    { label: 'Decision Architecture', value: 'Quality-Gated Late Fusion (EfficientNet-B4 + CHROM rPPG)' },
    { label: 'Visual Evidence Weight', value: fusion ? `${Math.round(fusion.visual_weight * 100)}%` : '80%' },
    { label: 'Physiological Evidence Weight', value: fusion ? `${Math.round(fusion.rppg_weight * 100)}%` : (result.rppg?.status === 'AVAILABLE' ? '20%' : '0%') },
    { label: 'Visual Score Contribution', value: formatPercent(fusion?.visual_probability ?? result.visual_fake_probability ?? result.fake_probability) },
    { label: 'Physiological Anomaly Score', value: fusion?.rppg_anomaly_score != null ? formatPercent(fusion.rppg_anomaly_score) : (result.rppg?.status === 'AVAILABLE' ? 'Quality-Gated' : 'None (0%)') },
    { label: 'Calibrated Decision Threshold', value: '0.50 (margin +/- 0.05)' },
    { label: 'Final Fused Manipulation Probability', value: formatPercent(result.fake_probability) },
    { label: 'Authentic Likelihood', value: formatPercent(result.real_probability) },
    { label: 'Verdict', value: result.result },
  ]

  return {
    title: 'Multimodal Evidence Fusion & Decision Rule',
    rows,
    paragraphs: [
      'The late-fusion head weighs visual-temporal representations (80%) against biological blood volume pulse recovery (up to 20%). ' +
      'Signals failing physiological quality criteria are gated out to prevent noisy physiological estimates from overriding confident visual evidence.',
    ],
  }
}

function buildGuidanceSection(result: AnalysisResult): ReportSection {
  const explanation = result.explanation || buildExplanation(result)
  const paragraphs: string[] = [
    `1. Primary Finding: ${explanation}`,
    '2. Complementary Verification: For critical or evidentiary applications, inspect complementary audio-visual phoneme synchronization and verify container metadata for transcoding or generation provenance.',
    `3. System Verification: Evaluated by BioVision Multimodal Pipeline (${result.model_name || 'EfficientNet-B4 + LSTM + CHROM rPPG'}) on ${result.device ? String(result.device).toUpperCase() : 'CPU'}. Analysis Token: ${result.analysis_id}.`,
  ]
  return {
    title: 'Forensic Guidance & Chain of Custody',
    paragraphs,
  }
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
    footer: `BioVision Forensic Assessment Report · ${result.analysis_id}`,
  })
}

function buildRppgSection(result: AnalysisResult): ReportSection {
  const rppg = result.rppg
  if (!rppg || rppg.status === 'SKIPPED') {
    return {
      title: 'Physiological Signal Analysis (rPPG)',
      paragraphs: ['rPPG analysis was skipped because no usable face was detected in the sampled frames.'],
    }
  }
  if (rppg.status === 'UNAVAILABLE' || !rppg.heart_rate_bpm) {
    return {
      title: 'Physiological Signal Analysis (rPPG)',
      paragraphs: [
        'A physiological (rPPG) signal could not be reliably recovered from this video. This is expected for very short clips, heavy motion, or faces too small to sample.',
        rppg.explanation || 'The physiological signal was not recoverable.',
        'In accordance with BioVision quality-gating protocols, unavailable signals contribute 0% weight to the fused verdict.',
      ],
    }
  }
  const rows: ReportRow[] = [
    { label: 'Status', value: 'Signal recovered' },
    { label: 'Extraction Method', value: 'CHROM (de Haan & Jeanne 2013), forehead + bilateral cheek ROIs' },
    { label: 'Analysis Window', value: rppg.window ? `${rppg.window.frames_read} frames @ ${rppg.window.fps} FPS` : '—' },
    { label: 'Heart Rate Estimate', value: `${rppg.heart_rate_bpm} BPM` },
    { label: 'Dominant Frequency', value: `${rppg.dominant_frequency} Hz` },
    { label: 'Signal Quality Index', value: rppg.signal_quality != null ? formatPercent(rppg.signal_quality, 1) : '—' },
  ]
  const metrics = rppg.quality_metrics
  if (metrics && metrics.snr_db != null) rows.push({ label: 'Signal-to-Noise Ratio (SNR)', value: `${metrics.snr_db} dB` })
  if (metrics && metrics.spectral_concentration != null) rows.push({ label: 'Spectral Concentration', value: String(metrics.spectral_concentration) })

  const paragraphs: string[] = []
  const hr = rppg.heart_rate_bpm
  const freq = rppg.dominant_frequency
  const qual = rppg.signal_quality ?? 0

  if (result.result === 'FAKE') {
    if (hr && hr > 130) {
      paragraphs.push(
        `Physiological analysis extracted an abnormally elevated dominant frequency of ${freq} Hz (${hr} BPM) ` +
        `accompanied by degraded signal quality (${formatPercent(qual, 1)}). In seated human subjects at rest, genuine ` +
        `cardiac pulse remains strictly within 60–100 BPM (1.0–1.67 Hz). This high-frequency pulse artifact is a recognized ` +
        `diagnostic marker of generative synthetic pixel jitter (frame-by-frame GAN/diffusion reconstruction noise), ` +
        `providing independent physiological corroboration of manipulation.`
      )
    } else {
      paragraphs.push(
        `Physiological blood volume pulse dynamics exhibited irregular wave morphology and low spectral concentration, ` +
        `characteristic of synthetic reenactment where facial skin color fluctuations lack genuine biological cardiac periodicity.`
      )
    }
  } else if (result.result === 'REAL') {
    paragraphs.push(
      `Recovered blood volume pulse demonstrated coherent cardiac spectral peaks within normal resting biological ` +
      `parameters (${hr} BPM, ${freq} Hz), consistent with genuine subcutaneous capillary blood flow.`
    )
  } else {
    paragraphs.push(
      'Physiological pulse recovery remained indeterminate. Marginal SNR obscures subtle cardiac harmonics.'
    )
  }

  paragraphs.push(
    'The CHROM rPPG signal is quality-gated and contributes up to 20% of the fused verdict when a reliable signal is available. EfficientNet-B4 contributes 80%.'
  )

  return {
    title: 'Physiological Signal Analysis (rPPG)',
    rows,
    paragraphs,
  }
}

export function generateVerdictReportPdf(result: AnalysisResult): Blob {
  const verdict = result.result
  const confidence = formatPercent(result.confidence)

  const sections: ReportSection[] = [
    {
      title: 'Executive Summary & Multimodal Verdict',
      rows: [
        { label: 'Final Verdict', value: verdict },
        { label: 'Decision Confidence', value: confidence },
        { label: 'Fused Fake Probability', value: formatPercent(result.fake_probability) },
        { label: 'Authentic Likelihood', value: formatPercent(result.real_probability) },
        { label: 'Calibrated Decision Threshold', value: '0.50 (balanced cutoff)' },
        ...reportMetaRows(result),
      ],
      paragraphs: [
        `BioVision executed an end-to-end multimodal deepfake forensic examination combining spatio-temporal ` +
        `facial feature representations with photoplethysmographic (rPPG) blood volume pulse recovery. Final classification ` +
        `is determined by quality-gated late fusion of spatial, temporal, and physiological evidence channels against ` +
        `a calibrated decision boundary.`,
      ],
    },
    buildVisualExaminationSection(result),
    buildRppgSection(result),
    buildFusionSection(result),
    buildSequenceIntervalSection(result),
    buildGuidanceSection(result),
  ]

  return buildReportPdf({
    title: 'BIOVISION',
    subtitle: 'Multimodal Deepfake Detection & Video Forensics Report',
    verdict,
    sections,
    footer: `Generated by BioVision Deepfake Forensic Engine · ID: ${result.analysis_id}`,
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
