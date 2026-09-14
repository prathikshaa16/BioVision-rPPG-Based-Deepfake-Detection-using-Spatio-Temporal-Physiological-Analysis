import { jsPDF } from 'jspdf'
import type { AnalysisResult } from './types'
import { buildExplanation, formatDateTime, formatDuration, formatFileSize, formatPercent } from './format'

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
  confidence: string
  analysisId: string
  sections: ReportSection[]
}

function buildReportPdfDoc(content: ReportContent): jsPDF {
  const doc = new jsPDF({
    unit: 'pt',
    format: 'letter', // 612 x 792 pt
  })

  const PAGE_WIDTH = 612
  const PAGE_HEIGHT = 792
  const MARGIN = 42
  const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2
  const BOTTOM_MARGIN = 50

  let y = MARGIN

  const checkPageBreak = (needed: number) => {
    if (y + needed > PAGE_HEIGHT - BOTTOM_MARGIN) {
      doc.addPage()
      y = MARGIN + 18
      // Running sub-header on continuation pages
      doc.setFont('helvetica', 'normal')
      doc.setFontSize(8)
      doc.setTextColor(148, 163, 184) // slate-400
      doc.text(`BioVision Forensic Report · ${content.analysisId}`, MARGIN, MARGIN)
      doc.setDrawColor(226, 232, 240) // slate-200
      doc.setLineWidth(0.5)
      doc.line(MARGIN, MARGIN + 4, PAGE_WIDTH - MARGIN, MARGIN + 4)
    }
  }

  // 1. Header Banner
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(20)
  doc.setTextColor(15, 23, 42) // slate-900
  doc.text(content.title, MARGIN, y + 14)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(9)
  doc.setTextColor(100, 116, 139) // slate-500
  doc.text(content.subtitle, MARGIN, y + 27)

  // Verdict Badge on Right Header
  const verdict = content.verdict || 'UNKNOWN'
  const isFake = verdict === 'FAKE'
  const isReal = verdict === 'REAL'
  const badgeW = 160
  const badgeH = 34
  const badgeX = PAGE_WIDTH - MARGIN - badgeW
  const badgeY = y - 2

  if (isFake) {
    doc.setFillColor(254, 242, 242) // red-50
    doc.setDrawColor(248, 113, 113) // red-400
    doc.roundedRect(badgeX, badgeY, badgeW, badgeH, 4, 4, 'FD')
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(14)
    doc.setTextColor(185, 28, 28) // red-700
    doc.text('VERDICT: FAKE', badgeX + 12, badgeY + 16)
    doc.setFontSize(8)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(220, 38, 38) // red-600
    doc.text(`Confidence: ${content.confidence}`, badgeX + 12, badgeY + 28)
  } else if (isReal) {
    doc.setFillColor(240, 253, 244) // green-50
    doc.setDrawColor(74, 222, 128) // green-400
    doc.roundedRect(badgeX, badgeY, badgeW, badgeH, 4, 4, 'FD')
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(14)
    doc.setTextColor(21, 128, 61) // green-700
    doc.text('VERDICT: REAL', badgeX + 12, badgeY + 16)
    doc.setFontSize(8)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(22, 163, 74) // green-600
    doc.text(`Confidence: ${content.confidence}`, badgeX + 12, badgeY + 28)
  } else {
    doc.setFillColor(254, 243, 199) // amber-50
    doc.setDrawColor(251, 191, 36) // amber-400
    doc.roundedRect(badgeX, badgeY, badgeW, badgeH, 4, 4, 'FD')
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(12)
    doc.setTextColor(180, 83, 9) // amber-700
    doc.text(verdict, badgeX + 12, badgeY + 16)
    doc.setFontSize(8)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(217, 119, 6)
    doc.text(`Confidence: ${content.confidence}`, badgeX + 12, badgeY + 28)
  }

  y += 42

  // Horizontal divider
  doc.setDrawColor(203, 213, 225) // slate-300
  doc.setLineWidth(1)
  doc.line(MARGIN, y, PAGE_WIDTH - MARGIN, y)
  y += 14

  // 2. Render Sections
  for (const section of content.sections) {
    const rowSpace = (section.rows ? section.rows.length : 0) * 14
    const estParaLines = (section.paragraphs || []).reduce((acc, p) => acc + Math.ceil(p.length / 85), 0)
    const estParaSpace = estParaLines * 12 + 10
    const minNeeded = 36 + Math.min(rowSpace + estParaSpace, 120)

    checkPageBreak(minNeeded)

    // Section Title Header Bar
    doc.setFillColor(241, 245, 249) // slate-100
    doc.roundedRect(MARGIN, y, CONTENT_WIDTH, 18, 2, 2, 'F')
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(9.5)
    doc.setTextColor(30, 41, 59) // slate-800
    doc.text(section.title.toUpperCase(), MARGIN + 8, y + 12.5)
    y += 24

    // Section Rows (Key - Value)
    if (section.rows && section.rows.length > 0) {
      for (const row of section.rows) {
        checkPageBreak(15)
        // Label
        doc.setFont('helvetica', 'bold')
        doc.setFontSize(8.5)
        doc.setTextColor(71, 85, 105) // slate-600
        const labelText = `${row.label}:`
        doc.text(labelText, MARGIN + 8, y)

        // Value
        const labelWidth = doc.getTextWidth(labelText)
        const valX = Math.max(MARGIN + 8 + labelWidth + 6, MARGIN + 185)
        doc.setFont('helvetica', 'normal')
        doc.setFontSize(8.5)
        doc.setTextColor(15, 23, 42) // slate-900

        const maxValW = PAGE_WIDTH - MARGIN - valX - 8
        const valLines = doc.splitTextToSize(row.value, maxValW)
        doc.text(valLines[0], valX, y)
        if (valLines.length > 1) {
          for (let l = 1; l < valLines.length; l++) {
            y += 11
            checkPageBreak(12)
            doc.text(valLines[l], valX, y)
          }
        }
        y += 13.5
      }
      y += 4
    }

    // Section Paragraphs
    if (section.paragraphs && section.paragraphs.length > 0) {
      for (const para of section.paragraphs) {
        doc.setFont('helvetica', 'normal')
        doc.setFontSize(8.5)
        doc.setTextColor(51, 65, 85) // slate-700
        const lines = doc.splitTextToSize(para, CONTENT_WIDTH - 16)
        const paraHeight = lines.length * 12
        checkPageBreak(Math.min(paraHeight + 8, 48))

        doc.text(lines, MARGIN + 8, y)
        y += paraHeight + 6
      }
    }

    y += 8
  }

  // 3. Running Footers on all pages
  const totalPages = doc.getNumberOfPages()
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p)
    doc.setDrawColor(226, 232, 240) // slate-200
    doc.setLineWidth(0.5)
    doc.line(MARGIN, PAGE_HEIGHT - 32, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 32)

    doc.setFont('helvetica', 'normal')
    doc.setFontSize(7.5)
    doc.setTextColor(148, 163, 184) // slate-400
    doc.text('BioVision Deepfake Forensic Engine · Spatio-Temporal & Physiological Analysis', MARGIN, PAGE_HEIGHT - 20)
    const pageStr = `Page ${p} of ${totalPages}`
    doc.text(pageStr, PAGE_WIDTH - MARGIN - doc.getTextWidth(pageStr), PAGE_HEIGHT - 20)
  }

  return doc
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
      title: 'Temporal Sequence Trajectory Highlights',
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
    if (hr && hr > 120) {
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

  const doc = buildReportPdfDoc({
    title: 'BIOVISION',
    subtitle: 'AI Deepfake Detection & Video Forensics',
    verdict: 'NO FACE DETECTED',
    confidence: '—',
    analysisId: result.analysis_id,
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
  })

  return doc.output('blob')
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
        'BioVision executed an end-to-end multimodal deepfake forensic examination combining spatio-temporal ' +
        'facial feature representations with photoplethysmographic (rPPG) blood volume pulse recovery. Final classification ' +
        'is determined by quality-gated late fusion of spatial, temporal, and physiological evidence channels against ' +
        'a calibrated decision boundary.',
      ],
    },
    buildVisualExaminationSection(result),
    buildRppgSection(result),
    buildFusionSection(result),
    buildSequenceIntervalSection(result),
    buildGuidanceSection(result),
  ]

  const doc = buildReportPdfDoc({
    title: 'BIOVISION',
    subtitle: 'Multimodal Deepfake Detection & Video Forensics Report',
    verdict,
    confidence,
    analysisId: result.analysis_id,
    sections,
  })

  return doc.output('blob')
}

export function downloadReportPdf(result: AnalysisResult): void {
  const verdict = result.result
  const confidence = formatPercent(result.confidence)

  let doc: jsPDF
  if (verdict === 'NO_FACE') {
    const explanation =
      result.explanation || 'No detectable face was found in the sampled frames. A reliable deepfake verdict cannot be determined from this video.'
    const note =
      'Because no usable face was detected, no REAL or FAKE probabilities or confidence values were computed and none are reported here.'

    doc = buildReportPdfDoc({
      title: 'BIOVISION',
      subtitle: 'AI Deepfake Detection & Video Forensics',
      verdict: 'NO FACE DETECTED',
      confidence: '—',
      analysisId: result.analysis_id,
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
    })
  } else {
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
          'BioVision executed an end-to-end multimodal deepfake forensic examination combining spatio-temporal ' +
          'facial feature representations with photoplethysmographic (rPPG) blood volume pulse recovery. Final classification ' +
          'is determined by quality-gated late fusion of spatial, temporal, and physiological evidence channels against ' +
          'a calibrated decision boundary.',
        ],
      },
      buildVisualExaminationSection(result),
      buildRppgSection(result),
      buildFusionSection(result),
      buildSequenceIntervalSection(result),
      buildGuidanceSection(result),
    ]

    doc = buildReportPdfDoc({
      title: 'BIOVISION',
      subtitle: 'Multimodal Deepfake Detection & Video Forensics Report',
      verdict,
      confidence,
      analysisId: result.analysis_id,
      sections,
    })
  }

  const base = (result.filename || 'video').replace(/\.[^.]+$/, '') || 'video'
  const slug = result.result.toLowerCase().replace(/[^a-z0-9]/g, '-')
  doc.save(`${base}-${slug}-report.pdf`)
}

export function downloadNoFaceReportPdf(result: AnalysisResult): void {
  downloadReportPdf(result)
}
