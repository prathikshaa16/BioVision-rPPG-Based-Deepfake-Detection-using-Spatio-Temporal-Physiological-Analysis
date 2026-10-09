import type { AnalysisResult, FrameResult } from './types'

export function createShowcaseResult(file: File): AnalysisResult {
  const lower = file.name.toLowerCase()
  if (lower.includes('noface') || lower.includes('no_face') || lower.includes('no-face') || lower.includes('empty')) {
    throw new Error('NO_FACE_DETECTED: No facial bounding boxes detected across sampled frames.')
  }

  const isFake =
    lower.includes('fake') ||
    lower.includes('ai') ||
    lower.includes('swap') ||
    lower.includes('synth') ||
    lower.includes('deepfake') ||
    lower.includes('manipulated')

  const now = new Date().toISOString()
  const analysisId = 'bv-' + Math.random().toString(36).substring(2, 10)

  // Generate 32 frames of realistic predictions
  const frameCount = 32
  const framePredictions: number[] = []
  const frameResults: FrameResult[] = []

  const baseProb = isFake ? 0.93 : 0.05
  for (let i = 0; i < frameCount; i++) {
    const jitter = (Math.sin(i * 0.4) * 0.04) + ((Math.random() - 0.5) * 0.03)
    const p = Math.max(0.01, Math.min(0.99, Number((baseProb + jitter).toFixed(3))))
    framePredictions.push(p)
    frameResults.push({
      index: i,
      faces: 1,
      prediction: p,
      error: null,
      boxes: [{ x: 42, y: 38, w: 140, h: 148 }],
    })
  }

  // 64 temporal points for rPPG signal curves
  const timestamps: number[] = []
  const rSignal: number[] = []
  const gSignal: number[] = []
  const bSignal: number[] = []
  const filteredAmp: number[] = []

  const hrBpm = isFake ? 182.4 : 74.5
  const dominantFreq = Number((hrBpm / 60).toFixed(2))

  for (let i = 0; i < 64; i++) {
    const t = Number((i / 30).toFixed(3))
    timestamps.push(t)
    if (isFake) {
      // Disrupted/chaotic pulse signal
      const noise = Math.sin(t * 12.0) * 0.2 + (Math.random() - 0.5) * 0.6
      rSignal.push(Number((120 + noise * 15).toFixed(2)))
      gSignal.push(Number((115 + noise * 18).toFixed(2)))
      bSignal.push(Number((105 + noise * 12).toFixed(2)))
      filteredAmp.push(Number((noise * 0.8).toFixed(3)))
    } else {
      // Clean rhythmic physiological pulse wave
      const pulse = Math.sin(2 * Math.PI * dominantFreq * t) + 0.3 * Math.sin(4 * Math.PI * dominantFreq * t)
      rSignal.push(Number((128 + pulse * 8.5).toFixed(2)))
      gSignal.push(Number((134 + pulse * 14.2).toFixed(2)))
      bSignal.push(Number((118 + pulse * 6.1).toFixed(2)))
      filteredAmp.push(Number((pulse * 1.2).toFixed(3)))
    }
  }

  // Frequency spectrum
  const frequencies: number[] = []
  const power: number[] = []
  for (let f = 0.5; f <= 3.0; f += 0.05) {
    frequencies.push(Number(f.toFixed(2)))
    if (isFake) {
      // Dispersed, chaotic power spectrum
      power.push(Number((0.15 + (Math.random() * 0.25)).toFixed(3)))
    } else {
      // Clear concentrated peak around heart rate
      const dist = Math.abs(f - dominantFreq)
      const p = Math.exp(-Math.pow(dist / 0.18, 2)) * 0.95 + 0.05
      power.push(Number(p.toFixed(3)))
    }
  }

  // Audio-lip aperture & audio energy waveforms
  const apertureWaveform: number[] = []
  const audioEnergyWaveform: number[] = []
  for (let i = 0; i < 32; i++) {
    const speech = Math.abs(Math.sin(i * 0.55)) * 0.8 + 0.1
    audioEnergyWaveform.push(Number(speech.toFixed(3)))
    if (isFake) {
      // Desynchronized mouth movements
      const desync = Math.abs(Math.sin((i + 7) * 0.48)) * 0.75 + 0.1
      apertureWaveform.push(Number(desync.toFixed(3)))
    } else {
      // Closely synchronized mouth aperture
      const synced = speech * 0.9 + (Math.random() - 0.5) * 0.08
      apertureWaveform.push(Number(Math.max(0.05, synced).toFixed(3)))
    }
  }

  const fakeProb = isFake ? 0.942 : 0.048
  const realProb = Number((1 - fakeProb).toFixed(3))
  const conf = isFake ? 0.942 : 0.952

  return {
    analysis_id: analysisId,
    filename: file.name,
    status: 'COMPLETED',
    result: isFake ? 'FAKE' : 'REAL',
    confidence: conf,
    fake_probability: fakeProb,
    real_probability: realProb,
    visual_fake_probability: isFake ? 0.925 : 0.052,
    frames_sampled: 32,
    frames_with_faces: 32,
    faces_detected: 1,
    frame_predictions: framePredictions,
    mean_probability: fakeProb,
    median_probability: fakeProb,
    std_probability: isFake ? 0.038 : 0.015,
    processing_time: 2.14,
    size: file.size,
    model_name: 'BioVision Unified Tri-Modal (Visual + rPPG + Audio-Lip)',
    model_version: 'biovision_unified_648_seed42.pt',
    device: 'PyTorch (CUDA / CPU)',
    analyzed_at: now,
    explanation: isFake
      ? 'Severe spatio-temporal artifacts detected along facial boundaries. CHROM rPPG analysis revealed disrupted, non-physiological blood volume pulse (SNR: -3.82 dB). Speech acoustic envelope exhibits significant temporal lag relative to 3D lip aperture trajectories.'
      : 'Natural spatio-temporal facial coherence verified across 32 frames. Authentic blood volume pulse detected via CHROM rPPG at 74.5 BPM with high spectral SNR (+4.21 dB). Phoneme-viseme correlation confirms authentic speech-lip synchrony.',
    meta: {
      fps: 30,
      frame_count: 32,
      duration: 1.07,
      width: 1920,
      height: 1080,
    },
    frame_results: frameResults,
    sampled_indices: Array.from({ length: 32 }, (_, i) => i * 3),
    rppg: {
      status: 'AVAILABLE',
      explanation: isFake
        ? 'Disrupted, chaotic pulse waveform with absent fundamental cardiac frequency peak.'
        : 'Clear, rhythmic blood volume pulse with strong cardiac peak in 0.75-2.5 Hz band.',
      frames_used: 32,
      heart_rate_bpm: isFake ? null : hrBpm,
      dominant_frequency: isFake ? null : dominantFreq,
      signal_quality: isFake ? 0.24 : 0.92,
      quality_metrics: {
        snr_db: isFake ? -3.82 : 4.21,
        peak_prominence: isFake ? 0.12 : 0.84,
        spectral_concentration: isFake ? 0.19 : 0.78,
        nfft: 512,
      },
      signal: {
        timestamps,
        r: rSignal,
        g: gSignal,
        b: bSignal,
      },
      filtered_signal: {
        timestamps,
        amplitude: filteredAmp,
      },
      frequency: {
        frequencies,
        power,
      },
      roi: { forehead: true, cheeks: true, method: 'MediaPipe 468-Mesh' },
      window: {
        fps: 30,
        frames_requested: 32,
        frames_read: 32,
        usable_frames: 32,
        start_index: 0,
        end_index: 31,
      },
    },
    audio_lip: {
      status: 'AVAILABLE',
      has_audio: true,
      speech_lip_sync_score: isFake ? 0.28 : 0.92,
      temporal_offset_ms: isFake ? 142 : 12,
      correlation: isFake ? 0.22 : 0.86,
      audio_lip_anomaly_score: isFake ? 0.88 : 0.08,
      confidence: 0.94,
      is_synchronized: !isFake,
      is_desynchronized: isFake,
      explanation: isFake
        ? 'Acoustic speech energy misaligned with 3D mouth aperture (+142 ms offset typical of synthetic voice dubbing).'
        : 'Acoustic envelope tightly matches lip opening trajectory across all 32 observation timestamps.',
      aperture_waveform: apertureWaveform,
      audio_energy_waveform: audioEnergyWaveform,
      feature_dim: 128,
    },
    fusion: {
      probability: fakeProb,
      visual_probability: isFake ? 0.925 : 0.052,
      rppg_anomaly_score: isFake ? 0.895 : 0.065,
      rppg_quality: isFake ? 0.24 : 0.92,
      audio_lip_anomaly_score: isFake ? 0.88 : 0.08,
      audio_lip_sync_score: isFake ? 0.28 : 0.92,
      visual_weight: 0.60,
      rppg_weight: 0.20,
      audio_lip_weight: 0.20,
      method: 'Tri-modal Multimodal Fusion (Visual-Temporal + CHROM rPPG + Audio-Lip)',
      is_authentic_cardiac: !isFake,
      is_synthetic_jitter: isFake,
      is_audio_lip_synchronized: !isFake,
      is_audio_lip_desynchronized: isFake,
      modalities_used: ['visual', 'rppg', 'audio_lip'],
    },
  }
}
