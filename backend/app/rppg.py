"""Real remote-photoplethysmography (rPPG) analysis for BioVision.

Implements the CHROM method (de Haan & Jeanne, 2013) using the physiological
regions of the detected face (forehead + cheeks), with:

  - per-frame mean-RGB extraction from MTCNN face boxes
  - CHROM signal projection
  - polynomial detrending
  - 0.8-3.0 Hz bandpass filtering (Butterworth biquads, pure numpy)
  - FFT-based dominant-frequency / heart-rate estimation
  - signal-quality indicators

Nothing here is fabricated. If the video cannot support a physiological
signal the module returns status 'UNAVAILABLE' with an honest explanation.
The rPPG output is returned as physiological forensic evidence and is later
quality-gated by the late-fusion layer before it can affect the verdict.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .config import (
    RPPG_WINDOW_SECONDS,
    RPPG_MAX_FRAMES,
    RPPG_MIN_FRAMES,
    RPPG_LOW_HZ,
    RPPG_HIGH_HZ,
)

# --- Signal filtering (pure numpy) -----------------------------------------


def _butterworth_biquad(kind: str, fc_hz: float, fs: float):
    """RBJ cookbook 2nd-order Butterworth low/high-pass coefficients.

    Returns normalized (b0, b1, b2, a1, a2) so a0 == 1.
    """
    w0 = 2.0 * np.pi * fc_hz / fs
    alpha = np.sin(w0) / np.sqrt(2.0)
    cos_w0 = np.cos(w0)
    if kind == 'lowpass':
        b = np.array([(1.0 - cos_w0) / 2.0, 1.0 - cos_w0, (1.0 - cos_w0) / 2.0])
    else:  # highpass
        b = np.array([(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    a0 = a[0]
    return (b[0] / a0, b[1] / a0, b[2] / a0, a[1] / a0, a[2] / a0)


def _biquad_filter(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    """Direct Form II transposed 2nd-order IIR filtering."""
    out = np.zeros(len(x), dtype=np.float64)
    z0 = 0.0
    z1 = 0.0
    for i in range(len(x)):
        v = x[i]
        y = b0 * v + z0
        out[i] = y
        z0 = b1 * v - a1 * y + z1
        z1 = b2 * v - a2 * y
    return out


def bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = RPPG_LOW_HZ, high_hz: float = RPPG_HIGH_HZ) -> np.ndarray:
    """Apply a 4th-order (2x2nd order) Butterworth bandpass [low_hz, high_hz]."""
    if fs <= 2.0 * high_hz:
        raise ValueError(f"Sampling rate {fs:.2f} Hz too low to bandpass to {high_hz} Hz")
    x = np.asarray(signal, dtype=np.float64)
    b0, b1, b2, a1, a2 = _butterworth_biquad('highpass', low_hz, fs)
    x = _biquad_filter(x, b0, b1, b2, a1, a2)
    b0, b1, b2, a1, a2 = _butterworth_biquad('lowpass', high_hz, fs)
    x = _biquad_filter(x, b0, b1, b2, a1, a2)
    return x


def detrend_signal(signal: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove a polynomial trend (order 2 by default) from the signal."""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < order + 2:
        return x - np.mean(x)
    t = np.arange(n)
    coeffs = np.polyfit(t, x, order)
    return x - np.polyval(coeffs, t)


# --- CHROM ------------------------------------------------------------------


def chrom_pulse(rgb_signals: np.ndarray) -> np.ndarray:
    """Compute the CHROM pulse signal.

    rgb_signals: array of shape (3, N) with rows [R, G, B] mean values.
    """
    r = np.asarray(rgb_signals[0], dtype=np.float64)
    g = np.asarray(rgb_signals[1], dtype=np.float64)
    b = np.asarray(rgb_signals[2], dtype=np.float64)

    if len(r) < 3 or np.std(r) == 0 or np.std(g) == 0 or np.std(b) == 0:
        raise ValueError('insufficient temporal variance in RGB signal')

    # Chromaticity normalization
    rn = r / np.mean(r) - 1.0
    gn = g / np.mean(g) - 1.0
    bn = b / np.mean(b) - 1.0

    # CHROM projection
    x = 3.0 * rn - 2.0 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_y < 1e-12:
        raise ValueError('degenerate CHROM projection (no Y variance)')

    alpha = std_x / std_y
    pulse = x - alpha * y
    return pulse


# --- Heart-rate / quality estimation ---------------------------------------


def estimate_heart_rate(signal: np.ndarray, fs: float, low_hz: float = RPPG_LOW_HZ, high_hz: float = RPPG_HIGH_HZ):
    """Estimate dominant frequency and heart rate from a bandpassed signal.

    Returns (heart_rate_bpm, dominant_frequency_hz, metrics) or
    (None, None, metrics) when the spectrum is degenerate.
    """
    metrics = {'snr_db': None, 'peak_prominence': None, 'spectral_concentration': None, 'nfft': 0}

    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 4:
        return None, None, metrics

    x = x - np.mean(x)
    nfft = 1
    while nfft < n * 8:
        nfft *= 2

    spectrum = np.abs(np.fft.rfft(x, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    metrics['nfft'] = int(nfft)

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    band_freqs = freqs[mask]
    band_power = spectrum[mask]
    if band_power.size == 0 or np.sum(band_power) <= 0:
        return None, None, metrics

    peak_idx = int(np.argmax(band_power))
    dominant_hz = float(band_freqs[peak_idx])
    peak_power = float(band_power[peak_idx])
    mean_band_power = float(np.mean(band_power))
    total_band_power = float(np.sum(band_power))

    snr_db = 10.0 * np.log10(peak_power / mean_band_power) if mean_band_power > 0 else None

    # Spectral concentration around the peak (+/- 0.25 Hz)
    half = 0.25
    window = (band_freqs >= dominant_hz - half) & (band_freqs <= dominant_hz + half)
    concentration = float(np.sum(band_power[window]) / total_band_power) if total_band_power > 0 else 0.0

    peak_prominence = peak_power / mean_band_power if mean_band_power > 0 else None

    metrics['snr_db'] = round(snr_db, 3) if snr_db is not None else None
    metrics['peak_prominence'] = round(peak_prominence, 3) if peak_prominence is not None else None
    metrics['spectral_concentration'] = round(concentration, 4)

    heart_rate_bpm = dominant_hz * 60.0
    return round(heart_rate_bpm, 2), round(dominant_hz, 4), metrics


def signal_quality_score(metrics: Dict[str, Any], usable_frames: int, min_frames: int) -> Optional[float]:
    """Combine quality indicators into a single 0..1 score.

    Returns None when no heart rate estimate exists (cannot score it).
    """
    if metrics.get('snr_db') is None or metrics.get('spectral_concentration') is None:
        return None
    snr = metrics['snr_db']
    concentration = metrics['spectral_concentration']

    # SNR -> 0..1 (6 dB = weak, 12 dB = moderate, 18+ dB = strong)
    snr_score = max(0.0, min(1.0, (snr - 6.0) / 12.0))
    # Spectral concentration -> 0..1 (target ~0.5+)
    concentration_score = max(0.0, min(1.0, concentration / 0.7))
    # Frame sufficiency
    frame_score = max(0.0, min(1.0, usable_frames / (min_frames * 2.0)))

    score = 0.45 * snr_score + 0.4 * concentration_score + 0.15 * frame_score
    return round(max(0.0, min(1.0, score)), 3)


# --- ROI / region extraction ------------------------------------------------


def extract_roi_mean_rgb(bgr_frame: np.ndarray, box) -> Optional[np.ndarray]:
    """Extract mean RGB from forehead + cheeks of a face box in a BGR frame.

    Returns array [R, G, B] or None if the box is too small / invalid.
    Region bounds are fractions of the detected face box (not the 224 crop),
    consistent with the BioVision methodology (forehead + cheek regions).
    """
    h, w = bgr_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 20 or bh < 20:
        return None

    def region_mean(fx0: float, fy0: float, fx1: float, fy1: float) -> Optional[np.ndarray]:
        rx0 = int(x1 + fx0 * bw)
        ry0 = int(y1 + fy0 * bh)
        rx1 = int(x1 + fx1 * bw)
        ry1 = int(y1 + fy1 * bh)
        if rx1 <= rx0 or ry1 <= ry0:
            return None
        patch = bgr_frame[ry0:ry1, rx0:rx1]
        # OpenCV returns BGR; return in RGB order.
        b, g, r = patch.astype(np.float64).mean(axis=(0, 1))
        return np.array([r, g, b], dtype=np.float64)

    forehead = region_mean(0.20, 0.08, 0.80, 0.30)
    cheek_left = region_mean(0.05, 0.38, 0.40, 0.68)
    cheek_right = region_mean(0.60, 0.38, 0.95, 0.68)

    valid = [reg for reg in (forehead, cheek_left, cheek_right) if reg is not None]
    if not valid:
        return None
    return np.mean(valid, axis=0)


# --- Window / orchestration -------------------------------------------------


def _longest_valid_run(valid: List[bool]) -> Optional[Dict[str, int]]:
    """Return {start, end, length} of the longest contiguous True run (or None)."""
    best = None
    cur_start = None
    length = 0
    n = len(valid)
    for i, ok in enumerate(valid):
        if ok:
            if cur_start is None:
                cur_start = i
            length += 1
        else:
            if cur_start is not None and (best is None or length > best['length']):
                best = {'start': cur_start, 'end': i - 1, 'length': length}
            cur_start = None
            length = 0
    if cur_start is not None and (best is None or length > best['length']):
        best = {'start': cur_start, 'end': n - 1, 'length': length}
    return best


def run_rppg_analysis(video_path: str, meta: Dict[str, Any], device: str = 'cpu',
                      window_seconds: float = RPPG_WINDOW_SECONDS,
                      max_frames: int = RPPG_MAX_FRAMES,
                      min_frames: int = RPPG_MIN_FRAMES) -> Dict[str, Any]:
    """Run the full rPPG pipeline on a contiguous window of the video.

    The signal is extracted from the actual face boxes (MTCNN) of the actual
    frames.  Returns an honest result dict; status is 'AVAILABLE' only when a
    physiological signal could genuinely be recovered.
    """
    from .face_processor import FaceProcessor
    from .video_processor import read_frames_by_indices

    fps = float(meta.get('fps') or 0.0)
    frame_count = int(meta.get('frame_count') or 0)

    def unavailable(reason: str) -> Dict[str, Any]:
        return {
            'status': 'UNAVAILABLE',
            'explanation': reason,
            'frames_used': 0,
            'heart_rate_bpm': None,
            'dominant_frequency': None,
            'signal_quality': None,
            'quality_metrics': None,
            'signal': None,
            'filtered_signal': None,
            'frequency': None,
            'roi': None,
            'window': {'fps': round(fps, 3), 'frames_requested': 0, 'frames_read': 0, 'usable_frames': 0},
        }

    if fps <= 0:
        return unavailable('The video has no reliable frame rate, so a temporal physiological signal cannot be sampled.')
    if frame_count < min_frames:
        return unavailable(
            f'The video is too short ({frame_count} frames) for a physiological signal; at least {min_frames} frames are required.'
        )

    window_frames = int(fps * window_seconds)
    window_frames = max(min_frames, min(window_frames, max_frames))
    window_frames = min(window_frames, frame_count)

    start_index = max(0, (frame_count - window_frames) // 2)
    indices = list(range(start_index, start_index + window_frames))
    frames = read_frames_by_indices(video_path, indices)

    fp = FaceProcessor(device=device)
    rgb_values: List[Optional[np.ndarray]] = []
    valid: List[bool] = []
    frames_read = 0

    for frame in frames:
        if frame is None:
            rgb_values.append(None)
            valid.append(False)
            continue
        frames_read += 1
        boxes, _ = fp.detect_faces(frame)
        if not boxes:
            rgb_values.append(None)
            valid.append(False)
            continue
        largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        rgb = extract_roi_mean_rgb(frame, largest)
        if rgb is None:
            rgb_values.append(None)
            valid.append(False)
            continue
        rgb_values.append(rgb)
        valid.append(True)

    run = _longest_valid_run(valid)
    usable_frames = sum(1 for ok in valid if ok)

    if run is None or run['length'] < min_frames:
        return unavailable(
            f'Only {usable_frames} usable face frame(s) were recovered in the analysis window '
            f'({len(indices)} frames attempted, {frames_read} read). At least {min_frames} contiguous '
            f'frames with a detected face are required for a physiological signal.'
        )

    rgb_matrix = np.stack([rgb_values[i] for i in range(run['start'], run['end'] + 1)])
    run_frames = run['length']

    try:
        pulse = chrom_pulse(rgb_matrix.T)
    except ValueError as exc:
        return unavailable(f'Physiological signal could not be recovered: {exc}.')

    pulse_detrended = detrend_signal(pulse)
    pulse_filtered = bandpass_filter(pulse_detrended, fs=fps)

    heart_rate_bpm, dominant_hz, metrics = estimate_heart_rate(pulse_filtered, fs=fps)

    if heart_rate_bpm is None or dominant_hz is None:
        return unavailable('The frequency spectrum was degenerate; a reliable pulse frequency could not be estimated.')

    quality = signal_quality_score(metrics, run_frames, min_frames)

    # Downsample displayed signals to keep the payload sane (max ~360 points).
    def downsample(arr: np.ndarray, limit: int = 360) -> np.ndarray:
        n = len(arr)
        if n <= limit:
            return arr
        idx = np.linspace(0, n - 1, num=limit, dtype=int)
        return arr[idx]

    run_start_sec = run['start'] / fps
    t_full = run_start_sec + np.arange(run_frames) / fps
    t_down = downsample(t_full)

    def pack(x: np.ndarray):
        x = downsample(x)
        return [round(float(v), 6) for v in x]

    # Spectrum limited to the analysis band with margin.
    nfft = metrics['nfft']
    spec = np.abs(np.fft.rfft(pulse_filtered - np.mean(pulse_filtered), n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fps)
    band_mask = (freqs >= RPPG_LOW_HZ - 0.1) & (freqs <= RPPG_HIGH_HZ + 0.1)
    spec_freqs = freqs[band_mask]
    spec_power = spec[band_mask]

    return {
        'status': 'AVAILABLE',
        'explanation': (
            'CHROM rPPG signal extracted from forehead + cheek regions of the detected face over '
            f'a {window_frames}-frame analysis window; dominant frequency {dominant_hz} Hz '
            f'({heart_rate_bpm} BPM). The waveform and heart rate remain diagnostics; the '
            'quality score is passed to the configured late-fusion layer as physiological evidence.'
        ),
        'frames_used': run_frames,
        'window': {
            'fps': round(fps, 3),
            'frames_requested': len(indices),
            'frames_read': frames_read,
            'usable_frames': usable_frames,
            'start_index': run['start'] + start_index,
            'end_index': run['end'] + start_index,
        },
        'roi': {
            'forehead': True,
            'cheeks': True,
            'method': 'MTCNN face box relative regions (forehead + left/right cheek)',
        },
        'signal': {
            'timestamps': [round(float(v), 3) for v in t_down],
            'r': pack(rgb_matrix[:, 0]),
            'g': pack(rgb_matrix[:, 1]),
            'b': pack(rgb_matrix[:, 2]),
        },
        'filtered_signal': {
            'timestamps': [round(float(v), 3) for v in t_down],
            'amplitude': pack(pulse_filtered),
        },
        'frequency': {
            'frequencies': [round(float(v), 4) for v in spec_freqs],
            'power': [round(float(v), 6) for v in spec_power],
        },
        'heart_rate_bpm': heart_rate_bpm,
        'dominant_frequency': dominant_hz,
        'signal_quality': quality,
        'quality_metrics': metrics,
    }
