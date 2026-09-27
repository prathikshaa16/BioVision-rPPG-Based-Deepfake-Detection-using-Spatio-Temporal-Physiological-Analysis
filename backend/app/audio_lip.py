"""Audio-Lip correspondence and phoneme-viseme synchrony analysis for BioVision.

Extracts:
1. Audio stream (16 kHz mono) via bundled imageio_ffmpeg.
2. 40-bin Mel-Frequency Cepstral Coefficients (MFCC) aligned to video frame steps.
3. MediaPipe / geometric mouth landmark contours (20 coordinates x 2 = 40 features).
4. Audio-visual temporal synchronization, cross-correlation lag (ms), and anomaly scoring.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.fftpack
import scipy.io.wavfile
import scipy.signal
import torch

from .config import BASE_DIR

MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318]
LANDMARK_MODEL_PATH = BASE_DIR / 'models' / 'face_landmarker.task'

_LANDMARKER = None


def _get_ffmpeg_exe() -> str:
    try:
        from shutil import which
        exe = which('ffmpeg')
        if exe:
            return exe
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(f"Could not locate ffmpeg executable: {exc}")


def _get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER

    if not LANDMARK_MODEL_PATH.exists():
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp.tasks.BaseOptions(model_asset_path=str(LANDMARK_MODEL_PATH))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return _LANDMARKER
    except Exception:
        return None


def extract_audio_track(video_path: str, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], bool, str]:
    """Extract audio track as 16 kHz mono float32 array.
    
    Returns (audio_array, has_audio, status).
    """
    if not os.path.exists(video_path):
        return None, False, 'FILE_NOT_FOUND'

    ffmpeg = _get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_name = tmp.name

    try:
        cmd = [
            ffmpeg, '-y', '-loglevel', 'error',
            '-i', str(video_path),
            '-vn', '-ac', '1', '-ar', str(sample_rate),
            '-f', 'wav', tmp_name
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if res.returncode != 0 or not os.path.exists(tmp_name) or os.path.getsize(tmp_name) < 44:
            return None, False, 'NO_AUDIO'

        sr, data = scipy.io.wavfile.read(tmp_name)
        if data.size == 0:
            return None, False, 'NO_AUDIO'

        # Convert to float32 in [-1.0, 1.0]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype == np.float32:
            data = np.copy(data)
        else:
            data = data.astype(np.float32)

        peak = float(np.max(np.abs(data)))
        if peak < 1e-4:
            return data, False, 'SILENT'

        return data, True, 'AVAILABLE'
    except Exception as exc:
        return None, False, f'ERROR_{type(exc).__name__}'
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def compute_mfcc(signal: np.ndarray, sr: int = 16000, n_steps: int = 32, n_mfcc: int = 40, n_mels: int = 40, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    """Compute 40-bin MFCC tensor [n_steps, 40] across uniform timestamps."""
    if signal is None or signal.size == 0 or np.max(np.abs(signal)) < 1e-6:
        return np.zeros((n_steps, n_mfcc), dtype=np.float32)

    # 1. Pre-emphasis
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # 2. Framing
    num_frames = 1 + int(np.floor((len(emphasized) - n_fft) / hop_length))
    if num_frames <= 0:
        return np.zeros((n_steps, n_mfcc), dtype=np.float32)

    indices = np.tile(np.arange(0, n_fft), (num_frames, 1)) + np.tile(np.arange(0, num_frames * hop_length, hop_length), (n_fft, 1)).T
    frames = emphasized[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(n_fft)

    # 3. FFT & Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # 4. Mel Filterbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / max(1, (bins[m] - bins[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / max(1, (bins[m + 1] - bins[m]))

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # 5. DCT-II
    mfcc_all = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # 6. Sample n_steps evenly spaced frames
    if mfcc_all.shape[0] >= n_steps:
        sample_indices = np.linspace(0, mfcc_all.shape[0] - 1, num=n_steps).round().astype(int)
        sampled_mfcc = mfcc_all[sample_indices]
    else:
        pad = n_steps - mfcc_all.shape[0]
        sampled_mfcc = np.pad(mfcc_all, ((0, pad), (0, 0)), mode='edge')

    return sampled_mfcc.astype(np.float32)


def extract_mouth_landmarks(frames: List[np.ndarray], boxes: List[Tuple[int, int, int, int]], n_steps: int = 32) -> np.ndarray:
    """Extract normalized mouth landmark coordinates [n_steps, 40] (20 points x 2)."""
    landmarker = _get_landmarker()
    observations = []

    for i in range(min(n_steps, len(frames))):
        frame = frames[i]
        box = boxes[i] if i < len(boxes) and boxes[i] is not None else None

        if frame is None or box is None:
            observations.append(np.zeros(40, dtype=np.float32))
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_w = max(float(x2 - x1), 1.0)
        face_h = max(float(y2 - y1), 1.0)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            observations.append(np.zeros(40, dtype=np.float32))
            continue

        extracted = False
        if landmarker is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if result.face_landmarks:
                    pts = result.face_landmarks[0]
                    coords = []
                    for idx in MOUTH_INDICES:
                        pt = pts[idx]
                        coords.extend([pt.x * (x2 - x1) / face_w, pt.y * (y2 - y1) / face_w])
                    observations.append(np.asarray(coords, dtype=np.float32))
                    extracted = True
            except Exception:
                pass

        if not extracted:
            # Geometric lip fallback: estimate 20 lip landmark points from lower 1/3 mouth box
            mouth_y_start = 0.65
            mouth_y_center = 0.78
            mouth_x_center = 0.50
            mouth_w = 0.40
            mouth_h = 0.18

            # Model 20 points along upper, lower, left, and right lip contours
            coords = []
            for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                px = mouth_x_center + (mouth_w / 2.0) * np.cos(angle)
                py = mouth_y_center + (mouth_h / 2.0) * np.sin(angle)
                coords.extend([float(px), float(py)])
            observations.append(np.asarray(coords, dtype=np.float32))

    while len(observations) < n_steps:
        observations.append(np.zeros(40, dtype=np.float32))

    return np.asarray(observations[:n_steps], dtype=np.float32)


def analyze_audio_lip_correspondence(
    audio: Optional[np.ndarray],
    mouth: np.ndarray,
    fps: float = 30.0,
    sample_rate: int = 16000
) -> Dict[str, Any]:
    """Forensic cross-modal phoneme-viseme synchrony and desynchronization analysis."""
    n_steps = mouth.shape[0]

    # Compute vertical lip aperture across time (distance between upper and lower lip)
    # In MOUTH_INDICES, index 14 is upper inner lip, index 17 is lower inner lip
    # Our indices are: point 5 (idx 17), point 16 (idx 14)
    # Coords are [x0, y0, x1, y1, ...]
    apertures = []
    for step in range(n_steps):
        row = mouth[step]
        # Upper lip center y: index 16 -> row[16*2 + 1]
        # Lower lip center y: index 5 -> row[5*2 + 1]
        if len(row) >= 34:
            y_upper = row[16 * 2 + 1] if len(row) > 33 else 0.70
            y_lower = row[5 * 2 + 1] if len(row) > 11 else 0.85
            aperture = max(0.0, float(abs(y_lower - y_upper)))
        else:
            aperture = 0.05
        apertures.append(aperture)

    aperture_arr = np.asarray(apertures, dtype=np.float32)
    # Normalize aperture variation
    ap_std = float(np.std(aperture_arr))
    if ap_std > 1e-4:
        norm_aperture = (aperture_arr - np.mean(aperture_arr)) / ap_std
    else:
        norm_aperture = aperture_arr - np.mean(aperture_arr)

    if audio is None or audio.size == 0 or np.max(np.abs(audio)) < 1e-4:
        return {
            'status': 'NO_AUDIO' if audio is None else 'SILENT',
            'has_audio': False,
            'speech_lip_sync_score': 0.50,
            'temporal_offset_ms': 0.0,
            'correlation': 0.0,
            'audio_lip_anomaly_score': 0.20,
            'confidence': 0.50,
            'is_synchronized': True,
            'is_desynchronized': False,
            'explanation': 'Video audio track is absent or silent; audio-lip phoneme-viseme analysis skipped without penalty.',
            'aperture_waveform': [round(float(v), 4) for v in aperture_arr],
            'audio_energy_waveform': [0.0] * n_steps,
            'feature_dim': 128,
        }

    # Audio RMS energy sampled at video frame steps
    audio_len = len(audio)
    step_samples = max(1, audio_len // n_steps)
    energies = []
    for s in range(n_steps):
        start = s * step_samples
        end = min(audio_len, (s + 1) * step_samples)
        segment = audio[start:end]
        if segment.size > 0:
            rms = float(np.sqrt(np.mean(segment ** 2)))
        else:
            rms = 0.0
        energies.append(rms)

    energy_arr = np.asarray(energies, dtype=np.float32)
    en_std = float(np.std(energy_arr))
    if en_std > 1e-4:
        norm_energy = (energy_arr - np.mean(energy_arr)) / en_std
    else:
        norm_energy = energy_arr - np.mean(energy_arr)

    # 1. Pearson cross-correlation at lag 0
    if ap_std > 1e-4 and en_std > 1e-4:
        r_corr = float(np.clip(np.corrcoef(norm_aperture, norm_energy)[0, 1], -1.0, 1.0))
        if np.isnan(r_corr):
            r_corr = 0.0
    else:
        r_corr = 0.0

    # 2. Optimal cross-correlation lag / offset search (±15 frames = ±500 ms at 30 fps)
    max_lag_frames = min(15, n_steps // 2)
    best_corr = r_corr
    best_lag = 0

    if ap_std > 1e-4 and en_std > 1e-4:
        for lag in range(-max_lag_frames, max_lag_frames + 1):
            if lag < 0:
                c = np.corrcoef(norm_aperture[:lag], norm_energy[-lag:])[0, 1]
            elif lag > 0:
                c = np.corrcoef(norm_aperture[lag:], norm_energy[:-lag])[0, 1]
            else:
                c = r_corr
            if not np.isnan(c) and c > best_corr:
                best_corr = float(c)
                best_lag = lag

    # Lag in milliseconds
    frame_ms = 1000.0 / max(1.0, fps)
    offset_ms = round(float(best_lag * frame_ms), 1)

    # 3. Synchrony and Anomaly Scoring
    # Genuine speech: r_corr is positive (0.25 - 0.70), offset within ±60 ms
    # Dubbed / Desynchronized deepfake: offset >= 100 ms or r_corr <= 0.08
    is_synchronized = (r_corr >= 0.22) and (abs(offset_ms) <= 80.0)
    is_desynchronized = (abs(offset_ms) >= 120.0 and best_corr > 0.30) or (r_corr < 0.05 and en_std > 0.01 and ap_std > 0.01)

    if is_synchronized:
        sync_score = min(1.0, 0.50 + 0.50 * r_corr)
        anomaly_score = max(0.02, min(0.18, 0.25 * (1.0 - r_corr)))
        explanation = (
            f"Audio-lip phoneme-viseme correspondence is tightly synchronized (Pearson r={round(r_corr, 3)}, "
            f"offset {offset_ms} ms). Speech acoustic envelopes match mouth aperture trajectory, confirming genuine audiovisual recording."
        )
    elif is_desynchronized:
        sync_score = max(0.05, min(0.35, 0.50 - 0.50 * abs(r_corr)))
        anomaly_score = min(0.95, 0.65 + 0.30 * min(1.0, abs(offset_ms) / 500.0))
        explanation = (
            f"Audiovisual desynchronization detected: {abs(offset_ms)} ms temporal offset between acoustic energy "
            f"and lip articulation (synchrony r={round(r_corr, 3)}). Characteristic of artificial lip sync or audio-dubbed deepfake."
        )
    else:
        sync_score = 0.50 + 0.20 * r_corr
        anomaly_score = 0.35 + 0.20 * (1.0 - max(0.0, r_corr))
        explanation = (
            f"Audio-lip alignment is moderate (correlation r={round(r_corr, 3)}, temporal drift {offset_ms} ms). "
            f"Subtle speech or ambient noise observed."
        )

    return {
        'status': 'AVAILABLE',
        'has_audio': True,
        'speech_lip_sync_score': round(float(sync_score), 4),
        'temporal_offset_ms': offset_ms,
        'correlation': round(float(r_corr), 4),
        'audio_lip_anomaly_score': round(float(anomaly_score), 4),
        'confidence': round(float(abs(sync_score - 0.50) * 2.0), 4),
        'is_synchronized': is_synchronized,
        'is_desynchronized': is_desynchronized,
        'explanation': explanation,
        'aperture_waveform': [round(float(v), 4) for v in aperture_arr],
        'audio_energy_waveform': [round(float(v), 4) for v in energy_arr],
        'feature_dim': 128,
    }
