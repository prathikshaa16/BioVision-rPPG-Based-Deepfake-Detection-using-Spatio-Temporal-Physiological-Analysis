import cv2
import os
from typing import List, Tuple, Dict, Any
from .config import DEFAULT_SAMPLE_FRAMES


def validate_video(path: str) -> Dict[str, Any]:
    """Validate video can be opened by OpenCV and return metadata.

    Returns metadata dict: fps, frame_count, duration_sec, width, height
    Raises ValueError on invalid/corrupt video.
    """
    if not os.path.exists(path):
        raise ValueError('File does not exist')

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('Cannot open video file')

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    if frame_count == 0:
        raise ValueError('Video contains no frames')

    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height,
    }


def sample_frame_indices(frame_count: int, n_samples: int = DEFAULT_SAMPLE_FRAMES) -> List[int]:
    """Return a list of frame indices uniformly sampled from the video.
    If frame_count < n_samples, return all indices.
    """
    if frame_count <= 0:
        return []
    if frame_count <= n_samples:
        return list(range(frame_count))

    # uniform sampling over the frame range [0, frame_count-1]
    import numpy as np

    indices = np.linspace(0, frame_count - 1, num=n_samples, dtype=int)
    return indices.tolist()


def read_frames_by_indices(path: str, indices: List[int]) -> List[Any]:
    """Read specific frames (BGR numpy arrays) from video without loading entire video."""
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('Cannot open video file for reading frames')

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frames.append(None)
        else:
            frames.append(frame)

    cap.release()
    return frames
