import os
import tempfile

import cv2
import torch

from ..inference import load_model
from ..multimodal_model import build_cached_biovision_visual_rppg_model
from ..video_processor import validate_video, sample_frame_indices, read_frames_by_indices


def make_test_video(path: str, n_frames: int = 30, width: int = 160, height: int = 120, fps: int = 15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    for i in range(n_frames):
        frame = (255 * (i % 2) * 1).astype('uint8') if False else None
        # create a simple gradient frame
        import numpy as np
        frame = np.zeros((height, width, 3), dtype='uint8')
        frame[:, :, 0] = (i * 5) % 255
        frame[:, :, 1] = (i * 3) % 255
        frame[:, :, 2] = (i * 7) % 255
        writer.write(frame)
    writer.release()


def test_video_sampling_and_reading(tmp_path):
    p = tmp_path / "test_video.mp4"
    make_test_video(str(p))

    meta = validate_video(str(p))
    assert meta['frame_count'] > 0
    assert meta['fps'] > 0 or meta['duration'] > 0

    indices = sample_frame_indices(meta['frame_count'], n_samples=5)
    assert len(indices) <= 5

    frames = read_frames_by_indices(str(p), indices)
    assert len(frames) == len(indices)
    # at least one frame should be non-None
    assert any([f is not None for f in frames])


def test_cached_model_checkpoint_loads(tmp_path):
    checkpoint_path = tmp_path / 'cached_biovision.pt'
    model = build_cached_biovision_visual_rppg_model()
    torch.save({'model_state_dict': model.state_dict(), 'epoch': 1, 'best_val_auc': 0.5}, checkpoint_path)

    loaded_model, info = load_model(device='cpu', model_type='cached', checkpoint_path=str(checkpoint_path))

    assert info['model_type'] == 'cached'
    assert type(loaded_model) is type(model)
    with torch.inference_mode():
        output = loaded_model(torch.randn(1, 32, 1792), torch.randn(1, 240))
    assert output.shape == (1,)
