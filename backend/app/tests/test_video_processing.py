import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.app.inference import load_model
from backend.app.multimodal_model import build_cached_biovision_visual_rppg_model
from backend.app.video_processor import validate_video, sample_frame_indices, read_frames_by_indices


def make_test_video(path: str, n_frames: int = 30, width: int = 160, height: int = 120, fps: int = 15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype='uint8')
        frame[:, :, 0] = (i * 5) % 255
        frame[:, :, 1] = (i * 3) % 255
        frame[:, :, 2] = (i * 7) % 255
        writer.write(frame)
    writer.release()


class TestVideoProcessing(unittest.TestCase):
    def test_video_sampling_and_reading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "test_video.mp4"
            make_test_video(str(p))

            meta = validate_video(str(p))
            self.assertGreater(meta['frame_count'], 0)
            self.assertTrue(meta['fps'] > 0 or meta['duration'] > 0)

            indices = sample_frame_indices(meta['frame_count'], n_samples=5)
            self.assertLessEqual(len(indices), 5)

            frames = read_frames_by_indices(str(p), indices)
            self.assertEqual(len(frames), len(indices))
            self.assertTrue(any(f is not None for f in frames))

    def test_cached_model_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'cached_biovision.pt'
            model = build_cached_biovision_visual_rppg_model()
            torch.save({'model_state_dict': model.state_dict(), 'epoch': 1, 'best_val_auc': 0.5}, checkpoint_path)

            loaded_model, info = load_model(device='cpu', model_type='cached', checkpoint_path=str(checkpoint_path))

            self.assertEqual(info['model_type'], 'cached')
            self.assertIsInstance(loaded_model, type(model))
            with torch.inference_mode():
                output = loaded_model(torch.randn(1, 32, 1792), torch.randn(1, 240))
            self.assertEqual(output.shape, (1,))

    def test_cardiac_spectral_deployed_model(self):
        """Verify production biovision_best.pt loads and executes forward pass."""
        prod_path = 'backend/models/biovision_best.pt'
        if not os.path.exists(prod_path):
            return
        loaded_model, info = load_model(device='cpu', model_type='cached', checkpoint_path=prod_path)
        self.assertEqual(info['model_type'], 'cached')
        self.assertIn(type(loaded_model).__name__, ('BioVisionCardiacSpectral', 'BioVisionMultiHarmonic'))
        self.assertGreater(float(info.get('optimal_threshold', 0)), 0.50)
        self.assertGreaterEqual(float(info.get('best_bal_acc', 0)), 0.75)
        auc_val = float(info.get('auc', info.get('best_val_auc', 0)))
        self.assertGreaterEqual(auc_val, 0.85)

        with torch.inference_mode():
            v = torch.randn(2, 32, 1792)
            r = torch.randn(2, 240)
            out = loaded_model(v, r)
            self.assertEqual(out.shape, (2,))
            probs = torch.sigmoid(out)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))


if __name__ == '__main__':
    unittest.main()

