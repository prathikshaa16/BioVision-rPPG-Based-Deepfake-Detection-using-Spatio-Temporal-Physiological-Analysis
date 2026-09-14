"""Comprehensive test suite for the BioVision Deepfake Detection System.

Covers:
1. Multi-Harmonic Architecture & Physio-Spectral Feature Extraction
2. 3-Seed Soft-Voting Ensemble & Probability Aggregation
3. Production Checkpoint Loading & Metadata Validation
4. Video Processing, Face Detection Fallback, & CHROM rPPG Extraction
5. FastAPI Endpoints (/health, /model/info, /dashboard/stats, /upload, /analyses, /evaluation/metrics)
"""

import os
import json
import tempfile
import unittest

import cv2
import numpy as np
import torch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.inference import load_model, analyze_video
from backend.app.config import resolve_model_paths
from backend.app.multimodal_model import (
    BioVisionMultiHarmonic,
    BioVisionEnsemble,
    build_biovision_model_from_checkpoint,
    build_cached_biovision_visual_rppg_model
)
from backend.app.video_processor import (
    validate_video,
    sample_frame_indices,
    read_frames_by_indices,
)
from backend.app.rppg import chrom_pulse


class TestBioVisionArchitecture(unittest.TestCase):
    """Unit tests for BioVisionMultiHarmonic & BioVisionEnsemble models."""

    def test_multi_harmonic_forward_pass(self):
        model = BioVisionMultiHarmonic()
        vis = torch.randn(2, 32, 1792)
        rppg = torch.randn(2, 240)
        out = model(vis, rppg)
        self.assertEqual(out.shape, (2,), "Multi-Harmonic model must output shape (B,)")
        probs = torch.sigmoid(out)
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in [0, 1]")

    def test_multi_harmonic_single_item_forward(self):
        model = BioVisionMultiHarmonic()
        model.eval()
        vis = torch.randn(32, 1792)
        rppg = torch.randn(240)
        out = model(vis, rppg)
        self.assertEqual(out.shape, (1,), "Single item input must output shape (1,)")

    def test_extract_physio_features_dimensions(self):
        model = BioVisionMultiHarmonic()
        rppg = torch.randn(4, 240)
        feats = model.extract_physio_features(rppg)
        self.assertEqual(feats.shape, (4, 32), "Physiological feature vector must have exactly 32 dimensions")

    def test_ensemble_forward_pass(self):
        m1 = BioVisionMultiHarmonic()
        m2 = BioVisionMultiHarmonic()
        m3 = BioVisionMultiHarmonic()
        ensemble = BioVisionEnsemble([m1, m2, m3])
        vis = torch.randn(2, 32, 1792)
        rppg = torch.randn(2, 240)
        out = ensemble(vis, rppg)
        self.assertEqual(out.shape, (2,), "Ensemble must output shape (B,)")
        probs = torch.sigmoid(out)
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))


class TestProductionCheckpoints(unittest.TestCase):
    """Unit tests verifying production checkpoints in backend/models."""

    def test_best_checkpoint_exists_and_loads(self):
        ckpt_path = 'backend/models/biovision_best.pt'
        self.assertTrue(os.path.exists(ckpt_path), "biovision_best.pt must exist in backend/models/")
        model, info = load_model(device='cpu', checkpoint_path=ckpt_path)
        self.assertIsInstance(model, BioVisionMultiHarmonic)
        self.assertIn('optimal_threshold', info)
        self.assertAlmostEqual(float(info['optimal_threshold']), 0.835, places=2)
        self.assertGreaterEqual(float(info.get('auc', 0)), 0.85)

    def test_ensemble_checkpoint_exists_and_loads(self):
        ckpt_path = 'backend/models/biovision_ensemble.pt'
        self.assertTrue(os.path.exists(ckpt_path), "biovision_ensemble.pt must exist in backend/models/")
        model, info = load_model(device='cpu', checkpoint_path=ckpt_path)
        self.assertIsInstance(model, BioVisionEnsemble)
        self.assertEqual(len(model.models), 3, "Ensemble must contain exactly 3 sub-models")
        self.assertEqual(info.get('seeds'), [42, 101, 777])


class TestVideoAndPhysioProcessing(unittest.TestCase):
    """Unit tests for video decoding, facial ROI extraction, and CHROM rPPG signal filtering."""

    def setUp(self):
        self.temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.temp_video_path = self.temp_video.name
        self.temp_video.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.temp_video_path, fourcc, 15.0, (160, 120))
        for i in range(30):
            frame = np.zeros((120, 160, 3), dtype='uint8')
            frame[:, :, 0] = (i * 7) % 255
            frame[:, :, 1] = (i * 11) % 255
            frame[:, :, 2] = (i * 13) % 255
            writer.write(frame)
        writer.release()

    def tearDown(self):
        if os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
            except Exception:
                pass

    def test_video_metadata_validation(self):
        meta = validate_video(self.temp_video_path)
        self.assertGreater(meta['frame_count'], 0)
        self.assertGreater(meta['fps'], 0)

    def test_frame_sampling(self):
        indices = sample_frame_indices(30, n_samples=16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices, sorted(indices))

    def test_chrom_rppg_with_synthetic_frames(self):
        # 3 color channels with realistic temporal variance
        t = np.linspace(0, 5, 100)
        r = 150.0 + 5.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.randn(100) * 0.5
        g = 120.0 + 6.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.randn(100) * 0.5
        b = 100.0 + 3.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.randn(100) * 0.5
        rgb_signals = np.vstack([r, g, b])
        pulse = chrom_pulse(rgb_signals)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), 100)


class TestFastAPIEndpoints(unittest.TestCase):
    """Integration tests for all FastAPI endpoints using TestClient."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_model_info_endpoint(self):
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("BioVision Multi-Harmonic", data.get("model_name", ""))
        self.assertAlmostEqual(float(data.get("optimal_threshold", 0)), 0.835, places=2)
        self.assertGreaterEqual(len(data.get("analysis_components", [])), 2)

    def test_dashboard_stats_endpoint(self):
        response = self.client.get("/dashboard/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_analyses", data)
        self.assertIn("fake_detected", data)
        self.assertIn("real_detected", data)

    def test_evaluation_metrics_dfd_endpoint(self):
        response = self.client.get("/evaluation/metrics?dataset=dfd")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("dataset"), "dfd")
        self.assertIn("metrics", data)
        self.assertIn("Google/Jigsaw", data["metrics"].get("dataset", ""))
        self.assertIn("balanced_accuracy", data["metrics"])
        self.assertIn("roc_auc", data["metrics"])

    def test_upload_synthetic_no_face_video(self):
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_path, fourcc, 15.0, (120, 120))
        for i in range(20):
            frame = np.zeros((120, 120, 3), dtype='uint8')
            writer.write(frame)
        writer.release()

        try:
            with open(temp_path, 'rb') as f:
                resp = self.client.post("/upload", files={"file": ("noface.mp4", f, "video/mp4")})
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("result"), "NO_FACE")

    def test_analyses_crud_lifecycle(self):
        # 1. Create analysis
        payload = {
            "analysis_id": "test-uuid-12345",
            "filename": "unit_test_video.mp4",
            "result": "REAL",
            "confidence": 0.942,
            "fake_probability": 0.058
        }
        create_resp = self.client.post("/analyses", json=payload)
        self.assertEqual(create_resp.status_code, 200)

        # 2. Get specific analysis
        get_resp = self.client.get("/analyses/test-uuid-12345")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json().get("result"), "REAL")

        # 3. Delete analysis
        del_resp = self.client.delete("/analyses/test-uuid-12345")
        self.assertEqual(del_resp.status_code, 200)

        # 4. Verify 404 after deletion
        not_found = self.client.get("/analyses/test-uuid-12345")
        self.assertEqual(not_found.status_code, 404)


if __name__ == '__main__':
    unittest.main()
