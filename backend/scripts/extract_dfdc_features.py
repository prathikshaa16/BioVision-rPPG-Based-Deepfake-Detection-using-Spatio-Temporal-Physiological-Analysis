"""Extract BioVision multimodal features (32x1792 visual + 240 rPPG) for DFDC.

Extracts:
  - 32 sampled facial crops -> EfficientNet-B4 features [32, 1792]
  - Forehead & cheeks skin ROIs -> CHROM rPPG pulse vector [240]
Saves shard records compatible with BioVision training and evaluation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from backend.app.face_processor import FaceProcessor
from backend.app.rppg import run_rppg_analysis
from backend.app.video_processor import sample_frame_indices, read_frames_by_indices, validate_video


def extract_single_video(video_path: str, fp: FaceProcessor, backbone: torch.nn.Module, device: str = 'cpu', n_frames: int = 32, rppg_size: int = 240):
    meta = validate_video(video_path)
    total_frames = meta['frame_count']
    if total_frames < n_frames:
        raise ValueError(f'Video has fewer than {n_frames} frames ({total_frames})')

    indices = sample_frame_indices(total_frames, n_samples=n_frames)
    frames = read_frames_by_indices(video_path, indices)

    face_crops = []
    for f in frames:
        if f is None:
            continue
        crops = fp.detect_and_crop(f)
        if crops:
            face_crops.append(crops[0])

    if not face_crops:
        raise ValueError('NO_FACE_DETECTED')

    weights = EfficientNet_B4_Weights.DEFAULT
    transform = weights.transforms()

    crops_tensor = torch.stack([transform(c) for c in face_crops[:n_frames]])
    if crops_tensor.shape[0] < n_frames:
        pad = n_frames - crops_tensor.shape[0]
        crops_tensor = torch.cat([crops_tensor, crops_tensor[-1:].repeat(pad, 1, 1, 1)], dim=0)

    with torch.inference_mode():
        batch = crops_tensor.to(device)
        feats = backbone.features(batch)
        feats = backbone.avgpool(feats)
        visual_features = torch.flatten(feats, 1).cpu()

    # rPPG extraction
    rppg_data = run_rppg_analysis(video_path, meta, device=device)
    signal = rppg_data.get('filtered_signal', {}).get('amplitude') if rppg_data.get('status') == 'AVAILABLE' else None
    if signal is None:
        rppg_vec = np.zeros(rppg_size, dtype=np.float32)
    else:
        sig_arr = np.asarray(signal, dtype=np.float32)
        if sig_arr.size < rppg_size:
            padded = np.zeros(rppg_size, dtype=np.float32)
            padded[:sig_arr.size] = sig_arr
            rppg_vec = padded
        else:
            rppg_vec = sig_arr[:rppg_size]

    return {
        'visual_features': visual_features,
        'rppg': torch.as_tensor(rppg_vec, dtype=torch.float32),
        'rppg_status': rppg_data.get('status', 'UNAVAILABLE'),
        'heart_rate_bpm': rppg_data.get('heart_rate_bpm'),
        'dominant_frequency': rppg_data.get('dominant_frequency'),
        'signal_quality': rppg_data.get('signal_quality'),
    }


def main():
    parser = argparse.ArgumentParser(description='Extract BioVision multimodal features for DFDC')
    parser.add_argument('--manifest', type=Path, required=True, help='Path to jsonl manifest')
    parser.add_argument('--output', type=Path, required=True, help='Output .pt file for extracted records')
    parser.add_argument('--batch-size', type=int, default=32, help='Save interval')
    parser.add_argument('--max-videos', type=int, default=None, help='Limit number of videos (for testing)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    print('Loading EfficientNet-B4 backbone...')
    weights = EfficientNet_B4_Weights.DEFAULT
    backbone = efficientnet_b4(weights=weights).to(device)
    backbone.eval()
    fp = FaceProcessor(device=device)

    records = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.max_videos:
        records = records[:args.max_videos]

    print(f'Processing {len(records)} videos from manifest: {args.manifest}')

    results = []
    failed = []
    start_time = time.time()

    for idx, rec in enumerate(records):
        video_path = rec['video']
        try:
            feats = extract_single_video(video_path, fp, backbone, device=device)
            item = dict(rec)
            item.update(feats)
            results.append(item)
            if (idx + 1) % 10 == 0 or idx == len(records) - 1:
                print(f'[{idx + 1}/{len(records)}] Processed: {Path(video_path).name} (Success={len(results)}, Fail={len(failed)})')
        except Exception as e:
            failed.append({'video': video_path, 'error': str(e)})
            print(f'[{idx + 1}/{len(records)}] Failed: {Path(video_path).name} -> {e}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'records': results,
        'failed': failed,
        'total_manifest': len(records),
        'usable': len(results),
        'failed_count': len(failed),
        'elapsed_seconds': round(time.time() - start_time, 2),
    }

    torch.save(payload, args.output)
    print(f'[OK] Saved {len(results)} records to {args.output} (Failed: {len(failed)}) in {payload["elapsed_seconds"]}s')


if __name__ == '__main__':
    main()
