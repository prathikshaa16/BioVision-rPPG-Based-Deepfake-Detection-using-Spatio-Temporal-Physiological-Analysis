#!/usr/bin/env python3
import requests
import json
import time

# Test upload endpoint with real video
video_path = 'uploads/real_test.mp4'
print(f"Uploading {video_path}...")
start = time.time()

with open(video_path, 'rb') as f:
    files = {'file': ('real_test.mp4', f, 'video/mp4')}
    try:
        resp = requests.post('http://127.0.0.1:8000/upload', files=files, timeout=120)
        elapsed = time.time() - start
        print(f'\n=== UPLOAD TEST RESULTS ===')
        print(f'Status Code: {resp.status_code}')
        print(f'Time elapsed: {elapsed:.2f}s\n')
        
        data = resp.json()
        print(f'Filename: {data.get("filename")}')
        print(f'Status: {data.get("status")}')
        print(f'Analysis ID: {data.get("analysis_id")}')
        print(f'\n--- Processing Metrics ---')
        print(f'Frames sampled: {data.get("frames_sampled")}')
        print(f'Frames analyzed (with faces): {data.get("frames_with_faces")}')
        print(f'Processing time: {data.get("processing_time")}s')
        print(f'\n--- Detection Results ---')
        print(f'Result: {data.get("result")}')
        print(f'Confidence: {data.get("confidence"):.4f}')
        print(f'Mean probability (FAKE): {data.get("mean_probability"):.4f}')
        print(f'Median probability (FAKE): {data.get("median_probability"):.4f}')
        print(f'Std Dev: {data.get("std_probability"):.4f}')
        print(f'\n--- Probabilities ---')
        print(f'Fake probability: {data.get("fake_probability"):.4f}')
        print(f'Real probability: {data.get("real_probability"):.4f}')
        print(f'\n--- Frame-level predictions ---')
        preds = data.get('frame_predictions', [])
        print(f'Total predictions: {len(preds)}')
        if preds:
            for i, p in enumerate(preds[:5]):
                print(f'  Frame {i}: {p:.4f}')
            if len(preds) > 5:
                print(f'  ... and {len(preds) - 5} more frames')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
