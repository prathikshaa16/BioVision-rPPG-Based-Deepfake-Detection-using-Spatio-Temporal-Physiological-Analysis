"""Extract BioVision training features from raw-video JSONL manifests.

This extractor requires librosa, soundfile, mediapipe, and ffmpeg on PATH.
It writes strict records compatible with train_multimodal.py and rejects videos
where a required modality cannot be recovered.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.app.face_processor import FaceProcessor
from backend.app.rppg import run_rppg_analysis
from backend.app.video_processor import read_frames_by_indices, sample_frame_indices, validate_video


MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318]


def extract_audio(video: Path, sample_rate: int = 16000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.wav') as temporary:
        try:
            from shutil import which
            ffmpeg = which('ffmpeg')
            if ffmpeg is None:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg = get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError('ffmpeg is required for audio extraction; install ffmpeg or imageio-ffmpeg') from exc
        command = [ffmpeg, '-y', '-loglevel', 'error', '-i', str(video), '-ac', '1', '-ar', str(sample_rate), temporary.name]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError('ffmpeg is required for audio extraction but was not found on PATH') from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'Audio could not be decoded from {video.name}') from exc
        import soundfile as sf
        audio, _ = sf.read(temporary.name, dtype='float32')
    if audio.size == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise ValueError(f'Audio is missing or silent in {video.name}')
    return audio


def extract_mfcc(audio: np.ndarray, sample_rate: int, steps: int) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=512, hop_length=160, win_length=400)
    timestamps = np.linspace(0, mfcc.shape[1] - 1, num=steps).round().astype(int)
    return mfcc[:, timestamps].T.astype(np.float32)


def extract_mouth(frames, boxes, steps: int, model_path: Path) -> np.ndarray:
    import mediapipe as mp
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    observations = []
    try:
        for frame, box in zip(frames, boxes):
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = [int(value) for value in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                raise ValueError('Mouth landmarks could not be recovered for every selected frame')
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = mesh.detect(image)
            if not result.face_landmarks or box is None:
                raise ValueError('Mouth landmarks could not be recovered for every selected frame')
            face_width = max(float(x2 - x1), 1.0)
            landmarks = result.face_landmarks[0]
            values = []
            for index in MOUTH_INDICES:
                point = landmarks[index]
                values.extend([point.x * (x2 - x1) / face_width, point.y * (y2 - y1) / face_width])
            observations.append(values)
    finally:
        mesh.close()
    return np.asarray(observations, dtype=np.float32).reshape(steps, 40)


def extract_record(record, output_dir: Path, face_processor: FaceProcessor, device: str, landmark_model: Path, mode: str):
    video = Path(record['video'])
    meta = validate_video(str(video))
    steps = 32
    indices = sample_frame_indices(meta['frame_count'], n_samples=steps)
    frames = read_frames_by_indices(str(video), indices)
    if len(frames) != steps or any(frame is None for frame in frames):
        raise ValueError('Could not read all selected frames')
    crops = []
    boxes = []
    for frame in frames:
        detected, _ = face_processor.detect_faces(frame)
        if not detected:
            raise ValueError('Face missing from selected visual sequence')
        box = max(detected, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
        cropped = face_processor.detect_and_crop(frame)
        if not cropped:
            raise ValueError('Face crop could not be produced')
        largest = max(cropped, key=lambda item: item.size[0] * item.size[1])
        crops.append(face_processor.preprocess_pil(largest).numpy())
        boxes.append(box)
    visual = np.stack(crops).astype(np.float32)
    rppg = run_rppg_analysis(str(video), meta, device=device)
    if rppg['status'] != 'AVAILABLE' or not rppg.get('filtered_signal'):
        raise ValueError('Reliable CHROM-rPPG signal unavailable')
    pulse = np.asarray(rppg['filtered_signal']['amplitude'], dtype=np.float32)
    stem = str(Path(record['relative_video']).with_suffix('')).replace('\\', '_').replace('/', '_')
    paths = {}
    arrays = [('visual', visual), ('rppg', pulse)]
    if mode == 'full':
        arrays.extend([
            ('mfcc', extract_mfcc(extract_audio(video), 16000, steps)),
            ('mouth', extract_mouth(frames, boxes, steps, landmark_model)),
        ])
    for name, array in arrays:
        destination = output_dir / f'{stem}_{name}.npy'
        np.save(destination, array)
        paths[name] = str(destination)
    return {**paths, 'label': int(record['label']), 'source_video': record['relative_video']}


def main():
    parser = argparse.ArgumentParser(description='Extract BioVision multimodal features')
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--face-landmarker-model', type=Path, required=True)
    parser.add_argument('--mode', choices=('full', 'visual-rppg'), default='full')
    parser.add_argument('--resume', action='store_true', help='Continue from an existing output manifest')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    records = [json.loads(line) for line in (args.manifest_dir / f'{args.split}.jsonl').read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    face_processor = FaceProcessor(device=device)
    output_manifest = args.output_dir / f'{args.split}.jsonl'
    rejected_manifest = args.output_dir / f'{args.split}.rejected.jsonl'
    output_records = []
    completed = set()
    if args.resume and output_manifest.exists():
        output_records = [json.loads(line) for line in output_manifest.read_text().splitlines() if line.strip()]
        completed = {record.get('source_video') for record in output_records}
    elif not args.resume:
        output_manifest.unlink(missing_ok=True)
        rejected_manifest.unlink(missing_ok=True)
    for index, record in enumerate(records, start=1):
        if record['relative_video'] in completed:
            print(f'[{index}/{len(records)}] {record["relative_video"]} already complete', flush=True)
            continue
        print(f'[{index}/{len(records)}] {record["relative_video"]}', flush=True)
        try:
            extracted = extract_record(record, args.output_dir, face_processor, device, args.face_landmarker_model, args.mode)
            output_records.append(extracted)
            with output_manifest.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(extracted) + '\n')
            completed.add(record['relative_video'])
        except (RuntimeError, ValueError) as exc:
            print(f'  rejected: {exc}', flush=True)
            with rejected_manifest.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'source_video': record['relative_video'], 'reason': str(exc)}) + '\n')
    print(f'accepted={len(output_records)} rejected={len(records) - len(output_records)} manifest={output_manifest}')


if __name__ == '__main__':
    main()
