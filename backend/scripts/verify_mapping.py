"""Verify the existing EfficientNet-B4 checkpoint mapping: output -> prob -> label.

Run from the repository root:

    python backend/scripts/verify_mapping.py
    python backend/scripts/verify_mapping.py --validate
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Verify model output -> probability -> label mapping')
    parser.add_argument('--validate', action='store_true', help='Run validation on labelled videos found in the repo')
    args = parser.parse_args()

    from backend.app.model_verification import probe_model, run_labelled_validation

    report = probe_model(device='cpu')
    print(json.dumps(report, indent=2))

    if args.validate:
        print('\n=== LABELLED VALIDATION ===')
        labelled = []
        real_candidates = [
            'uploads/real_test.mp4',
            'uploads/test_face_video.mp4',
        ]
        for path in real_candidates:
            if os.path.exists(path):
                labelled.append({'path': path, 'ground_truth': 'REAL'})

        if not labelled:
            print('No labelled videos available for validation.')
            return

        validation = run_labelled_validation(labelled, device='cpu')
        print(json.dumps(validation, indent=2))
        print(
            '\nNOTE: No labelled FAKE evaluation video is present in this repository, so '
            'fake-class accuracy cannot be proven. The reported metrics reflect only the '
            'labelled samples that actually exist.'
        )


if __name__ == '__main__':
    main()
