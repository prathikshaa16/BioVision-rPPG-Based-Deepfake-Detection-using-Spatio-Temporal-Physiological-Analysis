import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

modules = [
    'torch',
    'torchvision',
    'cv2',
    'facenet_pytorch',
    'backend.app.video_processor',
    'backend.app.face_processor',
    'backend.app.inference',
]

results = {}
for m in modules:
    try:
        __import__(m)
        results[m] = {'ok': True}
    except Exception as e:
        results[m] = {'ok': False, 'error': str(e)}

print(json.dumps(results, indent=2))
