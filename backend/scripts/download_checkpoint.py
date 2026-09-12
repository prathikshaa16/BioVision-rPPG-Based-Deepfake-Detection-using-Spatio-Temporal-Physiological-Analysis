import hashlib
import os
import requests

URL = 'https://huggingface.co/honi05/deepfake-detection/resolve/main/best_model.pt'
DST = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pt')

os.makedirs(os.path.dirname(DST), exist_ok=True)

print('Downloading', URL)
r = requests.get(URL, stream=True)
print('HTTP', r.status_code)
if r.status_code != 200:
    raise SystemExit(f'HTTP {r.status_code}')

hasher = hashlib.sha256()
size = 0
with open(DST, 'wb') as f:
    for chunk in r.iter_content(1024 * 1024):
        if not chunk:
            continue
        f.write(chunk)
        hasher.update(chunk)
        size += len(chunk)

print('Saved', DST)
print('Size', size)
print('SHA256', hasher.hexdigest())
