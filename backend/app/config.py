import os
from pathlib import Path

# Backend configuration constants
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / '..' / 'uploads'
UPLOAD_DIR = UPLOAD_DIR.resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Video processing
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
DEFAULT_SAMPLE_FRAMES = 15
FACE_MARGIN = 20  # pixels margin around detected face when cropping
TARGET_FACE_SIZE = (224, 224)

# Model
# Resolve the checkpoint from the config file location so it is stable regardless of CWD.
LEGACY_MODEL_FILENAME = 'best_model.pt'
CACHED_MODEL_FILENAME = 'biovision_best.pt'
LEGACY_MODEL_PATH = (BASE_DIR / 'models' / LEGACY_MODEL_FILENAME).resolve()
CACHED_MODEL_PATH = (BASE_DIR / 'models' / CACHED_MODEL_FILENAME).resolve()
if CACHED_MODEL_PATH.exists():
    DEFAULT_MODEL_TYPE = 'cached'
    MODEL_PATH = CACHED_MODEL_PATH
else:
    DEFAULT_MODEL_TYPE = os.environ.get('BIOVISION_MODEL_TYPE', 'legacy').strip().lower()
    MODEL_PATH = LEGACY_MODEL_PATH


def resolve_model_paths(model_type: str | None = None, checkpoint_path: str | None = None):
    """Return the resolved model kind and checkpoint path."""
    chosen_type = (model_type or DEFAULT_MODEL_TYPE or 'legacy').strip().lower()
    if checkpoint_path:
        return chosen_type, Path(checkpoint_path).resolve()
    if chosen_type == 'cached':
        return chosen_type, CACHED_MODEL_PATH
    return chosen_type, LEGACY_MODEL_PATH

# Device selection (will be validated at runtime)
TORCH_DEVICE = 'cuda' if (os.environ.get('CUDA_VISIBLE_DEVICES') or False) else 'cpu'

# rPPG (CHROM) physiological analysis — pure numpy, no scipy required
RPPG_WINDOW_SECONDS = 10.0      # contiguous window used for the temporal signal
RPPG_MAX_FRAMES = 300           # hard cap on frames read in the window
RPPG_MIN_FRAMES = 60            # minimum contiguous usable face frames for a signal
RPPG_LOW_HZ = 0.8               # bandpass lower edge (48 BPM)
RPPG_HIGH_HZ = 3.0              # bandpass upper edge (180 BPM)
