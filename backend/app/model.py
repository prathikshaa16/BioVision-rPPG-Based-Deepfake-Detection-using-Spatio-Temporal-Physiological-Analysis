import os
from typing import Tuple
import torch
from torchvision import models

from .config import MODEL_PATH


def build_efficientnet_b4(num_classes: int = 1) -> torch.nn.Module:
    """Construct the exact EfficientNet-B4 classifier head used by the upstream repo.

    Upstream architecture from the model card:
      Dropout(0.4) -> Linear(1792, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 1)
    """
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(256, num_classes),
    )
    return model


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, device: str = 'cpu') -> Tuple[torch.nn.Module, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)

    # The checkpoint is a plain OrderedDict state_dict, but allow the common nested form too.
    if isinstance(state, dict) and 'model_state_dict' in state:
        state_dict = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    # Some checkpoints may include a 'module.' prefix; strip it without silently renaming anything else.
    new_state = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        new_state[new_k] = v

    try:
        model.load_state_dict(new_state, strict=True)
        return model, {'missing_keys': [], 'unexpected_keys': []}
    except RuntimeError as exc:
        message = str(exc)
        # Show the exact mismatch without forcing a partial load.
        raise RuntimeError(message) from exc


def _safe_load_state_dict(model, state_dict):
    # Strict load only: if the checkpoint architecture differs, stop and report the mismatch.
    model.load_state_dict(state_dict, strict=True)
    return [], []
