"""Trainable BioVision multimodal architecture from the project specification.

Inputs are already aligned by the dataset layer:
visual [B, T, 3, 224, 224], rppg [B, N], MFCC [B, T, 40], and mouth [B, T, 40].
The model returns a single fake logit and preserves the 256 + 64 + 128 = 448
branch interface described in the paper.
"""

from typing import Dict

import torch
from torch import nn
from torchvision import models


class VisualTemporalBranch(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        backbone = models.efficientnet_b4(weights=None)
        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.LSTM(
            input_size=1792,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.30,
        )

    def forward(self, visual: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, steps = visual.shape[:2]
        encoded = self.encoder(visual.reshape(batch * steps, *visual.shape[2:]))
        encoded = self.pool(encoded).flatten(1).reshape(batch, steps, 1792)
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.temporal(packed)
        return hidden[-1]


class RppgBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.Dropout(0.20))

    def forward(self, rppg: torch.Tensor) -> torch.Tensor:
        if rppg.ndim == 2:
            rppg = rppg.unsqueeze(1)
        return self.projection(self.encoder(rppg))


class AudioLipBranch(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.audio_projection = nn.Linear(40, 128)
        self.lip_projection = nn.Linear(40, 64)
        self.temporal = nn.LSTM(input_size=192, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_size, 128), nn.Dropout(0.20))

    def forward(self, mfcc: torch.Tensor, mouth: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        paired = torch.cat((self.audio_projection(mfcc), self.lip_projection(mouth)), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            paired, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.temporal(packed)
        return self.output(hidden[-1])


class BioVisionMultimodalModel(nn.Module):
    """EfficientNet-B4-LSTM + learned CHROM + audio-lip fusion classifier."""

    def __init__(self):
        super().__init__()
        self.visual_temporal = VisualTemporalBranch()
        self.rppg = RppgBranch()
        self.audio_lip = AudioLipBranch()
        self.visual_norm = nn.LayerNorm(256)
        self.rppg_norm = nn.LayerNorm(64)
        self.audio_lip_norm = nn.LayerNorm(128)
        self.classifier = nn.Sequential(
            nn.Linear(448, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        visual: torch.Tensor,
        rppg: torch.Tensor,
        mfcc: torch.Tensor,
        mouth: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        visual_features = self.visual_norm(self.visual_temporal(visual, lengths))
        rppg_features = self.rppg_norm(self.rppg(rppg))
        audio_lip_features = self.audio_lip_norm(self.audio_lip(mfcc, mouth, lengths))
        fused = torch.cat((visual_features, rppg_features, audio_lip_features), dim=-1)
        return self.classifier(fused).squeeze(-1)

class BioVisionVisualRppgModel(nn.Module):
    """Audio-free fallback for datasets whose videos contain no audio stream."""

    def __init__(self):
        super().__init__()
        self.visual_temporal = VisualTemporalBranch()
        self.rppg = RppgBranch()
        self.visual_norm = nn.LayerNorm(256)
        self.rppg_norm = nn.LayerNorm(64)
        self.classifier = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, visual, rppg, lengths):
        visual_features = self.visual_norm(self.visual_temporal(visual, lengths))
        rppg_features = self.rppg_norm(self.rppg(rppg))
        return self.classifier(torch.cat((visual_features, rppg_features), dim=-1)).squeeze(-1)


class CachedBioVisionVisualRppg(nn.Module):
    """Exact cached-feature model contract used by the Colab baseline notebook."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1792,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
        )
        self.phys = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.phys_fc = nn.Linear(64, 64)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.phys_norm = nn.LayerNorm(64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, visual_features: torch.Tensor, rppg: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(visual_features)
        visual = self.visual_norm(hidden[-1])
        physiological = self.phys(rppg.unsqueeze(1)).squeeze(-1)
        physiological = self.phys_norm(self.phys_fc(physiological))
        fused = torch.cat((visual, physiological), dim=1)
        return self.classifier(fused).squeeze(1)


def build_biovision_multimodal_model() -> BioVisionMultimodalModel:
    return BioVisionMultimodalModel()

def build_biovision_visual_rppg_model() -> BioVisionVisualRppgModel:
    return BioVisionVisualRppgModel()


def build_cached_biovision_visual_rppg_model() -> CachedBioVisionVisualRppg:
    return CachedBioVisionVisualRppg()