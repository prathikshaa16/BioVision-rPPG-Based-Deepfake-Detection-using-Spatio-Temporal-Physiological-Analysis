"""Advanced Anti-Overfitting Multimodal BioVision Trainer.

Features:
  1. On-the-fly feature-space data augmentation:
     - Temporal sequence dropout (randomly zeroing 1-5 timesteps)
     - Gaussian embedding jitter (sigma=0.02)
     - Temporal circular shift and reversal
     - rPPG phase and amplitude jitter
  2. Balanced mini-batch sampling via WeightedRandomSampler (50/50 Real/Fake)
  3. Label smoothing (epsilon=0.05) to halt validation loss explosion
  4. Regularized architecture: Dropout=0.35 in LSTM, 0.40 in classifier, Weight Decay=1e-3
  5. Optimal threshold calibration using Youden's J index
  6. Tracks Accuracy, Balanced Accuracy, Specificity, Sensitivity, Precision, F1, and ROC-AUC
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from backend.app.multimodal_model import build_cached_biovision_visual_rppg_model


class AugmentedCachedDataset(Dataset):
    """Dataset with on-the-fly feature augmentations to prevent overfitting."""

    def __init__(self, path: Path, is_train: bool = True, noise_sigma: float = 0.02, seq_drop_max: int = 5):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.records = payload['records'] if isinstance(payload, dict) else payload
        if not self.records:
            raise ValueError(f'No records found in {path}')
        self.labels = np.asarray([int(record['label']) for record in self.records], dtype=np.int64)
        self.is_train = is_train
        self.noise_sigma = noise_sigma
        self.seq_drop_max = seq_drop_max

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        visual = torch.as_tensor(record['visual_features'], dtype=torch.float32).clone()
        rppg = torch.as_tensor(record['rppg'], dtype=torch.float32).flatten().clone()

        # Fixed rPPG padding/truncation
        if not 1 <= rppg.numel() <= 240:
            padded = torch.zeros(240, dtype=torch.float32)
            padded[:min(240, rppg.numel())] = rppg[:240]
            rppg = padded
        elif rppg.numel() < 240:
            padded = torch.zeros(240, dtype=torch.float32)
            padded[:rppg.numel()] = rppg
            rppg = padded
        else:
            rppg = rppg[:240]

        # In-memory feature augmentation during training
        if self.is_train:
            # 1. Temporal Sequence Dropout: zero out k random timesteps
            if random.random() < 0.70:
                k = random.randint(1, self.seq_drop_max)
                drop_indices = random.sample(range(visual.shape[0]), k)
                visual[drop_indices, :] = 0.0

            # 2. Gaussian feature jitter on visual embeddings
            if self.noise_sigma > 0:
                noise = torch.randn_like(visual) * self.noise_sigma
                visual = visual + noise

            # 3. Temporal Circular Shift / Reversal
            if random.random() < 0.30:
                shift = random.randint(-4, 4)
                visual = torch.roll(visual, shifts=shift, dims=0)

            # 4. rPPG phase and amplitude jitter
            if random.random() < 0.40:
                phase_shift = random.randint(-8, 8)
                rppg = torch.roll(rppg, shifts=phase_shift, dims=0)
                scale = random.uniform(0.90, 1.10)
                rppg = rppg * scale

        label = float(record['label'])
        return visual, rppg, torch.tensor(label, dtype=torch.float32)


class QualityGatedFusionModel(nn.Module):
    """BioVision Multimodal model with Quality-Gated Late Fusion and higher regularization."""

    def __init__(self, hidden_size: int = 256, dropout_lstm: float = 0.35, dropout_head: float = 0.40):
        super().__init__()
        self.visual_lstm = nn.LSTM(
            input_size=1792,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_lstm,
        )
        self.visual_norm = nn.LayerNorm(hidden_size)

        self.rppg_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.20),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rppg_norm = nn.LayerNorm(64)

        # Quality gating layer: learns to gate physiological signal influence
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + 64, 64),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_head * 0.75),
            nn.Linear(64, 1),
        )

    def forward(self, visual_features: torch.Tensor, rppg: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.visual_lstm(visual_features)
        visual = self.visual_norm(hidden[-1])

        physiological = self.rppg_conv(rppg.unsqueeze(1)).squeeze(-1)
        physiological = self.rppg_norm(physiological)

        # Compute adaptive gate
        gate_input = torch.cat((visual, physiological), dim=1)
        g = self.gate(gate_input)
        gated_physiological = physiological * g

        fused = torch.cat((visual, gated_physiological), dim=1)
        return self.classifier(fused).squeeze(1)


def compute_comprehensive_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        'threshold': round(threshold, 4),
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'specificity': specificity,
        'sensitivity': sensitivity,
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else 0.5,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def find_optimal_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_j = -1.0
    best_thresh = 0.5
    best_metrics = {}

    thresholds = np.linspace(0.10, 0.90, 81)
    for thresh in thresholds:
        m = compute_comprehensive_metrics(labels, probabilities, threshold=thresh)
        j = m['sensitivity'] + m['specificity'] - 1.0
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)
            best_metrics = m

    return best_thresh, best_metrics


def run_training_epoch(model, loader, device, optimizer, label_smoothing: float = 0.05):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    labels = []
    probs = []

    for visual, rppg, target in loader:
        visual, rppg, target = visual.to(device), rppg.to(device), target.to(device)

        # Smooth targets: 0 -> epsilon, 1 -> 1 - epsilon
        smoothed_target = target * (1.0 - 2.0 * label_smoothing) + label_smoothing

        optimizer.zero_grad(set_to_none=True)
        logits = model(visual, rppg)
        loss = criterion(logits, smoothed_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * target.numel()
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(batch_probs)
        labels.extend(target.detach().cpu().numpy().tolist())

    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    return total_loss / len(labels_arr), labels_arr, probs_arr


def run_eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    labels = []
    probs = []

    with torch.no_grad():
        for visual, rppg, target in loader:
            visual, rppg, target = visual.to(device), rppg.to(device), target.to(device)
            logits = model(visual, rppg)
            loss = criterion(logits, target)
            total_loss += loss.item() * target.numel()
            batch_probs = torch.sigmoid(logits).cpu().numpy().tolist()
            probs.extend(batch_probs)
            labels.extend(target.cpu().numpy().tolist())

    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    return total_loss / len(labels_arr), labels_arr, probs_arr


def main():
    parser = argparse.ArgumentParser(description='Train Advanced BioVision with Anti-Overfitting and Balanced Sampling')
    parser.add_argument('--train', type=Path, required=True, help='Train records (.pt)')
    parser.add_argument('--val', type=Path, required=True, help='Val records (.pt)')
    parser.add_argument('--output', type=Path, required=True, help='Output model checkpoint (.pt)')
    parser.add_argument('--history', type=Path, default=None, help='History log path (.json)')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay (L2 regularization)')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    print(f'Loading train dataset from: {args.train}')
    train_dataset = AugmentedCachedDataset(args.train, is_train=True)
    print(f'Loading val dataset from: {args.val}')
    val_dataset = AugmentedCachedDataset(args.val, is_train=False)

    # Balanced Sampler: each class gets 50% representation in expectation
    class_counts = np.bincount(train_dataset.labels, minlength=2)
    sample_weights = np.asarray([1.0 / class_counts[y] for y in train_dataset.labels], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f'Train samples: {len(train_dataset)} (REAL={class_counts[0]}, FAKE={class_counts[1]})')
    val_counts = np.bincount(val_dataset.labels, minlength=2)
    print(f'Val samples  : {len(val_dataset)} (REAL={val_counts[0]}, FAKE={val_counts[1]})')

    model = QualityGatedFusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history_path = args.history or args.output.with_suffix('.history.json')
    history = {
        'model_architecture': 'QualityGatedBioVision',
        'train_file': str(args.train.resolve()),
        'val_file': str(args.val.resolve()),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'device': str(device),
        'epochs': [],
    }

    best_auc = -1.0
    best_bal_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    print('\n' + '=' * 78)
    print('STARTING ADVANCED BIOVISION ANTI-OVERFITTING TRAINING')
    print('=' * 78)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_labels, t_probs = run_training_epoch(model, train_loader, device, optimizer, label_smoothing=args.label_smoothing)
        v_loss, v_labels, v_probs = run_eval_epoch(model, val_loader, device)

        t_metrics = compute_comprehensive_metrics(t_labels, t_probs, threshold=0.5)
        v_metrics_default = compute_comprehensive_metrics(v_labels, v_probs, threshold=0.5)
        opt_thresh, v_metrics_opt = find_optimal_youden_threshold(v_labels, v_probs)

        record = {
            'epoch': epoch,
            'train_loss': round(t_loss, 4),
            'val_loss': round(v_loss, 4),
            'lr': optimizer.param_groups[0]['lr'],
            'train_auc': round(t_metrics['auc'], 4),
            'val_auc': round(v_metrics_default['auc'], 4),
            'val_acc_default': round(v_metrics_default['accuracy'], 4),
            'val_bal_acc_default': round(v_metrics_default['balanced_accuracy'], 4),
            'val_opt_threshold': round(opt_thresh, 4),
            'val_acc_opt': round(v_metrics_opt['accuracy'], 4),
            'val_bal_acc_opt': round(v_metrics_opt['balanced_accuracy'], 4),
            'val_sensitivity_opt': round(v_metrics_opt['sensitivity'], 4),
            'val_specificity_opt': round(v_metrics_opt['specificity'], 4),
            'val_f1_opt': round(v_metrics_opt['f1'], 4),
        }
        history['epochs'].append(record)

        print(
            f'Epoch {epoch:02d}/{args.epochs} | '
            f'TrainLoss={t_loss:.4f} ValLoss={v_loss:.4f} | '
            f'ValAUC={record["val_auc"]} | '
            f'BalAcc@0.5={record["val_bal_acc_default"]} | '
            f'OptThresh={opt_thresh:.2f} -> BalAcc@Opt={record["val_bal_acc_opt"]} (Sens={record["val_sensitivity_opt"]}, Spec={record["val_specificity_opt"]})'
        )

        # Model selection based on Validation AUC and Balanced Accuracy
        current_score = record['val_auc'] + record['val_bal_acc_opt']
        best_score = best_auc + best_bal_acc
        if current_score > best_score:
            best_auc = record['val_auc']
            best_bal_acc = record['val_bal_acc_opt']
            best_epoch = epoch
            patience_counter = 0

            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_auc,
                'best_val_bal_acc': best_bal_acc,
                'optimal_threshold': opt_thresh,
                'metrics_at_optimal_threshold': v_metrics_opt,
                'history_summary': record,
            }, args.output)
            print(f'  [BEST CHECKPOINT SAVED] Epoch {epoch} (AUC={best_auc:.4f}, BalAcc@Opt={best_bal_acc:.4f}, Thresh={opt_thresh:.2f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'\n[EARLY STOPPING] Triggered at epoch {epoch} (Patience={args.patience} reached without improvement).')
                break

        scheduler.step()

    history['best_epoch'] = best_epoch
    history['best_val_auc'] = best_auc
    history['best_val_bal_acc'] = best_bal_acc
    history['total_training_time'] = round(time.time() - start_time, 2)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print('\n' + '=' * 78)
    print('TRAINING COMPLETED')
    print(f'Best Epoch       : {best_epoch}')
    print(f'Best Val ROC-AUC : {best_auc:.4f}')
    print(f'Best Val BalAcc  : {best_bal_acc:.4f}')
    print(f'Checkpoint saved : {args.output}')
    print(f'History log saved: {history_path}')
    print('=' * 78)


if __name__ == '__main__':
    main()
