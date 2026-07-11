"""
cbramod_baseline_multidataset.py
=================================
CBraMod EEG Foundation Model baseline for StylePrior-MOMENT.

Designed to reuse the existing StylePrior-MOMENT data splits:
    - APAVA: 5-fold cross-subject validation
    - SleepEDF: leave-one-subject-out
    - REFED: leave-one-subject-out

Two transfer modes:
    --mode head_only : freeze CBraMod, train only classifier head
    --mode full      : fine-tune CBraMod + classifier head

Expected server layout:
    project_root/
      CBraMod/                         # official repo clone, or path passed by --cbramod_root
        models/cbramod.py
        pretrained_weights/pretrained_weights.pth
      preprocessing.py
      preprocessing_sleepedf.py
      preprocessing_refed.py
      utils.py
      cbramod_baseline_multidataset.py

Example smoke test:
    CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
        --dataset APAVA --mode head_only --end_fold 1 --epochs 2

Example APAVA full baseline:
    CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
        --dataset APAVA --mode full --epochs 30

Notes:
    1. The official CBraMod quick start uses input shape [B, C, S, 200].
       This script accepts your existing [B, C, T] EEG batches, resamples
       them from --input_sr to 200 Hz, pads to a multiple of 200, and reshapes
       to [B, C, S, 200].
    2. Validation balanced accuracy is used for model selection to match your
       StylePrior-MOMENT setup.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils import set_all_seeds, compute_comprehensive_metrics, clear_gpu_memory


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    dataset: str = "APAVA"
    mode: str = "full"
    cbramod_root: str = "./CBraMod"
    checkpoint: str = ""
    gpu_id: int = 0
    seed: int = 2025
    batch_size: int = 32
    epochs: int = 30
    lr: float = 5e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop: int = 5
    input_sr: float = 256.0
    target_sr: float = 200.0
    num_workers: int = 4
    start_fold: int = 0
    end_fold: Optional[int] = None
    output_dir: str = "./results_cbramod"
    save_logits: bool = False


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="APAVA", choices=["APAVA", "SleepEDF", "REFED"])
    p.add_argument("--mode", default="full", choices=["head_only", "full"])
    p.add_argument("--cbramod_root", default="./CBraMod")
    p.add_argument("--checkpoint", default="", help="Path to pretrained_weights.pth. Default: <cbramod_root>/pretrained_weights/pretrained_weights.pth")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-5, help="Backbone LR in full mode; ignored in head_only mode")
    p.add_argument("--head_lr", type=float, default=1e-4, help="Classifier LR")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--input_sr", type=float, default=256.0)
    p.add_argument("--target_sr", type=float, default=200.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--start_fold", type=int, default=0)
    p.add_argument("--end_fold", type=int, default=None)
    p.add_argument("--output_dir", default="./results_cbramod")
    p.add_argument("--save_logits", action="store_true")
    return RunConfig(**vars(p.parse_args()))


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "APAVA": dict(num_classes=2, num_subjects=23, k_folds=5, data_dir=None),
    "SleepEDF": dict(num_classes=5, num_subjects=20, k_folds=20, data_dir="./datasets/sleepedf"),
    "REFED": dict(num_classes=2, num_subjects=32, k_folds=32, data_dir="./datasets/refed"),
}


def split_dataset_by_subject(dataset, train_ratio: float = 0.75, random_state: int = 2025):
    all_sids = np.asarray(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_sids)
    n_train = int(len(unique_sids) * train_ratio)
    train_sids = set(unique_sids[:n_train])
    train_idx = [i for i in range(len(dataset)) if all_sids[i] in train_sids]
    val_idx = [i for i in range(len(dataset)) if all_sids[i] not in train_sids]
    print(f"  Subject split: {len(train_sids)} train / {len(unique_sids) - len(train_sids)} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def load_fold_datasets(dataset_name: str, seed: int):
    if dataset_name == "APAVA":
        from preprocessing import apava_k_fold_split
        return apava_k_fold_split(k=DATASET_CFG[dataset_name]["k_folds"], random_state=seed, use_cache=True)
    if dataset_name == "SleepEDF":
        from preprocessing_sleepedf import sleepedf_loso_split
        return sleepedf_loso_split(DATASET_CFG[dataset_name]["data_dir"])
    if dataset_name == "REFED":
        from preprocessing_refed import refed_loso_split
        return refed_loso_split(DATASET_CFG[dataset_name]["data_dir"])
    raise ValueError(dataset_name)


def make_loaders(cfg: RunConfig):
    fold_datasets = load_fold_datasets(cfg.dataset, cfg.seed)
    fold_loaders = []
    for fold_idx, (train_dataset_raw, test_dataset) in enumerate(fold_datasets):
        train_dataset, val_dataset = split_dataset_by_subject(train_dataset_raw, train_ratio=0.75, random_state=cfg.seed)
        common = dict(num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
        fold_loaders.append((
            DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, **common),
            DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, **common),
            DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, **common),
        ))
    return fold_loaders


def unpack_batch(batch, device: torch.device):
    if len(batch) >= 3:
        eeg, labels, subject_ids = batch[0], batch[1], batch[2]
    else:
        eeg, labels, subject_ids = batch[0], batch[1], None
    eeg = eeg.float().to(device)
    labels = labels.long().to(device)
    if subject_ids is not None:
        subject_ids = subject_ids.to(device)
    return eeg, labels, subject_ids


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class CBraModInputAdapter(nn.Module):
    """Convert [B,C,T] EEG at input_sr into CBraMod [B,C,S,200]."""

    def __init__(self, input_sr: float = 256.0, target_sr: float = 200.0, patch_points: int = 200):
        super().__init__()
        self.input_sr = float(input_sr)
        self.target_sr = float(target_sr)
        self.patch_points = int(patch_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected EEG tensor [B,C,T], got {tuple(x.shape)}")
        b, c, t = x.shape
        target_len = max(self.patch_points, int(round(t * self.target_sr / self.input_sr)))
        # Linear interpolation over temporal dimension.
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        n_segments = int(np.ceil(target_len / self.patch_points))
        padded_len = n_segments * self.patch_points
        if padded_len > target_len:
            x = F.pad(x, (0, padded_len - target_len))
        return x.reshape(b, c, n_segments, self.patch_points)


class CBraModClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, input_sr: float = 256.0, target_sr: float = 200.0):
        super().__init__()
        self.adapter = CBraModInputAdapter(input_sr=input_sr, target_sr=target_sr)
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x4 = self.adapter(x)
        feats = self.backbone(x4)
        return self.classifier(feats)


def import_cbramod(cbramod_root: str):
    cbramod_root = os.path.abspath(cbramod_root)
    if cbramod_root not in sys.path:
        sys.path.insert(0, cbramod_root)
    try:
        from models.cbramod import CBraMod
        return CBraMod
    except Exception as e:
        raise ImportError(
            f"Failed to import CBraMod from {cbramod_root}.\n"
            "Expected official repo layout with models/cbramod.py.\n"
            "Try: git clone https://github.com/wjq-learning/CBraMod.git ./CBraMod\n"
            f"Original error: {repr(e)}"
        )


def load_pretrained_backbone(cfg: RunConfig, device: torch.device) -> nn.Module:
    CBraMod = import_cbramod(cfg.cbramod_root)
    backbone = CBraMod().to(device)
    ckpt_path = cfg.checkpoint or os.path.join(cfg.cbramod_root, "pretrained_weights", "pretrained_weights.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"CBraMod checkpoint not found: {ckpt_path}\n"
            "Download official pretrained_weights.pth or pass --checkpoint."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if isinstance(ckpt, dict):
        cleaned = {}
        for k, v in ckpt.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v
        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        print(f"Loaded CBraMod checkpoint: {ckpt_path}")
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")
    # Official quick start removes projection head before downstream classifier.
    if hasattr(backbone, "proj_out"):
        backbone.proj_out = nn.Identity()
    return backbone


def build_model(cfg: RunConfig, num_classes: int, device: torch.device) -> CBraModClassifier:
    backbone = load_pretrained_backbone(cfg, device)
    model = CBraModClassifier(backbone, num_classes=num_classes, input_sr=cfg.input_sr, target_sr=cfg.target_sr).to(device)
    if cfg.mode == "head_only":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif cfg.mode == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(cfg.mode)
    return model


def make_optimizer(model: CBraModClassifier, cfg: RunConfig):
    if cfg.mode == "head_only":
        return optim.AdamW(model.classifier.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    # Different LR for backbone and head is safer for full fine-tuning.
    return optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr},
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


# -----------------------------------------------------------------------------
# Train/eval
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        eeg, y, _ = unpack_batch(batch, device)
        logits = model(eeg)
        prob = torch.softmax(logits, dim=1)
        preds.append(torch.argmax(logits, dim=1).cpu())
        labels.append(y.cpu())
        if num_classes == 2:
            probs.append(prob[:, 1].cpu())
        else:
            probs.append(prob.cpu())
    p = torch.cat(preds).numpy()
    y = torch.cat(labels).numpy()
    pr = torch.cat(probs).numpy() if num_classes == 2 else torch.cat(probs, 0).numpy()
    return compute_comprehensive_metrics(y, p, pr, num_classes)


def train_one_fold(cfg: RunConfig, train_loader, val_loader, num_classes: int, device: torch.device):
    model = build_model(cfg, num_classes, device)
    optimizer = make_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best_state, best_bacc = None, -1.0
    bad_epochs = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            eeg, y, _ = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, num_classes, device)
        val_bacc = float(val_metrics.get("balanced_accuracy", val_metrics.get("accuracy", 0.0)))
        scheduler.step(val_bacc)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | loss={np.mean(losses):.4f} | val_acc={val_metrics.get('accuracy',0):.4f} | val_bacc={val_bacc:.4f} | val_f1={val_metrics.get('f1_macro',0):.4f}")

        if val_bacc > best_bacc:
            best_bacc = val_bacc
            bad_epochs = 0
            best_state = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "val_metrics": copy.deepcopy(val_metrics),
                "epoch": epoch,
            }
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop:
                print(f"Early stopping at epoch {epoch}; best val_bacc={best_bacc:.4f}")
                break

    del model
    clear_gpu_memory()
    return best_state


def aggregate_metrics(fold_results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    vals: Dict[str, List[float]] = {}
    for r in fold_results:
        metrics = r.get("test_metrics")
        if not metrics:
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                vals.setdefault(k, []).append(float(v))
    means = {k: float(np.mean(v)) for k, v in vals.items()}
    stds = {k: float(np.std(v)) for k, v in vals.items()}
    return means, stds


def run(cfg: RunConfig):
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    ds_cfg = DATASET_CFG[cfg.dataset]
    num_classes = int(ds_cfg["num_classes"])
    set_all_seeds(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(json.dumps(asdict(cfg), indent=2))
    print(f"Device: {device}")

    fold_loaders = make_loaders(cfg)
    end_fold = cfg.end_fold if cfg.end_fold is not None else len(fold_loaders)
    fold_results = []

    for fold_idx in range(cfg.start_fold, min(end_fold, len(fold_loaders))):
        train_loader, val_loader, test_loader = fold_loaders[fold_idx]
        print("\n" + "=" * 70)
        print(f"CBraMod | dataset={cfg.dataset} | mode={cfg.mode} | fold {fold_idx+1}/{len(fold_loaders)}")
        print("=" * 70)
        set_all_seeds(cfg.seed)
        try:
            best_state = train_one_fold(cfg, train_loader, val_loader, num_classes, device)
            if best_state is None:
                raise RuntimeError("No best checkpoint produced")
            model = build_model(cfg, num_classes, device)
            model.load_state_dict(best_state["state_dict"], strict=True)
            test_metrics = evaluate(model, test_loader, num_classes, device)
            print(f"Fold {fold_idx+1} test: acc={test_metrics.get('accuracy',0):.4f}, bacc={test_metrics.get('balanced_accuracy',0):.4f}, f1={test_metrics.get('f1_macro',0):.4f}")
            fold_results.append({
                "fold": fold_idx + 1,
                "status": "ok",
                "best_epoch": best_state.get("epoch"),
                "val_metrics": best_state.get("val_metrics"),
                "test_metrics": test_metrics,
            })
            del model
            clear_gpu_memory()
        except Exception as e:
            print(f"Fold {fold_idx+1} FAILED: {repr(e)}")
            fold_results.append({"fold": fold_idx + 1, "status": "failed", "error": repr(e)})
            clear_gpu_memory()

        # Save after every fold so interrupted runs still leave usable results.
        partial_path = os.path.join(cfg.output_dir, f"cbramod_{cfg.dataset}_{cfg.mode}_seed{cfg.seed}_partial.pkl")
        with open(partial_path, "wb") as f:
            pickle.dump({"config": asdict(cfg), "fold_results": fold_results}, f)

    successful = [r for r in fold_results if r.get("status") == "ok"]
    means, stds = aggregate_metrics(successful)
    print("\n" + "=" * 70)
    print(f"CBraMod Summary | dataset={cfg.dataset} | mode={cfg.mode} | folds={len(successful)}/{end_fold-cfg.start_fold}")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "average_precision"]:
        if k in means:
            print(f"  {k:20s}: {means[k]:.4f} ± {stds[k]:.4f}")

    final_path = os.path.join(cfg.output_dir, f"cbramod_{cfg.dataset}_{cfg.mode}_seed{cfg.seed}.pkl")
    with open(final_path, "wb") as f:
        pickle.dump({
            "config": asdict(cfg),
            "dataset": cfg.dataset,
            "mode": cfg.mode,
            "seed": cfg.seed,
            "completed_folds": len(successful),
            "metrics_mean": means,
            "metrics_std": stds,
            "fold_results": fold_results,
        }, f)
    print(f"Saved: {final_path}")
    return final_path


if __name__ == "__main__":
    run(parse_args())
