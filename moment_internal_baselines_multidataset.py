#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moment_internal_baselines_multidataset.py

Run internal MOMENT baselines for StylePrior-MOMENT paper:
  - linear: frozen MOMENT backbone + trainable classification head
  - full: full fine-tuning of MOMENT
  - lora: frozen MOMENT backbone + LoRA on attention q/v + trainable classification head

It reuses the same dataset split functions as the existing project:
  APAVA      -> preprocessing.apava_k_fold_split
  SleepEDF   -> preprocessing_sleepedf.sleepedf_loso_split
  REFED      -> preprocessing_refed.refed_loso_split

Example:
  python moment_internal_baselines_multidataset.py --dataset APAVA --mode linear
  python moment_internal_baselines_multidataset.py --dataset REFED --mode lora --start_fold 0 --end_fold 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize
except Exception as e:
    raise RuntimeError("This script requires scikit-learn. Please install sklearn/scikit-learn.") from e


# Keep local project imports available.
sys.path.append(".")
sys.path.append("./MoE_moment")


DATASET_CFG = {
    "APAVA": dict(seq_len=256, input_channels=16, num_classes=2,
                  num_subjects=23, k_folds=5, data_dir=None, sampling_rate=256.0),
    "SleepEDF": dict(seq_len=256, input_channels=16, num_classes=5,
                     num_subjects=20, k_folds=20, data_dir="./datasets/sleepedf", sampling_rate=256.0),
    "REFED": dict(seq_len=256, input_channels=16, num_classes=2,
                  num_subjects=32, k_folds=32, data_dir="./datasets/refed", sampling_rate=256.0),
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def split_dataset_by_subject(dataset, train_ratio: float = 0.75, random_state: int = 2025):
    all_sids = np.array(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_sids)

    n_train = int(len(unique_sids) * train_ratio)
    train_subjects = set(unique_sids[:n_train])
    train_idx = [i for i in range(len(dataset)) if all_sids[i] in train_subjects]
    val_idx = [i for i in range(len(dataset)) if all_sids[i] not in train_subjects]

    print(f"  Subject split: {n_train} train / {len(unique_sids)-n_train} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def load_fold_datasets(dataset_name: str, cfg: Dict[str, Any], seed: int):
    if dataset_name == "APAVA":
        from preprocessing import apava_k_fold_split
        print("Loading APAVA k-fold split...")
        return apava_k_fold_split(k=cfg["k_folds"], random_state=seed, use_cache=True)

    if dataset_name == "SleepEDF":
        from preprocessing_sleepedf import sleepedf_loso_split
        print("Loading SleepEDF LOSO split...")
        return sleepedf_loso_split(cfg["data_dir"])

    if dataset_name == "REFED":
        from preprocessing_refed import refed_loso_split
        print("Loading REFED LOSO split...")
        return refed_loso_split(cfg["data_dir"])

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_loaders(dataset_name: str, cfg: Dict[str, Any], seed: int,
                 batch_size: int, num_workers: int):
    fold_datasets = load_fold_datasets(dataset_name, cfg, seed)

    loaders = []
    for fold_idx, (train_dataset_raw, test_dataset) in enumerate(fold_datasets):
        train_dataset, val_dataset = split_dataset_by_subject(
            train_dataset_raw, train_ratio=0.75, random_state=seed
        )
        loaders.append((
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=False, pin_memory=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, drop_last=False, pin_memory=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, drop_last=False, pin_memory=True),
        ))
    return loaders


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear.")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(dropout)

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x: [..., in_features]
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)
        return base_out + self.scaling * lora


def get_root_module(model):
    # MOMENTPipeline usually has .model. If not, the model itself is nn.Module.
    return getattr(model, "model", model)


def iter_named_modules_with_parent(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        yield full_name, module, name, child
        yield from iter_named_modules_with_parent(child, full_name)


def is_lora_target(full_name: str, local_name: str, target_modules: str) -> bool:
    targets = [t.strip().lower() for t in target_modules.split(",") if t.strip()]
    ln = local_name.lower()
    fn = full_name.lower()
    for t in targets:
        if ln == t:
            return True
        # Common T5 attention names: SelfAttention.q / SelfAttention.v
        if fn.endswith("." + t):
            return True
        if f".selfattention.{t}" in fn:
            return True
        if f".attention.{t}" in fn:
            return True
    return False


def inject_lora(model, target_modules: str = "q,v", r: int = 8, alpha: int = 16, dropout: float = 0.05) -> int:
    root = get_root_module(model)
    replacements = []
    for full_name, parent, local_name, child in iter_named_modules_with_parent(root):
        if isinstance(child, nn.Linear) and is_lora_target(full_name, local_name, target_modules):
            replacements.append((full_name, parent, local_name, child))

    for full_name, parent, local_name, child in replacements:
        setattr(parent, local_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    return len(replacements)


def set_requires_grad_all(model, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def set_head_trainable(model, flag: bool = True) -> int:
    count = 0
    candidates = []

    # Official MOMENTPipeline usually uses model.model.head.
    if hasattr(model, "model") and hasattr(model.model, "head"):
        candidates.append(model.model.head)
    if hasattr(model, "head"):
        candidates.append(model.head)

    # Avoid duplicate modules.
    seen = set()
    for m in candidates:
        if id(m) in seen:
            continue
        seen.add(id(m))
        for p in m.parameters():
            p.requires_grad = flag
            count += p.numel()
    return count


def import_moment_pipeline():
    errors = []
    try:
        from momentfm import MOMENTPipeline
        return MOMENTPipeline
    except Exception as e:
        errors.append(f"from momentfm import MOMENTPipeline failed: {repr(e)}")

    try:
        from MoE_moment.momentfm import MOMENTPipeline
        return MOMENTPipeline
    except Exception as e:
        errors.append(f"from MoE_moment.momentfm import MOMENTPipeline failed: {repr(e)}")

    try:
        from MoE_moment.momentfm.models.moment import MOMENT
        # MOMENT alone is not enough for from_pretrained pipeline.
        raise ImportError("Found MOMENT class but not MOMENTPipeline.")
    except Exception as e:
        errors.append(f"fallback failed: {repr(e)}")

    raise ImportError(
        "Could not import MOMENTPipeline. Please make sure the original MOMENT package "
        "is available in this environment.\n" + "\n".join(errors)
    )


def build_moment_model(args, cfg: Dict[str, Any], device: torch.device):
    MOMENTPipeline = import_moment_pipeline()

    model_kwargs = {
        "task_name": "classification",
        "n_channels": cfg["input_channels"],
        "num_class": cfg["num_classes"],
        "freeze_embedder": False,  # We set trainability manually below.
        "freeze_encoder": False,
        "freeze_head": False,
        "seq_len": cfg["seq_len"],
        "reduction": args.reduction,
        "add_positional_embedding": False,
    }

    # Try both official and local calling conventions.
    try:
        model = MOMENTPipeline.from_pretrained(
            model_path=args.model_path,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        model = MOMENTPipeline.from_pretrained(
            args.model_path,
            model_kwargs=model_kwargs,
        )

    if hasattr(model, "init"):
        try:
            model.init()
        except Exception as e:
            print(f"Warning: model.init() failed or unnecessary: {repr(e)}")

    model = model.to(device)

    # Disable gradient checkpointing for internal baselines.
    # This is especially important for LoRA with a frozen backbone, where
    # checkpointing may interact poorly with non-grad inputs.
    try:
        root = get_root_module(model)
        if hasattr(root, "encoder") and hasattr(root.encoder, "gradient_checkpointing_disable"):
            root.encoder.gradient_checkpointing_disable()
        elif hasattr(root, "gradient_checkpointing_disable"):
            root.gradient_checkpointing_disable()
    except Exception as e:
        print(f"Warning: failed to disable gradient checkpointing: {repr(e)}")

    model.task_name = "classification" if hasattr(model, "task_name") else getattr(model, "task_name", None)

    if args.mode == "linear":
        set_requires_grad_all(model, False)
        trainable_head = set_head_trainable(model, True)
        print(f"Mode linear: frozen backbone, trainable head params={trainable_head}")

    elif args.mode == "full":
        set_requires_grad_all(model, True)
        print("Mode full: all MOMENT parameters trainable.")

    elif args.mode == "lora":
        set_requires_grad_all(model, False)
        n_lora = inject_lora(
            model,
            target_modules=args.lora_targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        trainable_head = set_head_trainable(model, True)
        print(f"Mode lora: injected LoRA modules={n_lora}, trainable head params={trainable_head}")
        if n_lora == 0:
            print("WARNING: no LoRA modules were injected. Check module names with --print_modules.")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # LoRA modules may be injected after the original model has already been
    # moved to device. Move once more so newly created parameters are on device.
    model = model.to(device)
    return model



def _patch_timeseries_outputs_for_logits(model) -> None:
    """
    Some local MOMENT versions call:
        TimeseriesOutputs(..., logits=logits, ...)
    while the installed TimeseriesOutputs class does not accept logits.
    We patch the symbol inside classify().__globals__ with a lightweight
    compatible container. This avoids editing MOMENT source files.
    """
    class CompatTimeseriesOutputs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getitem__(self, key):
            return getattr(self, key)

        def get(self, key, default=None):
            return getattr(self, key, default)

    modules_to_patch = []

    if hasattr(model, "classify"):
        modules_to_patch.append(model)

    root = get_root_module(model)
    if root is not model and hasattr(root, "classify"):
        modules_to_patch.append(root)

    for m in modules_to_patch:
        try:
            glb = m.classify.__globals__
            glb["TimeseriesOutputs"] = CompatTimeseriesOutputs
        except Exception:
            pass


def get_logits(model, x: torch.Tensor) -> torch.Tensor:
    _patch_timeseries_outputs_for_logits(model)

    # Official MOMENTPipeline and local variants usually expose classify().
    try:
        out = model.classify(x_enc=x)
    except TypeError:
        try:
            out = model.classify(x_enc=x, input_mask=None)
        except Exception:
            out = model(x_enc=x)
    except AttributeError:
        out = model(x_enc=x)

    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if torch.is_tensor(out):
        return out

    raise RuntimeError(f"Cannot extract logits from output type {type(out)}")

def batch_to_xy(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True).long()
        return x, y
    raise TypeError("Expected batch to be tuple/list.")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], num_classes: int):
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    if y_prob is not None:
        try:
            if num_classes == 2 and y_prob.shape[1] >= 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                out["average_precision"] = float(average_precision_score(y_true, y_prob[:, 1]))
            elif num_classes > 2 and y_prob.shape[1] == num_classes:
                y_bin = label_binarize(y_true, classes=list(range(num_classes)))
                out["roc_auc"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
                out["average_precision"] = float(average_precision_score(y_bin, y_prob, average="macro"))
        except Exception:
            pass
    return out


@torch.no_grad()
def evaluate(model, loader, device: torch.device, num_classes: int):
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        x, y = batch_to_xy(batch, device)
        logits = get_logits(model, x)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.argmax(prob, dim=-1)
        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
        probs.append(prob.detach().cpu().numpy())
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    y_prob = np.concatenate(probs, axis=0) if probs else None
    return compute_metrics(y_true, y_pred, y_prob, num_classes)


def train_one_fold(args, cfg, fold_idx: int, train_loader, val_loader, test_loader, device):
    model = build_moment_model(args, cfg, device)

    if args.print_modules:
        root = get_root_module(model)
        for name, module in root.named_modules():
            if isinstance(module, nn.Linear):
                print(name, module)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Check mode/model.")

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_bacc = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x, y = batch_to_xy(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = get_logits(model, x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device, cfg["num_classes"])
        val_bacc = val_metrics.get("balanced_accuracy", 0.0)
        scheduler.step(val_bacc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={np.mean(losses):.4f} | "
            f"val_acc={val_metrics.get('accuracy', float('nan')):.4f} | "
            f"val_bacc={val_bacc:.4f} | "
            f"val_f1={val_metrics.get('f1_macro', float('nan')):.4f}",
            flush=True,
        )

        if val_bacc > best_val_bacc + 1e-8:
            best_val_bacc = val_bacc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.early_stop:
            print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best val_bacc={best_val_bacc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, cfg["num_classes"])

    # Clean GPU memory aggressively between folds.
    del model
    torch.cuda.empty_cache()

    return {
        "fold": fold_idx + 1,
        "status": "ok",
        "best_epoch": best_epoch,
        "best_val_bacc": best_val_bacc,
        "test_metrics": test_metrics,
    }


def aggregate(fold_results: List[Dict[str, Any]]):
    vals = {}
    for r in fold_results:
        if r.get("status") != "ok":
            continue
        for k, v in r.get("test_metrics", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.setdefault(k, []).append(float(v))
    means = {k: float(np.mean(v)) for k, v in vals.items()}
    stds = {k: float(np.std(v)) for k, v in vals.items()}
    return means, stds


def save_result(path: str, obj: Dict[str, Any]):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["APAVA", "SleepEDF", "REFED"])
    parser.add_argument("--mode", required=True, choices=["linear", "full", "lora"])
    parser.add_argument("--model_path", default="./MOMENT-1-small")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default: linear/lora 1e-4, full 5e-5")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=None)
    parser.add_argument("--output_dir", default="./results_moment_internal")
    parser.add_argument("--reduction", default="concat", choices=["concat", "mean"])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # LoRA options.
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q,v")
    parser.add_argument("--print_modules", action="store_true")

    args = parser.parse_args()

    if args.lr is None:
        args.lr = 5e-5 if args.mode == "full" else 1e-4

    cfg = DATASET_CFG[args.dataset]
    set_all_seeds(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(json.dumps(vars(args), indent=2), flush=True)
    print(f"Device: {device}", flush=True)

    loaders = make_loaders(args.dataset, cfg, args.seed, args.batch_size, args.num_workers)

    start = args.start_fold
    end = args.end_fold if args.end_fold is not None else len(loaders)
    end = min(end, len(loaders))

    fold_results = []
    for fold_idx in range(start, end):
        print("=" * 70, flush=True)
        print(f"MOMENT internal baseline | dataset={args.dataset} | mode={args.mode} | fold {fold_idx+1}/{len(loaders)}", flush=True)
        print("=" * 70, flush=True)

        train_loader, val_loader, test_loader = loaders[fold_idx]
        try:
            r = train_one_fold(args, cfg, fold_idx, train_loader, val_loader, test_loader, device)
            print(f"Fold {fold_idx+1} test metrics: {r['test_metrics']}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"fold": fold_idx + 1, "status": "failed", "error": repr(e)}
            print(f"Fold {fold_idx+1} FAILED: {repr(e)}", flush=True)
        fold_results.append(r)

        partial = {
            "dataset": args.dataset,
            "model": f"MOMENT-{args.mode}",
            "mode": args.mode,
            "seed": args.seed,
            "config": vars(args),
            "fold_results": fold_results,
        }
        save_result(
            os.path.join(args.output_dir, f"moment_{args.mode}_{args.dataset}_seed{args.seed}_partial.pkl"),
            partial,
        )

    means, stds = aggregate(fold_results)
    final = {
        "dataset": args.dataset,
        "model": f"MOMENT-{args.mode}",
        "mode": args.mode,
        "seed": args.seed,
        "config": vars(args),
        "fold_results": fold_results,
        "metrics_mean": means,
        "metrics_std": stds,
    }

    final_path = os.path.join(args.output_dir, f"moment_{args.mode}_{args.dataset}_seed{args.seed}.pkl")
    save_result(final_path, final)

    ok = sum(1 for r in fold_results if r.get("status") == "ok")
    print("=" * 70, flush=True)
    print(f"MOMENT-{args.mode} Summary | dataset={args.dataset} | folds={ok}/{end-start}", flush=True)
    for k in sorted(means):
        print(f"  {k:20s}: {means[k]:.4f} ± {stds.get(k, 0.0):.4f}", flush=True)
    print(f"Saved: {final_path}", flush=True)


if __name__ == "__main__":
    main()
