#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_trainable_params.py

Count total/trainable parameters for:
  - MOMENT-linear
  - MOMENT-full
  - MOMENT-LoRA
  - StylePrior-MOMENT
  - StylePrior-MOMENT STSA adapter params
  - CBraMod-full

Example:
  python count_trainable_params.py --dataset APAVA --out ./paper_tables/param_efficiency_APAVA.csv
  python count_trainable_params.py --all_datasets --out ./paper_tables/param_efficiency_all.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn


sys.path.append(".")
sys.path.append("./MoE_moment")
sys.path.append("./CBraMod")

from moment_internal_baselines_multidataset import (
    DATASET_CFG,
    build_moment_model,
    inject_lora,
    set_requires_grad_all,
    set_head_trainable,
)


def count_params(model) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
    }


class DummyArgs:
    pass


def build_args(mode: str, dataset: str, model_path: str):
    a = DummyArgs()
    a.dataset = dataset
    a.mode = mode
    a.model_path = model_path
    a.reduction = "concat"
    a.lora_targets = "q,v"
    a.lora_r = 8
    a.lora_alpha = 16
    a.lora_dropout = 0.05
    a.print_modules = False
    return a


def count_moment_variants(dataset: str, cfg: Dict[str, Any], model_path: str, device):
    rows = []
    for mode in ["linear", "full", "lora"]:
        args = build_args(mode, dataset, model_path)
        model = build_moment_model(args, cfg, device)
        c = count_params(model)
        rows.append({
            "dataset": dataset,
            "model": f"MOMENT-{mode}",
            "training_setting": {
                "linear": "frozen backbone + trainable head",
                "full": "full fine-tuning",
                "lora": "LoRA(q,v) + trainable head",
            }[mode],
            "tta_updated_params": "",
            **c,
        })
        del model
        torch.cuda.empty_cache()
    return rows


def build_styleprior(dataset: str, cfg: Dict[str, Any], model_path: str, device):
    from MoE_moment.momentfm.models.SS_MOMENT import SageStreamPipeline

    decoupling_config = {
        "shared_config": {
            "num_experts": 5,
            "top_k": 2,
            "dropout": 0.1,
            "freq_learning_mode": "lightweight_biomedical_filter",
            "routing_strategy": "simple",
            "expert_dim_ratio": 1/8,
            "max_freq": 100.0,
            "sampling_rate": cfg["sampling_rate"],
            "aux_loss_weight": 1.0,
            "enable_shared_backbone_hypernetwork": True,
            "num_subjects": cfg["num_subjects"],
            "subject_embedding_dim": 64,
            "expert_embedding_dim": 32,
            "hyper_expert_hidden_dim": 64,
            "num_channels": cfg["input_channels"],
            "moe_conditioning_dim": 64,
        }
    }

    model = SageStreamPipeline.from_pretrained(
        model_path=model_path,
        decoupling_config=decoupling_config,
        model_kwargs={
            "task_name": "classification",
            "n_channels": cfg["input_channels"],
            "num_class": cfg["num_classes"],
            "freeze_embedder": True,
            "freeze_encoder": True,
            "freeze_head": False,
            "seq_len": cfg["seq_len"],
            "reduction": "concat",
            "add_positional_embedding": False,
        },
    ).to(device)
    model.task_name = "classification"
    model.set_training_stage("source_domain")
    return model


def count_styleprior(dataset: str, cfg: Dict[str, Any], model_path: str, device):
    rows = []
    try:
        model = build_styleprior(dataset, cfg, model_path, device)
        c = count_params(model)

        # STSA adapter params.
        try:
            from MoE_moment.momentfm.models.layers.SA_MoE import StyleAdaptor
            d_model = model.model.config.d_model
            adapter = StyleAdaptor(num_channels=cfg["input_channels"], feature_dim=d_model).to(device)
            adapter_params = sum(p.numel() for p in adapter.parameters())
        except Exception:
            adapter_params = ""

        rows.append({
            "dataset": dataset,
            "model": "StylePrior-MOMENT",
            "training_setting": "frozen backbone + trainable SA-MoE + head",
            "tta_updated_params": adapter_params,
            **c,
        })
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        rows.append({
            "dataset": dataset,
            "model": "StylePrior-MOMENT",
            "training_setting": f"FAILED: {repr(e)}",
            "total_params": "",
            "trainable_params": "",
            "trainable_ratio": "",
            "tta_updated_params": "",
        })
    return rows


class CBraModClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, num_classes: int, d_model: int = 200):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        raise NotImplementedError("Parameter-count wrapper only.")


def count_cbramod(dataset: str, cfg: Dict[str, Any], cbramod_root: str, device):
    rows = []
    try:
        sys.path.insert(0, cbramod_root)
        from models.cbramod import CBraMod

        backbone = CBraMod().to(device)
        # Match our baseline: full fine-tuning + classifier.
        model = CBraModClassifier(backbone, cfg["input_channels"], cfg["num_classes"]).to(device)

        # Materialize LazyLinear with a representative feature shape if possible.
        # CBraMod output shape may vary; use official input one segment at 200 points.
        with torch.no_grad():
            dummy = torch.zeros(2, cfg["input_channels"], 1, 200, device=device)
            feat = model.backbone(dummy)
            _ = model.classifier(feat)

        for p in model.parameters():
            p.requires_grad = True

        c = count_params(model)
        rows.append({
            "dataset": dataset,
            "model": "CBraMod-full",
            "training_setting": "full fine-tuning + classifier",
            "tta_updated_params": "",
            **c,
        })
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        rows.append({
            "dataset": dataset,
            "model": "CBraMod-full",
            "training_setting": f"FAILED: {repr(e)}",
            "total_params": "",
            "trainable_params": "",
            "trainable_ratio": "",
            "tta_updated_params": "",
        })
    return rows


def fmt_int(x):
    if x == "" or x is None:
        return ""
    return f"{int(x):,}"


def fmt_ratio(x):
    if x == "" or x is None:
        return ""
    return f"{100*float(x):.3f}%"


def write_outputs(rows: List[Dict[str, Any]], out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "model",
        "training_setting",
        "total_params",
        "trainable_params",
        "trainable_ratio",
        "tta_updated_params",
    ]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    md = out.with_suffix(".md")
    with open(md, "w") as f:
        f.write("| Dataset | Model | Setting | Total Params | Trainable Params | Trainable Ratio | TTA-updated Params |\n")
        f.write("|---|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.get('dataset','')} | {r.get('model','')} | {r.get('training_setting','')} | "
                f"{fmt_int(r.get('total_params'))} | {fmt_int(r.get('trainable_params'))} | "
                f"{fmt_ratio(r.get('trainable_ratio'))} | {fmt_int(r.get('tta_updated_params'))} |\n"
            )

    print(f"Saved: {out}")
    print(f"Saved: {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["APAVA", "SleepEDF", "REFED"], default="APAVA")
    ap.add_argument("--all_datasets", action="store_true")
    ap.add_argument("--model_path", default="./MOMENT-1-small")
    ap.add_argument("--styleprior_model_path", default=None)
    ap.add_argument("--cbramod_root", default="./CBraMod")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out", default="./paper_tables/param_efficiency.csv")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    datasets = list(DATASET_CFG.keys()) if args.all_datasets else [args.dataset]
    rows = []
    for ds in datasets:
        cfg = DATASET_CFG[ds]
        print("=" * 70)
        print(f"Counting params for {ds}")
        rows.extend(count_moment_variants(ds, cfg, args.model_path, device))
        styleprior_path = args.styleprior_model_path or args.model_path
        rows.extend(count_styleprior(ds, cfg, styleprior_path, device))
        rows.extend(count_cbramod(ds, cfg, args.cbramod_root, device))

    write_outputs(rows, args.out)


if __name__ == "__main__":
    main()
