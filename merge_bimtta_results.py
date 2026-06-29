"""
merge_bimtta_results.py
=======================
合并多个并行进程的 BiMTTA 结果，生成完整的汇总指标

用法（8个进程跑完后）：
    python merge_bimtta_results.py --dataset REFED

会自动找 ./results/bimtta_REFED_fold*_seed2025.pkl 并合并
"""

import argparse
import os
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="REFED",
                    choices=["APAVA", "BCI2a", "SleepEDF", "REFED"])
parser.add_argument("--seed", type=int, default=2025)
args = parser.parse_args()

dataset_name = args.dataset
seed         = args.seed
results_dir  = "./results"

# ── 找所有对应的 pkl 文件 ──────────────────────────────────────────────────────
pattern = f"bimtta_{dataset_name}_fold"
pkl_files = sorted([
    f for f in os.listdir(results_dir)
    if f.startswith(pattern) and f.endswith(f"_seed{seed}.pkl")
])

if not pkl_files:
    print(f"找不到任何 {pattern}*_seed{seed}.pkl 文件")
    print(f"请先运行所有并行进程")
    exit(1)

print(f"找到 {len(pkl_files)} 个结果文件：")
for f in pkl_files:
    print(f"  {f}")

# ── 合并所有 fold 结果 ─────────────────────────────────────────────────────────
all_fold_results = []
k_folds = None

for fname in pkl_files:
    fpath = os.path.join(results_dir, fname)
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    if k_folds is None:
        k_folds = data['k_folds']

    fold_results = data.get('fold_results', [])
    all_fold_results.extend(fold_results)
    print(f"  {fname}: {len(fold_results)} folds loaded")

# 按 fold 编号排序
all_fold_results.sort(key=lambda x: x.get('fold', 0))

successful = [r for r in all_fold_results if r.get('status') != 'failed'
              and r.get('test_metrics') is not None]

print(f"\n合并后：{len(successful)}/{k_folds} folds 成功")

# ── 计算汇总指标 ───────────────────────────────────────────────────────────────
def _agg(metric_key):
    vals = [r[metric_key] for r in successful if r.get(metric_key)]
    all_m, std_m = {}, {}
    for r in vals:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                all_m.setdefault(k, []).append(v)
    for k, v in all_m.items():
        std_m[k] = np.std(v)
        all_m[k] = np.mean(v)
    return all_m, std_m

mean_m, std_m = _agg('test_metrics')
def _p(key): return f"{mean_m.get(key,0.0):.4f} ± {std_m.get(key,0.0):.4f}"

print(f"\n{'='*55}")
print(f"=== BiMTTA K-Fold Results (Merged) ===")
print(f"Dataset: {dataset_name}, K={k_folds}, Seed={seed}")
print(f"Completed folds: {len(successful)}/{k_folds}")

print("\n🏆 Baseline Metrics:")
print(f"  Accuracy:          {_p('accuracy')}")
print(f"  Balanced Accuracy: {_p('balanced_accuracy')}")
print(f"  F1 Score (Macro):  {_p('f1_macro')}")
print(f"  Precision (Macro): {_p('precision_macro')}")
print(f"  Recall (Macro):    {_p('recall_macro')}")
if 'roc_auc'           in mean_m: print(f"  ROC AUC:           {_p('roc_auc')}")
if 'average_precision' in mean_m: print(f"  Avg Prec:          {_p('average_precision')}")

tta_m, tta_std = _agg('tta_metrics')
print("\n🚀 TTA Metrics:")
print(f"  Accuracy:          {tta_m.get('accuracy',          0.0):.4f}")
print(f"  Balanced Accuracy: {tta_m.get('balanced_accuracy', 0.0):.4f}")
print(f"  F1 Score (Macro):  {tta_m.get('f1_macro',          0.0):.4f}")
print(f"  Precision (Macro): {tta_m.get('precision_macro',   0.0):.4f}")
print(f"  Recall (Macro):    {tta_m.get('recall_macro',      0.0):.4f}")
if 'roc_auc'           in tta_m: print(f"  ROC AUC:           {tta_m.get('roc_auc',           0.0):.4f}")
if 'average_precision' in tta_m: print(f"  Avg Prec:          {tta_m.get('average_precision',  0.0):.4f}")

# ── 保存合并后的完整结果 ──────────────────────────────────────────────────────
save_path = os.path.join(results_dir, f"bimtta_{dataset_name}_seed{seed}.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'dataset':          dataset_name,
        'seed':             seed,
        'k_folds':          k_folds,
        'completed_folds':  len(successful),
        'baseline_metrics': mean_m,
        'tta_metrics':      tta_m,
        'fold_results':     all_fold_results,
    }, f)
print(f"\n合并结果保存至: {save_path}")
