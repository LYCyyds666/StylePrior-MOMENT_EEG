"""
Aggregate StylePrior / baseline pickle results into CSV and Markdown tables.

Examples:
    python aggregate_pkl_results.py --glob './results/**/*.pkl' --out results_summary.csv
    python aggregate_pkl_results.py --glob './results_cbramod/*.pkl' --out cbramod_summary.csv
"""
from __future__ import annotations
import argparse, glob, os, pickle, csv
from typing import Any, Dict, List
import numpy as np


def load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def infer_model(path: str, obj: Dict[str, Any]) -> str:
    name = os.path.basename(path).lower()
    if 'cbramod' in name:
        mode = obj.get('mode') or obj.get('config', {}).get('mode', '')
        return 'CBraMod' + (f' ({mode})' if mode else '')
    if 'ss_moment' in name or 'styleprior' in name:
        return 'StylePrior-MOMENT'
    for m in ['eegnet', 'graphsleepnet', 'salientsleepnet', 'mmcnn', 'bimtta', 'lora', 'linear', 'fullft']:
        if m in name:
            return m
    return os.path.splitext(os.path.basename(path))[0]


def extract_summary(path: str) -> Dict[str, Any]:
    obj = load(path)
    dataset = obj.get('dataset') or obj.get('config', {}).get('dataset') or 'UNKNOWN'
    model = infer_model(path, obj)
    mode = obj.get('mode') or obj.get('config', {}).get('mode') or ''
    # Prefer TTA metrics for StylePrior if present; otherwise main/test metrics.
    mean = obj.get('metrics_mean') or obj.get('tta_metrics') or obj.get('baseline_metrics') or obj.get('test_metrics') or {}
    std = obj.get('metrics_std') or {}
    # If no aggregated std, compute from fold_results.
    if not mean and 'fold_results' in obj:
        vals = {}
        for r in obj['fold_results']:
            metrics = r.get('tta_metrics') or r.get('test_metrics')
            if not metrics: continue
            for k, v in metrics.items():
                if isinstance(v, (int,float,np.floating)):
                    vals.setdefault(k, []).append(float(v))
        mean = {k: float(np.mean(v)) for k, v in vals.items()}
        std = {k: float(np.std(v)) for k, v in vals.items()}
    row = {'file': path, 'dataset': dataset, 'model': model, 'mode': mode}
    for k in ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc', 'average_precision']:
        if k in mean:
            row[k+'_mean'] = mean[k]
            row[k+'_std'] = std.get(k, '')
    return row


def fmt(row, metric):
    m = row.get(metric+'_mean', '')
    s = row.get(metric+'_std', '')
    if m == '': return ''
    if s == '': return f"{100*float(m):.2f}"
    return f"{100*float(m):.2f} ± {100*float(s):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', default='results_summary.csv')
    args = ap.parse_args()
    paths = sorted(glob.glob(args.glob, recursive=True))
    rows = [extract_summary(p) for p in paths]
    fields = ['dataset','model','mode','accuracy_mean','accuracy_std','balanced_accuracy_mean','balanced_accuracy_std','f1_macro_mean','f1_macro_std','precision_macro_mean','precision_macro_std','recall_macro_mean','recall_macro_std','roc_auc_mean','roc_auc_std','average_precision_mean','average_precision_std','file']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    md = os.path.splitext(args.out)[0] + '.md'
    with open(md, 'w') as f:
        f.write('| Dataset | Model | Acc | BAcc | Macro-F1 | Macro-Prec | Macro-Recall |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r.get('dataset','')} | {r.get('model','')} | {fmt(r,'accuracy')} | {fmt(r,'balanced_accuracy')} | {fmt(r,'f1_macro')} | {fmt(r,'precision_macro')} | {fmt(r,'recall_macro')} |\n")
    print(f"Wrote {args.out} and {md} from {len(rows)} files")

if __name__ == '__main__':
    main()
