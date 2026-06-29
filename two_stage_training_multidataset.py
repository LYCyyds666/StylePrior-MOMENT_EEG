"""
two_stage_training_multidataset.py
====================================
SS-MOMENT 训练脚本，支持 APAVA / SleepEDF / REFED 三个数据集
通过命令行参数切换，无需修改代码

用法：
    python two_stage_training_multidataset.py --dataset APAVA
    python two_stage_training_multidataset.py --dataset SleepEDF
    python two_stage_training_multidataset.py --dataset REFED
"""

import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('./MoE_moment')
import numpy as np

from MoE_moment.momentfm.models.SS_MOMENT import SageStreamPipeline
from MoE_moment.momentfm.models.layers.SA_MoE import StyleAdaptor
from utils import set_all_seeds, compute_comprehensive_metrics, \
    print_validation_results, clear_gpu_memory

# ── 命令行参数 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="APAVA",
                    choices=["APAVA", "SleepEDF", "REFED"])
args = parser.parse_args()

# ── 全局配置 ───────────────────────────────────────────────────────────────────
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = args.dataset
model_name   = "MOMENT-1-small"
seed         = 2025
batch_size   = 32
reduction    = "concat"
epochs       = 30
learning_rate = 5e-5
weight_decay  = 1e-5
early_stop    = 5
aux_loss_weight = 0.001

enable_tta_in_kfold = True
tta_method          = "STSA"
tta_learning_rate   = 5e-4
tta_batch_size      = 64

# ── 数据集参数 ─────────────────────────────────────────────────────────────────
DATASET_CFG = {
    "APAVA": dict(
        seq_len=256, input_channels=16, num_classes=2,
        num_subjects=23, k_folds=5, data_dir=None,
        sampling_rate=256.0,
    ),
    "SleepEDF": dict(
        seq_len=256, input_channels=16, num_classes=5,
        num_subjects=20, k_folds=20, data_dir="./datasets/sleepedf",
        sampling_rate=256.0,
    ),
    "REFED": dict(
        seq_len=256, input_channels=16, num_classes=2,
        num_subjects=32, k_folds=32, data_dir="./datasets/refed",
        sampling_rate=256.0,
    ),
}

cfg            = DATASET_CFG[dataset_name]
seq_len        = cfg["seq_len"]
input_channels = cfg["input_channels"]
num_classes    = cfg["num_classes"]
num_subjects   = cfg["num_subjects"]
k_folds        = cfg["k_folds"]
data_dir       = cfg["data_dir"]
sampling_rate  = cfg["sampling_rate"]

decoupling_config = {
    'shared_config': {
        'num_experts':    5,
        'top_k':          2,
        'dropout':        0.1,
        'freq_learning_mode':    'lightweight_biomedical_filter',
        'routing_strategy':      'simple',
        'expert_dim_ratio':      1/8,
        'max_freq':              100.0,
        'sampling_rate':         sampling_rate,
        'aux_loss_weight':       1.0,
        'enable_shared_backbone_hypernetwork': True,
        'num_subjects':          num_subjects,
        'subject_embedding_dim': 64,
        'expert_embedding_dim':  32,
        'hyper_expert_hidden_dim': 64,
        'num_channels':          input_channels,
        'moe_conditioning_dim':  64,
    },
}

set_all_seeds(seed)

print(f"\n{'='*55}")
print(f"SS-MOMENT  |  Dataset: {dataset_name}")
print(f"Classes: {num_classes}  |  Subjects: {num_subjects}  |  K-Folds: {k_folds}")
print(f"{'='*55}\n")


# ── 数据加载 ───────────────────────────────────────────────────────────────────

def load_k_fold_data():
    if dataset_name == "APAVA":
        from preprocessing import apava_k_fold_split
        print("Loading APAVA k-fold (cross-subject)...")
        fold_datasets = apava_k_fold_split(k=k_folds, random_state=seed, use_cache=True)

    elif dataset_name == "SleepEDF":
        from preprocessing_sleepedf import sleepedf_loso_split
        print("Loading SleepEDF LOSO (cross-subject)...")
        fold_datasets = sleepedf_loso_split(data_dir)

    elif dataset_name == "REFED":
        from preprocessing_refed import refed_loso_split
        print("Loading REFED LOSO (cross-subject)...")
        fold_datasets = refed_loso_split(data_dir)

    fold_loaders = []
    for fold_idx, (train_dataset, test_dataset) in enumerate(fold_datasets):
        train_ds, val_ds = split_dataset_by_subject(
            train_dataset, train_ratio=0.75, random_state=seed)
        fold_loaders.append((
            DataLoader(train_ds,    batch_size=batch_size, shuffle=True,
                       num_workers=4, drop_last=False, pin_memory=True),
            DataLoader(val_ds,      batch_size=batch_size, shuffle=False,
                       num_workers=4, drop_last=False, pin_memory=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, drop_last=False, pin_memory=True),
        ))
    return fold_loaders


def split_dataset_by_subject(dataset, train_ratio=0.75, random_state=42):
    all_sids    = np.array(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
    np.random.seed(random_state)
    np.random.shuffle(unique_sids)
    n_train   = int(len(unique_sids) * train_ratio)
    train_set = set(unique_sids[:n_train])
    train_idx = [i for i in range(len(dataset)) if all_sids[i] in train_set]
    val_idx   = [i for i in range(len(dataset)) if all_sids[i] not in train_set]
    print(f"  Subject split: {n_train} train / {len(unique_sids)-n_train} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ── 模型构建 ───────────────────────────────────────────────────────────────────

def build_model():
    model = SageStreamPipeline.from_pretrained(
        model_path="./" + model_name,
        decoupling_config=decoupling_config,
        model_kwargs={
            "task_name":                "classification",
            "n_channels":               input_channels,
            "num_class":                num_classes,
            "freeze_embedder":          True,
            "freeze_encoder":           True,
            "freeze_head":              False,
            "seq_len":                  seq_len,
            "reduction":                reduction,
            "add_positional_embedding": False,
        }
    ).to(device)
    model.task_name = "classification"
    model.set_training_stage("source_domain")
    return model


# ── 训练 ───────────────────────────────────────────────────────────────────────

def train_one_fold(train_loader, val_loader):
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_val_bacc  = -1.0
    no_improve     = 0
    best_state     = None

    for epoch in range(epochs):
        model.train()
        tr_p, tr_l, tr_pr = [], [], []

        for bd in train_loader:
            if len(bd) == 3:
                eeg, lbl, sid = bd[0].to(device), bd[1].to(device), bd[2].to(device)
            else:
                eeg, lbl, sid = bd[0].to(device), bd[1].to(device), None

            optimizer.zero_grad()
            out    = model.classify(x_enc=eeg, subject_ids=sid)
            logits = out.logits if hasattr(out, 'logits') else out
            aux    = getattr(out, 'aux_loss', 0.0)
            loss   = criterion(logits, lbl)
            if isinstance(aux, torch.Tensor) and aux.numel() > 0:
                loss = loss + aux_loss_weight * aux
            loss.backward()
            optimizer.step()

            for blk in model.model.encoder.block:
                if hasattr(blk, 'clear_aux_losses'):
                    blk.clear_aux_losses()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            tr_p.extend(preds.detach().cpu().numpy())
            tr_l.extend(lbl.detach().cpu().numpy())
            if num_classes == 2:
                tr_pr.extend(probs[:, 1].detach().cpu().numpy())
            else:
                tr_pr.append(probs.detach().cpu().numpy())

        tr_arr = np.array(tr_pr) if num_classes == 2 else np.concatenate(tr_pr, 0)
        tr_m   = compute_comprehensive_metrics(tr_l, tr_p, tr_arr, num_classes)

        model.eval()
        vl_p, vl_l, vl_pr = [], [], []
        with torch.no_grad():
            for bd in val_loader:
                if len(bd) == 3:
                    eeg, lbl, sid = bd[0].to(device), bd[1].to(device), bd[2].to(device)
                else:
                    eeg, lbl, sid = bd[0].to(device), bd[1].to(device), None
                out    = model.classify(x_enc=eeg, subject_ids=sid)
                logits = out.logits if hasattr(out, 'logits') else out
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(logits, dim=1)
                vl_p.extend(preds.detach().cpu().numpy())
                vl_l.extend(lbl.detach().cpu().numpy())
                if num_classes == 2:
                    vl_pr.extend(probs[:, 1].detach().cpu().numpy())
                else:
                    vl_pr.append(probs.detach().cpu().numpy())

        vl_arr = np.array(vl_pr) if num_classes == 2 else np.concatenate(vl_pr, 0)
        vl_m   = compute_comprehensive_metrics(vl_l, vl_p, vl_arr, num_classes)

        scheduler.step(vl_m['balanced_accuracy'])
        print(f"Epoch {epoch+1}/{epochs}: Train: ", end="")
        print_validation_results(tr_m)
        print("Val: ", end="")
        print_validation_results(vl_m)

        if vl_m['balanced_accuracy'] > best_val_bacc:
            best_val_bacc, no_improve = vl_m['balanced_accuracy'], 0
            best_state = {
                'epoch':      epoch + 1,
                'state_dict': {k: v.clone() for k, v in model.state_dict().items()},
                'val_metrics': vl_m,
            }
        else:
            no_improve += 1
        if no_improve >= early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_state


# ── 评估 ───────────────────────────────────────────────────────────────────────

def evaluate_on_test_set(model, loader):
    model.eval()
    p, l, pr = [], [], []
    with torch.no_grad():
        for bd in loader:
            if len(bd) == 3:
                eeg, lbl, sid = bd[0].to(device), bd[1], bd[2].to(device)
            else:
                eeg, lbl, sid = bd[0].to(device), bd[1], None
            out    = model.classify(x_enc=eeg, subject_ids=sid)
            logits = out.logits if hasattr(out, 'logits') else out
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
            p.extend(preds.detach().cpu().numpy())
            l.extend(lbl.numpy())
            if num_classes == 2:
                pr.extend(probs[:, 1].detach().cpu().numpy())
            else:
                pr.append(probs.detach().cpu().numpy())

    pa = np.array(pr) if num_classes == 2 else np.concatenate(pr, 0)
    m  = compute_comprehensive_metrics(l, p, pa, num_classes)
    m['predictions']   = p
    m['true_labels']   = l
    m['probabilities'] = pa
    return m


def initialize_unknown_subject_embeddings(model, train_sids, test_sids):
    unk = set(test_sids) - set(train_sids)
    if not unk:
        return
    for blk in model.model.encoder.block:
        if hasattr(blk, 'shared_knowledge') and \
           hasattr(blk.shared_knowledge, 'subject_embedding') and \
           blk.shared_knowledge.subject_embedding is not None:
            se = blk.shared_knowledge.subject_embedding
            with torch.no_grad():
                mean_e = se.weight[[s - 1 for s in train_sids]].mean(0)
                for u in unk:
                    se.weight[u - 1].copy_(mean_e)


# ── STSA 测试时自适应 ──────────────────────────────────────────────────────────

def STSA(model, test_loader, tta_lr=5e-4, tta_steps_per_batch=1, tta_batch_size=64):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    lids = [blk.shared_knowledge.layer_id
            for blk in model.model.encoder.block
            if hasattr(blk, 'shared_knowledge') and
            hasattr(blk.shared_knowledge, 'layer_id')]
    if not lids:
        return None

    ada = StyleAdaptor(num_channels=input_channels,
                       feature_dim=model.model.config.d_model).to(device)
    ada.train()

    for blk in model.model.encoder.block:
        if hasattr(blk, 'shared_knowledge') and \
           hasattr(blk.shared_knowledge, 'switch_to_STSA'):
            blk.shared_knowledge.switch_to_STSA(ada)

    opt = optim.Adam(ada.parameters(), lr=tta_lr)

    tta_loader = (
        DataLoader(test_loader.dataset, batch_size=tta_batch_size,
                   shuffle=False, drop_last=False,
                   num_workers=test_loader.num_workers,
                   pin_memory=test_loader.pin_memory)
        if tta_batch_size != test_loader.batch_size else test_loader
    )

    all_p, all_l, all_pr = [], [], []

    for bd in tta_loader:
        if len(bd) == 3:
            inputs, _, sid = bd[0].to(device), bd[1], bd[2].to(device)
        else:
            inputs, sid = bd[0].to(device), None

        for _ in range(tta_steps_per_batch):
            with torch.enable_grad():
                opt.zero_grad()
                out    = model.classify(x_enc=inputs, subject_ids=sid)
                logits = out.logits if hasattr(out, 'logits') else out

                confidence_weights = []
                for blk in model.model.encoder.block:
                    if not (hasattr(blk, 'shared_knowledge') and
                            hasattr(blk.shared_knowledge, 'get_STSA_tta_features')):
                        continue
                    raw_f, norm_f, gamma, beta = blk.shared_knowledge.get_STSA_tta_features()
                    if raw_f is None:
                        continue
                    B, C = gamma.shape[0], gamma.shape[1]
                    rf4  = raw_f.view(B, C, raw_f.shape[1], raw_f.shape[2])
                    eps  = 1e-8
                    t_err_m = (rf4.mean(2) - beta).abs()  / (rf4.mean(2).abs()  + eps)
                    t_err_s = (rf4.std(2)  - gamma).abs() / (rf4.std(2).abs()   + eps)
                    t_disc  = (t_err_m + t_err_s).mean(-1)
                    s_err_m = (rf4.mean(1) - beta.mean(1,  keepdim=True)).abs() / \
                               (rf4.mean(1).abs() + eps)
                    s_err_s = (rf4.std(1)  - gamma.mean(1, keepdim=True)).abs() / \
                               (rf4.std(1).abs()  + eps)
                    s_disc  = (s_err_m + s_err_s).mean(-1)
                    confidence_weights.append(
                        (t_disc.mean(1) + s_disc.mean(1)) / 2)

                if confidence_weights:
                    final_conf = torch.stack(confidence_weights).mean(0)
                else:
                    final_conf = torch.ones(inputs.shape[0], device=device)

                with torch.no_grad():
                    pl = torch.argmax(logits, 1)
                loss = (final_conf * F.cross_entropy(
                    logits, pl, reduction='none')).mean()
                loss.backward()
                opt.step()

        with torch.no_grad():
            eo = model.classify(x_enc=inputs, subject_ids=sid)
            el = eo.logits if hasattr(eo, 'logits') else eo
            ep = torch.softmax(el, 1)
            all_p.append(torch.argmax(el, 1).cpu())
            all_l.append(bd[1].cpu())
            if num_classes == 2:
                all_pr.append(ep[:, 1].cpu())
            else:
                all_pr.append(ep.cpu())

    for blk in model.model.encoder.block:
        if hasattr(blk, 'shared_knowledge') and \
           hasattr(blk.shared_knowledge, 'switch_to_pretrain_mode'):
            blk.shared_knowledge.switch_to_pretrain_mode()

    fp = torch.cat(all_p).numpy()
    fl = torch.cat(all_l).numpy()
    fpr = torch.cat(all_pr).numpy() if num_classes == 2 \
        else torch.cat(all_pr, 0).numpy()

    m = compute_comprehensive_metrics(fl, fp, fpr, num_classes)
    del ada, opt
    return {'overall_metrics': m, 'predictions': fp,
            'true_labels': fl, 'probabilities': fpr}


# ── K-Fold 主循环 ──────────────────────────────────────────────────────────────

def k_fold_cross_validation():
    fold_loaders     = load_k_fold_data()
    all_fold_results = []

    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
        print(f"\n{'='*55}")
        print(f"Fold {fold_idx+1}/{k_folds}  |  Dataset: {dataset_name}")
        print(f"{'='*55}")
        set_all_seeds(seed)

        best_state = train_one_fold(train_loader, val_loader)
        if best_state is None:
            all_fold_results.append({
                'fold': fold_idx+1, 'status': 'failed',
                'test_metrics': None, 'tta_metrics': None})
            continue

        model = build_model()
        model.load_state_dict(best_state['state_dict'])

        # 初始化未见受试者的 embedding
        tr_s = [x for b in train_loader if len(b) == 3 for x in b[2].tolist()]
        te_s = [x for b in test_loader  if len(b) == 3 for x in b[2].tolist()]
        initialize_unknown_subject_embeddings(
            model, sorted(set(tr_s)), sorted(set(te_s)))

        # Baseline 评测
        test_metrics = evaluate_on_test_set(model, test_loader)
        print_validation_results(test_metrics, fold_idx+1,
                                 f"Fold {fold_idx+1} Baseline: ")

        # TTA 评测
        tta_metrics = None
        if enable_tta_in_kfold and tta_method == "STSA":
            tta_result = STSA(model, test_loader,
                              tta_lr=tta_learning_rate,
                              tta_steps_per_batch=1,
                              tta_batch_size=tta_batch_size)
            if tta_result:
                tta_metrics = tta_result['overall_metrics']
                print_validation_results(tta_metrics, fold_idx+1,
                                         f"Fold {fold_idx+1} TTA: ")

        all_fold_results.append({
            'fold':          fold_idx + 1,
            'test_metrics':  test_metrics,
            'tta_metrics':   tta_metrics,
            'train_metrics': best_state.get('val_metrics', {}),
            'seed':          seed,
        })

        del model
        clear_gpu_memory()

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    successful = [r for r in all_fold_results if r.get('status') != 'failed']
    if not successful:
        print("No successful folds completed")
        return

    print(f"\n{'='*55}")
    print(f"=== K-Fold Cross Validation Results ===")
    print(f"Dataset: {dataset_name}, K={k_folds}, Seed={seed}")
    print(f"Completed folds: {len(successful)}/{k_folds}")

    def _agg(key):
        vals = [r[key] for r in successful if r.get(key)]
        m, s = {}, {}
        for r in vals:
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    m.setdefault(k, []).append(v)
        for k, v in m.items():
            s[k] = np.std(v)
            m[k] = np.mean(v)
        return m, s

    bm, bs = _agg('test_metrics')
    def _p(d, k): return f"{d.get(k,0.0):.4f}"
    def _ps(m, s, k): return f"{m.get(k,0.0):.4f} ± {s.get(k,0.0):.4f}"

    print("\n🏆 Baseline Metrics:")
    print(f"  Accuracy:          {_ps(bm,bs,'accuracy')}")
    print(f"  Balanced Accuracy: {_ps(bm,bs,'balanced_accuracy')}")
    print(f"  F1 Score (Macro):  {_ps(bm,bs,'f1_macro')}")
    print(f"  Precision (Macro): {_ps(bm,bs,'precision_macro')}")
    print(f"  Recall (Macro):    {_ps(bm,bs,'recall_macro')}")
    if 'roc_auc'           in bm: print(f"  ROC AUC:           {_ps(bm,bs,'roc_auc')}")
    if 'average_precision' in bm: print(f"  Avg Prec:          {_ps(bm,bs,'average_precision')}")

    tta_folds = [r for r in successful if r.get('tta_metrics')]
    if tta_folds:
        tm, ts = _agg('tta_metrics')
        print("\n🚀 TTA Metrics:")
        print(f"  Accuracy:          {_p(tm,'accuracy')}")
        print(f"  Balanced Accuracy: {_p(tm,'balanced_accuracy')}")
        print(f"  F1 Score (Macro):  {_p(tm,'f1_macro')}")
        print(f"  Precision (Macro): {_p(tm,'precision_macro')}")
        print(f"  Recall (Macro):    {_p(tm,'recall_macro')}")
        if 'roc_auc'           in tm: print(f"  ROC AUC:           {_p(tm,'roc_auc')}")
        if 'average_precision' in tm: print(f"  Avg Prec:          {_p(tm,'average_precision')}")
    else:
        tm = None

    os.makedirs("./results", exist_ok=True)
    sp = f"./results/ss_moment_{dataset_name}_seed{seed}.pkl"
    with open(sp, 'wb') as f:
        pickle.dump({
            'dataset':          dataset_name,
            'seed':             seed,
            'k_folds':          k_folds,
            'completed_folds':  len(successful),
            'baseline_metrics': bm,
            'tta_metrics':      tm,
            'fold_results':     all_fold_results,
        }, f)
    print(f"\n结果保存至: {sp}")
    return bm, tm


def main():
    k_fold_cross_validation()


if __name__ == "__main__":
    main()
