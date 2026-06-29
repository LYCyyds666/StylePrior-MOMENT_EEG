"""
ablation_sleepedf.py
====================
消融实验脚本，基于 SleepEDF 数据集
基于修复后的 SA_MoE.py（三个 bug 已修复）

变体：
    full        : 完整模型，结果与 two_stage_training_multidataset.py SleepEDF 完全一致
    wo_sisl     : w/o SISL  gamma=1, beta=0，风格变换退化为恒等变换
    wo_hse      : w/o HSE   去掉 SharedHyperExpert 的贡献
    wo_stsa     : w/o STSA  去掉测试时自适应

用法：
    python ablation_sleepedf.py --variant full
    python ablation_sleepedf.py --variant wo_sisl
    python ablation_sleepedf.py --variant wo_hse
    python ablation_sleepedf.py --variant wo_stsa
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('./MoE_moment')

from MoE_moment.momentfm.models.SS_MOMENT import SageStreamPipeline
from MoE_moment.momentfm.models.layers.SA_MoE import StyleAdaptor, SA_MoE
from utils import set_all_seeds, compute_comprehensive_metrics, \
    print_validation_results, clear_gpu_memory

# ── 命令行参数 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="full",
                    choices=["full", "wo_sisl", "wo_hse", "wo_stsa"])
parser.add_argument("--data_dir", default="./datasets/sleepedf")
args = parser.parse_args()

# ── 全局配置 ───────────────────────────────────────────────────────────────────
device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
variant        = args.variant
data_dir       = args.data_dir
seed           = 2025
model_name     = "MOMENT-1-small"
seq_len        = 256
input_channels = 16
num_classes    = 5
num_subjects   = 20
sampling_rate  = 256.0
k_folds        = 20
batch_size     = 32
epochs         = 30
learning_rate  = 5e-5
weight_decay   = 1e-5
early_stop     = 5
reduction      = "concat"
aux_loss_weight = 0.001
enable_stsa    = (variant != "wo_stsa")

print(f"\n{'='*55}")
print(f"消融变体: {variant}  |  enable_stsa: {enable_stsa}")
print(f"{'='*55}\n")

# ── decoupling_config（所有变体保持一致）─────────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════════════════
# 消融 Monkey-Patch
# ════════════════════════════════════════════════════════════════════════════════

# ── w/o SISL：让 gamma=1, beta=0，风格变换退化为恒等变换 ──────────────────────
# 实现：patch _forward_with_shared_backbone_hypernetwork 的非 TTA 分支
# 只改 gamma/beta 的计算，其他完全不变（保证和 full 代码路径一致）

if variant == "wo_sisl":
    _orig_fwd = SA_MoE._forward_with_shared_backbone_hypernetwork
    _SOFTPLUS_INV_ONE = 0.5413  # softplus(0.5413) + 1e-8 ≈ 1.0
    _GAMMA_VAL = float(torch.nn.functional.softplus(torch.tensor(_SOFTPLUS_INV_ONE)).item()) + 1e-8

    def _fwd_wo_sisl(self, hidden_states, subject_ids, residual,
                     channels_times_batch, seq_len_arg, d_model, actual_batch_size):
        # TTA 模式：不改，让 STSA 正常工作
        if self.in_tta_mode and self.tta_adaptor is not None:
            return _orig_fwd(self, hidden_states, subject_ids, residual,
                             channels_times_batch, seq_len_arg, d_model, actual_batch_size)

        # 非 TTA 模式：gamma=1, beta=0
        zero_based_ids = torch.clamp(
            subject_ids - 1, 0, self.subject_embedding.num_embeddings - 1)
        subject_embeddings = self.subject_embedding(zero_based_ids)

        if actual_batch_size is None:
            actual_batch_size = subject_embeddings.shape[0]

        B, C, S, F = actual_batch_size, self.num_channels, seq_len_arg, d_model

        global_style_context = self.shared_backbone_hypernetwork.shared_backbone(
            subject_embeddings)

        # gamma=1, beta=0：恒等变换（去掉受试者风格学习）
        gamma = torch.full((B, C, F),
                           _GAMMA_VAL,
                           device=hidden_states.device, dtype=hidden_states.dtype)
        beta  = torch.zeros(B, C, F,
                            device=hidden_states.device, dtype=hidden_states.dtype)

        # 以下和修复后的原版完全一致
        aligned_features = self._apply_style_alignment(
            hidden_states, gamma, beta,
            actual_batch_size, channels_times_batch, seq_len_arg, d_model)

        router_output = self.shared_router(aligned_features, layer_id=self.layer_id)
        if len(router_output) == 3:
            gates, indices, router_probs = router_output
            self._last_router_probs = router_probs
        else:
            gates, indices = router_output
            self._last_router_probs = None
            router_probs = F.softmax(
                gates.sum(dim=-1, keepdim=True).expand(-1, -1, self.num_experts), dim=-1)

        self._last_gates   = gates
        self._last_indices = indices

        router_probs_flat = router_probs.view(-1, self.num_experts)
        indices_flat      = indices.view(-1, self.top_k)

        selection_emb_flat = self.expert_embeddings.get_selection_embedding(
            router_probs_flat, indices_flat)

        global_ctx_flat = (global_style_context.unsqueeze(1)
                           .expand(-1, C * S, -1).contiguous()
                           .view(-1, global_style_context.shape[-1]))

        moe_head_input    = torch.cat([global_ctx_flat, selection_emb_flat], dim=-1)
        conditioning_flat = self.shared_backbone_hypernetwork.moe_head(moe_head_input)

        aligned_flat = aligned_features.view(-1, F)
        gates_flat   = gates.view(-1, self.top_k)

        final_output_flat = self._calculate_moe_output(
            aligned_flat, gates_flat, indices_flat, conditioning_flat)

        output = final_output_flat.view(channels_times_batch, seq_len_arg, d_model)
        output = residual + output

        if self.training and self.aux_loss_weight > 0:
            self.aux_losses.append(self._calculate_aux_loss(gates))

        return output

    SA_MoE._forward_with_shared_backbone_hypernetwork = _fwd_wo_sisl
    print("wo_sisl: gamma=1, beta=0 patch applied")


# ── w/o HSE：_calculate_moe_output 里去掉 hse_output ────────────────────────
# 只用 main_expert_output，不加 hse_output

if variant == "wo_hse":
    def _calc_moe_wo_hse(self, aligned_flat, gates_flat, indices_flat,
                          conditioning_vectors):
        """w/o HSE：只用标准 MoE expert，不加 SharedHyperExpert 的贡献"""
        final_output = torch.zeros_like(aligned_flat)
        for k in range(self.top_k):
            expert_indices = indices_flat[:, k]
            gate_values    = gates_flat[:, k].unsqueeze(-1)
            for expert_idx in range(self.num_experts):
                expert_mask = (expert_indices == expert_idx)
                if expert_mask.any():
                    token_indices = torch.where(expert_mask)[0]
                    if len(token_indices) > 0:
                        selected_features  = aligned_flat[token_indices]
                        expert             = self.optimized_global_pool.get_expert(expert_idx)
                        main_expert_output = expert(selected_features)
                        # 不加 hse_output（去掉 HSE 的贡献）
                        weighted_output    = gate_values[token_indices] * main_expert_output
                        final_output[token_indices] += weighted_output
        return final_output

    SA_MoE._calculate_moe_output = _calc_moe_wo_hse
    print("wo_hse: SharedHyperExpert disabled")


# ── 数据加载 ───────────────────────────────────────────────────────────────────

def load_k_fold_data():
    from preprocessing_sleepedf import sleepedf_loso_split
    print("Loading SleepEDF LOSO (cross-subject)...")
    fold_datasets = sleepedf_loso_split(data_dir)
    fold_loaders  = []
    for _, (train_dataset, test_dataset) in enumerate(fold_datasets):
        train_ds, val_ds = split_by_subject(train_dataset)
        fold_loaders.append((
            DataLoader(train_ds,     batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=False),
            DataLoader(val_ds,       batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False),
        ))
    return fold_loaders


def split_by_subject(dataset, train_ratio=0.75):
    all_sids    = np.array(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
    np.random.seed(seed)
    np.random.shuffle(unique_sids)
    n_train   = int(len(unique_sids) * train_ratio)
    train_set = set(unique_sids[:n_train])
    print(f"  Subject split: {n_train} train / {len(unique_sids)-n_train} val")
    return (
        Subset(dataset, [i for i in range(len(dataset)) if all_sids[i] in train_set]),
        Subset(dataset, [i for i in range(len(dataset)) if all_sids[i] not in train_set]),
    )


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


def get_class_weights(loader):
    from sklearn.utils.class_weight import compute_class_weight
    lbls = []
    for b in loader:
        lbls.extend(b[1].numpy())
    w = compute_class_weight('balanced', classes=np.arange(num_classes),
                              y=np.array(lbls))
    return torch.tensor(w, dtype=torch.float32).to(device)


# ── 训练（与 two_stage_training_multidataset.py 完全一致）────────────────────

def train_one_fold(train_loader, val_loader):
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    best_val_bacc, no_improve, best_state = -1.0, 0, None

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
            tr_p.extend(torch.argmax(logits, 1).cpu().numpy())
            tr_l.extend(lbl.cpu().numpy())
            tr_pr.append(torch.softmax(logits, 1).detach().cpu().numpy())

        tr_m = compute_comprehensive_metrics(
            tr_l, tr_p, np.concatenate(tr_pr, 0), num_classes)

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
                vl_p.extend(torch.argmax(logits, 1).cpu().numpy())
                vl_l.extend(lbl.cpu().numpy())
                vl_pr.append(torch.softmax(logits, 1).cpu().numpy())

        vl_m = compute_comprehensive_metrics(
            vl_l, vl_p, np.concatenate(vl_pr, 0), num_classes)
        scheduler.step(vl_m['balanced_accuracy'])

        print(f"Epoch {epoch+1}/{epochs}: Train: ", end="")
        print_validation_results(tr_m)
        print("Val: ", end="")
        print_validation_results(vl_m)

        if vl_m['balanced_accuracy'] > best_val_bacc:
            best_val_bacc, no_improve = vl_m['balanced_accuracy'], 0
            best_state = {
                'epoch':       epoch + 1,
                'state_dict':  {k: v.clone() for k, v in model.state_dict().items()},
                'val_metrics': vl_m,
            }
        else:
            no_improve += 1
        if no_improve >= early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_state


# ── 评估 ───────────────────────────────────────────────────────────────────────

def evaluate_baseline(model, loader):
    model.eval()
    p, l, pr = [], [], []
    with torch.no_grad():
        for bd in loader:
            eeg, lbl = bd[0].to(device), bd[1]
            sid = bd[2].to(device) if len(bd) == 3 else None
            out    = model.classify(x_enc=eeg, subject_ids=sid)
            logits = out.logits if hasattr(out, 'logits') else out
            p.extend(torch.argmax(logits, 1).cpu().numpy())
            l.extend(lbl.numpy())
            pr.append(torch.softmax(logits, 1).cpu().numpy())
    pa = np.concatenate(pr, 0)
    m  = compute_comprehensive_metrics(l, p, pa, num_classes)
    m['predictions'] = p
    m['true_labels']   = l
    m['probabilities'] = pa
    return m


def init_unknown_embeddings(model, train_sids, test_sids):
    unk = set(test_sids) - set(train_sids)
    if not unk:
        return
    for blk in model.model.encoder.block:
        se = getattr(getattr(blk, 'shared_knowledge', None),
                     'subject_embedding', None)
        if se is not None:
            with torch.no_grad():
                mean_e = se.weight[[s - 1 for s in train_sids]].mean(0)
                for u in unk:
                    se.weight[u - 1].copy_(mean_e)


# ── STSA（与 two_stage_training_multidataset.py 的 STSA 函数完全一致）─────────

def run_stsa(model, test_loader, tta_lr=5e-4, tta_batch_size=64):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

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
                B_C, S, feat_dim = raw_f.shape
                B, C = gamma.shape[0], gamma.shape[1]
                raw_features_4d    = raw_f.view(B, C, S, feat_dim)
                true_mean_temporal = raw_features_4d.mean(dim=2)
                true_std_temporal  = raw_features_4d.std(dim=2)
                true_mean_spatial  = raw_features_4d.mean(dim=1)
                true_std_spatial   = raw_features_4d.std(dim=1)
                prior_gamma_spatial = gamma.mean(dim=1, keepdim=True)
                prior_beta_spatial  = beta.mean(dim=1, keepdim=True)
                epsilon = 1e-8
                temporal_err_mean = (true_mean_temporal - beta).abs() \
                                    / (true_mean_temporal.abs() + epsilon)
                temporal_err_std  = (true_std_temporal  - gamma).abs() \
                                    / (true_std_temporal.abs()  + epsilon)
                temporal_discrepancy = (temporal_err_mean + temporal_err_std).mean(dim=-1)
                spatial_err_mean  = (true_mean_spatial - prior_beta_spatial).abs() \
                                    / (true_mean_spatial.abs() + epsilon)
                spatial_err_std   = (true_std_spatial  - prior_gamma_spatial).abs() \
                                    / (true_std_spatial.abs()  + epsilon)
                spatial_discrepancy  = (spatial_err_mean + spatial_err_std).mean(dim=-1)
                temporal_confidence  = temporal_discrepancy.mean(dim=1)
                spatial_confidence   = spatial_discrepancy.mean(dim=1)
                confidence_weights.append((temporal_confidence + spatial_confidence) / 2)

            if confidence_weights:
                final_confidence = torch.stack(confidence_weights).mean(dim=0)
            else:
                final_confidence = torch.ones(inputs.shape[0], device=device)

            with torch.no_grad():
                pseudo_labels = torch.argmax(logits, dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            loss    = (final_confidence * ce_loss).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            eo = model.classify(x_enc=inputs, subject_ids=sid)
            el = eo.logits if hasattr(eo, 'logits') else eo
            ep = torch.softmax(el, 1)
            all_p.extend(torch.argmax(el, 1).cpu().numpy())
            all_l.extend(bd[1].numpy())
            all_pr.append(ep.cpu().numpy())

    for blk in model.model.encoder.block:
        if hasattr(blk, 'shared_knowledge') and \
           hasattr(blk.shared_knowledge, 'switch_to_pretrain_mode'):
            blk.shared_knowledge.switch_to_pretrain_mode()

    pa = np.concatenate(all_pr, 0)
    m  = compute_comprehensive_metrics(all_l, all_p, pa, num_classes)
    del ada, opt
    return m


# ── K-Fold 主循环 ──────────────────────────────────────────────────────────────

def k_fold_cross_validation():
    fold_loaders     = load_k_fold_data()
    all_fold_results = []

    for fi, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
        print(f"\n{'='*55}")
        print(f"[{variant}] Fold {fi+1}/{k_folds}")
        print(f"{'='*55}")
        set_all_seeds(seed)

        bs = train_one_fold(train_loader, val_loader)
        if bs is None:
            all_fold_results.append({'fold': fi+1, 'status': 'failed'})
            continue

        model = build_model()
        model.load_state_dict(bs['state_dict'])

        tr_s = [x for b in train_loader if len(b) == 3 for x in b[2].tolist()]
        te_s = [x for b in test_loader  if len(b) == 3 for x in b[2].tolist()]
        init_unknown_embeddings(model, sorted(set(tr_s)), sorted(set(te_s)))

        bm = evaluate_baseline(model, test_loader)
        print(f"  Fold {fi+1} Baseline Acc: {bm['accuracy']:.4f}")

        tm = None
        if enable_stsa:
            tm = run_stsa(model, test_loader)
            if tm:
                print(f"  Fold {fi+1} TTA Acc:      {tm['accuracy']:.4f}")

        all_fold_results.append({
            'fold':             fi + 1,
            'baseline_acc':     bm['accuracy'],
            'tta_acc':          tm['accuracy'] if tm else None,
            'baseline_metrics': bm,
            'tta_metrics':      tm,
        })
        del model
        clear_gpu_memory()

    successful = [r for r in all_fold_results if r.get('status') != 'failed']
    if not successful:
        print("No successful folds")
        return

    accs     = [r['baseline_acc'] for r in successful]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    print(f"\n{'='*55}")
    print(f"[{variant}] SleepEDF Results (K={k_folds})")
    print(f"{'='*55}")
    print(f"\n🏆 Baseline Metrics:\n  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    if enable_stsa:
        ta = [r['tta_acc'] for r in successful if r['tta_acc'] is not None]
        if ta:
            print(f"\n🚀 TTA Metrics:\n  Accuracy: {np.mean(ta):.4f} ± {np.std(ta):.4f}")

    os.makedirs("./results", exist_ok=True)
    sp = f"./results/ablation_sleepedf_{variant}_seed{seed}.pkl"
    with open(sp, 'wb') as f:
        pickle.dump({
            'variant':      variant,
            'dataset':      'SleepEDF',
            'seed':         seed,
            'k_folds':      k_folds,
            'mean_acc':     mean_acc,
            'std_acc':      std_acc,
            'fold_results': all_fold_results,
        }, f)
    print(f"\n结果保存至: {sp}")
    return mean_acc, std_acc


if __name__ == "__main__":
    k_fold_cross_validation()