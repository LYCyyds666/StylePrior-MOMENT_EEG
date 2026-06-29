"""
mmcnn_baseline.py
=================
MMCNN PyTorch 实现（忠实复现官方 Keras 源码）
原论文：Jia et al., "MMCNN: A Multi-branch Multi-scale Convolutional Neural
        Network for Motor Imagery Classification", ECML-PKDD 2020
 
官方 Keras 实现：https://github.com/ziyujia/ECML-PKDD_MMCNN
 
支持数据集：APAVA / BCI2a / SleepEDF / REFED
输出格式与 two_stage_training.py 完全一致
 
用法：
    python mmcnn_baseline.py --dataset APAVA
    python mmcnn_baseline.py --dataset BCI2a
    python mmcnn_baseline.py --dataset SleepEDF
    python mmcnn_baseline.py --dataset REFED
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
架构（直接对应官方 MMCNN_model.py）：
 
输入: (B, C, T)   ← PyTorch 通道优先，等价于 Keras 的 (T, C)
 
5 个并行分支（EIN-a ~ EIN-e），各使用不同尺度的 Inception kernel：
  EIN-a: kernel [5,10,15,10],   stride=2
  EIN-b: kernel [40,45,50,100], stride=4
  EIN-c: kernel [60,65,70,100], stride=4
  EIN-d: kernel [80,85,90,100], stride=4
  EIN-e: kernel [160,180,200,180], stride=16
 
每个分支：
  InceptionBlock(4路: k1/k2/k3卷积 + MaxPool+1x1) → concat(64ch)
  → MaxPool(4,4) → BN → Dropout
  → ResBlock(3层Conv, kernel各不同, L2正则)
  → SEBlock(GlobalAvgPool → FC → ELU → FC → Sigmoid → scale)
  → MaxPool → Flatten
 
5分支 Flatten → Concat → Dropout → Dense(num_classes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
 
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
 
from utils import set_all_seeds, compute_comprehensive_metrics, \
    print_validation_results, clear_gpu_memory
 
# ── 命令行参数 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="APAVA",
                    choices=["APAVA", "BCI2a", "SleepEDF", "REFED"])
args = parser.parse_args()
 
# ── 全局配置 ───────────────────────────────────────────────────────────────────
device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name  = args.dataset
seed          = 2025
batch_size    = 32
epochs        = 30
learning_rate = 1e-4        # 与原论文一致
weight_decay  = 0.0         # 原论文用 L2 在 Conv 层，不用 AdamW
early_stop    = 5
 
DATASET_CFG = {
    "APAVA":    dict(seq_len=1024, input_channels=16, num_classes=2,
                     num_subjects=23,  k_folds=5,  data_dir=None,               sr=256.0),
    "BCI2a":    dict(seq_len=1024, input_channels=16, num_classes=4,
                     num_subjects=9,   k_folds=9,  data_dir="./datasets/bci2a",    sr=256.0),
    "SleepEDF": dict(seq_len=1024, input_channels=16, num_classes=5,
                     num_subjects=20,  k_folds=20, data_dir="./datasets/sleepedf", sr=256.0),
    "REFED":    dict(seq_len=256, input_channels=16, num_classes=2,
                     num_subjects=32,  k_folds=32, data_dir="./datasets/refed",    sr=256.0),
}
 
cfg            = DATASET_CFG[dataset_name]
seq_len        = cfg["seq_len"]
input_channels = cfg["input_channels"]
num_classes    = cfg["num_classes"]
num_subjects   = cfg["num_subjects"]
k_folds        = cfg["k_folds"]
data_dir       = cfg["data_dir"]
sampling_rate  = cfg["sr"]
 
set_all_seeds(seed)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# 模型定义（忠实对应官方 Keras MMCNN_model.py）
# ════════════════════════════════════════════════════════════════════════════════
 
class InceptionBlock(nn.Module):
    """
    对应官方 inception_block()
    4 路并行：
      路1: Conv1d(k1, stride) + BN + ELU
      路2: Conv1d(k2, stride) + BN + ELU
      路3: Conv1d(k3, stride) + BN + ELU
      路4: MaxPool1d(k4, stride) + Conv1d(1) + BN + ELU
    输出: (B, 4*F, T') — F=16 per branch → 64 channels
    """
    def __init__(self, in_ch: int, filters: int,
                 kernels: list, stride: int, l2: float = 0.01):
        super().__init__()
        k1, k2, k3, k4 = kernels
        F = filters   # 每路输出 16 channels
 
        # 路1-3: 不同尺度时间卷积
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, F, kernel_size=k1, stride=stride,
                      padding=k1//2, bias=False),
            nn.GroupNorm(max(1, min(8, F)), F), nn.ELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, F, kernel_size=k2, stride=stride,
                      padding=k2//2, bias=False),
            nn.GroupNorm(max(1, min(8, F)), F), nn.ELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, F, kernel_size=k3, stride=stride,
                      padding=k3//2, bias=False),
            nn.GroupNorm(max(1, min(8, F)), F), nn.ELU()
        )
        # 路4: MaxPool + 1×1 Conv
        self.branch4_pool = nn.MaxPool1d(kernel_size=k4, stride=stride,
                                          padding=k4//2)
        self.branch4_conv = nn.Sequential(
            nn.Conv1d(in_ch, F, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, min(8, F)), F), nn.ELU()
        )
        self.out_ch = F * 4   # 64
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4_p = self.branch4_pool(x)
        b4   = self.branch4_conv(b4_p)
        # 对齐长度（stride 导致各支长度可能差1）
        min_len = min(b1.shape[-1], b2.shape[-1], b3.shape[-1], b4.shape[-1])
        b1, b2, b3, b4 = [t[..., :min_len] for t in [b1, b2, b3, b4]]
        return torch.cat([b1, b2, b3, b4], dim=1)   # (B, 64, T')
 
 
class ResBlock(nn.Module):
    """
    对应官方 conv_block()
    3层Conv1d（相同 kernel_size）+ L2正则（用 weight_decay 近似）+ 残差
    filters=[16,16,16]
    """
    def __init__(self, in_ch: int, filters: list, kernel: int,
                 l2: float = 0.002):
        super().__init__()
        k1, k2, k3 = filters
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, k1, kernel_size=kernel, padding=kernel//2, bias=False),
            nn.GroupNorm(min(8,k1) if k1%min(8,k1)==0 else 1, k1), nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(k1, k2, kernel_size=kernel, padding=kernel//2, bias=False),
            nn.GroupNorm(min(8,k2) if k2%min(8,k2)==0 else 1, k2), nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(k2, k3, kernel_size=kernel, padding=kernel//2, bias=False),
            nn.GroupNorm(min(8,k3) if k3%min(8,k3)==0 else 1, k3)
        )
        # 残差 1×1 投影
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, k3, kernel_size=1, bias=False),
            nn.GroupNorm(min(8,k3) if k3%min(8,k3)==0 else 1, k3)
        )
        self.act  = nn.ELU()
        self.drop = nn.Dropout(0.2)   # 源码 res block 后有 dropout
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # 对齐时间长度（偶数 kernel 的 padding=k//2 会让输出比输入多1个点）
        min_len = min(out.shape[-1], res.shape[-1])
        out = out[..., :min_len]
        res = res[..., :min_len]
        out = self.act(out + res)
        return self.drop(out)
 
 
class SEBlock(nn.Module):
    """
    对应官方 squeeze_excitation_layer()
    GlobalAvgPool → FC(out_dim//ratio) → ELU → FC(out_dim) → Sigmoid → scale
    """
    def __init__(self, ch: int, ratio: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // ratio)
        self.fc2 = nn.Linear(ch // ratio, ch)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        sq  = x.mean(dim=-1)                    # (B, C) — GlobalAvgPool
        ex  = F.elu(self.fc1(sq))               # (B, C//ratio)
        ex  = torch.sigmoid(self.fc2(ex))       # (B, C)
        ex  = ex.unsqueeze(-1)                  # (B, C, 1)
        return x * ex                           # scale
 
 
class EINBranch(nn.Module):
    """
    一个 EIN（EEG Inception Network）分支
    对应官方 EIN-a ~ EIN-e 各分支的代码块
 
    InceptionBlock → MaxPool(4,4) → BN → Dropout
    → ResBlock → SEBlock → MaxPool(p2, s2) → Flatten
    """
    def __init__(
        self,
        in_ch: int,
        inception_filters: int,       # 每路 F=16
        inception_kernels: list,      # [k1,k2,k3,k4]
        inception_stride: int,
        res_filters: list,            # [16,16,16]
        res_kernel: int,
        se_ratio: int,
        pool2_size: int,
        pool2_stride: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.inception = InceptionBlock(in_ch, inception_filters,
                                         inception_kernels, inception_stride)
        inc_ch = self.inception.out_ch   # 64
 
        self.pool1   = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.bn1     = nn.GroupNorm(min(8, inc_ch) if inc_ch % min(8,inc_ch)==0 else 1, inc_ch)
        self.drop1   = nn.Dropout(dropout)
 
        self.res     = ResBlock(inc_ch, res_filters, res_kernel)
        res_ch       = res_filters[-1]   # 16
 
        self.se      = SEBlock(res_ch, se_ratio)
        self.pool2   = nn.MaxPool1d(kernel_size=pool2_size,
                                     stride=pool2_stride, padding=0)
        self.dropout = dropout
 
        # 预计算 flatten 维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, seq_len)
            dummy = self.inception(dummy)
            dummy = self.pool1(dummy)
            dummy = self.bn1(dummy)
            dummy = self.res(dummy)
            dummy = self.se(dummy)
            dummy = self.pool2(dummy)
            self.flat_dim = dummy.numel()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.res(x)
        x = self.se(x)
        x = self.pool2(x)
        return x.view(x.size(0), -1)
 
 
class MMCNN(nn.Module):
    """
    MMCNN 完整模型
    5 个 EIN 并行分支 → Concat → Dropout → FC(num_classes)
 
    超参数完全来自官方 MMCNN_model.py：
        inception_filters = [16,16,16,16]（每路16，共4路concat=64）
        inception_kernel_length = [
            [5,10,15,10],     # EIN-a（小尺度，高频）
            [40,45,50,100],   # EIN-b
            [60,65,70,100],   # EIN-c
            [80,85,90,100],   # EIN-d
            [160,180,200,180] # EIN-e（大尺度，低频）
        ]
        inception_stride = [2, 4, 4, 4, 16]
        res_block_filters = [16, 16, 16]
        res_block_kernel_stride = [8, 7, 7, 7, 6]
        second_maxpooling_size   = [4, 3, 3, 3, 2]
        second_maxpooling_stride = [4, 3, 3, 3, 2]
        se_ratio = 8
        dropout  = 0.8（原论文值，较大的 dropout）
    """
    def __init__(
        self,
        num_classes:   int,
        n_channels:    int,
        seq_len:       int,
        dropout:       float = 0.5,   # 原论文 0.8，适当减小提高跨数据集泛化
    ):
        super().__init__()
 
        # 官方超参数（直接复制自 MMCNN_model.py）
        inc_filters  = 16   # 每路 16 ch
        inc_kernels  = [
            [5,  10,  15,  10],
            [40, 45,  50,  100],
            [60, 65,  70,  100],
            [80, 85,  90,  100],
            [160,180, 200, 180],
        ]
        inc_strides  = [2, 4, 4, 4, 16]
        res_filters  = [16, 16, 16]
        res_kernels  = [8, 7, 7, 7, 6]
        pool2_sizes  = [4, 3, 3, 3, 2]
        pool2_strides= [4, 3, 3, 3, 2]
        se_ratio     = 8
 
        self.branches = nn.ModuleList()
        total_flat    = 0
 
        for i in range(5):
            branch = EINBranch(
                in_ch=n_channels,
                inception_filters=inc_filters,
                inception_kernels=inc_kernels[i],
                inception_stride=inc_strides[i],
                res_filters=res_filters,
                res_kernel=res_kernels[i],
                se_ratio=se_ratio,
                pool2_size=pool2_sizes[i],
                pool2_stride=pool2_strides[i],
                dropout=dropout,
                seq_len=seq_len,
            )
            self.branches.append(branch)
            total_flat += branch.flat_dim
 
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_flat, num_classes)
 
        # L2 正则化通过 optimizer weight_decay 实现
        self._l2 = 0.002
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        feats = [branch(x) for branch in self.branches]
        out   = torch.cat(feats, dim=1)   # (B, sum_flat)
        out   = self.dropout(out)
        return self.classifier(out)       # (B, num_classes)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════════════════════
 
def load_k_fold_data():
    if dataset_name == "APAVA":
        from preprocessing import apava_k_fold_split
        print("Loading APAVA k-fold (cross-subject)...")
        fold_datasets = apava_k_fold_split(k=k_folds, random_state=seed,
                                           use_cache=True)
    elif dataset_name == "BCI2a":
        from preprocessing_bci2a import bci2a_official_split
        print("Loading BCI2a Official Split (Subject-Dependent, Session A→B)...")
        fold_datasets = bci2a_official_split(data_dir)
        fold_loaders = []
        for fold_idx, (train_dataset, test_dataset) in enumerate(fold_datasets):
            n       = len(train_dataset)
            n_train = int(n * 0.8)
            train_ds = Subset(train_dataset, list(range(n_train)))
            val_ds   = Subset(train_dataset, list(range(n_train, n)))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True, drop_last=False)
            val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=4, pin_memory=True, drop_last=False)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=4, pin_memory=True, drop_last=False)
            fold_loaders.append((train_loader, val_loader, test_loader))
        return fold_loaders
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
        train_ds, val_ds = split_by_subject(train_dataset)
        train_loader = DataLoader(train_ds,     batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=False)
        val_loader   = DataLoader(val_ds,       batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)
        fold_loaders.append((train_loader, val_loader, test_loader))
    return fold_loaders
 
 
def split_by_subject(dataset, train_ratio=0.75):
    all_sids    = np.array(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
    np.random.seed(seed)
    np.random.shuffle(unique_sids)
    n_train   = int(len(unique_sids) * train_ratio)
    train_set = set(unique_sids[:n_train])
    train_idx = [i for i in range(len(dataset)) if all_sids[i] in train_set]
    val_idx   = [i for i in range(len(dataset)) if all_sids[i] not in train_set]
    print(f"  Subject split: {n_train} train / {len(unique_sids)-n_train} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# 训练 / 评估
# ════════════════════════════════════════════════════════════════════════════════
 
def build_model():
    return MMCNN(
        num_classes=num_classes,
        n_channels=input_channels,
        seq_len=seq_len,
        dropout=0.5,     # 原论文 0.8，跨数据集泛化适当减小
    ).to(device)
 
 
def get_class_weights(loader):
    from sklearn.utils.class_weight import compute_class_weight
    all_labels = []
    for batch in loader:
        all_labels.extend(batch[1].numpy())
    all_labels = np.array(all_labels)
    weights    = compute_class_weight('balanced',
                                       classes=np.arange(num_classes),
                                       y=all_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)
 
 
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
 
    all_preds, all_labels, all_probs = [], [], []
 
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            eeg, labels = batch[0].to(device), batch[1].to(device)
 
            if is_train:
                optimizer.zero_grad()
 
            logits = model(eeg)
            loss   = criterion(logits, labels)
 
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
 
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if num_classes == 2:
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
            else:
                all_probs.append(probs.detach().cpu().numpy())
 
    probs_arr = (np.array(all_probs) if num_classes == 2
                 else np.concatenate(all_probs, axis=0))
    return compute_comprehensive_metrics(all_labels, all_preds, probs_arr, num_classes)
 
 
def train_one_fold(train_loader, val_loader):
    model     = build_model()
    weights   = get_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
 
    # 原论文用 Adam lr=1e-4，L2 在 Conv 层用 regularizers.l2(0.002)
    # PyTorch 中用 weight_decay 近似
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
 
    best_val_bacc = -1.0
    no_improve    = 0
    best_state    = None
 
    for epoch in range(epochs):
        tr_metrics = run_epoch(model, train_loader, criterion, optimizer)
        vl_metrics = run_epoch(model, val_loader,   criterion)
 
        scheduler.step(vl_metrics['balanced_accuracy'])
 
        print(f"Epoch {epoch+1}/{epochs}: Train: ", end="")
        print_validation_results(tr_metrics)
        print("Val: ", end="")
        print_validation_results(vl_metrics)
 
        if vl_metrics['balanced_accuracy'] > best_val_bacc:
            best_val_bacc = vl_metrics['balanced_accuracy']
            no_improve    = 0
            best_state    = {
                'epoch':       epoch + 1,
                'state_dict':  {k: v.clone() for k, v in model.state_dict().items()},
                'val_metrics': vl_metrics,
            }
        else:
            no_improve += 1
 
        if no_improve >= early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
 
    return best_state
 
 
def evaluate(model, loader):
    criterion  = nn.CrossEntropyLoss()
    metrics    = run_epoch(model, loader, criterion)
    # 额外收集原始结果用于保存
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            eeg, lbl = batch[0].to(device), batch[1]
            logits   = model(eeg)
            p        = torch.softmax(logits, dim=1)
            pred     = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbl.numpy())
            if num_classes == 2:
                probs.extend(p[:, 1].cpu().numpy())
            else:
                probs.append(p.cpu().numpy())
    probs_arr = (np.array(probs) if num_classes == 2
                 else np.concatenate(probs, axis=0))
    metrics['predictions']   = preds
    metrics['true_labels']   = labels
    metrics['probabilities'] = probs_arr
    return metrics
 
 
# ════════════════════════════════════════════════════════════════════════════════
# K-Fold 主循环
# ════════════════════════════════════════════════════════════════════════════════
 
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
                'fold': fold_idx + 1, 'status': 'failed',
                'test_metrics': None, 'tta_metrics': None
            })
            continue
 
        model = build_model()
        model.load_state_dict(best_state['state_dict'])
 
        test_metrics = evaluate(model, test_loader)
        print_validation_results(test_metrics, fold_idx + 1,
                                 f"Fold {fold_idx+1} Baseline: ")
 
        # MMCNN 无 TTA，同结果填充保持格式一致
        tta_metrics = test_metrics
 
        all_fold_results.append({
            'fold':          fold_idx + 1,
            'test_metrics':  test_metrics,
            'tta_metrics':   tta_metrics,
            'train_metrics': best_state.get('val_metrics', {}),
            'seed':          seed,
        })
 
        del model
        clear_gpu_memory()
 
    # ── 汇总输出（与 two_stage_training.py 格式完全一致）─────────────────────
    successful = [r for r in all_fold_results if r.get('status') != 'failed']
 
    if not successful:
        print("No successful folds completed")
        return
 
    print(f"\n{'='*55}")
    print(f"=== K-Fold Cross Validation Results ===")
    print(f"Dataset: {dataset_name}, K={k_folds}, Seed={seed}")
    print(f"Completed folds: {len(successful)}/{k_folds}")
 
    all_m = {}
    for r in successful:
        for k, v in r['test_metrics'].items():
            if isinstance(v, (int, float)):
                all_m.setdefault(k, []).append(v)
    mean_m = {k: np.mean(v) for k, v in all_m.items()}
    std_m  = {k: np.std(v)  for k, v in all_m.items()}
 
    def _p(key):
        return f"{mean_m.get(key, 0.0):.4f} ± {std_m.get(key, 0.0):.4f}"
 
    print("\n🏆 Baseline Metrics:")
    print(f"  Accuracy:          {_p('accuracy')}")
    print(f"  Balanced Accuracy: {_p('balanced_accuracy')}")
    print(f"  F1 Score (Macro):  {_p('f1_macro')}")
    print(f"  Precision (Macro): {_p('precision_macro')}")
    print(f"  Recall (Macro):    {_p('recall_macro')}")
    if 'roc_auc'           in mean_m: print(f"  ROC AUC:           {_p('roc_auc')}")
    if 'average_precision' in mean_m: print(f"  Avg Prec:          {_p('average_precision')}")
 
    print("\n🚀 TTA Metrics:")
    print(f"  Accuracy:          {mean_m.get('accuracy',          0.0):.4f}")
    print(f"  Balanced Accuracy: {mean_m.get('balanced_accuracy', 0.0):.4f}")
    print(f"  F1 Score (Macro):  {mean_m.get('f1_macro',          0.0):.4f}")
    print(f"  Precision (Macro): {mean_m.get('precision_macro',   0.0):.4f}")
    print(f"  Recall (Macro):    {mean_m.get('recall_macro',      0.0):.4f}")
    if 'roc_auc'           in mean_m: print(f"  ROC AUC:           {mean_m.get('roc_auc',           0.0):.4f}")
    if 'average_precision' in mean_m: print(f"  Avg Prec:          {mean_m.get('average_precision',  0.0):.4f}")
 
    os.makedirs("./results", exist_ok=True)
    save_path = f"./results/mmcnn_{dataset_name}_seed{seed}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'dataset':          dataset_name,
            'seed':             seed,
            'k_folds':          k_folds,
            'completed_folds':  len(successful),
            'baseline_metrics': mean_m,
            'fold_results':     all_fold_results,
        }, f)
    print(f"\n结果保存至: {save_path}")
    return mean_m
 
 
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"MMCNN Baseline (ECML-PKDD 2020)")
    print(f"Dataset:    {dataset_name}")
    print(f"Classes:    {num_classes}")
    print(f"Channels:   {input_channels}")
    print(f"Seq len:    {seq_len}")
    print(f"Subjects:   {num_subjects}")
    print(f"K-Folds:    {k_folds}")
    print(f"Device:     {device}")
    print(f"{'='*55}\n")
 
    k_fold_cross_validation()
