"""
graphsleepnet_baseline.py
=========================
GraphSleepNet PyTorch 实现
原论文：Jia et al., "GraphSleepNet: Adaptive Spatial-Temporal Graph
        Convolutional Networks for Sleep Stage Classification", IJCAI 2020
 
支持数据集：APAVA / BCI2a / SleepEDF / REFED
输出格式与 two_stage_training.py 完全一致
 
用法：
    python graphsleepnet_baseline.py --dataset APAVA
    python graphsleepnet_baseline.py --dataset BCI2a
    python graphsleepnet_baseline.py --dataset SleepEDF
    python graphsleepnet_baseline.py --dataset REFED
 
架构说明（对应论文 Figure 3）：
    输入 (B, C, T)
    → DE特征提取：把时间维按频段切分，计算微分熵特征 → (B, C, F_bands)
    → Adaptive Graph Learning：学习 C×C 邻接矩阵 A
    → ST-GCN × K层：每层包含
        - Spatial Attention：对 C 维度加权
        - Graph Convolution：A × X × W
        - Temporal Attention：对 T 维度加权
        - Temporal Convolution：1D Conv
        - BN + ReLU
    → Global Average Pooling
    → Classifier：FC → num_classes
"""
 
import argparse
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
learning_rate = 1e-3
weight_decay  = 1e-4
early_stop    = 5
 
DATASET_CFG = {
    "APAVA":    dict(seq_len=1024, input_channels=16, num_classes=2,
                     num_subjects=23,  k_folds=5,  data_dir=None,       sr=256.0),
    "BCI2a":    dict(seq_len=1024, input_channels=16, num_classes=4,
                     num_subjects=9,   k_folds=9,  data_dir="./datasets/bci2a",    sr=256.0),
    "SleepEDF": dict(seq_len=256, input_channels=16, num_classes=5,
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
# 模型定义
# ════════════════════════════════════════════════════════════════════════════════
 
class DEFeatureExtractor(nn.Module):
    """
    差分熵（Differential Entropy）特征提取
    论文中把EEG按频段分成5个子带，计算每段DE值作为节点特征
    子带：δ(0.5-4Hz), θ(4-8Hz), α(8-13Hz), β(13-30Hz), γ(30-45Hz)
 
    输入：(B, C, T)
    输出：(B, C, n_bands)
 
    实现：用 Welch 法近似，直接用可学习的1D卷积来提取各频段的功率谱特征
    （完全可微分，比离线计算DE更适合端对端训练）
    """
    def __init__(self, seq_len: int, sr: float = 256.0, n_bands: int = 5):
        super().__init__()
        self.n_bands = n_bands
        self.sr      = sr
 
        # 频段边界（Hz）→ 对应FFT bin
        band_edges_hz = [0.5, 4.0, 8.0, 13.0, 30.0, 45.0]
        n_fft = seq_len
        freq_res = sr / n_fft
 
        self.band_bins = []
        for i in range(n_bands):
            lo = int(band_edges_hz[i]   / freq_res)
            hi = int(band_edges_hz[i+1] / freq_res) + 1
            lo = max(lo, 1)
            hi = min(hi, n_fft // 2)
            self.band_bins.append((lo, hi))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        返回: (B, C, n_bands)
        """
        B, C, T = x.shape
 
        # FFT → 功率谱
        xf  = torch.fft.rfft(x, n=T, dim=-1)          # (B, C, T//2+1)
        psd = (xf.real ** 2 + xf.imag ** 2) / T        # 功率谱密度
 
        # 各频段平均功率 → 近似 DE（DE = 0.5*log(2πe*σ²) ≈ 0.5*log(power)）
        band_feats = []
        for (lo, hi) in self.band_bins:
            band_power = psd[:, :, lo:hi].mean(dim=-1, keepdim=True)  # (B,C,1)
            de = 0.5 * torch.log(band_power.clamp(min=1e-8))
            band_feats.append(de)
 
        out = torch.cat(band_feats, dim=-1)  # (B, C, n_bands)
        return out
 
 
class AdaptiveGraphLearning(nn.Module):
    """
    自适应图学习（论文 Section 3.1）
    学习节点间的连接关系，输出 C×C 邻接矩阵
 
    A = softmax(ReLU(E₁ · E₂ᵀ))
    E₁, E₂ 是可学习的节点嵌入，各 shape (C, d_embed)
    """
    def __init__(self, n_nodes: int, d_embed: int = 16):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n_nodes, d_embed) * 0.01)
        self.E2 = nn.Parameter(torch.randn(n_nodes, d_embed) * 0.01)
 
    def forward(self) -> torch.Tensor:
        """返回归一化邻接矩阵 (C, C)"""
        A = F.relu(self.E1 @ self.E2.t())          # (C, C)
        A = F.softmax(A, dim=-1)                    # row-wise normalize
        return A
 
 
class SpatialAttention(nn.Module):
    """
    空间注意力：对通道维度（节点维度）加权
    输入: (B, C, F)
    输出: (B, C, F)
    """
    def __init__(self, n_channels: int, n_features: int):
        super().__init__()
        self.W  = nn.Linear(n_features, n_features, bias=False)
        self.bs = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.Vs = nn.Parameter(torch.ones(n_channels, n_channels) * 0.01)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F)
        # S = Vs ⊙ sigmoid(X·W·Xᵀ + bs)
        Xw   = self.W(x)                             # (B, C, F)
        S    = torch.bmm(Xw, x.transpose(1, 2))      # (B, C, C)
        S    = torch.sigmoid(S + self.bs)             # broadcast bs
        S    = self.Vs * S                            # element-wise scale
        # row-wise softmax → attention map
        S    = F.softmax(S, dim=-1)                   # (B, C, C)
        out  = torch.bmm(S, x)                        # (B, C, F)
        return out
 
 
class TemporalAttention(nn.Module):
    """
    时间注意力：对时间维度加权
    输入: (B, C, T)
    输出: (B, C, T)
 
    修复：bt/Vt 改为动态生成（消除固定 n_time 参数的维度依赖），
    使同一模块可复用于任意特征维大小（5→32→64）。
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.W  = nn.Linear(n_channels, n_channels, bias=False)
        # 用可学习的标量代替固定大小矩阵参数
        self.scale = nn.Parameter(torch.tensor(0.01))
        self.bias  = nn.Parameter(torch.tensor(0.0))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — T 这里是特征维（DE bands 或 hidden_dim），大小动态
        xt   = x.transpose(1, 2)                      # (B, T, C)
        Xw   = self.W(xt)                              # (B, T, C)
        E    = torch.bmm(Xw, xt.transpose(1, 2))      # (B, T, T)
        E    = torch.sigmoid(E + self.bias)            # 标量 bias，可广播任意 T
        E    = self.scale * E                          # 标量 scale
        E    = F.softmax(E, dim=-1)                    # (B, T, T)
        out  = torch.bmm(E, xt).transpose(1, 2)       # (B, C, T)
        return out
 
 
class GraphConvolution(nn.Module):
    """
    图卷积：X' = A · X · W
    输入: (B, C, F_in)
    输出: (B, C, F_out)
    A: (C, C) 邻接矩阵（外部传入）
    """
    def __init__(self, f_in: int, f_out: int, bias: bool = True):
        super().__init__()
        self.W = nn.Linear(f_in, f_out, bias=bias)
 
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F_in), A: (C, C)
        AX  = torch.matmul(A, x)          # (B, C, F_in)
        out = self.W(AX)                   # (B, C, F_out)
        return out
 
 
class STGCNBlock(nn.Module):
    """
    一个 ST-GCN Block：
    Spatial Attention → Graph Conv → Temporal Attention → Temporal Conv → BN + ReLU
    """
    def __init__(self, n_channels: int, f_in: int, f_out: int,
                 t_in: int = None, t_out: int = None, kernel_size: int = 3):
        super().__init__()
        self.spatial_att  = SpatialAttention(n_channels, f_in)
        self.graph_conv   = GraphConvolution(f_in, f_out)
        self.temporal_att = TemporalAttention(n_channels)   # 动态，不依赖固定 t_in
        self.temporal_conv = nn.Conv1d(
            f_out, f_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn   = nn.BatchNorm1d(f_out)
        self.relu = nn.ReLU()
 
        # 残差连接（若维度变化则用1×1卷积对齐）
        self.residual = nn.Identity() if f_in == f_out else nn.Linear(f_in, f_out)
 
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F_in)  ← 特征维在最后
        A: (C, C)
        返回: (B, C, F_out)
        """
        residual = self.residual(x)                    # (B, C, F_out)
 
        # Spatial Attention
        x = self.spatial_att(x)                        # (B, C, F_in)
 
        # Graph Convolution
        x = self.graph_conv(x, A)                      # (B, C, F_out)
 
        # Temporal Attention（这里把 F 维视作 "时间"，C 视作 "通道"）
        # 为了复用 TemporalAttention，reshape 一下
        x = self.temporal_att(x)                       # (B, C, F_out)
 
        # Temporal Conv：沿 F 维（最后维）
        # Conv1d 期望 (B, C_in, L)，这里 C_in=n_channels, L=F_out
        x = x.transpose(1, 2)                          # (B, F_out, C)
        x = self.temporal_conv(x)                      # (B, F_out, C)
        x = x.transpose(1, 2)                          # (B, C, F_out)
 
        # BN 沿特征维
        x = x.transpose(1, 2)                          # (B, F_out, C)
        x = self.bn(x)
        x = x.transpose(1, 2)                          # (B, C, F_out)
 
        x = self.relu(x + residual)
        return x
 
 
class GraphSleepNet(nn.Module):
    """
    GraphSleepNet 完整模型
 
    流程：
        (B, C, T)
        → DEFeatureExtractor    → (B, C, n_bands=5)
        → AdaptiveGraphLearning → A (C, C)
        → STGCNBlock × n_layers → (B, C, hidden_dim)
        → Global Mean Pooling   → (B, hidden_dim)
        → Dropout → FC          → (B, num_classes)
    """
    def __init__(
        self,
        num_classes:    int,
        n_channels:     int,
        seq_len:        int,
        sr:             float = 256.0,
        n_bands:        int   = 5,
        d_embed:        int   = 16,
        hidden_dims:    list  = None,   # 每个 ST-GCN block 的输出维度
        n_layers:       int   = 2,
        dropout:        float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]
 
        self.n_channels = n_channels
        self.n_bands    = n_bands
 
        # ── DE 特征提取 ──
        self.de_extractor = DEFeatureExtractor(seq_len, sr, n_bands)
 
        # ── 自适应图学习 ──
        self.graph_learner = AdaptiveGraphLearning(n_channels, d_embed)
 
        # ── ST-GCN Blocks ──
        dims    = [n_bands] + hidden_dims
        t_sizes = [n_bands] + hidden_dims   # 时间/特征维大小变化
 
        self.stgcn_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            block = STGCNBlock(
                n_channels=n_channels,
                f_in=dims[i],
                f_out=dims[i+1],
                t_in=t_sizes[i],
                t_out=t_sizes[i+1],
                kernel_size=3,
            )
            self.stgcn_blocks.append(block)
 
        # ── 分类头 ──
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → logits (B, num_classes)"""
        # DE 特征提取
        x = self.de_extractor(x)          # (B, C, n_bands)
 
        # 自适应图学习
        A = self.graph_learner()           # (C, C)
 
        # ST-GCN
        for block in self.stgcn_blocks:
            x = block(x, A)               # (B, C, hidden_dim)
 
        # 全局平均池化（跨节点维度）
        x = x.mean(dim=1)                 # (B, hidden_dim)
 
        x = self.dropout(x)
        return self.classifier(x)         # (B, num_classes)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════════════════════
 
def load_k_fold_data():
    if dataset_name == "APAVA":
        from preprocessing import apava_k_fold_split
        print("Loading APAVA k-fold (cross-subject)...")
        fold_datasets = apava_k_fold_split(k=k_folds, random_state=seed, use_cache=True)
 
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
 
        train_loader = DataLoader(train_ds,      batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=False)
        val_loader   = DataLoader(val_ds,        batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)
        fold_loaders.append((train_loader, val_loader, test_loader))
 
    return fold_loaders
 
 
def split_by_subject(dataset, train_ratio=0.75):
    all_sids    = np.array(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
    np.random.seed(seed)
    np.random.shuffle(unique_sids)
    n_train     = int(len(unique_sids) * train_ratio)
    train_set   = set(unique_sids[:n_train])
    train_idx   = [i for i in range(len(dataset)) if all_sids[i] in train_set]
    val_idx     = [i for i in range(len(dataset)) if all_sids[i] not in train_set]
    print(f"  Subject split: {n_train} train / {len(unique_sids)-n_train} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# 训练 / 评估
# ════════════════════════════════════════════════════════════════════════════════
 
def build_model():
    return GraphSleepNet(
        num_classes=num_classes,
        n_channels=input_channels,
        seq_len=seq_len,
        sr=sampling_rate,
        n_bands=5,
        d_embed=16,
        hidden_dims=[32, 64],
        n_layers=2,
        dropout=0.5,
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
 
 
def train_one_fold(train_loader, val_loader):
    model     = build_model()
    weights   = get_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
 
    best_val_bacc = -1.0
    no_improve    = 0
    best_state    = None
 
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        tr_preds, tr_labels, tr_probs = [], [], []
 
        for batch in train_loader:
            eeg, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(eeg)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            tr_preds.extend(preds.cpu().numpy())
            tr_labels.extend(labels.cpu().numpy())
            if num_classes == 2:
                tr_probs.extend(probs[:, 1].detach().cpu().numpy())
            else:
                tr_probs.append(probs.detach().cpu().numpy())
 
        tr_probs_arr = (np.array(tr_probs) if num_classes == 2
                        else np.concatenate(tr_probs, axis=0))
        tr_metrics = compute_comprehensive_metrics(
            tr_labels, tr_preds, tr_probs_arr, num_classes)
 
        # ── Val ──
        model.eval()
        vl_preds, vl_labels, vl_probs = [], [], []
 
        with torch.no_grad():
            for batch in val_loader:
                eeg, labels = batch[0].to(device), batch[1].to(device)
                logits = model(eeg)
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(logits, dim=1)
                vl_preds.extend(preds.cpu().numpy())
                vl_labels.extend(labels.cpu().numpy())
                if num_classes == 2:
                    vl_probs.extend(probs[:, 1].cpu().numpy())
                else:
                    vl_probs.append(probs.cpu().numpy())
 
        vl_probs_arr = (np.array(vl_probs) if num_classes == 2
                        else np.concatenate(vl_probs, axis=0))
        vl_metrics = compute_comprehensive_metrics(
            vl_labels, vl_preds, vl_probs_arr, num_classes)
 
        scheduler.step(vl_metrics['balanced_accuracy'])
 
        print(f"Epoch {epoch+1}/{epochs}: Train: ", end="")
        print_validation_results(tr_metrics)
        print("Val: ", end="")
        print_validation_results(vl_metrics)
 
        if vl_metrics['balanced_accuracy'] > best_val_bacc:
            best_val_bacc = vl_metrics['balanced_accuracy']
            no_improve    = 0
            best_state    = {
                'epoch':      epoch + 1,
                'state_dict': {k: v.clone() for k, v in model.state_dict().items()},
                'val_metrics': vl_metrics,
            }
        else:
            no_improve += 1
 
        if no_improve >= early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
 
    return best_state
 
 
def evaluate(model, loader):
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
    metrics = compute_comprehensive_metrics(labels, preds, probs_arr, num_classes)
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
 
        # GraphSleepNet 无 TTA，用相同结果填充保持格式一致
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
 
    # ── 汇总输出（与 two_stage_training.py 格式完全一致）───────────────────────
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
 
    print("\n🏆 Baseline Metrics:")
    print(f"  Accuracy:          {mean_m.get('accuracy',          0.0):.4f} ± {std_m.get('accuracy',          0.0):.4f}")
    print(f"  Balanced Accuracy: {mean_m.get('balanced_accuracy', 0.0):.4f} ± {std_m.get('balanced_accuracy', 0.0):.4f}")
    print(f"  F1 Score (Macro):  {mean_m.get('f1_macro',          0.0):.4f} ± {std_m.get('f1_macro',          0.0):.4f}")
    print(f"  Precision (Macro): {mean_m.get('precision_macro',   0.0):.4f} ± {std_m.get('precision_macro',   0.0):.4f}")
    print(f"  Recall (Macro):    {mean_m.get('recall_macro',      0.0):.4f} ± {std_m.get('recall_macro',      0.0):.4f}")
    if 'roc_auc' in mean_m:
        print(f"  ROC AUC:           {mean_m.get('roc_auc',           0.0):.4f} ± {std_m.get('roc_auc',           0.0):.4f}")
    if 'average_precision' in mean_m:
        print(f"  Avg Prec:          {mean_m.get('average_precision',  0.0):.4f} ± {std_m.get('average_precision',  0.0):.4f}")
 
    print("\n🚀 TTA Metrics:")
    print(f"  Accuracy:          {mean_m.get('accuracy',          0.0):.4f}")
    print(f"  Balanced Accuracy: {mean_m.get('balanced_accuracy', 0.0):.4f}")
    print(f"  F1 Score (Macro):  {mean_m.get('f1_macro',          0.0):.4f}")
    print(f"  Precision (Macro): {mean_m.get('precision_macro',   0.0):.4f}")
    print(f"  Recall (Macro):    {mean_m.get('recall_macro',      0.0):.4f}")
    if 'roc_auc' in mean_m:
        print(f"  ROC AUC:           {mean_m.get('roc_auc',           0.0):.4f}")
    if 'average_precision' in mean_m:
        print(f"  Avg Prec:          {mean_m.get('average_precision',  0.0):.4f}")
 
    import os
    os.makedirs("./results", exist_ok=True)
    save_path = f"./results/graphsleepnet_{dataset_name}_seed{seed}.pkl"
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
    print(f"GraphSleepNet Baseline")
    print(f"Dataset:    {dataset_name}")
    print(f"Classes:    {num_classes}")
    print(f"Channels:   {input_channels}")
    print(f"Seq len:    {seq_len}")
    print(f"Subjects:   {num_subjects}")
    print(f"K-Folds:    {k_folds}")
    print(f"Device:     {device}")
    print(f"{'='*55}\n")
 
    k_fold_cross_validation()
