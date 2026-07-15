"""
salientsleepnet_baseline.py
===========================
SalientSleepNet PyTorch 实现（Single-Stream 变体）
原论文：Jia et al., "SalientSleepNet: Multimodal Salient Wave Detection
        Network for Sleep Staging", IJCAI 2021
 
支持数据集：APAVA / BCI2a / SleepEDF / REFED
输出格式与 two_stage_training.py 完全一致
 
用法：
    python salientsleepnet_baseline.py --dataset APAVA
    python salientsleepnet_baseline.py --dataset BCI2a
    python salientsleepnet_baseline.py --dataset SleepEDF
    python salientsleepnet_baseline.py --dataset REFED
 
架构说明（对应论文 Figure 3，Single-Stream 变体）：
    输入 (B, C, T)
    → Channel Merge：把 C 通道合并成 2 路（EEG/EOG 模拟）
    → U²-Structure（单流）
        Encoder：5 × U-Unit（每个 U-Unit 内部是深度l=4的U-like结构）
        Decoder：4 × U-Unit + skip connections
    → MSE（Multi-Scale Extraction）：膨胀率 1/2/3/4 的空洞卷积 + 瓶颈层
    → Global Average Pooling + Classifier
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
learning_rate = 1e-3
weight_decay  = 1e-4
early_stop    = 5
 
DATASET_CFG = {
    "APAVA":    dict(seq_len=1024, input_channels=16, num_classes=2,
                     num_subjects=23,  k_folds=5,  data_dir=None,               sr=256.0),
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
 
def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1):
    """
    基础卷积块：Conv1d → GroupNorm → ReLU
    用 GroupNorm 代替 BatchNorm1d：
      - BN 在 batch_size=1 或序列长度=1 时会报错（无法计算统计量）
      - GN 只依赖通道维，batch_size=1 完全没问题
      num_groups=min(8, out_ch)：保证 out_ch 能被 num_groups 整除
    """
    num_groups = min(8, out_ch)
    # 确保整除
    while out_ch % num_groups != 0:
        num_groups -= 1
    num_groups = max(num_groups, 1)
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.GroupNorm(num_groups, out_ch),
        nn.ReLU(inplace=True),
    )
 
 
class ULikeStructure(nn.Module):
    """
    U-like 结构（论文 4.1 节）：深度 l=4 的编解码器
    编码阶段：l 次下采样
    解码阶段：l 次上采样 + skip connection
    输入/输出 shape：(B, C, T)
    """
    def __init__(self, channels: int, depth: int = 4):
        super().__init__()
        self.depth = depth
 
        # Encoder：每层 conv + pooling
        self.enc_convs = nn.ModuleList()
        self.pools     = nn.ModuleList()
        for d in range(depth):
            self.enc_convs.append(conv_bn_relu(channels, channels))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
 
        # Bottleneck
        self.bottleneck = conv_bn_relu(channels, channels)
 
        # Decoder：每层 upsample + skip + conv
        self.dec_convs   = nn.ModuleList()
        self.upsamples   = nn.ModuleList()
        for d in range(depth):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='linear',
                                               align_corners=False))
            # skip connection 拼接后通道×2 → 还原
            self.dec_convs.append(conv_bn_relu(channels * 2, channels))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for d in range(self.depth):
            x = self.enc_convs[d](x)
            skips.append(x)
            x = self.pools[d](x)
 
        # Bottleneck
        x = self.bottleneck(x)
 
        # Decoder
        for d in reversed(range(self.depth)):
            x = self.upsamples[d](x)
            sk = skips[d]
            # 对齐长度（池化可能导致差1）
            if x.shape[-1] != sk.shape[-1]:
                x = F.interpolate(x, size=sk.shape[-1], mode='linear',
                                  align_corners=False)
            x = torch.cat([x, sk], dim=1)
            x = self.dec_convs[d](x)
 
        return x
 
 
class UUnit(nn.Module):
    """
    U-Unit（论文公式1-3）：
        1. Channel-Reshape（投影到 mid_ch）
        2. U-like Structure（深度 l=4）
        3. 残差连接（投影回原始通道数）
 
    输入 (B, in_ch, T) → 输出 (B, out_ch, T)
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None,
                 u_depth: int = 4):
        super().__init__()
        if mid_ch is None:
            mid_ch = max(out_ch // 2, 8)
 
        # Channel reshape（输入投影）
        self.reshape_in  = conv_bn_relu(in_ch,  mid_ch, kernel=1, padding=0)
        # U-like 结构
        self.u_like      = ULikeStructure(mid_ch, depth=u_depth)
        # 投影回 out_ch
        self.reshape_out = conv_bn_relu(mid_ch, out_ch, kernel=1, padding=0)
        # 残差（in_ch ≠ out_ch 时需要投影）
        self.residual    = (nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
                            if in_ch != out_ch else nn.Identity())
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)           # (B, out_ch, T)
        xm  = self.reshape_in(x)         # (B, mid_ch, T)
        xm  = self.u_like(xm)            # (B, mid_ch, T)
        xm  = self.reshape_out(xm)       # (B, out_ch, T)
        return xm + res                  # 残差融合
 
 
class U2Structure(nn.Module):
    """
    U²-Structure（论文 4.1 节）
    Encoder：5 个 U-Unit，每层 2× 下采样
    Decoder：4 个 U-Unit，2× 上采样 + skip
 
    输入 (B, in_ch, T) → 输出 (B, base_ch, T)
    """
    def __init__(self, in_ch: int, base_ch: int = 16, u_depth: int = 4):
        super().__init__()
        # 各层通道数
        enc_chs = [in_ch,
                   base_ch,
                   base_ch * 2,
                   base_ch * 4,
                   base_ch * 8,
                   base_ch * 8]   # bottleneck
 
        # Encoder：5 个 U-Unit
        self.enc_units = nn.ModuleList()
        self.pools     = nn.ModuleList()
        for i in range(5):
            self.enc_units.append(UUnit(enc_chs[i], enc_chs[i+1],
                                        mid_ch=max(enc_chs[i+1]//2, 8),
                                        u_depth=u_depth))
            if i < 4:   # 前4层下采样，第5层（bottleneck）不采样
                self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
 
        # Decoder：4 个 U-Unit（逆序）
        dec_chs = list(reversed(enc_chs[1:]))  # [8B,8B,4B,2B,B]
        self.dec_units   = nn.ModuleList()
        self.upsamples   = nn.ModuleList()
        for i in range(4):
            # skip 来自对称 encoder 层，concat 后通道 ×2
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='linear',
                                              align_corners=False))
            self.dec_units.append(UUnit(dec_chs[i] + dec_chs[i+1], dec_chs[i+1],
                                        mid_ch=max(dec_chs[i+1]//2, 8),
                                        u_depth=u_depth))
 
        self.out_ch = dec_chs[-1]  # = base_ch
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc_feats = []
        for i in range(5):
            x = self.enc_units[i](x)
            enc_feats.append(x)
            if i < 4:
                x = self.pools[i](x)
 
        # x = bottleneck output (enc_feats[4])
        x = enc_feats[4]
 
        # Decoder（从 enc_feats[3] 开始取 skip）
        for i in range(4):
            x = self.upsamples[i](x)
            skip = enc_feats[3 - i]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear',
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_units[i](x)
 
        return x   # (B, base_ch, T)
 
 
class MSEModule(nn.Module):
    """
    Multi-Scale Extraction 模块（论文 4.2 节）
    4 个膨胀率不同的空洞卷积（rate=1,2,3,4）→ concat → 瓶颈层
 
    输入 (B, in_ch, T) → 输出 (B, in_ch//4, T)
    """
    def __init__(self, in_ch: int, bottleneck_rate: int = 4):
        super().__init__()
        # 4个空洞卷积，各生成 in_ch 通道
        self.dconv1 = conv_bn_relu(in_ch, in_ch, kernel=3,
                                   padding=1,  dilation=1)
        self.dconv2 = conv_bn_relu(in_ch, in_ch, kernel=3,
                                   padding=2,  dilation=2)
        self.dconv3 = conv_bn_relu(in_ch, in_ch, kernel=3,
                                   padding=3,  dilation=3)
        self.dconv4 = conv_bn_relu(in_ch, in_ch, kernel=3,
                                   padding=4,  dilation=4)
 
        # 瓶颈层：4*in_ch → in_ch // bottleneck_rate
        out_ch = max(in_ch // bottleneck_rate, 8)
        self.bottleneck = nn.Sequential(
            conv_bn_relu(in_ch * 4, in_ch, kernel=1, padding=0),
            conv_bn_relu(in_ch, out_ch,    kernel=1, padding=0),
        )
        self.out_ch = out_ch
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dconv1(x)
        d2 = self.dconv2(x)
        d3 = self.dconv3(x)
        d4 = self.dconv4(x)
        xms = torch.cat([d1, d2, d3, d4], dim=1)   # (B, 4*in_ch, T)
        return self.bottleneck(xms)                  # (B, out_ch, T)
 
 
class MMAModule(nn.Module):
    """
    MultiModal Attention 模块（论文 4.3 节）
    融合：X_fuse = X1 + X2 + X1⊙X2
    SE 通道注意力：GAP → FC → ReLU → FC → Sigmoid → element-wise scale
 
    输入：两个 (B, ch, T) 特征图
    输出：(B, ch, T)
    """
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.se = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )
 
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 模态融合（公式7）
        x_fuse = x1 + x2 + x1 * x2               # (B, ch, T)
        # SE 通道注意力（公式8-9）
        gap    = x_fuse.mean(dim=-1)               # (B, ch)
        att    = self.se(gap).unsqueeze(-1)        # (B, ch, 1)
        return x_fuse * att                        # (B, ch, T)
 
 
class SalientSleepNet(nn.Module):
    """
    SalientSleepNet（Single-Stream 变体）
 
    由于输入只有 EEG（无独立 EOG），
    我们把 16 通道分成前8通道（模拟EEG）和后8通道（模拟EOG）作为双流输入。
    这与论文 SingleSalientModel 单流版本精神一致。
 
    流程：
        (B, C=16, T=256)
        → split → EEG stream (B, 8, T) & EOG stream (B, 8, T)
        → U²-Structure (各自独立)    → (B, base_ch, T)
        → MSE                         → (B, mse_ch, T)
        → MMA (两流融合)              → (B, mse_ch, T)
        → Global Average Pooling      → (B, mse_ch)
        → Dropout → FC                → (B, num_classes)
    """
    def __init__(
        self,
        num_classes:  int,
        n_channels:   int,
        seq_len:      int,
        base_ch:      int  = 16,
        u_depth:      int  = 4,
        mse_rate:     int  = 4,
        dropout:      float = 0.5,
    ):
        super().__init__()
        # 通道分配
        eeg_ch = n_channels // 2       # 8
        eog_ch = n_channels - eeg_ch   # 8
 
        # 双流 U²-Structure
        self.u2_eeg = U2Structure(eeg_ch, base_ch=base_ch, u_depth=u_depth)
        self.u2_eog = U2Structure(eog_ch, base_ch=base_ch, u_depth=u_depth)
 
        # 各流接 MSE
        stream_ch = self.u2_eeg.out_ch   # = base_ch
        self.mse_eeg = MSEModule(stream_ch, bottleneck_rate=mse_rate)
        self.mse_eog = MSEModule(stream_ch, bottleneck_rate=mse_rate)
 
        # MMA 融合
        mse_ch = self.mse_eeg.out_ch
        self.mma = MMAModule(mse_ch, reduction=4)
 
        # 分类头
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(mse_ch, num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        C = x.shape[1]
        half = C // 2
 
        # 分流
        x_eeg = x[:, :half,  :]      # (B, 8, T)
        x_eog = x[:, half:,  :]      # (B, 8, T)
 
        # U² 特征提取
        f_eeg = self.u2_eeg(x_eeg)   # (B, base_ch, T)
        f_eog = self.u2_eog(x_eog)   # (B, base_ch, T)
 
        # MSE
        f_eeg = self.mse_eeg(f_eeg)  # (B, mse_ch, T)
        f_eog = self.mse_eog(f_eog)  # (B, mse_ch, T)
 
        # MMA 融合
        f_out = self.mma(f_eeg, f_eog)   # (B, mse_ch, T)
 
        # Global Average Pooling
        f_out = f_out.mean(dim=-1)        # (B, mse_ch)
 
        f_out = self.dropout(f_out)
        return self.classifier(f_out)     # (B, num_classes)
 
 
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
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds,       batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=True)
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
    return SalientSleepNet(
        num_classes=num_classes,
        n_channels=input_channels,
        seq_len=seq_len,
        base_ch=16,
        u_depth=4,
        mse_rate=4,
        dropout=0.5,
    ).to(device)
 
 
def get_class_weights(loader):
    from sklearn.utils.class_weight import compute_class_weight
    all_labels = []
    for batch in loader:
        all_labels.extend(batch[1].numpy())
    all_labels = np.array(all_labels)
    weights = compute_class_weight('balanced',
                                    classes=np.arange(num_classes),
                                    y=all_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)
 
 
def run_epoch(model, loader, criterion, optimizer=None):
    """统一的 train/eval epoch 函数"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
 
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
 
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
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
 
            total_loss += loss.item()
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
    metrics = compute_comprehensive_metrics(
        all_labels, all_preds, probs_arr, num_classes)
    metrics['loss'] = total_loss / len(loader)
    return metrics
 
 
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
    criterion = nn.CrossEntropyLoss()
    metrics   = run_epoch(model, loader, criterion)
    # 收集 predictions / true_labels / probabilities 用于保存
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
# K-Fold 主循环（与 two_stage_training.py 输出格式完全一致）
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
 
        # SalientSleepNet 无 TTA，用相同结果填充保持格式一致
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
 
    # ── 汇总输出 ───────────────────────────────────────────────────────────────
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
 
    def _fmt(key):
        return f"{mean_m.get(key, 0.0):.4f} ± {std_m.get(key, 0.0):.4f}"
 
    print("\n🏆 Baseline Metrics:")
    print(f"  Accuracy:          {_fmt('accuracy')}")
    print(f"  Balanced Accuracy: {_fmt('balanced_accuracy')}")
    print(f"  F1 Score (Macro):  {_fmt('f1_macro')}")
    print(f"  Precision (Macro): {_fmt('precision_macro')}")
    print(f"  Recall (Macro):    {_fmt('recall_macro')}")
    if 'roc_auc'           in mean_m: print(f"  ROC AUC:           {_fmt('roc_auc')}")
    if 'average_precision' in mean_m: print(f"  Avg Prec:          {_fmt('average_precision')}")
 
    print("\n🚀 TTA Metrics:")
    print(f"  Accuracy:          {mean_m.get('accuracy',          0.0):.4f}")
    print(f"  Balanced Accuracy: {mean_m.get('balanced_accuracy', 0.0):.4f}")
    print(f"  F1 Score (Macro):  {mean_m.get('f1_macro',          0.0):.4f}")
    print(f"  Precision (Macro): {mean_m.get('precision_macro',   0.0):.4f}")
    print(f"  Recall (Macro):    {mean_m.get('recall_macro',      0.0):.4f}")
    if 'roc_auc'           in mean_m: print(f"  ROC AUC:           {mean_m.get('roc_auc',           0.0):.4f}")
    if 'average_precision' in mean_m: print(f"  Avg Prec:          {mean_m.get('average_precision',  0.0):.4f}")
 
    os.makedirs("./results", exist_ok=True)
    save_path = f"./results/salientsleepnet_{dataset_name}_seed{seed}.pkl"
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
    print(f"SalientSleepNet Baseline")
    print(f"Dataset:    {dataset_name}")
    print(f"Classes:    {num_classes}")
    print(f"Channels:   {input_channels}")
    print(f"Seq len:    {seq_len}")
    print(f"Subjects:   {num_subjects}")
    print(f"K-Folds:    {k_folds}")
    print(f"Device:     {device}")
    print(f"{'='*55}\n")
 
    k_fold_cross_validation()
