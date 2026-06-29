"""
eegnet_baseline.py
==================
EEGNet Baseline，支持 APAVA / BCI2a / SleepEDF / REFED 四个数据集
输出格式与 two_stage_training.py 完全一致
 
用法：
    python eegnet_baseline.py                        # 默认 APAVA
    python eegnet_baseline.py --dataset BCI2a
    python eegnet_baseline.py --dataset SleepEDF
    python eegnet_baseline.py --dataset REFED
 
参考论文：
    Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
    EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018
"""
 
import argparse
import numpy as np
import torch
import torch.nn as nn
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
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = args.dataset
seed         = 2025
batch_size   = 32
epochs       = 30
learning_rate = 1e-3
weight_decay  = 1e-4
early_stop    = 5
 
# ── 数据集参数 ─────────────────────────────────────────────────────────────────
DATASET_CFG = {
    "APAVA": dict(
        seq_len=1024, input_channels=16, num_classes=2,
        num_subjects=23, k_folds=5, data_dir=None,
        sampling_rate=256.0
    ),
    "BCI2a": dict(
        seq_len=1024, input_channels=16, num_classes=4,
        num_subjects=9, k_folds=9, data_dir="./datasets/bci2a",
        sampling_rate=256.0
    ),
    "SleepEDF": dict(
        seq_len=256, input_channels=16, num_classes=5,
        num_subjects=20, k_folds=20, data_dir="./datasets/sleepedf",
        sampling_rate=256.0
    ),
    "REFED": dict(
        seq_len=256, input_channels=16, num_classes=2,
        num_subjects=32, k_folds=32, data_dir="./datasets/refed",
        sampling_rate=256.0
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
 
set_all_seeds(seed)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# EEGNet 模型定义
# ════════════════════════════════════════════════════════════════════════════════
 
class EEGNet(nn.Module):
    """
    EEGNet: 紧凑型 EEG 分类网络
    输入: (batch, n_channels, seq_len)
    输出: (batch, num_classes)
 
    超参数选择依据：
        F1=8, D=2, F2=16 为原论文推荐默认值
        kernel_size = sampling_rate // 2 = 128（对应0.5秒时间感受野）
        dropout = 0.5（原论文推荐）
    """
    def __init__(
        self,
        num_classes: int,
        n_channels: int,
        seq_len: int,
        sampling_rate: float = 256.0,
        F1: int = 8,          # 时频滤波器数量
        D: int = 2,           # Depthwise 深度乘子
        F2: int = 16,         # Separable 滤波器数量（= F1 * D）
        dropout: float = 0.5,
    ):
        super().__init__()
        self.F1  = F1
        self.D   = D
        self.F2  = F2
        F2       = F1 * D     # 确保一致
 
        # ── Block 1: Temporal + Depthwise Spatial ──────────────────────────
        # 时间卷积：捕获频率特征（感受野 = 0.5s）
        kernel_t = int(sampling_rate // 2)   # 128 @ 256Hz
        if kernel_t % 2 == 0:
            kernel_t += 1                    # 保持奇数padding方便计算
 
        self.block1 = nn.Sequential(
            # 时间卷积（same padding）
            nn.Conv2d(1, F1, kernel_size=(1, kernel_t),
                      padding=(0, kernel_t // 2), bias=False),
            nn.BatchNorm2d(F1),
 
            # Depthwise 空间卷积（跨通道，捕获空间滤波）
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
 
            nn.AvgPool2d(kernel_size=(1, 4)),   # 时间下采样 4x
            nn.Dropout(dropout),
        )
 
        # ── Block 2: Separable Depthwise Conv ─────────────────────────────
        kernel_s = 16    # 原论文推荐 16 点（约 0.0625s）
 
        self.block2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, kernel_s),
                      padding=(0, kernel_s // 2), groups=F1 * D, bias=False),
            # Pointwise
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
 
            nn.AvgPool2d(kernel_size=(1, 8)),   # 时间下采样 8x
            nn.Dropout(dropout),
        )
 
        # ── 分类头 ─────────────────────────────────────────────────────────
        # 计算展平后维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, seq_len)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            self._flatten_dim = dummy.numel()
 
        self.classifier = nn.Linear(self._flatten_dim, num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, seq_len)
        返回: (batch, num_classes) logits
        """
        # 添加 channel 维度 → (batch, 1, n_channels, seq_len)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
 
 
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
        train_ds, val_ds = split_by_subject(train_dataset, train_ratio=0.75)
 
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)
        fold_loaders.append((train_loader, val_loader, test_loader))
 
    return fold_loaders
 
 
def split_by_subject(dataset, train_ratio=0.75):
    """与 two_stage_training.py 的 split_dataset_by_subject 完全相同"""
    all_sids    = np.array(dataset.subject_ids)
    unique_sids = np.unique(all_sids)
 
    np.random.seed(seed)
    np.random.shuffle(unique_sids)
 
    n_train = int(len(unique_sids) * train_ratio)
    train_set = set(unique_sids[:n_train])
 
    train_idx = [i for i in range(len(dataset)) if all_sids[i] in train_set]
    val_idx   = [i for i in range(len(dataset)) if all_sids[i] not in train_set]
 
    print(f"  Subject split: {n_train} train / {len(unique_sids)-n_train} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
 
 
# ════════════════════════════════════════════════════════════════════════════════
# 训练 / 评估
# ════════════════════════════════════════════════════════════════════════════════
 
def build_model():
    return EEGNet(
        num_classes=num_classes,
        n_channels=input_channels,
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        F1=8, D=2, F2=16,
        dropout=0.5,
    ).to(device)
 
 
def get_class_weights(loader):
    """从 DataLoader 自动计算类别权重（解决类别不平衡）"""
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
        optimizer, mode='max', factor=0.5, patience=3
    )
 
    best_val_bacc  = -1.0
    no_improve     = 0
    best_state     = None
 
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
# K-Fold 主循环（与 two_stage_training.py 的 k_fold_cross_validation 完全对齐）
# ════════════════════════════════════════════════════════════════════════════════
 
def k_fold_cross_validation():
    fold_loaders     = load_k_fold_data()
    all_fold_results = []
 
    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx+1}/{k_folds}  |  Dataset: {dataset_name}")
        print(f"{'='*50}")
 
        set_all_seeds(seed)
        best_state = train_one_fold(train_loader, val_loader)
 
        if best_state is None:
            all_fold_results.append({
                'fold': fold_idx + 1, 'status': 'failed',
                'test_metrics': None, 'tta_metrics': None
            })
            continue
 
        # 加载最佳模型
        model = build_model()
        model.load_state_dict(best_state['state_dict'])
 
        # Baseline 评测
        test_metrics = evaluate(model, test_loader)
        print_validation_results(test_metrics, fold_idx + 1,
                                 f"Fold {fold_idx+1} Baseline: ")
 
        # EEGNet 无 TTA，用相同结果填充（保持输出格式一致）
        tta_metrics = test_metrics
 
        all_fold_results.append({
            'fold':         fold_idx + 1,
            'test_metrics': test_metrics,
            'tta_metrics':  tta_metrics,
            'train_metrics': best_state.get('val_metrics', {}),
            'seed':          seed,
        })
 
        del model
        clear_gpu_memory()
 
    # ── 汇总（与 two_stage_training.py 输出格式完全相同）──────────────────────
    successful = [r for r in all_fold_results if r.get('status') != 'failed']
 
    if not successful:
        print("No successful folds completed")
        return
 
    print(f"\n{'='*50}")
    print(f"=== K-Fold Cross Validation Results ===")
    print(f"Dataset: {dataset_name}, K={k_folds}, Seed={seed}")
    print(f"Completed folds: {len(successful)}/{k_folds}")
 
    # 聚合 baseline 指标
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
 
    # EEGNet 无 TTA，打印相同结果保持格式统一
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
 
    # 保存结果
    import pickle, os
    os.makedirs("./results", exist_ok=True)
    save_path = f"./results/eegnet_{dataset_name}_seed{seed}.pkl"
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
    print(f"\n{'='*50}")
    print(f"EEGNet Baseline")
    print(f"Dataset:    {dataset_name}")
    print(f"Classes:    {num_classes}")
    print(f"Channels:   {input_channels}")
    print(f"Seq len:    {seq_len}")
    print(f"Subjects:   {num_subjects}")
    print(f"K-Folds:    {k_folds}")
    print(f"Device:     {device}")
    print(f"{'='*50}\n")
 
    k_fold_cross_validation()