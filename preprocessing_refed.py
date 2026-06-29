"""
preprocessing_refed.py
======================
REFED 数据集预处理脚本（修复版）
依赖：pip install scipy numpy torch
不需要 MNE / pyedflib
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集实际格式（已验证）】
    EEG_videos.mat：
        变量名 video_1 ~ video_15
        shape: (64, 134000)  → 64通道，1000Hz，约134秒
    annotations/{sid}_label.mat：
        变量名 video_1 ~ video_15
        shape: (134, 2)      → 134个时间点（每秒1个），2列=[valence, arousal]
        值范围: 0~200（SAM量表），中点100，>100=正向，≤100=负向
    标签与EEG对齐方式：每秒1个标签点对应1000个EEG样本
 
【任务定义】
    valence 二分类：均值 > 100 → label=1（正向），≤ 100 → label=0（负向）
    与 APAVA 保持二分类，num_classes=2
 
【用法】
    python preprocessing_refed.py --data_dir ./datasets/REFED --output_dir ./datasets/refed
"""
 
import os
import glob
import pickle
import argparse
import warnings
import numpy as np
from math import gcd
from scipy.signal import butter, filtfilt, resample_poly
from scipy.io import loadmat
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
import torch
 
warnings.filterwarnings('ignore')
 
# ── 全局参数（与 two_stage_training.py 完全对齐）─────────────────────────────
TARGET_SR   = 256    # 目标采样率
ORIG_SR     = 1000   # REFED EEG 原始采样率
SEQ_LEN     = 256    # 1秒窗口 = 256pts @ 256Hz
N_CH_OUT    = 16     # 模型输入通道数（从64选16）
N_SUBJECTS  = 32
N_TRIALS    = 15
N_CLASSES   = 2      # valence 二分类
 
# 滑动窗口
WINDOW_SEC = 1.0     # 1秒
STRIDE_SEC = 0.5     # 0.5秒步长，50% overlap
 
# 带通滤波（情绪EEG常用频段）
BP_LOW, BP_HIGH = 1.0, 45.0
 
# 标签参数
# shape (134, 2)：第0列=valence，第1列=arousal，值范围0~200
VALENCE_COL  = 0
LABEL_SR     = 1.0   # 标签采样率：每秒1个点
LABEL_MID    = 100.0 # SAM量表中点，>100=正向
 
# 从64通道均匀选16个（覆盖全脑）
N_EEG_CH_IN  = 64
SEL_CH_IDX   = np.linspace(0, N_EEG_CH_IN - 1, N_CH_OUT, dtype=int).tolist()
 
 
# ── 信号处理 ──────────────────────────────────────────────────────────────────
 
def bandpass(x, low, high, fs, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype='band')
    return filtfilt(b, a, x, axis=-1).astype(np.float32)
 
def do_resample(x, orig, target):
    if abs(orig - target) < 0.5:
        return x.astype(np.float32)
    g    = gcd(int(round(target)), int(round(orig)))
    up   = int(round(target)) // g
    down = int(round(orig))   // g
    return resample_poly(x, up, down, axis=-1).astype(np.float32)
 
def zscore(epoch):
    mu    = epoch.mean(axis=-1, keepdims=True)
    sigma = epoch.std(axis=-1,  keepdims=True) + 1e-8
    return (epoch - mu) / sigma
 
 
# ── MAT 读取 ──────────────────────────────────────────────────────────────────
 
def load_mat_safe(path):
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        try:
            import h5py
            data = {}
            with h5py.File(path, 'r') as f:
                for k in f.keys():
                    data[k] = np.array(f[k]).T
            return data
        except ImportError:
            raise RuntimeError(f"{path} 是 v7.3 格式，需要：pip install h5py")
 
 
# ── 单个受试者处理 ────────────────────────────────────────────────────────────
 
def process_subject(subj_id, data_dir):
    """
    处理单个受试者全部 trial
    返回 X (n_windows, 16, 256), y (n_windows,)
    """
    eeg_path   = os.path.join(data_dir, "data",        str(subj_id), "EEG_videos.mat")
    label_path = os.path.join(data_dir, "annotations", f"{subj_id}_label.mat")
 
    if not os.path.exists(eeg_path):
        print(f"    ⚠ 找不到 EEG: {eeg_path}")
        return None, None
    if not os.path.exists(label_path):
        print(f"    ⚠ 找不到标签: {label_path}")
        return None, None
 
    eeg_mat   = load_mat_safe(eeg_path)
    label_mat = load_mat_safe(label_path)
 
    X_all, y_all = [], []
 
    for trial_idx in range(N_TRIALS):
        key = f"video_{trial_idx + 1}"
 
        # ── 读取 EEG (64, n_times_orig) ──
        if key not in eeg_mat:
            print(f"    trial {trial_idx+1}: 找不到 EEG key '{key}'，跳过")
            continue
        eeg_raw = np.array(eeg_mat[key], dtype=np.float32)  # (64, 134000)
        if eeg_raw.ndim == 1:
            eeg_raw = eeg_raw[np.newaxis, :]
        if eeg_raw.shape[0] > eeg_raw.shape[1]:
            eeg_raw = eeg_raw.T   # 防止转置
 
        # ── 读取标签 (134, 2) ──
        if key not in label_mat:
            print(f"    trial {trial_idx+1}: 找不到标签 key '{key}'，跳过")
            continue
        lbl_raw = np.array(label_mat[key], dtype=np.float32)  # (134, 2)
        if lbl_raw.ndim == 1:
            lbl_raw = lbl_raw[:, np.newaxis]
        # 确保 shape (n_label_times, 2)，如果是 (2, n) 则转置
        if lbl_raw.shape[0] == 2 and lbl_raw.shape[1] != 2:
            lbl_raw = lbl_raw.T
        # valence 列
        valence_raw = lbl_raw[:, VALENCE_COL]  # (134,)
 
        # ── EEG 预处理：带通 → 重采样 → 选通道 ──
        n_label_pts = len(valence_raw)         # 134
        n_eeg_orig  = eeg_raw.shape[1]         # 134000
 
        # 用标签点数反推 EEG 实际秒数（防止长度不一致）
        trial_sec = n_label_pts / LABEL_SR     # 134 秒
 
        # 带通滤波
        eeg_filt = bandpass(eeg_raw, BP_LOW, BP_HIGH, ORIG_SR)
 
        # 裁剪 EEG 到与标签对应的整秒长度
        n_eeg_expected = int(trial_sec * ORIG_SR)
        if n_eeg_orig < n_eeg_expected:
            # EEG 比标签短，截断标签
            trial_sec       = n_eeg_orig / ORIG_SR
            n_label_pts_use = int(trial_sec * LABEL_SR)
            valence_raw     = valence_raw[:n_label_pts_use]
            eeg_filt        = eeg_filt[:, :n_eeg_orig]
        else:
            eeg_filt = eeg_filt[:, :n_eeg_expected]
 
        # 重采样到 TARGET_SR
        eeg_rs = do_resample(eeg_filt, ORIG_SR, TARGET_SR)  # (64, n_new)
 
        # 选16个通道
        # 根据实际通道数动态选择
        n_ch_actual = eeg_rs.shape[0]
        if n_ch_actual >= N_CH_OUT:
            idx    = np.linspace(0, n_ch_actual - 1, N_CH_OUT, dtype=int)
            eeg_16 = eeg_rs[idx, :]
        else:
            repeats = (N_CH_OUT + n_ch_actual - 1) // n_ch_actual
            eeg_16  = np.tile(eeg_rs, (repeats, 1))[:N_CH_OUT, :]
 
        # ── 标签插值到 TARGET_SR ──
        # valence_raw: 每秒1个点 → 插值到 TARGET_SR
        n_eeg_new   = eeg_16.shape[1]
        t_label     = np.linspace(0, 1, len(valence_raw))
        t_eeg       = np.linspace(0, 1, n_eeg_new)
        valence_interp = interp1d(t_label, valence_raw, kind='linear',
                                  fill_value='extrapolate')(t_eeg).astype(np.float32)
        # shape: (n_eeg_new,)
 
        # ── 滑动窗口切分 ──
        win_len    = SEQ_LEN                      # 256
        stride_len = int(STRIDE_SEC * TARGET_SR)  # 128
 
        start = 0
        while start + win_len <= n_eeg_new:
            end   = start + win_len
            epoch = eeg_16[:, start:end]          # (16, 256)
 
            # 对应的 valence 段
            val_seg  = valence_interp[start:end]
            val_mean = float(val_seg.mean())
 
            # 离散化：SAM 量表中点100，>100=正向
            label = 1 if val_mean > LABEL_MID else 0
 
            epoch = zscore(epoch)
            X_all.append(epoch)
            y_all.append(label)
            start += stride_len
 
    if not X_all:
        return None, None
 
    X = np.array(X_all, dtype=np.float32)   # (n_windows, 16, 256)
    y = np.array(y_all,  dtype=np.int64)
    return X, y
 
 
# ── 预处理所有受试者 ──────────────────────────────────────────────────────────
 
def preprocess_all(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_data = {}
 
    # 自动发现实际存在的受试者 ID（有的可能缺失）
    label_files = glob.glob(os.path.join(data_dir, "annotations", "*_label.mat"))
    subj_ids    = sorted([
        int(os.path.basename(f).split('_')[0])
        for f in label_files
        if os.path.basename(f).split('_')[0].isdigit()
    ])
    print(f"发现 {len(subj_ids)} 个受试者: {subj_ids}")
 
    for subj_id in subj_ids:
        print(f"\n── Subject {subj_id:02d} ──")
        X, y = process_subject(subj_id, data_dir)
 
        if X is None:
            print(f"    跳过")
            continue
 
        dist = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
        ratio = (y == 1).mean()
        print(f"    {X.shape[0]} windows, shape {X.shape}")
        print(f"    标签分布: {dist}, 正向比例: {ratio:.3f}")
 
        all_data[subj_id] = dict(X=X, y=y)
        np.save(os.path.join(output_dir, f"S{subj_id:02d}_X.npy"), X)
        np.save(os.path.join(output_dir, f"S{subj_id:02d}_y.npy"), y)
 
    pkl_path = os.path.join(output_dir, "refed_all.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_data, f)
 
    # 全局统计
    all_y = np.concatenate([v['y'] for v in all_data.values()])
    print(f"\n{'='*40}")
    print(f"✓ 完成，共 {len(all_data)} 个受试者，{len(all_y)} 个窗口")
    print(f"  全局标签分布: 正向={( all_y==1).sum()}, 负向={(all_y==0).sum()}")
    print(f"  正向比例: {(all_y==1).mean():.3f}")
    print(f"  保存到: {pkl_path}")
    return all_data
 
 
# ── Dataset 类（与 APAVA 完全对齐）───────────────────────────────────────────
 
class REFEDDataset(Dataset):
    """
    返回 (eeg_data, label, subject_id)
    - eeg_data  : Tensor (16, 256)  float32
    - label     : Tensor scalar     int64   0=负向, 1=正向
    - subject_id: Tensor scalar     int64   原始受试者ID（1-indexed）
    """
    def __init__(self, samples):
        self.data        = np.stack([s[0] for s in samples]).astype(np.float32)
        self.labels      = np.array([s[1] for s in samples], dtype=np.int64)
        self.subject_ids = np.array([s[2] for s in samples], dtype=np.int64)
 
    def __len__(self): return len(self.data)
 
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]),
            torch.tensor(self.labels[idx],      dtype=torch.long),
            torch.tensor(self.subject_ids[idx], dtype=torch.long),
        )
 
 
def _build_samples(all_data, sids):
    samples = []
    for sid in sids:
        d = all_data[sid]
        for i in range(len(d['X'])):
            samples.append((d['X'][i], int(d['y'][i]), sid))
    return samples
 
 
# ── 划分函数（与 apava_k_fold_split 接口完全相同）────────────────────────────
 
def refed_loso_split(data_dir):
    """
    Leave-One-Subject-Out
    返回 list of (train_dataset, test_dataset)
    """
    pkl_path = os.path.join(data_dir, "refed_all.pkl")
    assert os.path.exists(pkl_path), f"找不到 {pkl_path}，请先运行预处理"
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
 
    sids  = sorted(all_data.keys())
    folds = []
    for test_sid in sids:
        train_sids = [s for s in sids if s != test_sid]
        train_ds   = REFEDDataset(_build_samples(all_data, train_sids))
        test_ds    = REFEDDataset(_build_samples(all_data, [test_sid]))
        folds.append((train_ds, test_ds))
        dist = {0: int((all_data[test_sid]['y'] == 0).sum()),
                1: int((all_data[test_sid]['y'] == 1).sum())}
        print(f"  LOSO test=S{test_sid:02d}: "
              f"train={len(train_ds)}, test={len(test_ds)}, dist={dist}")
    return folds
 
 
def refed_k_fold_split(data_dir, k=5, random_state=42):
    """
    K-Fold 跨受试者（与 apava_k_fold_split 接口完全相同）
    推荐在受试者数多时用，比 LOSO 快
    """
    pkl_path = os.path.join(data_dir, "refed_all.pkl")
    assert os.path.exists(pkl_path), f"找不到 {pkl_path}，请先运行预处理"
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
 
    sids = sorted(all_data.keys())
    np.random.seed(random_state)
    sids_shuffled = np.random.permutation(sids).tolist()
 
    groups = [[] for _ in range(k)]
    for i, sid in enumerate(sids_shuffled):
        groups[i % k].append(sid)
 
    folds = []
    for fold_idx in range(k):
        test_sids  = groups[fold_idx]
        train_sids = [s for i, g in enumerate(groups) for s in g if i != fold_idx]
        train_ds   = REFEDDataset(_build_samples(all_data, train_sids))
        test_ds    = REFEDDataset(_build_samples(all_data, test_sids))
        folds.append((train_ds, test_ds))
        print(f"  K-Fold {fold_idx+1}/{k}: "
              f"test={test_sids}, train={len(train_ds)}, test={len(test_ds)}")
    return folds
 
 
# ── 入口 ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="./datasets/REFED")
    ap.add_argument("--output_dir", default="./datasets/refed")
    args = ap.parse_args()
 
    all_data = preprocess_all(args.data_dir, args.output_dir)
 
    # shape 验证
    print("\n── shape 验证 ──")
    folds = refed_loso_split(args.output_dir)
    x, y, sid = folds[0][0][0]
    assert tuple(x.shape) == (16, 256), f"❌ shape 错误: {x.shape}"
    assert y.item() in [0, 1],          f"❌ 标签错误: {y.item()}"
    print(f"✓ shape={tuple(x.shape)}, label={y.item()}, subject_id={sid.item()}")