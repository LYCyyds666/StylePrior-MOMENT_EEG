"""
preprocessing_bci2a.py（MNE版本）
==================================
BCI Competition IV Dataset 2a 预处理脚本
使用 MNE 正确读取 GDF 文件

依赖：pip install mne scipy numpy

用法：
    python preprocessing_bci2a.py --data_dir ./datasets/raw_bci2a --output_dir ./datasets/bci2a
"""

import os
import pickle
import argparse
import numpy as np
from math import gcd
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import Dataset
import torch

# ── 全局参数 ──────────────────────────────────────────────────────────────────
TARGET_SR  = 256
# BCI2a 标准协议：4秒 epoch，cue后0.5s开始取，共4秒
# 4s x 256Hz = 1024 点
SEQ_LEN    = 1024
N_CH_OUT   = 16
ORIG_SR    = 250
N_CLASSES  = 4
N_SUBJECTS = 9
SEL_CH     = list(range(16))
BP_LOW, BP_HIGH = 4.0, 40.0
TMIN_SEC   = 0.5   # cue 后 0.5s 开始
EPOCH_SEC  = 4.0   # 取 4 秒


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


def process_session(gdf_path, mat_label_path=None):
    """用 MNE 读取 GDF，返回 (X, y)"""
    import mne
    from scipy.io import loadmat

    print(f"    读取: {os.path.basename(gdf_path)}")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    # 带通滤波
    raw.filter(BP_LOW, BP_HIGH, fir_design='firwin', verbose=False)

    # 获取 events
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # MNE 把原始事件码映射为整数 ID，通过 event_id 字典反查
    # 原始码: 769=左手, 770=右手, 771=脚, 772=舌, 768=trial start
    # MNE 映射: '769'->7, '770'->8, '771'->9, '772'->10, '768'->6（见实际输出）
    # 用 event_id 字典动态构建映射，不硬编码，兼容不同版本 MNE
    label_map  = {}   # mne_event_id → class_label (0-3)
    cue_id_set = set()  # Session E 用的 cue event id

    for key_str, mne_id in event_id.items():
        try:
            orig_code = int(key_str)
        except (ValueError, TypeError):
            continue
        if orig_code == 769: label_map[mne_id] = 0
        if orig_code == 770: label_map[mne_id] = 1
        if orig_code == 771: label_map[mne_id] = 2
        if orig_code == 772: label_map[mne_id] = 3
        if orig_code in (768, 769, 770, 771, 772):
            cue_id_set.add(mne_id)

    if mat_label_path is not None and os.path.exists(mat_label_path):
        # Session E：找所有 MI 相关 cue（768~772），标签从 mat 读
        from scipy.io import loadmat as _loadmat
        mat   = _loadmat(mat_label_path)
        lbls  = mat['classlabel'].flatten().astype(np.int64) - 1  # 1-4 → 0-3

        cue_events = [ev[0] for ev in events if ev[2] in cue_id_set]

        trial_list = [(cue_events[i], int(lbls[i]))
                      for i in range(min(len(cue_events), len(lbls)))]
    else:
        # Session T：事件码直接映射为标签
        trial_list = [(ev[0], label_map[ev[2]])
                      for ev in events if ev[2] in label_map]

    if not trial_list:
        print(f"    ⚠ 未找到 trial events")
        return None, None

    # 获取原始数据
    data = raw.get_data()  # (n_ch, n_times)
    n_eeg = min(data.shape[0], 22)
    eeg   = data[:n_eeg, :]

    # 重采样
    eeg = do_resample(eeg, sfreq, TARGET_SR)
    eeg = eeg[SEL_CH, :]  # (16, N)

    sr_ratio  = TARGET_SR / sfreq
    tmin_samp = int(TMIN_SEC * TARGET_SR)
    epoch_len = SEQ_LEN

    X_list, y_list = [], []
    for raw_pos, label in trial_list:
        start = int(round(raw_pos * sr_ratio)) + tmin_samp
        end   = start + epoch_len
        if end > eeg.shape[1]:
            continue
        epoch = zscore(eeg[:, start:end].copy())
        X_list.append(epoch)
        y_list.append(label)

    if not X_list:
        return None, None

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def preprocess_all(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_data = {}

    for subj in range(1, N_SUBJECTS + 1):
        sid_s    = f"A{subj:02d}"
        t_path   = os.path.join(data_dir, f"{sid_s}T.gdf")
        e_path   = os.path.join(data_dir, f"{sid_s}E.gdf")
        mat_path = os.path.join(data_dir, f"{sid_s}E.mat")

        if not os.path.exists(t_path):
            print(f"  跳过 S{subj:02d}: 找不到 {t_path}")
            continue

        print(f"\n── Subject {subj} ──")
        Xt, yt = process_session(t_path)
        if Xt is None:
            continue
        print(f"    T: {Xt.shape[0]} trials, dist={dict(zip(*np.unique(yt, return_counts=True)))}")

        Xe, ye = None, None
        if os.path.exists(e_path) and os.path.exists(mat_path):
            Xe, ye = process_session(e_path, mat_path)
            if Xe is not None:
                print(f"    E: {Xe.shape[0]} trials, dist={dict(zip(*np.unique(ye, return_counts=True)))}")

        X_all = np.concatenate([Xt, Xe], 0) if Xe is not None else Xt
        y_all = np.concatenate([yt, ye], 0) if ye is not None else yt

        all_data[subj] = dict(X=X_all, y=y_all, X_train=Xt, y_train=yt, X_test=Xe, y_test=ye)
        np.save(os.path.join(output_dir, f"S{subj:02d}_X.npy"), X_all)
        np.save(os.path.join(output_dir, f"S{subj:02d}_y.npy"), y_all)

    pkl = os.path.join(output_dir, "bci2a_all.pkl")
    with open(pkl, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"\n✓ 保存到 {pkl}（{len(all_data)} 个受试者）")
    return all_data


class BCI2aDataset(Dataset):
    def __init__(self, samples):
        self.data        = np.stack([s[0] for s in samples]).astype(np.float32)
        self.labels      = np.array([s[1] for s in samples], dtype=np.int64)
        self.subject_ids = np.array([s[2] for s in samples], dtype=np.int64)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]),
                torch.tensor(self.labels[idx],      dtype=torch.long),
                torch.tensor(self.subject_ids[idx], dtype=torch.long))


def _samples(all_data, sids, train_only=False):
    out = []
    for sid in sids:
        d = all_data[sid]
        X = d['X_train'] if train_only and d['X_train'] is not None else d['X']
        y = d['y_train'] if train_only and d['y_train'] is not None else d['y']
        for i in range(len(X)):
            out.append((X[i], int(y[i]), sid))
    return out


def bci2a_loso_split(data_dir):
    pkl = os.path.join(data_dir, "bci2a_all.pkl")
    assert os.path.exists(pkl), f"找不到 {pkl}，请先运行预处理"
    with open(pkl, 'rb') as f:
        all_data = pickle.load(f)

    sids  = sorted(all_data.keys())
    folds = []
    for test_sid in sids:
        train_sids = [s for s in sids if s != test_sid]
        train_ds   = BCI2aDataset(_samples(all_data, train_sids))
        test_ds    = BCI2aDataset(_samples(all_data, [test_sid]))
        folds.append((train_ds, test_ds))
        print(f"  LOSO fold test=S{test_sid:02d}: train={len(train_ds)}, test={len(test_ds)}")
    return folds


def bci2a_official_split(data_dir):
    pkl = os.path.join(data_dir, "bci2a_all.pkl")
    assert os.path.exists(pkl), f"找不到 {pkl}，请先运行预处理"
    with open(pkl, 'rb') as f:
        all_data = pickle.load(f)

    folds = []
    for sid in sorted(all_data.keys()):
        d = all_data[sid]
        if d['X_test'] is None:
            print(f"  S{sid:02d} 无 Session E，跳过")
            continue
        train_ds = BCI2aDataset([(d['X_train'][i], int(d['y_train'][i]), sid) for i in range(len(d['X_train']))])
        test_ds  = BCI2aDataset([(d['X_test'][i],  int(d['y_test'][i]),  sid) for i in range(len(d['X_test']))])
        folds.append((train_ds, test_ds))
        print(f"  Official S{sid:02d}: train={len(train_ds)}, test={len(test_ds)}")
    return folds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="./datasets/raw_bci2a")
    ap.add_argument("--output_dir", default="./datasets/bci2a")
    args = ap.parse_args()

    preprocess_all(args.data_dir, args.output_dir)

    print("\n── shape 验证 ──")
    folds = bci2a_official_split(args.output_dir)
    x, y, sid = folds[0][0][0]
    assert tuple(x.shape) == (16, 1024), f"shape 错误: {x.shape}"
    print(f"✓ shape={tuple(x.shape)}, label={y.item()}, subject_id={sid.item()}")