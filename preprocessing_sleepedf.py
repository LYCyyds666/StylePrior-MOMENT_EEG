"""
preprocessing_sleepedf.py
=========================
Sleep-EDF-20 预处理脚本
依赖：pip install pyedflib scipy numpy
不需要 MNE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【下载清单】完全免费，无需注册
URL 根目录：https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
需要下载 20 对文件（共 40 个）：

    SC4001E0-PSG.edf    SC4001EC-Hypnogram.edf   <- Subject 1
    SC4011E0-PSG.edf    SC4011EC-Hypnogram.edf   <- Subject 2
    SC4021E0-PSG.edf    SC4021EC-Hypnogram.edf   <- Subject 3
    SC4031E0-PSG.edf    SC4031EC-Hypnogram.edf   <- Subject 4
    SC4041E0-PSG.edf    SC4041EC-Hypnogram.edf   <- Subject 5
    SC4051E0-PSG.edf    SC4051EC-Hypnogram.edf   <- Subject 6
    SC4061E0-PSG.edf    SC4061EC-Hypnogram.edf   <- Subject 7
    SC4071E0-PSG.edf    SC4071EC-Hypnogram.edf   <- Subject 8
    SC4081E0-PSG.edf    SC4081EC-Hypnogram.edf   <- Subject 9
    SC4091E0-PSG.edf    SC4091EC-Hypnogram.edf   <- Subject 10
    SC4101E0-PSG.edf    SC4101EC-Hypnogram.edf   <- Subject 11
    SC4111E0-PSG.edf    SC4111EC-Hypnogram.edf   <- Subject 12
    SC4121E0-PSG.edf    SC4121EC-Hypnogram.edf   <- Subject 13
    SC4131E0-PSG.edf    SC4131EC-Hypnogram.edf   <- Subject 14
    SC4141E0-PSG.edf    SC4141EC-Hypnogram.edf   <- Subject 15
    SC4151E0-PSG.edf    SC4151EC-Hypnogram.edf   <- Subject 16
    SC4161E0-PSG.edf    SC4161EC-Hypnogram.edf   <- Subject 17
    SC4171E0-PSG.edf    SC4171EC-Hypnogram.edf   <- Subject 18
    SC4181E0-PSG.edf    SC4181EC-Hypnogram.edf   <- Subject 19
    SC4191E0-PSG.edf    SC4191EC-Hypnogram.edf   <- Subject 20

服务器批量下载命令（直接粘贴执行）：
    mkdir -p raw_sleepedf && cd raw_sleepedf
    BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    for i in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19; do
        echo "正在下载受试者 $i ..."
        wget -nv -c "${BASE}/SC4${i}1E0-PSG.edf"
        wget -nv -c "${BASE}/SC4${i}1EC-Hypnogram.edf"
    done

注：若某个受试者 Night1 文件不存在，可尝试 Night2（把 *1E0* 换成 *2E0*）
    下载前可用以下命令查实际文件名：
    wget -qO- "${BASE}/RECORDS-PSG" | head -50

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【用法】
    python preprocessing_sleepedf.py --data_dir ./raw_sleepedf --output_dir ./data/sleepedf

【输出】
    ./data/sleepedf/
    ├── S01_X.npy  shape:(n_epochs, 16, 256)
    ├── S01_y.npy  shape:(n_epochs,)  int64 (0~4)
    ├── ...
    └── sleepedf_all.pkl   <- two_stage_training.py 调用入口

标签：0=W, 1=N1, 2=N2, 3=N3(含N4), 4=REM
"""

import os, re, glob, pickle, argparse
import numpy as np
from math import gcd
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import Dataset
import torch

# ── 全局参数（与 two_stage_training.py 完全对齐）─────────────────────────────
TARGET_SR  = 256    # 目标采样率，与 APAVA 一致
SEQ_LEN    = 256    # 1秒窗口（从30s epoch中心截取）
N_CH_OUT   = 16     # 输出通道数（2通道各重复8次）
ORIG_SR    = 100    # Sleep-EDF EEG 原始采样率
EPOCH_SEC  = 30     # 睡眠分期标准窗口
N_CLASSES  = 5      # W/N1/N2/N3/REM
N_EEG_CH   = 2      # Fpz-Cz, Pz-Oz

STAGE_MAP = {
    'Sleep stage W':  0,
    'Sleep stage 1':  1,
    'Sleep stage 2':  2,
    'Sleep stage 3':  3,
    'Sleep stage 4':  3,
    'Sleep stage R':  4,
    'Sleep stage ?': -1,
    'Movement time': -1,
}

BP_LOW, BP_HIGH = 0.5, 35.0


# ── 信号处理工具 ──────────────────────────────────────────────────────────────

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


# ── EDF 文件读取 ──────────────────────────────────────────────────────────────

def read_psg(psg_path):
    import pyedflib
    f      = pyedflib.EdfReader(psg_path)
    n_ch   = f.signals_in_file
    labels = [f.getLabel(i).strip() for i in range(n_ch)]
    sfreqs = [f.getSampleFrequency(i) for i in range(n_ch)]

    priority = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EEG FpzCz', 'EEG PzOz']
    eeg_idx  = []
    for target in priority:
        for i, lbl in enumerate(labels):
            if target.lower() in lbl.lower() and i not in eeg_idx:
                eeg_idx.append(i)
                break
    if len(eeg_idx) < 2:
        for i, lbl in enumerate(labels):
            if 'EEG' in lbl.upper() and i not in eeg_idx:
                eeg_idx.append(i)
            if len(eeg_idx) == 2:
                break
    while len(eeg_idx) < 2 and len(eeg_idx) < n_ch:
        eeg_idx.append(len(eeg_idx))
    eeg_idx = eeg_idx[:N_EEG_CH]

    sfreq   = float(sfreqs[eeg_idx[0]])
    signals = np.array([f.readSignal(i).astype(np.float32) for i in eeg_idx])
    f.close()
    return signals, sfreq


def read_hypnogram(hyp_path):
    import pyedflib
    f = pyedflib.EdfReader(hyp_path)
    try:
        onsets, durs, descs = f.readAnnotations()
    except Exception:
        f.close()
        return []
    f.close()

    stages = []
    for onset, dur, desc in zip(onsets, durs, descs):
        if isinstance(desc, bytes):
            desc = desc.decode('utf-8', errors='ignore')
        desc  = desc.strip()
        label = STAGE_MAP.get(desc, -1)
        stages.append((float(onset), float(dur), label))
    return stages


# ── 单对文件处理 ──────────────────────────────────────────────────────────────

def process_pair(psg_path, hyp_path):
    print(f"    读取: {os.path.basename(psg_path)}")
    eeg, sfreq = read_psg(psg_path)
    eeg = bandpass(eeg, BP_LOW, BP_HIGH, sfreq)
    eeg = do_resample(eeg, sfreq, TARGET_SR)
    n_total = eeg.shape[1]

    stages = read_hypnogram(hyp_path)
    if not stages:
        print(f"    ⚠ Hypnogram 为空")
        return None, None

    epoch_samp  = int(EPOCH_SEC * TARGET_SR)   # 7680
    center_half = SEQ_LEN // 2                  # 128

    X_list, y_list = [], []
    for onset_sec, dur_sec, label in stages:
        if label < 0:
            continue
        n_sub = max(1, int(round(dur_sec / EPOCH_SEC)))
        for sub_i in range(n_sub):
            start  = int(round((onset_sec + sub_i * EPOCH_SEC) * TARGET_SR))
            center = start + epoch_samp // 2
            s_snip = center - center_half
            e_snip = center + center_half
            if s_snip < 0 or e_snip > n_total:
                continue
            snippet   = eeg[:, s_snip:e_snip]                          # (2, 256)
            snippet16 = np.repeat(snippet, N_CH_OUT // N_EEG_CH, axis=0)  # (16, 256)
            snippet16 = zscore(snippet16)
            X_list.append(snippet16)
            y_list.append(label)

    if not X_list:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def parse_sid(filename):
    m = re.search(r'SC4(\d{2})', os.path.basename(filename))
    return int(m.group(1)) + 1 if m else None


# ── 预处理所有受试者 ──────────────────────────────────────────────────────────

def preprocess_all(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*PSG.edf")))
    if not psg_files:
        psg_files = sorted(glob.glob(os.path.join(data_dir, "**/*PSG.edf"), recursive=True))
    print(f"找到 {len(psg_files)} 个 PSG 文件")

    all_data = {}
    for psg_path in psg_files:
        hyp_path = psg_path.replace("E0-PSG.edf", "EC-Hypnogram.edf")
        if not os.path.exists(hyp_path):
            hyp_path = psg_path.replace("-PSG.edf", "-Hypnogram.edf")
        if not os.path.exists(hyp_path):
            print(f"  跳过: 找不到 Hypnogram for {os.path.basename(psg_path)}")
            continue

        sid = parse_sid(psg_path)
        if sid is None:
            continue

        print(f"\n── Subject {sid:02d} ──")
        X, y = process_pair(psg_path, hyp_path)
        if X is None:
            continue

        dist = {i: int((y == i).sum()) for i in range(N_CLASSES)}
        print(f"    {X.shape[0]} epochs，分布: {dist}")

        if sid in all_data:
            all_data[sid]['X'] = np.concatenate([all_data[sid]['X'], X])
            all_data[sid]['y'] = np.concatenate([all_data[sid]['y'], y])
        else:
            all_data[sid] = dict(X=X, y=y)

        np.save(os.path.join(output_dir, f"S{sid:02d}_X.npy"), all_data[sid]['X'])
        np.save(os.path.join(output_dir, f"S{sid:02d}_y.npy"), all_data[sid]['y'])

    pkl = os.path.join(output_dir, "sleepedf_all.pkl")
    with open(pkl, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"\n✓ 保存到 {pkl}（{len(all_data)} 个受试者）")
    return all_data


# ── Dataset 类（与 APAVA 完全对齐）───────────────────────────────────────────

class SleepEDFDataset(Dataset):
    """
    返回 (eeg_data, label, subject_id)
    - eeg_data  : Tensor (16, 256)  float32
    - label     : Tensor scalar     int64  0=W,1=N1,2=N2,3=N3,4=REM
    - subject_id: Tensor scalar     int64  1-indexed
    """
    def __init__(self, samples):
        self.data        = np.stack([s[0] for s in samples]).astype(np.float32)
        self.labels      = np.array([s[1] for s in samples], dtype=np.int64)
        self.subject_ids = np.array([s[2] for s in samples], dtype=np.int64)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]),
                torch.tensor(self.labels[idx],      dtype=torch.long),
                torch.tensor(self.subject_ids[idx], dtype=torch.long))


def _samples(all_data, sids):
    out = []
    for sid in sids:
        d = all_data[sid]
        for i in range(len(d['X'])):
            out.append((d['X'][i], int(d['y'][i]), sid))
    return out


# ── 划分函数（格式与 apava_k_fold_split 完全相同）────────────────────────────

def sleepedf_loso_split(data_dir):
    """
    Subject-Independent LOSO 划分
    返回 list of (train_dataset, test_dataset)，共 20 个 fold
    直接替换 two_stage_training.py 里的 apava_k_fold_split
    """
    pkl = os.path.join(data_dir, "sleepedf_all.pkl")
    assert os.path.exists(pkl), f"找不到 {pkl}，请先运行预处理"
    with open(pkl, 'rb') as f:
        all_data = pickle.load(f)

    sids  = sorted(all_data.keys())
    folds = []
    for test_sid in sids:
        train_sids = [s for s in sids if s != test_sid]
        train_ds   = SleepEDFDataset(_samples(all_data, train_sids))
        test_ds    = SleepEDFDataset(_samples(all_data, [test_sid]))
        folds.append((train_ds, test_ds))
        dist = {i: int((all_data[test_sid]['y'] == i).sum()) for i in range(N_CLASSES)}
        print(f"  LOSO test=S{test_sid:02d}: train={len(train_ds)}, test={len(test_ds)}, dist={dist}")
    return folds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="./raw_sleepedf")
    ap.add_argument("--output_dir", default="./data/sleepedf")
    args = ap.parse_args()

    preprocess_all(args.data_dir, args.output_dir)

    print("\n── shape 验证 ──")
    folds = sleepedf_loso_split(args.output_dir)
    x, y, sid = folds[0][0][0]
    assert tuple(x.shape) == (16, 256), f"❌ shape 错误: {x.shape}"
    print(f"✓ shape={tuple(x.shape)}, label={y.item()}, subject_id={sid.item()}")
