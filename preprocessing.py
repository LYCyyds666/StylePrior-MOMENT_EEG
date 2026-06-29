from torcheeg import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import random
import hashlib


class APAVADataset(Dataset):
    def __init__(self, root_path, subset='all', online_transform=None, label_transform=None,
                 use_cache=True, cache_dir='./cache', train_ids=None, test_ids=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        self.online_transform = online_transform
        self.label_transform = label_transform
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        data_list = np.load(self.label_path)
        all_ids = list(data_list[:, 1].astype(int))

        self.train_ids, self.test_ids = train_ids, test_ids

        if subset == 'train':
            self.ids = self.train_ids
            self.subset_name = 'train'
        elif subset == 'test':
            self.ids = self.test_ids
            self.subset_name = 'test'
        else:
            self.ids = all_ids
            self.subset_name = 'all'

        ids_str = '_'.join(map(str, sorted(self.ids)))
        cache_key = f"{self.subset_name}_{ids_str}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        self.cache_file = os.path.join(self.cache_dir, f'apava_{self.subset_name}_{cache_hash}.pkl')

        self.cache_key = cache_key
        self.cache_hash = cache_hash

        self.X, self.y, self.subject_ids = self.load_apava()

        self.data = self.X
        self.labels = self.y

    def load_apava(self):
        if self.use_cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['X'], cached_data['y'], cached_data['subject_ids']

        feature_list = []
        label_list = []
        subject_id_list = []
        filenames = []

        subject_label = np.load(self.label_path)

        for filename in os.listdir(self.data_path):
            filenames.append(filename)
        filenames.sort()

        for j in range(len(filenames)):
            if j + 1 in self.ids:
                trial_label = subject_label[j]
                path = os.path.join(self.data_path, filenames[j])
                subject_feature = np.load(path)

                for trial_feature in subject_feature:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
                    subject_id_list.append(j + 1)

        X = np.array(feature_list)
        y = np.array(label_list)
        subject_ids = np.array(subject_id_list)

        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        subject_ids = subject_ids[indices]

        X = X.transpose(0, 2, 1)  # [samples, channels, time]

        if self.use_cache:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            with open(self.cache_file, 'wb') as f:
                cache_data = {
                    'X': X,
                    'y': y[:, 0],
                    'subject_ids': subject_ids,
                    'cache_key': self.cache_key,
                    'subset_name': self.subset_name,
                    'ids': sorted(self.ids)
                }
                pickle.dump(cache_data, f)

        return X, y[:, 0], subject_ids

    def get_subject_id(self, index):

        filenames = []
        for filename in os.listdir(self.data_path):
            filenames.append(filename)
        filenames.sort()

        cumulative_samples = 0
        for j, filename in enumerate(filenames):
            subject_feature = np.load(os.path.join(self.data_path, filename))
            subject_sample_count = len(subject_feature)
            if index < cumulative_samples + subject_sample_count:
                return j + 1
            cumulative_samples += subject_sample_count
        return 1

    def __getitem__(self, index):
        eeg_data = self.X[index].astype(np.float32)
        label = self.y[index].astype(np.int64)

        subject_id = self.subject_ids[index]

        if self.online_transform:
            sample = self.online_transform(eeg=eeg_data)
            eeg_data = sample['eeg']

        if self.label_transform:
            label = self.label_transform(label)

        return eeg_data, label, subject_id

    def __len__(self):
        return len(self.y)


def get_random_subject_split(all_subject_ids, test_ratio=0.2, random_state=42):
    random.seed(random_state)

    subject_ids = all_subject_ids.copy()
    random.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    n_test = max(1, int(n_subjects * test_ratio))
    n_train = n_subjects - n_test

    if n_train <= 0:
        n_train = 1
        n_test = n_subjects - n_train

    test_ids = subject_ids[:n_test]
    train_ids = subject_ids[n_test:]

    return train_ids, test_ids


def get_k_fold_subject_splits(all_subject_ids, k=5, random_state=42):
    random.seed(random_state)

    subject_ids = all_subject_ids.copy()
    random.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    fold_size = n_subjects // k
    remainder = n_subjects % k

    folds = []
    start_idx = 0

    for fold_idx in range(k):
        current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + current_fold_size

        test_ids = subject_ids[start_idx:end_idx]

        train_ids = subject_ids[:start_idx] + subject_ids[end_idx:]

        folds.append((train_ids, test_ids))

        print(f"  Train subjects: {sorted(train_ids)} (n={len(train_ids)})")
        print(f"  Test subjects: {sorted(test_ids)} (n={len(test_ids)})")
        print()

        start_idx = end_idx

    return folds


def apava_k_fold_split(k=5, random_state=42, use_cache=True):
    data_list = np.load('./datasets/APAVA/Label/label.npy')
    all_subject_ids = list(set(data_list[:, 1].astype(int)))

    fold_splits = get_k_fold_subject_splits(all_subject_ids, k=k, random_state=random_state)

    fold_datasets = []

    for fold_idx, (train_ids, test_ids) in enumerate(fold_splits):
        print("Creating datasets")

        train_dataset = APAVADataset(
            root_path='./datasets/APAVA',
            subset='train',
            online_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            label_transform=None,
            use_cache=use_cache,
            train_ids=train_ids,
            test_ids=test_ids
        )

        test_dataset = APAVADataset(
            root_path='./datasets/APAVA',
            subset='test',
            online_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            label_transform=None,
            use_cache=use_cache,
            train_ids=train_ids,
            test_ids=test_ids
        )

        fold_datasets.append((train_dataset, test_dataset))
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Test dataset: {len(test_dataset)} samples")
        print()

    return fold_datasets


def apava_cross_subject_split(dataset=None, test_size=0.2, random_state=42, use_cache=True):
    data_list = np.load('./datasets/APAVA/Label/label.npy')
    all_subject_ids = list(set(data_list[:, 1].astype(int)))
    
    train_ids, test_ids = get_random_subject_split(
        all_subject_ids, test_ratio=test_size, random_state=random_state
    )
    
    train_dataset = APAVADataset(
        root_path='./datasets/APAVA',
        subset='train',
        online_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        label_transform=None,
        use_cache=use_cache,
        train_ids=train_ids,
        test_ids=test_ids
    )
    
    test_dataset = APAVADataset(
        root_path='./datasets/APAVA',
        subset='test',
        online_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        label_transform=None,
        use_cache=use_cache,
        train_ids=train_ids,
        test_ids=test_ids
    )
    
    return train_dataset, test_dataset

def get_dataset(data_name='APAVA'):
    if data_name == "APAVA":
        dataset = APAVADataset(
            root_path='./datasets/APAVA',
            online_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            label_transform=None
        )
        return dataset
    else:
        raise ValueError("Unsupported dataset")
