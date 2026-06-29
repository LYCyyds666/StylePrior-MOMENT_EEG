import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import apava_k_fold_split
import sys
sys.path.append('./MoE_moment')
import torch.nn.functional as F
from preprocessing import apava_k_fold_split
from preprocessing_refed import refed_loso_split, refed_k_fold_split
from MoE_moment.momentfm.models.SS_MOMENT import SageStreamPipeline
from MoE_moment.momentfm.models.layers.SA_MoE import StyleAdaptor

import numpy as np

from utils import set_all_seeds, compute_comprehensive_metrics,\
      print_validation_results, clear_gpu_memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "MOMENT-1-small"
dataset_name = "refed"
seed = 2025
batch_size = 32  
reduction = "concat"
epochs = 30
learning_rate = 5e-5
weight_decay = 1e-5

if dataset_name == "APAVA":
    seq_len = 256
    input_channels = 16
    num_classes = 2
    num_subjects = 23
    sampling_rate = 256.0
elif dataset_name == "refed":   # ← ADD 整个 elif 块
    seq_len = 256
    input_channels = 16
    num_classes = 2              # valence 二分类
    num_subjects = 32
    sampling_rate = 256.0
    k_folds = 32                 # LOSO = 32 folds（或改为5做K-Fold）
    data_dir = "./datasets/refed"
else:
    raise ValueError("Unsupported dataset")

expert_dim_ratio = 1/8
max_freq = 100.0

enable_subject_style_normalization = True
subject_embedding_dim = 64

aux_loss_weight = 0.001

decoupling_config = {
    'shared_config': {
        'num_experts': 5,
        'top_k': 2,
        'dropout': 0.1,
        'freq_learning_mode': 'lightweight_biomedical_filter',
        'routing_strategy': 'simple',
        'expert_dim_ratio': expert_dim_ratio,
        'max_freq': max_freq,
        'sampling_rate': sampling_rate,
        'aux_loss_weight': 1.0,
        'enable_shared_backbone_hypernetwork': enable_subject_style_normalization,
        'num_subjects': num_subjects,
        'subject_embedding_dim': subject_embedding_dim,
        'expert_embedding_dim': 32,
        'hyper_expert_hidden_dim': 64,
        'num_channels': input_channels,
        'moe_conditioning_dim': 64,
    },
}

early_stop = 5

enable_k_fold = True
k_folds = 5

enable_tta_in_kfold = True
tta_method = "STSA"
tta_learning_rate = 5e-4
tta_batch_size = 64



set_all_seeds(seed)

def load_k_fold_data_with_validation():
    if dataset_name == "APAVA":
        print("Creating APAVA k-fold datasets (cross-subject)...")
        fold_datasets = apava_k_fold_split(
            k=k_folds, random_state=seed, use_cache=True
        )
    elif dataset_name == "refed":   # ← ADD 整个 elif 块
        print("Creating REFED LOSO datasets (cross-subject)...")
        # LOSO（推荐，体现跨被试泛化性）：
        fold_datasets = refed_loso_split(data_dir)
        # K-Fold（可选，和 APAVA 的5折对齐）：
        # fold_datasets = refed_k_fold_split(data_dir, k=k_folds, random_state=seed)
    fold_loaders = []
    for fold_idx, (original_train_dataset, test_dataset) in enumerate(fold_datasets):
        train_dataset, val_dataset = split_dataset_by_subject(
            original_train_dataset, train_ratio=0.75, random_state=seed
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, drop_last=False, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True
        )

        fold_loaders.append((train_loader, val_loader, test_loader))

    return fold_loaders

def split_dataset_by_subject(dataset, train_ratio=0.75, random_state=42):
    import numpy as np
    from torch.utils.data import Subset

    all_subject_ids = np.array(dataset.subject_ids)
    unique_subjects = np.unique(all_subject_ids)

    np.random.seed(random_state)
    np.random.shuffle(unique_subjects)

    n_train_subjects = int(len(unique_subjects) * train_ratio)
    train_subjects = unique_subjects[:n_train_subjects]
    val_subjects = unique_subjects[n_train_subjects:]

    train_indices = []
    val_indices = []

    for i in range(len(dataset)):
        subject_id = all_subject_ids[i]
        if subject_id in train_subjects:
            train_indices.append(i)
        else:
            val_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"  Subject split: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects")

    return train_dataset, val_dataset

def load_k_fold_data():
    return load_k_fold_data_with_validation()

def print_classification_metrics(metrics, title="Classification Metrics"):
    print(f"\n=== {title} ===")
    print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    print(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"F1 Macro: {metrics.get('f1_macro', 0.0):.4f}")
    # print(f"Cohen Kappa: {metrics.get('cohen_kappa', 0.0):.4f}")
    # print(f"ROC AUC: {metrics.get('roc_auc', 0.0):.4f}")
    print(f"Precision Macro: {metrics.get('precision_macro', 0.0):.4f}")
    print(f"Recall Macro: {metrics.get('recall_macro', 0.0):.4f}")

    if 'average_precision' in metrics:
        print(f"Average Precision: {metrics.get('average_precision', 0.0):.4f}")
    # if 'matthews_corrcoef' in metrics:
    #     print(f"Matthews Corrcoef: {metrics.get('matthews_corrcoef', 0.0):.4f}")
    # if 'jaccard_macro' in metrics:
    #     print(f"Jaccard Macro: {metrics.get('jaccard_macro', 0.0):.4f}")

    if 'f1_binary' in metrics:
        print(f"F1 Binary: {metrics.get('f1_binary', 0.0):.4f}")
        print(f"Precision Binary: {metrics.get('precision_binary', 0.0):.4f}")
        print(f"Recall Binary: {metrics.get('recall_binary', 0.0):.4f}")
    print()

def train_moment_model_with_validation(train_loader, val_loader):
    model_kwargs = {
        "task_name": "classification",
        "n_channels": input_channels,
        "num_class": num_classes,
        "freeze_embedder": True,
        "freeze_encoder": True,
        "freeze_head": False,
        "seq_len": seq_len,
        "reduction": reduction,
        "add_positional_embedding": False
    }

    moment_model = SageStreamPipeline.from_pretrained(
        model_path="./"+model_name,
        decoupling_config=decoupling_config,
        model_kwargs=model_kwargs
    ).to(device)

    moment_model.task_name = "classification"
    moment_model.set_training_stage("source_domain")

    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        moment_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3 #verbose=True
    )

    best_val_balanced_acc = -1.0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        moment_model.train()
        train_preds, train_labels, train_probs = [], [], []

        for _, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                eeg_data, labels, subject_ids = batch_data
                subject_ids = subject_ids.to(device)
            else:
                eeg_data, labels = batch_data
                subject_ids = None

            eeg_data = eeg_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = moment_model.classify(x_enc=eeg_data, subject_ids=subject_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                aux_loss = getattr(outputs, 'aux_loss', 0.0)
            else:
                logits = outputs
                aux_loss = 0.0

            classification_loss = classification_criterion(logits, labels)

            if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
                total_loss_batch = classification_loss + aux_loss_weight * aux_loss
            else:
                total_loss_batch = classification_loss

            total_loss_batch.backward()
            optimizer.step()
            # --- 关键修复：在这里添加清理代码 ---
            # 遍历模型中所有的 block 并清理 aux_loss
            for block in moment_model.model.encoder.block:
                if hasattr(block, 'clear_aux_losses'):
                    block.clear_aux_losses()
            # ------------------------------------
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

            if num_classes == 2:
                train_probs.extend(probs[:, 1].detach().cpu().numpy())
            else:
                train_probs.append(probs.detach().cpu().numpy())

        if num_classes == 2:
            train_probs_array = np.array(train_probs)
        else:
            train_probs_array = np.concatenate(train_probs, axis=0)

        train_metrics = compute_comprehensive_metrics(
            train_labels, train_preds, train_probs_array, num_classes
        )

        moment_model.eval()
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    eeg_data, labels, subject_ids = batch_data
                    subject_ids = subject_ids.to(device)
                else:
                    eeg_data, labels = batch_data
                    subject_ids = None

                eeg_data = eeg_data.to(device)
                labels = labels.to(device)

                outputs = moment_model.classify(x_enc=eeg_data, subject_ids=subject_ids)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

                if num_classes == 2:
                    val_probs.extend(probs[:, 1].detach().cpu().numpy())
                else:
                    val_probs.append(probs.detach().cpu().numpy())

        if num_classes == 2:
            val_probs_array = np.array(val_probs)
        else:
            val_probs_array = np.concatenate(val_probs, axis=0)

        val_metrics = compute_comprehensive_metrics(
            val_labels, val_preds, val_probs_array, num_classes
        )

        scheduler.step(val_metrics['balanced_accuracy'])

        print(f"Epoch {epoch+1}/{epochs}: Train: ", end="")
        print_validation_results(train_metrics)
        print("Val: ", end="")
        print_validation_results(val_metrics)

        if val_metrics['balanced_accuracy'] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics['balanced_accuracy']
            epochs_without_improvement = 0

            best_model_state = {
                'epoch': epoch + 1,
                'moment_model_state_dict': moment_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_balanced_acc': best_val_balanced_acc
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop:
            break

    if best_model_state:
        print('\nfinished')
        return best_model_state
    else:
        return None

def evaluate_on_test_set(model, test_loader):
    model.eval()
    test_preds, test_labels, test_probs = [], [], []

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                eeg_data, labels, subject_ids = batch_data
                subject_ids = subject_ids.to(device)
            else:
                eeg_data, labels = batch_data
                subject_ids = None

            eeg_data = eeg_data.to(device)
            labels = labels.to(device)

            outputs = model.classify(x_enc=eeg_data, subject_ids=subject_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.detach().cpu().numpy())
            test_labels.extend(labels.detach().cpu().numpy())

            if num_classes == 2:
                test_probs.extend(probs[:, 1].detach().cpu().numpy())
            else:
                test_probs.append(probs.detach().cpu().numpy())

    if num_classes == 2:
        test_probs_array = np.array(test_probs)
    else:
        test_probs_array = np.concatenate(test_probs, axis=0)

    test_metrics = compute_comprehensive_metrics(
        test_labels, test_preds, test_probs_array, num_classes
    )

    test_metrics['predictions'] = test_preds
    test_metrics['true_labels'] = test_labels
    test_metrics['probabilities'] = test_probs_array

    return test_metrics

def single_fold_training(fold_idx, train_loader, val_loader, test_loader):
    best_model = train_moment_model_with_validation(train_loader, val_loader)
    return best_model

def k_fold_single_training(fold_idx, train_loader, val_loader, test_loader):
    fold_seed = seed
    set_all_seeds(fold_seed)

    best_model = single_fold_training(fold_idx, train_loader, val_loader, test_loader)

    if best_model:
        fold_model_path = f'./best_model_fold_{fold_idx + 1}.pth'
        torch.save(best_model, fold_model_path)

        model_kwargs = {
            "task_name": "classification",
            "n_channels": input_channels,
            "num_class": num_classes,
            "freeze_embedder": True,
            "freeze_encoder": True,
            "freeze_head": False,
            "seq_len": seq_len,
            "reduction": reduction,
            "add_positional_embedding": False
        }

        eval_model = SageStreamPipeline.from_pretrained(
            model_path="./"+model_name,
            decoupling_config=decoupling_config,
            model_kwargs=model_kwargs
        ).to(device)

        eval_model.load_state_dict(best_model['moment_model_state_dict'])

        train_subject_ids = []
        test_subject_ids = []

        for batch_data in train_loader:
            if len(batch_data) == 3:
                _, _, subject_ids = batch_data
                train_subject_ids.extend(subject_ids.tolist())

        for batch_data in test_loader:
            if len(batch_data) == 3:
                _, _, subject_ids = batch_data
                test_subject_ids.extend(subject_ids.tolist())

        train_subject_ids = sorted(list(set(train_subject_ids)))
        test_subject_ids = sorted(list(set(test_subject_ids)))

        initialize_unknown_subject_embeddings(eval_model, train_subject_ids, test_subject_ids)

        test_metrics = evaluate_on_test_set(eval_model, test_loader)

        best_model['test_metrics'] = test_metrics

        print_validation_results(test_metrics, fold_idx + 1, f"Fold {fold_idx + 1} Baseline: ")

        tta_metrics = None
        if enable_tta_in_kfold:
            tta_model = eval_model
            if tta_method == "STSA":
                tta_result = STSA(
                    tta_model, test_loader,
                    tta_lr=tta_learning_rate,
                    tta_steps_per_batch=1,
                    tta_batch_size=tta_batch_size
                )

            if tta_result:
                tta_metrics = tta_result['overall_metrics']
                print_validation_results(tta_metrics, fold_idx + 1, f"Fold {fold_idx + 1} TTA: ")

            del eval_model
            del tta_result
            clear_gpu_memory()
        else:
            del eval_model
            clear_gpu_memory()

        fold_result = {
            'fold': fold_idx + 1,
            'model_path': fold_model_path,
            'test_metrics': test_metrics,
            'tta_metrics': tta_metrics,
            'train_metrics': best_model.get('train_metrics', {}),
            'seed': fold_seed
        }

        return fold_result

    return None

def k_fold_cross_validation():
    fold_loaders = load_k_fold_data()

    all_fold_results = []

    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
        fold_result = k_fold_single_training(fold_idx, train_loader, val_loader, test_loader)
        if fold_result:
            all_fold_results.append(fold_result)
        else:
            failed_fold_result = {
                'fold': fold_idx + 1,
                'status': 'failed',
                'error': 'Training returned None',
                'test_metrics': None,
                'tta_metrics': None
            }
            all_fold_results.append(failed_fold_result)

    successful_folds = [r for r in all_fold_results if r.get('status') != 'failed']
    failed_folds = [r for r in all_fold_results if r.get('status') == 'failed']

    if successful_folds:
        print(f"\n=== K-Fold Cross Validation Results ===")
        print(f"Dataset: {dataset_name}, K={k_folds}, Seed={seed}")
        print(f"Completed folds: {len(successful_folds)}/{k_folds}")
        
        # Calculate mean metrics
        all_metrics = {}
        for fold_result in successful_folds:
            test_metrics = fold_result.get('test_metrics', {})
            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Calculate mean and std for each metric
        mean_metrics = {}
        for key, values in all_metrics.items():
            mean_metrics[key] = np.mean(values)
            mean_metrics[f'{key}_std'] = np.std(values)
        
        # Print baseline results
        print("\n🏆 Baseline Metrics:")
        print(f"  Accuracy:     {mean_metrics.get('accuracy', 0.0):.4f}")
        print(f"  Balanced Accuracy: {mean_metrics.get('balanced_accuracy', 0.0):.4f}")
        print(f"  F1 Score (Macro): {mean_metrics.get('f1_macro', 0.0):.4f}")
        print(f"  Precision (Macro): {mean_metrics.get('precision_macro', 0.0):.4f}")
        print(f"  Recall (Macro): {mean_metrics.get('recall_macro', 0.0):.4f}")
        if 'roc_auc' in mean_metrics:
            print(f"  ROC AUC:    {mean_metrics.get('roc_auc', 0.0):.4f}")
        if 'average_precision' in mean_metrics:
            print(f"  Avg Prec:   {mean_metrics.get('average_precision', 0.0):.4f}")
        
        # Print TTA results if available
        tta_folds = [r for r in successful_folds if r.get('tta_metrics') is not None]
        if tta_folds and enable_tta_in_kfold:
            tta_metrics = {}
            for fold_result in tta_folds:
                tta_result = fold_result.get('tta_metrics', {})
                for key, value in tta_result.items():
                    if isinstance(value, (int, float)):
                        if key not in tta_metrics:
                            tta_metrics[key] = []
                        tta_metrics[key].append(value)
            
            # Calculate mean TTA metrics
            mean_tta_metrics = {}
            for key, values in tta_metrics.items():
                mean_tta_metrics[key] = np.mean(values)
                mean_tta_metrics[f'{key}_std'] = np.std(values)

            print("\n🚀 TTA Metrics:")
            print(f"  Accuracy:     {mean_tta_metrics.get('accuracy', 0.0):.4f}")
            print(f"  Balanced Accuracy: {mean_tta_metrics.get('balanced_accuracy', 0.0):.4f}")
            print(f"  F1 Score (Macro): {mean_tta_metrics.get('f1_macro', 0.0):.4f}")
            print(f"  Precision (Macro): {mean_tta_metrics.get('precision_macro', 0.0):.4f}")
            print(f"  Recall (Macro): {mean_tta_metrics.get('recall_macro', 0.0):.4f}")
            if 'roc_auc' in mean_tta_metrics:
                print(f"  ROC AUC:    {mean_tta_metrics.get('roc_auc', 0.0):.4f}")
            if 'average_precision' in mean_tta_metrics:
                print(f"  Avg Prec:   {mean_tta_metrics.get('average_precision', 0.0):.4f}")
        
        # Save simple summary
        summary_result = {
            'k_folds': k_folds,
            'dataset': dataset_name,
            'seed': seed,
            'completed_folds': len(successful_folds),
            'baseline_metrics': mean_metrics,
            'tta_metrics': mean_tta_metrics if tta_folds else None,
            'fold_results': all_fold_results
        }
        
        return summary_result
    else:
        print("No successful folds completed")
        failed_summary = {
            'k_folds': k_folds,
            'dataset': dataset_name,
            'seed': seed,
            'num_classes': num_classes,
            'fold_results': all_fold_results,
            'summary': {
                'completed_folds': len(successful_folds),
                'failed_folds': len(failed_folds),
                'status': 'all_failed'
            }
        }
        
        return failed_summary

def initialize_unknown_subject_embeddings(model, train_subject_ids, test_subject_ids):
    train_subjects = set(train_subject_ids)
    test_subjects = set(test_subject_ids)
    unknown_subjects = test_subjects - train_subjects

    if len(unknown_subjects) == 0:
        return

    subject_embedding_modules = []
    for block in model.model.encoder.block:
        if hasattr(block, 'shared_knowledge') and hasattr(block.shared_knowledge, 'subject_embedding'):
            if hasattr(block.shared_knowledge.subject_embedding, 'weight'):
                subject_embedding_modules.append(block.shared_knowledge.subject_embedding)

    for i, subject_embedding in enumerate(subject_embedding_modules):
        with torch.no_grad():
            train_subject_indices = [sid - 1 for sid in train_subjects]
            train_embeddings = subject_embedding.weight[train_subject_indices]
            mean_embedding = train_embeddings.mean(dim=0)

            for unknown_subject in unknown_subjects:
                unknown_index = unknown_subject - 1
                subject_embedding.weight[unknown_index].copy_(mean_embedding)

def STSA(model, test_loader_new_subject, tta_lr=1e-4, tta_steps_per_batch=1, tta_batch_size=None):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    layer_ids_to_adapt = []
    for i, block in enumerate(model.model.encoder.block):
        if hasattr(block, 'shared_knowledge') and hasattr(block.shared_knowledge, 'layer_id'):
            layer_ids_to_adapt.append(block.shared_knowledge.layer_id)

    if not layer_ids_to_adapt:
        return None

    tta_adaptor = StyleAdaptor(
        num_channels=input_channels,
        feature_dim=model.model.config.d_model
    )
    tta_adaptor.to(device)
    tta_adaptor.train()

    for block in model.model.encoder.block:
        if hasattr(block, 'shared_knowledge'):
            if hasattr(block.shared_knowledge, 'switch_to_STSA'):
                block.shared_knowledge.switch_to_STSA(tta_adaptor)

    optimizer = optim.Adam(tta_adaptor.parameters(), lr=tta_lr)

    if tta_batch_size is not None and tta_batch_size != test_loader_new_subject.batch_size:
        original_dataset = test_loader_new_subject.dataset

        from torch.utils.data import DataLoader
        tta_loader = DataLoader(
            original_dataset,
            batch_size=tta_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=test_loader_new_subject.num_workers,
            pin_memory=test_loader_new_subject.pin_memory
        )
    else:
        tta_loader = test_loader_new_subject

    all_preds, all_labels, all_probs = [], [], []

    for batch_idx, batch_data in enumerate(tta_loader):
        if len(batch_data) == 3:
            inputs, _, subject_ids = batch_data
            subject_ids = subject_ids.to(device)
        else:
            inputs, _ = batch_data
            subject_ids = None

        inputs = inputs.to(device)

        for step in range(tta_steps_per_batch):
            with torch.enable_grad():
                optimizer.zero_grad()

                outputs = model.classify(x_enc=inputs, subject_ids=subject_ids)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                confidence_weights = []

                for block in model.model.encoder.block:
                    if hasattr(block, 'shared_knowledge') and hasattr(block.shared_knowledge, 'get_STSA_tta_features'):
                        raw_features, norm_features, gamma, beta = block.shared_knowledge.get_STSA_tta_features()

                        B_times_C, S, feat_dim = raw_features.shape
                        B, C = gamma.shape[0], gamma.shape[1]

                        raw_features_4d = raw_features.view(B, C, S, feat_dim)
                        true_mean_temporal = raw_features_4d.mean(dim=2)
                        true_std_temporal = raw_features_4d.std(dim=2)

                        true_mean_spatial = raw_features_4d.mean(dim=1)
                        true_std_spatial = raw_features_4d.std(dim=1)

                        prior_gamma_temporal = gamma
                        prior_beta_temporal = beta

                        prior_gamma_spatial = gamma.mean(dim=1, keepdim=True)
                        prior_beta_spatial = beta.mean(dim=1, keepdim=True)

                        epsilon = 1e-8

                        temporal_err_mean = torch.abs(true_mean_temporal - prior_beta_temporal)\
                              / (torch.abs(true_mean_temporal) + epsilon)
                        temporal_err_std = torch.abs(true_std_temporal - prior_gamma_temporal)\
                              / (torch.abs(true_std_temporal) + epsilon)

                        temporal_discrepancy = (temporal_err_mean + temporal_err_std).mean(dim=-1)

                        spatial_err_mean = torch.abs(true_mean_spatial - prior_beta_spatial)\
                              / (torch.abs(true_mean_spatial) + epsilon)
                        spatial_err_std = torch.abs(true_std_spatial - prior_gamma_spatial)\
                              / (torch.abs(true_std_spatial) + epsilon)

                        spatial_discrepancy = (spatial_err_mean + spatial_err_std).mean(dim=-1)

                        temporal_confidence = temporal_discrepancy.mean(dim=1)
                        spatial_confidence = spatial_discrepancy.mean(dim=1)
                        combined_confidence = (temporal_confidence + spatial_confidence)/2

                        confidence_weights.append(combined_confidence)

                if confidence_weights:
                    final_confidence = torch.stack(confidence_weights).mean(dim=0)

                with torch.no_grad():
                    pseudo_labels = torch.argmax(logits, dim=1)
                
                ce_loss_per_sample = F.cross_entropy(logits, pseudo_labels, reduction='none')
                pseudo_loss_weighted = (final_confidence * ce_loss_per_sample).mean()

                total_loss = pseudo_loss_weighted

                total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            eval_outputs = model.classify(x_enc=inputs, subject_ids=subject_ids)

            if hasattr(eval_outputs, 'logits'):
                eval_logits = eval_outputs.logits
            else:
                eval_logits = eval_outputs

            eval_probs = torch.softmax(eval_logits, dim=1)
            eval_preds = torch.argmax(eval_logits, dim=1)

            if len(batch_data) == 3:
                _, batch_labels, _ = batch_data
            else:
                _, batch_labels = batch_data

            all_preds.append(eval_preds.cpu())
            all_labels.append(batch_labels.cpu())

            if num_classes == 2:
                all_probs.append(eval_probs[:, 1].cpu())
            else:
                all_probs.append(eval_probs.cpu())

    for block in model.model.encoder.block:
        if hasattr(block, 'shared_knowledge') and hasattr(block.shared_knowledge, 'switch_to_pretrain_mode'):
            block.shared_knowledge.switch_to_pretrain_mode()

    final_preds = torch.cat(all_preds).numpy()
    final_labels = torch.cat(all_labels).numpy()

    if num_classes == 2:
        final_probs = torch.cat(all_probs).numpy()
    else:
        final_probs = torch.cat(all_probs, dim=0).numpy()

    final_metrics = compute_comprehensive_metrics(
        final_labels, final_preds, final_probs, num_classes
    )

    del tta_adaptor
    del optimizer
    del all_preds, all_labels, all_probs

    return {
        'overall_metrics': final_metrics,
        'predictions': final_preds,
        'true_labels': final_labels,
        'probabilities': final_probs
    }

def main():
    if enable_k_fold:
        return k_fold_cross_validation()

if __name__ == "__main__":
    main()

