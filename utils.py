import torch
import random
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, cohen_kappa_score, recall_score,
    precision_score, matthews_corrcoef, average_precision_score, jaccard_score
)


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()

def compute_comprehensive_metrics(y_true, y_pred, y_prob=None, num_classes=2):
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Optional metrics with safe computation
    try:
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro')
    except:
        metrics['matthews_corrcoef'] = 0.0
        metrics['jaccard_macro'] = 0.0
    
    # Probability-based metrics
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if num_classes == 2:
                prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics['roc_auc'] = roc_auc_score(y_true, prob_pos)
                metrics['average_precision'] = average_precision_score(y_true, prob_pos)
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['average_precision_macro'] = 0.0  # Simplified for multi-class
        except:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
    else:
        metrics['roc_auc'] = 0.0 
        metrics['average_precision'] = 0.0
    
    return metrics



def print_validation_results(metrics, fold=None, prefix=""):
    acc = metrics.get('accuracy', 0.0)
    f1 = metrics.get('f1_macro', 0.0)
    
    if fold is not None:
        print(f"{prefix}Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
    else:
        print(f"{prefix}Acc={acc:.4f}, F1={f1:.4f}")
