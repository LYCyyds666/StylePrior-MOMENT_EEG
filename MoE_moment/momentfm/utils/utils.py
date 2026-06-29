import torch
import torch.nn as nn
from argparse import Namespace

class NamespaceWithDefaults(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def getattr(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_namespace(cls, namespace):
        # 补全这个报错缺失的方法
        return cls(**vars(namespace))

def get_huggingface_model_dimensions(model_name):
    return {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048
    }

def get_anomaly_criterion(criterion): return nn.MSELoss(reduction='none')
def get_forecasting_criterion(criterion): return nn.MSELoss()
def get_classification_criterion(criterion): return nn.CrossEntropyLoss()
def get_masking_criterion(criterion): return nn.MSELoss()
def get_imputation_criterion(criterion): return nn.MSELoss()

def get_activation_fn(activation):
    if activation == "relu": return nn.ReLU()
    return nn.GELU()

class ControlTokenConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def make_variable_structure(x): return x
