from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class TimeseriesOutputs:
    embeddings: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    forecast: Optional[torch.FloatTensor] = None
    input_mask: Optional[torch.BoolTensor] = None
    illegal_output: Optional[torch.FloatTensor] = None
