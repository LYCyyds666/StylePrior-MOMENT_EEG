import torch

class Masking:
    def __init__(self, mask_ratio=0.0):
        self.mask_ratio = mask_ratio

    @staticmethod
    def convert_seq_to_patch_view(mask, patch_len):
        # 如果传入的 mask 是 None，返回全 1 的 mask（表示不遮掩）
        if mask is None:
            return torch.ones(1, 1, 1) # 这里的维度会由后面的 repeat 补齐
        # 简单的 patch 转换逻辑，确保维度对得上
        if len(mask.shape) == 2:
            batch_size, seq_len = mask.shape
            num_patches = seq_len // patch_len
            return mask.view(batch_size, num_patches, patch_len).mean(dim=-1)
        return mask
