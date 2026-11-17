"""Minimal metadata base for the two unchanged upstream inference models."""
import torch.nn as nn

class I2IBaseModel(nn.Module):
    def __init__(self, kwargs, scale, offset, in_channels, blend_size):
        super().__init__()
        self.i2i_offset = offset

def register_model(cls):
    return cls
