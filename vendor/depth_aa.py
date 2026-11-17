# Derived from nagadomi/nunif (MIT); see NOTICE and licenses/nunif.txt.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_base import I2IBaseModel, register_model
from .replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
from .init import basic_module_init
from .compile_wrapper import conditional_compile
from .attention import WindowMHA2d, WindowScoreBias


class WABlock(nn.Module):
    def __init__(self, in_channels, window_size, shift, layer_norm=False):
        super(WABlock, self).__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=2, window_size=window_size, shift=shift)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            ReplicationPad2dNaive((1, 1, 1, 1), detach=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1, inplace=True))
        self.bias = WindowScoreBias(window_size)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


@register_model
class DepthAA(I2IBaseModel):
    name = "iw3.depth_aa"

    def __init__(self):
        super(DepthAA, self).__init__(locals(), scale=1, offset=0, in_channels=1, blend_size=0)
        C = 32
        self.proj_in = nn.Conv2d(4, C, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([
            WABlock(C, window_size=8, shift=True),
            WABlock(C, window_size=8, shift=False),
            WABlock(C, window_size=8, shift=True),
        ])
        self.proj_out = nn.Conv2d(C, 4, kernel_size=1, stride=1, padding=0)
        basic_module_init(self.proj_in)
        nn.init.constant_(self.proj_out.weight, 0)

    @torch.inference_mode()
    def infer(self, x):
        # TODO: not tested
        min_value, max_value = x.amin(), x.amax()
        scale = (max_value - min_value)
        x = (x - min_value) / scale
        x = torch.nan_to_num(x)
        x = self.forward(x, clamp=False)
        x = x * scale
        x = x + min_value
        return x

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x, clamp=None):
        src = x
        input_height, input_width = x.shape[2:]
        pad_w = (16 - input_width % 16)
        pad_h = (16 - input_height % 16)
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        x = replication_pad2d_naive(x, (pad_w1, pad_w2, pad_h1, pad_h2), detach=True)

        x = F.pixel_unshuffle(x, 2)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (-pad_w1, -pad_w2, -pad_h1, -pad_h2), mode="constant")
        x = src + x

        if clamp is None:
            if not self.training:
                x = torch.clamp(x, 0, 1)
        else:
            if clamp:
                x = torch.clamp(x, 0, 1)

        return x
