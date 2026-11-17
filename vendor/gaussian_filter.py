# Derived from nagadomi/nunif (MIT); see NOTICE and licenses/nunif.txt.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .permute import kernel1d_to_conv1d_weight, kernel2d_to_conv2d_weight
from .replication_pad2d import ReplicationPad1dNaive, ReplicationPad2dNaive, replication_pad2d_naive


def get_gaussian_kernel1d(kernel_size, dtype=None, device=None, sigma=None):
    if kernel_size == 1:
        return torch.ones((kernel_size,), dtype=dtype, device=device)

    # gaussin kernel formula is from torchvision
    if sigma is None:
        sigma = kernel_size * 0.15 + 0.35
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
    gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel


def get_gaussian_kernel2d(kernel_size, dtype=None, device=None, sigma=None):
    if isinstance(kernel_size, (int,)):
        kernel_size = [kernel_size, kernel_size]

    kernel1d_y = get_gaussian_kernel1d(kernel_size[0], sigma=sigma, dtype=dtype, device=device)
    kernel1d_x = get_gaussian_kernel1d(kernel_size[1], sigma=sigma, dtype=dtype, device=device)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def gaussian_blur2d(x, kernel_size, sigma=None):
    if isinstance(kernel_size, (int,)):
        kernel_size = [kernel_size, kernel_size]

    batch = True
    if x.ndim == 3:
        x = x.unsqueeze(0)
        batch = False

    padding = [kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2]

    kernel = get_gaussian_kernel2d(kernel_size, dtype=x.dtype, device=x.device, sigma=sigma)
    weight = kernel2d_to_conv2d_weight(x.shape[1], kernel)
    x = replication_pad2d_naive(x, padding, detach=True)
    x = F.conv2d(x, weight=weight, bias=None, groups=weight.shape[0])

    if not batch:
        x = x.squeeze(0)

    return x


class GaussianFilter2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None, sigma=None):
        super().__init__()
        with torch.no_grad():
            kernel = get_gaussian_kernel2d(kernel_size, sigma=sigma)
            kernel = kernel2d_to_conv2d_weight(in_channels, kernel)
        self.register_buffer("kernel", kernel, persistent=False)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 4
            self.pad = ReplicationPad2dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        return F.conv2d(self.pad(x), weight=self.kernel, bias=None, groups=self.kernel.shape[0])


class SeparableGaussianFilter2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None, sigma=None):
        super().__init__()
        with torch.no_grad():
            kernel_h = get_gaussian_kernel2d([1, kernel_size], sigma=sigma)
            kernel_v = get_gaussian_kernel2d([kernel_size, 1], sigma=sigma)
            kernel_h = kernel2d_to_conv2d_weight(in_channels, kernel_h)
            kernel_v = kernel2d_to_conv2d_weight(in_channels, kernel_v)
        self.register_buffer("kernel_h", kernel_h, persistent=False)
        self.register_buffer("kernel_v", kernel_v, persistent=False)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 4
            self.pad = ReplicationPad2dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, weight=self.kernel_h, bias=None, groups=self.kernel_h.shape[0])
        x = F.conv2d(x, weight=self.kernel_v, bias=None, groups=self.kernel_v.shape[0])

        return x


class GaussianFilter1d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=None):
        super().__init__()
        with torch.no_grad():
            kernel = get_gaussian_kernel1d(kernel_size)
            kernel = kernel1d_to_conv1d_weight(in_channels, kernel)
        self.register_buffer("kernel", kernel, persistent=False)

        if padding is not None:
            if isinstance(padding, int):
                padding = (padding,) * 2
            self.pad = ReplicationPad1dNaive(padding, detach=True)
        else:
            self.pad = nn.Identity()

    def forward(self, x):
        return F.conv1d(self.pad(x), weight=self.kernel, bias=None, groups=self.kernel.shape[0])
