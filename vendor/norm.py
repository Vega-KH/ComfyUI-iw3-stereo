# Derived from nagadomi/nunif (MIT); see NOTICE and licenses/nunif.txt.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .permute import bchw_to_bhwc, bhwc_to_bchw


class L2Normalize(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)


def LayerNormNoBias(normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
    # bias=False, requires pytorch 2.1
    return nn.LayerNorm(
        normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=False, device=device, dtype=dtype
    )


class LayerNormNoBias2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormNoBias(dim)

    def forward(self, x):
        x = bhwc_to_bchw(self.norm(bchw_to_bhwc(x)))
        return x


class GroupNormNoBias(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        else:
            self.weight = self.register_parameter("weight", None)

    def forward(self, x):
        x = F.group_norm(x, num_groups=self.num_groups, weight=self.weight, bias=None, eps=self.eps)
        return x


class RMSNorm1(nn.RMSNorm):
    # 0-centered ver
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight + 1.0, self.eps).to(x.dtype)


class RMSNorm2d(nn.Module):
    # 0-centered ver
    def __init__(self, in_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, in_channels, 1, 1), dtype=torch.float32))
        nn.init.zeros_(self.weight)

    def forward(self, x):
        scale = torch.rsqrt(torch.mean(x.float() ** 2, dim=1, keepdim=True) + 1e-5)
        scale = scale * (1.0 + self.weight)
        return x * scale.to(x.dtype)


class FastLayerNorm(nn.LayerNorm):
    """
    Idea from timm fast_layer_norm.
    Stop upcasting to fp32 when autocast.
    """

    def forward(self, input):
        if torch.jit.is_scripting():
            return super().forward(input)

        if torch.is_autocast_enabled(input.device.type):
            dtype = torch.get_autocast_dtype(input.device.type)
            input = input.to(dtype)
            if self.weight is not None:
                weight = self.weight.to(dtype)
            else:
                weight = None
            if self.bias is not None:
                bias = self.bias.to(dtype)
            else:
                bias = None
            with torch.amp.autocast(device_type=input.device.type, enabled=False):
                return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        else:
            return super().forward(input)


class FastLayerNorm2d(FastLayerNorm):
    def forward(self, x):
        return bhwc_to_bchw(super().forward(bchw_to_bhwc(x)))


class ReparamBatchNorm2d(nn.Module):
    # For ReparamConv2d
    # simple 0-1 interpolation version of Batch Renormalize
    max_iteration: torch.Tensor

    def __init__(self, num_features, eps=1e-5, momentum=0.02, max_iteration=500_000):
        super(ReparamBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum_max = momentum
        self.momentum_min = 1e-5
        self.interpolate_max = 0.9
        self.interpolate_min = 1e-5
        self.weight = nn.Parameter(torch.empty(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(num_features, dtype=torch.float32))

        self.register_buffer("max_iteration", torch.tensor(max_iteration, dtype=torch.long))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_mean", torch.empty(num_features, dtype=torch.float32))
        self.register_buffer("running_var", torch.empty(num_features, dtype=torch.float32))
        self.reset_parameters()

    def reset_running_stats(self):
        self.num_batches_tracked.zero_()
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        if self.training and self.num_batches_tracked < self.max_iteration:
            momentum = self.cosine_annealing(
                min_v=self.momentum_min,
                max_v=self.momentum_max,
                t=self.num_batches_tracked,
                max_t=self.max_iteration,
            )
            interpolation = self.cosine_annealing(
                min_v=self.interpolate_min,
                max_v=self.interpolate_max,
                t=self.num_batches_tracked,
                max_t=self.max_iteration,
            )
            batch_var, batch_mean = torch.var_mean(
                x.to(self.running_mean.dtype), dim=(0, 2, 3), correction=1, keepdim=False
            )
            self.running_mean.lerp_(batch_mean.detach(), momentum)
            self.running_var.lerp_(batch_var.detach(), momentum)
            self.num_batches_tracked.add_(1)

            var = self.running_var.lerp(batch_var, interpolation).reshape(1, -1, 1, 1)
            mean = self.running_var.lerp(batch_var, interpolation).reshape(1, -1, 1, 1)
        else:
            var = self.running_var.reshape(1, -1, 1, 1)
            mean = self.running_mean.reshape(1, -1, 1, 1)

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        scale = w * (var + self.eps).rsqrt()
        bias = b - mean * scale
        return (x * scale + bias).to(x.dtype)

    @staticmethod
    def cosine_annealing(min_v, max_v, t, max_t):
        if max_t > t:
            return min_v + 0.5 * (max_v - min_v) * (1.0 + torch.cos((t / max_t) * torch.pi))
        else:
            return min_v


def apply_batch_max_iteration_(model: nn.Module, max_iteration: int):
    for name, child in model.named_children():
        if isinstance(child, ReparamBatchNorm2d):
            child.max_iteration.fill_(max_iteration)
        else:
            apply_batch_max_iteration_(child, max_iteration)


def apply_frozen_bn_(model: nn.Module):
    from torchvision.ops import FrozenBatchNorm2d

    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen_bn = FrozenBatchNorm2d(child.weight.shape[0], eps=child.eps).to(
                device=child.weight.device, dtype=child.weight.dtype
            )
            frozen_bn.load_state_dict(child.state_dict())
            if child.training:
                frozen_bn.train()
            else:
                frozen_bn.eval()
            setattr(model, name, frozen_bn)
        else:
            apply_frozen_bn_(child)
