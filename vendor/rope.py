# Derived from nagadomi/nunif (MIT); see NOTICE and licenses/nunif.txt.
from collections.abc import Callable

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, training: bool = False) -> torch.Tensor:
    half = x.shape[-1] // 2
    i1 = (Ellipsis, slice(None, half))
    i2 = (Ellipsis, slice(half, None))
    if training or torch.onnx.is_in_onnx_export():
        return x * cos + torch.cat((-x[i2], x[i1]), dim=-1) * sin
    else:
        # This requires Torch 2.13 or later.
        sin = sin[i1]  # sin[i1] == sin[i2]
        # float32
        out = x * cos
        out[i1].addcmul_(x[i2], sin, value=-1)
        out[i2].addcmul_(x[i1], sin, value=1)
        return out


class RoPE2d(nn.Module):
    cos_h: torch.Tensor
    sin_h: torch.Tensor
    cos_w: torch.Tensor
    sin_w: torch.Tensor
    q_norm: nn.Module
    k_norm: nn.Module

    def __init__(
        self,
        head_dim: int,
        size: int | tuple[int, int],
        base: float = 10000.0,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be a multiple of 4"

        if isinstance(size, int):
            height = width = size
        else:
            height, width = size

        self.head_dim = head_dim
        self.height = height
        self.width = width
        self.dim_chunk = head_dim // 2

        cos_h, sin_h, cos_w, sin_w = self._precompute_freqs(base)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

        if norm_layer is not None:
            self.q_norm = norm_layer(head_dim)
            self.k_norm = norm_layer(head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    @torch.no_grad()
    def _precompute_freqs(self, base: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_h, sin_h = self._get_1d_sin_cos(base, self.height, self.dim_chunk)
        cos_w, sin_w = self._get_1d_sin_cos(base, self.width, self.dim_chunk)

        cos_h = cos_h.reshape(1, 1, self.height, 1, self.dim_chunk)
        sin_h = sin_h.reshape(1, 1, self.height, 1, self.dim_chunk)
        cos_w = cos_w.reshape(1, 1, 1, self.width, self.dim_chunk)
        sin_w = sin_w.reshape(1, 1, 1, self.width, self.dim_chunk)

        return cos_h, sin_h, cos_w, sin_w

    def _get_1d_sin_cos(self, base: float, max_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        exp = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (base**exp)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, num_heads, N, head_dim = x.shape
        x_h = x[..., : self.dim_chunk].reshape(B, num_heads, self.height, self.width, self.dim_chunk)
        x_w = x[..., self.dim_chunk :].reshape(B, num_heads, self.height, self.width, self.dim_chunk)

        out_h = apply_rope(x_h, self.cos_h.to(x.dtype), self.sin_h.to(x.dtype), training=self.training)
        out_w = apply_rope(x_w, self.cos_w.to(x.dtype), self.sin_w.to(x.dtype), training=self.training)

        out_h = out_h.reshape(B, num_heads, N, self.dim_chunk)
        out_w = out_w.reshape(B, num_heads, N, self.dim_chunk)
        return torch.cat([out_h, out_w], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: [B, num_heads, N, head_dim] (N = h * w)
        """
        q = self.apply_rope(self.q_norm(q).to(q.dtype))
        k = self.apply_rope(self.k_norm(k).to(k.dtype))
        return q, k
