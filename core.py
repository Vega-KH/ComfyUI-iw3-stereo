"""Tensor processing; ComfyUI loading and execution policy live in nodes.py."""
import torch

from .vendor.dilation import dilate_edge, dilate_inner, dilate_outer, mask_closing
from .vendor.forward_warp import apply_divergence_forward_warp


def check_depth(depth):
    if depth.ndim != 4 or depth.shape[1] != 1 or min(depth.shape[-2:]) < 2:
        raise ValueError('Depth must have shape [batch, 1, height, width], with both dimensions >= 2.')
    if not torch.isfinite(depth).all():
        raise ValueError('Depth contains NaN or infinity. Repair invalid depth upstream.')


def prepare_depth(depth, convention, aa_model=None, edge_x=2, edge_y=2,
                  invert=False, distance_shift=0.2, sky=None, normalize=True):
    check_depth(depth)
    depth = depth.float()
    if convention == 'distance':
        if (depth < 0).any() or (depth + distance_shift <= 0).any():
            raise ValueError('Distance must be nonnegative and distance + shift must be positive.')
        depth = 1.0 / (depth + distance_shift)
        if sky is not None:
            if sky.shape != depth.shape or not torch.isfinite(sky).all():
                raise ValueError('DA3 sky and depth must be aligned, finite tensors of the same shape.')
            weights = (sky.clamp(0.3, 1.0) - 0.3) / (1.0 - 0.3)
            count = depth.shape[-2] * depth.shape[-1] - (sky > 0.3).sum((-2, -1), keepdim=True)
            depth = torch.where(count < 10, torch.zeros_like(depth), depth * (1.0 - weights))
    elif convention in ('normalized_disparity', 'normalized_far'):
        if (depth < 0).any() or (depth > 1).any():
            raise ValueError('Normalized depth must be in [0, 1]. Choose inverse_depth or distance for raw values.')
        if convention == 'normalized_far':
            depth = 1.0 - depth
    elif convention == 'inverse_depth':
        if (depth < 0).any():
            raise ValueError('Inverse depth must be nonnegative.')
    else:
        raise ValueError(f'Unsupported depth convention: {convention}')
    if invert:
        if convention in ('normalized_disparity', 'normalized_far'):
            depth = 1.0 - depth
        else:
            depth = depth.amax((-2, -1), keepdim=True) + depth.amin((-2, -1), keepdim=True) - depth
    frames = []
    for frame in depth.split(1):
        # iw3 DA3: inverse distance/sky -> learned AA -> Edge Fix -> minmax -> mapper.
        if aa_model is not None:
            frame = aa_model.infer(frame)
        if edge_x or edge_y:
            frame = dilate_edge(frame, (edge_x, edge_y))
        if normalize:
            lo, hi = frame.amin(), frame.amax()
            frame = torch.where(hi > lo, (frame - lo) / (hi - lo).clamp(min=1e-6), frame).clamp(0, 1)
        frames.append(frame)
    return torch.cat(frames)


def repair_mask(raw, is_left, inner, outer, base_width):
    mask = raw.flip(-1) if is_left else raw
    mask = mask_closing(mask > 0)
    mask = dilate_outer(mask, outer, base_width=base_width)
    mask = dilate_inner(mask, inner, base_width=base_width)
    return mask.flip(-1) if is_left else mask


def synthesize(image, depth, model=None, method='forward_inpaint', divergence=2.0,
               convergence=0.5, balance=0.0, separation=0.0, exponent=1.0,
               inner=0, outer=0, progress=None):
    if method not in ('forward_inpaint', 'forward_fill'):
        raise ValueError(f'Unsupported synthesis method: {method}')
    if method == 'forward_inpaint' and model is None:
        raise ValueError('forward_inpaint requires the iw3 image inpainting model.')
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 2:
        raise ValueError('Image must be RGB with both dimensions >= 2.')
    if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
        raise ValueError('Source RGB must be finite and in [0, 1].')
    check_depth(depth)
    if depth.shape[0] not in (1, image.shape[0]):
        raise ValueError('Depth batch must contain one map or one map per source image.')
    if abs(balance) > 1 or divergence < 0 or exponent <= 0:
        raise ValueError('Balance must be in [-1, 1], divergence >= 0, and exponent > 0.')
    if exponent != 1.0:
        depth = depth.clamp(0, 1).pow(exponent)
    # A constant shift is expressed in the warp equation, before mask generation.
    effective = max(divergence, abs(2 * separation))
    if effective:
        depth = depth * (divergence / effective)
        convergence = (divergence * convergence + 2 * separation) / effective
    results = [[] for _ in range(6)]
    for i in range(image.shape[0]):
        x = image[i:i+1]
        d = depth[i:i+1] if depth.shape[0] != 1 else depth
        kwargs = dict(convergence=convergence, method=method, return_mask=True, width_base=False)
        if balance == 0:
            left, right, lm, rm = apply_divergence_forward_warp(x, d, effective, synthetic_view='both', **kwargs)
        else:
            if balance == -1:
                left, lm = x, None
            else:
                left, _, lm, _ = apply_divergence_forward_warp(x, d, effective * (1 + balance) / 2,
                                                              synthetic_view='left', **kwargs)
            if balance == 1:
                right, rm = x, None
            else:
                _, right, _, rm = apply_divergence_forward_warp(x, d, effective * (1 - balance) / 2,
                                                              synthetic_view='right', **kwargs)
        masks, raw_masks, eyes = [], [], []
        for eye, raw, is_left in ((left, lm, True), (right, rm, False)):
            if raw is None:
                raw = x.new_zeros((1, 1, *x.shape[-2:]))
            # Raw reports geometric holes (including layered holes), never RGB blackness.
            raw_masks.append((raw > 0).float())
            mask = repair_mask(raw, is_left, inner, outer, d.shape[-1])
            if method == 'forward_inpaint':
                if is_left:
                    eye = model.infer(eye.flip(-1), mask.flip(-1)).flip(-1)
                else:
                    eye = model.infer(eye, mask)
                # This is the model's final, blurred compositing region, not just warp holes.
                oriented = mask.flip(-1) if is_left else mask
                _, final_mask = model.preprocess(torch.zeros_like(eye), oriented)
                mask = final_mask.flip(-1) if is_left else final_mask
            else:
                mask = raw_masks[-1]
            masks.append(mask.float())
            eyes.append(eye)
        for bucket, value in zip(results, (*eyes, *raw_masks, *masks)):
            bucket.append(value)
        if progress:
            progress(i + 1, image.shape[0])
    return tuple(torch.cat(items) for items in results)


def pack(left, right, layout):
    if layout == 'left-right':
        return torch.cat((left, right), -1)
    if layout == 'right-left':
        return torch.cat((right, left), -1)
    if layout == 'top-bottom':
        return torch.cat((left, right), -2)
    if layout == 'bottom-top':
        return torch.cat((right, left), -2)
    if layout == 'red-cyan-anaglyph':
        return torch.cat((left[:, :1], right[:, 1:]), 1)
    raise ValueError(f'Unknown layout: {layout}')
