from dataclasses import dataclass
from pathlib import Path
import hashlib
import urllib.request

import torch
import torch.nn as nn
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .core import pack, prepare_depth, synthesize
from .vendor.light_inpaint_v1 import LightInpaintV1
from .vendor.depth_aa import DepthAA

MODELS = {
    'inpaint': ('iw3_light_inpaint_v1_20250919.pth', '8b1a85df321ede3ea45d0b37a7f2c509e941687c062e1d584aab8808c253db6d'),
    'aa': ('iw3_depth_aa_20250530.pth', '5a36c8ebfdde78b60784535b9cd77016cd617faab42228dfa9ae17b657ab9c18'),
}
ModelType = io.Custom('IW3_STEREO_MODELS')
DepthType = io.Custom('IW3_DEPTH')
GeometryType = io.Custom('DA3_GEOMETRY')


class StereoModels(nn.Module):
    def __init__(self, inpaint, aa):
        super().__init__()
        self.inpaint = inpaint
        self.aa = aa


@dataclass
class DepthPacket:
    tensor: torch.Tensor
    source_size: tuple


def load_checkpoint(kind, download):
    filename, digest = MODELS[kind]
    root = Path(folder_paths.models_dir) / 'iw3'
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'Place {filename} in {root}, or enable download_missing on Load iw3 Stereo Models.')
        url = 'https://github.com/nagadomi/nunif/releases/download/0.0.0/' + filename
        temporary = path.with_suffix('.pth.part')
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Checkpoint integrity check failed for {filename}.')
            temporary.replace(path)
        except (OSError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'Could not download {filename}. Download it from {url} and place it in {root}. {exc}') from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'{path} does not match the official checkpoint SHA-256. Restore the official file.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = 'inpaint.light_inpaint_v1' if kind == 'inpaint' else 'iw3.depth_aa'
    if checkpoint.get('name') != expected or 'state_dict' not in checkpoint:
        raise ValueError(f'{filename} is not the expected nunif {expected} checkpoint.')
    model = LightInpaintV1() if kind == 'inpaint' else DepthAA()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()


def activate(models, size):
    # Sequential images/eyes; reserve space for padded activations and splat indices.
    mm.load_models_gpu([models], memory_required=max(size[0] * size[1] * 900, 256 * 1024**2), force_full_load=True)
    return models.load_device


class LoadIW3StereoModels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LoadIW3StereoModels', display_name='Load iw3 Stereo Models', category='iw3 stereo',
                         inputs=[io.Combo.Input('device', options=['auto', 'cpu'], default='auto'),
                                 io.Boolean.Input('download_missing', default=True,
                                                  tooltip='Download only the two official, hash-verified iw3 checkpoints if missing.')],
                         outputs=[ModelType.Output('models')])

    @classmethod
    def execute(cls, device='auto', download_missing=True):
        if device not in ('auto', 'cpu'):
            raise ValueError('device must be auto or cpu.')
        model = StereoModels(load_checkpoint('inpaint', download_missing), load_checkpoint('aa', download_missing))
        target = mm.get_torch_device() if device == 'auto' else torch.device('cpu')
        patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=target, offload_device=torch.device('cpu'))
        return io.NodeOutput(patcher)


def processing_inputs():
    return [io.Boolean.Input('depth_aa', default=True),
            io.Int.Input('edge_fix_x', default=2, min=0, max=20,
                         tooltip='iw3 horizontal Edge Fix iterations at the incoming depth resolution; 0 disables.'),
            io.Int.Input('edge_fix_y', default=2, min=0, max=20,
                         tooltip='iw3 vertical Edge Fix iterations at the incoming depth resolution; 0 disables.'),
            io.Boolean.Input('invert_depth', default=False)]


class IW3DepthFromDA3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IW3DepthFromDA3', display_name='iw3 Depth from DA3 Geometry', category='iw3 stereo',
                         description='Use raw built-in DA3 geometry, not DA3Render. Converts distance with iw3 DA3 sky weighting and +0.2 shift, then AA, Edge Fix and minmax.',
                         inputs=[GeometryType.Input('da3_geometry'), ModelType.Input('models'), *processing_inputs(),
                                 io.Float.Input('distance_shift', default=0.2, min=0.0, max=100.0, step=0.01)],
                         outputs=[DepthType.Output('depth'), io.Image.Output('processed_depth')])

    @classmethod
    def execute(cls, da3_geometry, models, depth_aa=True, edge_fix_x=2, edge_fix_y=2, invert_depth=False, distance_shift=0.2):
        if not isinstance(da3_geometry, dict) or not {'depth', 'image'} <= da3_geometry.keys():
            raise ValueError('Connect DA3Inference da3_geometry directly. A rendered depth IMAGE is not raw geometry.')
        d, image = da3_geometry['depth'], da3_geometry['image']
        if d.ndim != 3 or tuple(d.shape) != tuple(image.shape[:3]):
            raise ValueError('DA3 depth and source image must be aligned and have matching batches and dimensions.')
        size = tuple(image.shape[1:3])
        device = activate(models, size) if depth_aa else torch.device('cpu')
        sky = da3_geometry.get('sky')
        sky = sky.unsqueeze(1).to(device) if sky is not None else None
        d = prepare_depth(d.unsqueeze(1).to(device), 'distance', models.model.aa if depth_aa else None,
                          edge_fix_x, edge_fix_y, invert_depth, distance_shift, sky, normalize=True).cpu()
        return io.NodeOutput(DepthPacket(d, size), d.movedim(1, -1).expand(-1, -1, -1, 3).contiguous())


class IW3DepthFromImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IW3DepthFromImage', display_name='iw3 Depth from Grayscale Image', category='iw3 stereo',
                         description='Explicit conventions for existing grayscale maps. Colorized depth is rejected. Prefer raw DA3 geometry for DA3.',
                         inputs=[io.Image.Input('depth_image'), ModelType.Input('models'),
                                 io.Combo.Input('convention', options=['normalized_disparity', 'normalized_far', 'inverse_depth', 'distance']),
                                 io.Boolean.Input('normalize', default=False,
                                                  tooltip='Enable for raw inverse depth or distance. Leave off for already normalized maps.'),
                                 *processing_inputs(),
                                 io.Float.Input('distance_shift', default=0.2, min=0.0, max=100.0, step=0.01)],
                         outputs=[DepthType.Output('depth'), io.Image.Output('processed_depth')])

    @classmethod
    def execute(cls, depth_image, models, convention='normalized_disparity', normalize=False,
                depth_aa=True, edge_fix_x=2, edge_fix_y=2, invert_depth=False, distance_shift=0.2):
        if depth_image.ndim != 4 or depth_image.shape[-1] not in (1, 3):
            raise ValueError('Provide a grayscale IMAGE with one channel or equal RGB channels.')
        if depth_image.shape[-1] == 3 and (depth_image - depth_image[..., :1]).abs().max() > 1e-4:
            raise ValueError('Colorized depth is incompatible. Use raw DA3 geometry or an uncolored grayscale depth map.')
        if convention in ('distance', 'inverse_depth') and not normalize:
            raise ValueError('Enable normalize for raw distance/inverse depth, or select normalized_disparity.')
        size = tuple(depth_image.shape[1:3])
        device = activate(models, size) if depth_aa else torch.device('cpu')
        d = prepare_depth(depth_image[..., :1].movedim(-1, 1).to(device), convention,
                          models.model.aa if depth_aa else None, edge_fix_x, edge_fix_y,
                          invert_depth, distance_shift, normalize=normalize).cpu()
        return io.NodeOutput(DepthPacket(d, size), d.movedim(1, -1).expand(-1, -1, -1, 3).contiguous())


class IW3Stereo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IW3Stereo', display_name='iw3 Stereo Image', category='iw3 stereo',
                         inputs=[io.Image.Input('image'), DepthType.Input('depth'), ModelType.Input('models'),
                                 io.Combo.Input('method', options=['forward_inpaint', 'forward_fill']),
                                 io.Float.Input('divergence', default=2.0, min=0, max=15, step=0.05,
                                                tooltip='Total stereo divergence, percent of the longest image side. Each eye gets half at balance=0 (iw3 convention).'),
                                 io.Float.Input('convergence', default=0.5, min=0, max=1, step=0.01,
                                                tooltip='Disparity at the screen plane: 0 far, 1 near, after the depth exponent.'),
                                 io.Float.Input('balance', default=0, min=-1, max=1, step=0.05,
                                                tooltip='0 balanced; +1 synthesize left only; -1 synthesize right only.'),
                                 io.Float.Input('separation', default=0, min=-5, max=5, step=0.01,
                                                tooltip='Additional per-eye separation, percent of longest side at balance=0. Positive shifts left leftward and right rightward.'),
                                 io.Float.Input('depth_exponent', default=1, min=0.1, max=3, step=0.05),
                                 io.Int.Input('mask_inner_dilation', default=0, min=0, max=30),
                                 io.Int.Input('mask_outer_dilation', default=0, min=0, max=30),
                                 io.Combo.Input('layout', options=['left-right', 'right-left', 'top-bottom', 'bottom-top', 'red-cyan-anaglyph']),
                                 io.Combo.Input('depth_resize', options=['require_aligned', 'resize_bilinear'],
                                                tooltip='Default requires matching dimensions. Explicit resize assumes identical field of view; it cannot repair crops.')],
                         outputs=[io.Image.Output('stereo'), io.Image.Output('left'), io.Image.Output('right'),
                                  io.Mask.Output('left_gap'), io.Mask.Output('right_gap'),
                                  io.Mask.Output('left_repair'), io.Mask.Output('right_repair')])

    @classmethod
    def execute(cls, image, depth, models, method='forward_inpaint', divergence=2., convergence=.5, balance=0.,
                separation=0., depth_exponent=1., mask_inner_dilation=0, mask_outer_dilation=0,
                layout='left-right', depth_resize='require_aligned'):
        size = tuple(image.shape[1:3])
        if size != depth.source_size and depth_resize != 'resize_bilinear':
            raise ValueError('Image/depth dimensions differ. Use matching inputs or explicitly enable depth_resize=resize_bilinear.')
        device = activate(models, size)
        d = depth.tensor.to(device)
        pbar = comfy.utils.ProgressBar(image.shape[0])
        def progress(done, total):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(done, total)
        left, right, lg, rg, lr, rr = synthesize(
            image.movedim(-1, 1).to(device), d, models.model.inpaint, method, divergence, convergence,
            balance, separation, depth_exponent, mask_inner_dilation, mask_outer_dilation, progress)
        stereo = pack(left, right, layout).movedim(1, -1).cpu()
        return io.NodeOutput(stereo, left.movedim(1, -1).cpu(), right.movedim(1, -1).cpu(),
                             lg[:, 0].cpu(), rg[:, 0].cpu(), lr[:, 0].cpu(), rr[:, 0].cpu())
