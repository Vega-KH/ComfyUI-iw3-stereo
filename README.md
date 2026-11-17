# iw3 Stereo for ComfyUI

Turn an image and its depth map into full-resolution stereoscopic images, entirely inside ComfyUI. Uses iw3's **forward_inpaint** synthesis and lightweight image inpainting, with **Depth Anti-aliasing** and independent horizontal/vertical **Edge Fix** controls.

- Balanced synthesis of both eyes by default; adjustable balance and single-eye options.
- Separate left/right images, full-resolution SBS, reversed SBS, top-bottom, bottom-top and red-cyan anaglyph.
- Per-eye geometric gap masks and final soft repair masks.
- Direct support for ComfyUI's built-in Depth Anything 3 geometry, plus explicitly interpreted grayscale depth maps.
- Ordinary image batches, CPU or ComfyUI's selected accelerator, and ComfyUI model caching/offload.
- No text prompts, manual masks, external iw3 installation, or viewer dependencies.

**[Read the user guide](docs/USER_GUIDE.md)** for what every setting does, useful starting ranges, scene-specific recipes, and a troubleshooting sequence for ghosting and edge artifacts.

## Installation

From your `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/Vega-KH/ComfyUI-iw3-stereo.git
```

Or download the repository ZIP and extract its folder into `custom_nodes`. Restart ComfyUI. No additional pip packages are required: the plugin reuses ComfyUI's existing PyTorch/CUDA installation.

Tested with ComfyUI 0.45.0, Python 3.13 and PyTorch 2.13 on CPU and NVIDIA CUDA. Requires the V3 node API and `CoreModelPatcher`; older ComfyUI versions may need updating. Built-in DA3 nodes and a DA3 model are required only for the DA3 example; you can also reuse an existing grayscale depth map.

## Basic workflow

Import **[workflows/DA3_to_SBS.json](workflows/DA3_to_SBS.json)** into ComfyUI:

1. Choose an image in **Load Image**.
2. In **Load Depth Anything 3**, select your installed `depth_anything_3_mono_large.safetensors` from `ComfyUI/models/geometry_estimation`.
3. Press **Run**. The full-resolution left-right SBS saves under `ComfyUI/output/iw3`.

The workflow connects raw `DA3Inference` geometry directly to the preparation node, avoiding a rendered-depth round trip. Its starting settings are DA3 resolution **504 / upper_bound_resize / mono**, divergence **2**, convergence **0.5**, balance **0**, learned AA **on**, Edge Fix **2/2**, and `forward_inpaint`. Increase DA3 resolution for a difficult silhouette before increasing stereo strength. DA3 estimation resolution does not reduce stereo output dimensions.

The stock `LoadImage` node may initially show `example.png`; select any image already in your input folder or upload one. This repository does not include source photographs or DA3 model weights.

## Models

**Load iw3 Stereo Models** downloads only these two official, SHA-256-verified artifacts when first executed. They are cached in `ComfyUI/models/iw3`; nothing downloads at ComfyUI startup.

| Model | Official download | Size |
|---|---|---:|
| Still-image LightInpaintV1 | [iw3_light_inpaint_v1_20250919.pth](https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_light_inpaint_v1_20250919.pth) | 8.65 MiB |
| Learned depth anti-aliasing | [iw3_depth_aa_20250530.pth](https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_depth_aa_20250530.pth) | 291 KiB |

For an offline installation, put both files in that folder and disable `download_missing`. Errors identify the missing/damaged file and manual installation location. Checkpoints are loaded with `weights_only=True`, architecture-name checks and strict state dictionaries. No PyTorch/CUDA reinstall is involved.

## Nodes

| Node | Purpose |
|---|---|
| Load iw3 Stereo Models | Load/cache the two small models; select automatic device or CPU |
| iw3 Depth from DA3 Geometry | Convert raw DA3 distance and sky probabilities into prepared disparity; apply AA and Edge Fix once |
| iw3 Depth from Grayscale Image | Explicitly interpret normalized disparity, far-white depth, inverse depth or raw distance |
| iw3 Stereo Image | Synthesize both eyes, pack the chosen output and return per-eye masks |

`left_gap` and `right_gap` are binary geometric holes before repair-mask preparation; **white means needs repair**. `left_repair` and `right_repair` include directional expansion and soft model blending. Valid black RGB is valid content. [Mask details and downstream use](docs/USER_GUIDE.md#outputs-and-masks).

## Scope and provenance

This is a still-image converter. Batches are independent frames, not temporally consistent video inpainting. It includes no OpenXR/viewer, diffusion generator, automatic convergence or full iw3 frontend. Large disparities, thin structures and inaccurate depth can still produce artifacts; the lightweight fill model cannot recover unseen scene detail reliably.

The default tensor path was compared directly with stock iw3 image inference on CPU/CUDA and a real image/depth pair. Matching float32 inputs/settings produced identical eye tensors. This does not imply that a different DA3 frontend, processing resolution or autocast configuration will generate identical depth or stereo.

Derived from [nunif / iw3](https://github.com/nagadomi/nunif), with ComfyStereo used as a control/workflow reference. The requested `forward_inpaint` warp is preserved; resolve-iw3's different MonoBW warp is not substituted. [UPSTREAM.json](UPSTREAM.json) records revisions, source hashes and model hashes. See [NOTICE](NOTICE), [LICENSE](LICENSE) and [the upstream license](licenses/nunif.txt) for attribution and licensing. Model weights are not redistributed here; DA3 weights retain their own terms.
