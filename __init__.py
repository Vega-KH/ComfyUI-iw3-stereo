from comfy_api.latest import ComfyExtension
from .nodes import LoadIW3StereoModels, IW3DepthFromDA3, IW3DepthFromImage, IW3Stereo


class IW3StereoExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadIW3StereoModels, IW3DepthFromDA3, IW3DepthFromImage, IW3Stereo]


async def comfy_entrypoint():
    return IW3StereoExtension()
