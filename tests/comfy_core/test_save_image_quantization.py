import numpy as np
from PIL import Image
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import nodes  # noqa: E402


def test_save_image_clamps_without_mutating_input(tmp_path):
    images = torch.tensor([[[[-0.1, 0.0, 0.5], [1.0, 1.1, 0.25]]]])
    original = images.clone()
    saver = nodes.SaveImage()
    saver.output_dir = str(tmp_path)
    result = saver.save_images(images, filename_prefix='clamped')
    filename = result['ui']['images'][0]['filename']
    with Image.open(tmp_path / filename) as saved:
        np.testing.assert_array_equal(
            np.asarray(saved), np.array([[[0, 0, 127], [255, 255, 63]]], dtype=np.uint8),
        )
    torch.testing.assert_close(images, original, rtol=0, atol=0)
