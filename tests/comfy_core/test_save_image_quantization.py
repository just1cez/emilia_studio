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


def test_headless_intermediate_prefixes_use_zero_compression_losslessly(tmp_path, monkeypatch):
    monkeypatch.setenv("EMILIA_HEADLESS_FAST_BOOT", "1")
    yy, xx = np.meshgrid(np.arange(128, dtype=np.uint8), np.arange(128, dtype=np.uint8), indexing="ij")
    arr = np.stack([xx, yy, (xx ^ yy).astype(np.uint8)], axis=-1)
    images = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
    saver = nodes.SaveImage()
    saver.output_dir = str(tmp_path)
    inter_res = saver.save_images(images, filename_prefix="emilia/quality/01_base")
    final_res = saver.save_images(images, filename_prefix="emilia/quality/03_finished_2x")
    inter_path = tmp_path / inter_res["ui"]["images"][0]["subfolder"] / inter_res["ui"]["images"][0]["filename"]
    final_path = tmp_path / final_res["ui"]["images"][0]["subfolder"] / final_res["ui"]["images"][0]["filename"]
    assert inter_path.stat().st_size > final_path.stat().st_size
    with Image.open(inter_path) as inter_im, Image.open(final_path) as final_im:
        np.testing.assert_array_equal(np.asarray(inter_im), arr)
        np.testing.assert_array_equal(np.asarray(final_im), arr)
