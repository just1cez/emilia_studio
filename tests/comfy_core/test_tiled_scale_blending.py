import math

import pytest
import torch

from comfy.utils import tiled_scale_multidim


@pytest.mark.parametrize(
    'shape,tile,overlap',
    [
        ((2, 3, 48, 40), (24, 24), 8),
        ((1, 3, 96, 80), (32, 32), 20),
        ((1, 3, 16, 16), (32, 32), 8),
        ((2, 3, 11, 19, 23), (8, 12, 16), (3, 4, 5)),
        ((2, 3, 43), (16,), 5),
    ],
)
def test_tiled_blending_preserves_affine_output_at_edges(shape, tile, overlap):
    samples = torch.linspace(-1, 1, steps=math.prod(shape)).reshape(shape)
    actual = tiled_scale_multidim(
        samples, lambda x: x * 0.75 + 0.125,
        tile=tile, overlap=overlap, upscale_amount=1, out_channels=3,
    )
    torch.testing.assert_close(actual, samples * 0.75 + 0.125, rtol=1e-5, atol=1e-6)
