import math
import weakref

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy import sd


def make_vae(monkeypatch, dtype):
    monkeypatch.setattr(sd.model_management, 'load_models_gpu', lambda *a, **kw: None)
    vae = sd.VAE.__new__(sd.VAE)
    vae.device = torch.device('cpu')
    vae.output_device = torch.device('cpu')
    vae.vae_dtype = dtype
    vae.vae_output_dtype = lambda: torch.float32
    vae.latent_dim = 2
    vae.extra_1d_channel = None
    vae.disable_offload = True
    vae.format_encoded = None
    vae.patcher = SimpleNamespace(get_free_memory=lambda device: 2)
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.memory_used_encode = lambda shape, dtype: 1
    vae.throw_exception_if_invalid = lambda: None
    vae.vae_encode_crop_pixels = lambda pixels: pixels
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x.add_(1).div_(2).clamp_(0, 1)
    return vae


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_decode_owns_output_and_preserves_aliased_source(monkeypatch, dtype):
    vae = make_vae(monkeypatch, dtype)
    vae.first_stage_model = SimpleNamespace(decode=lambda x: x.transpose(-1,-2))
    source = torch.linspace(-2,2,5*3*4*6).reshape(5,3,4,6).to(dtype)
    original = source.clone()
    result = vae.decode(source)
    expected = original.transpose(-1,-2).float().add(1).div(2).clamp(0,1).movedim(1,-1)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    torch.testing.assert_close(source,original,rtol=0,atol=0)
    result.zero_()
    torch.testing.assert_close(source,original,rtol=0,atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_encode_owns_strided_output_with_partial_last_batch(monkeypatch, dtype):
    vae = make_vae(monkeypatch,dtype)
    vae.first_stage_model = SimpleNamespace(encode=lambda x: x[:,:,::2,::2])
    source = torch.linspace(-1,1,5*6*4*3).reshape(5,6,4,3).to(dtype)
    original = source.clone()
    result = vae.encode(source)
    expected = original.movedim(-1,1)[:,:,::2,::2].float()
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    result.zero_()
    torch.testing.assert_close(source,original,rtol=0,atol=0)


@pytest.mark.parametrize('operation', ['encode', 'decode'])
def test_late_oom_releases_model_output_before_tiled_fallback(monkeypatch, operation):
    vae = make_vae(monkeypatch, torch.float32)
    references = []
    def model_output(x):
        out = x.clone()
        references.append(weakref.ref(out))
        return out
    vae.first_stage_model = SimpleNamespace(encode=model_output, decode=model_output)
    source = torch.ones(1, 3, 4, 4) if operation == 'decode' else torch.ones(1, 4, 4, 3)
    original = source.clone()
    def fail_allocation(*a, **kw):
        raise torch.OutOfMemoryError('late destination allocation')
    monkeypatch.setattr(sd.torch, 'empty', fail_allocation)
    monkeypatch.setattr(sd.model_management, 'soft_empty_cache', lambda: None)
    vae.handles_tiling = False
    def fallback(x):
        assert references and references[0]() is None
        return x.clone()
    vae.decode_tiled_ = fallback
    vae.encode_tiled_ = fallback
    result = getattr(vae, operation)(source)
    expected = source.movedim(1, -1) if operation == 'decode' else source.movedim(-1, 1)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(source, original)


@pytest.mark.parametrize('shape', [(2, 3, 19, 23), (1, 3, 4, 5)])
def test_tiled_decode_preserves_three_pass_values_and_input(monkeypatch, shape):
    vae = make_vae(monkeypatch, torch.float32)
    vae.upscale_ratio = 1
    vae.first_stage_model = SimpleNamespace(decode=lambda x: x * 0.75)
    source = torch.linspace(-1, 1, steps=math.prod(shape)).reshape(shape)
    original = source.clone()
    decode = lambda x: x * 0.75
    a = sd.comfy.utils.tiled_scale(source, decode, 4, 16, 2, upscale_amount=1)
    b = sd.comfy.utils.tiled_scale(source, decode, 16, 4, 2, upscale_amount=1)
    c = sd.comfy.utils.tiled_scale(source, decode, 8, 8, 2, upscale_amount=1)
    expected = ((a + b + c) / 3).add_(1).div_(2).clamp_(0, 1)
    actual = vae.decode_tiled_(source, tile_x=8, tile_y=8, overlap=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
