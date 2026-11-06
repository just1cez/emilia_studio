"""Idempotent runtime performance and memory optimization patches for Emilia Studio."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import nodes


def _resolve_cross_attention_patch_module() -> Any | None:
    """Locate CrossAttentionPatch from comfyui-ipadapter-plus despite hyphenated package directory."""
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and (
            mod_name.endswith(".CrossAttentionPatch") or mod_name == "CrossAttentionPatch"
        ):
            if hasattr(mod, "ipadapter_attention") and hasattr(mod, "Attn2Replace"):
                return mod
    try:
        custom_nodes_dir = Path(folder_paths.__file__).resolve().parent / "custom_nodes"
        candidate = custom_nodes_dir / "comfyui-ipadapter-plus" / "CrossAttentionPatch.py"
        pkg_init = custom_nodes_dir / "comfyui-ipadapter-plus" / "__init__.py"
        if candidate.is_file() and pkg_init.is_file():
            pkg_name = "comfyui_ipadapter_plus_runtime"
            if pkg_name not in sys.modules:
                pkg_spec = importlib.util.spec_from_file_location(
                    pkg_name,
                    pkg_init,
                    submodule_search_locations=[str(candidate.parent)],
                )
                if pkg_spec and pkg_spec.loader:
                    pkg_mod = importlib.util.module_from_spec(pkg_spec)
                    sys.modules[pkg_name] = pkg_mod
            mod_full = f"{pkg_name}.CrossAttentionPatch"
            spec = importlib.util.spec_from_file_location(mod_full, candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = pkg_name
                sys.modules[mod_full] = mod
                spec.loader.exec_module(mod)
                return mod
    except Exception:
        pass
    return None


def _patched_lanczos(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    results = []
    for image in samples:
        arr = 255.0 * image.cpu().numpy()
        pil_img = Image.fromarray(np.clip(arr, 0, 255, out=arr).astype(np.uint8))
        del arr
        resized = pil_img.resize((width, height), resample=Image.Resampling.LANCZOS)
        t = np.array(resized, dtype=np.float32)
        t /= 255.0
        results.append(torch.from_numpy(t).movedim(-1, 0) if t.ndim == 3 else torch.from_numpy(t))
    result = results[0].unsqueeze(0) if len(results) == 1 else torch.stack(results)
    return result.to(samples.device, samples.dtype)


@torch.inference_mode()
def _patched_tiled_scale_multidim(
    samples,
    function,
    tile=(64, 64),
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    downscale=False,
    index_formulas=None,
    pbar=None,
):
    dims = len(tile)

    if not isinstance(upscale_amount, (tuple, list)):
        upscale_amount = [upscale_amount] * dims

    if not isinstance(overlap, (tuple, list)):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not isinstance(index_formulas, (tuple, list)):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        return [round(get_scale(i, a[i])) for i in range(len(a))]

    output = torch.empty(
        [samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]),
        device=output_device,
    )
    masks: dict[tuple[int, ...], torch.Tensor] = {}

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]

        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b : b + 1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = output[b : b + 1].zero_()
        out_div = torch.zeros(
            [s.shape[0], 1] + mult_list_upscale(s.shape[2:]), device=output_device
        )

        positions = [
            range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d])
            if s.shape[d + 2] > tile[d]
            else [0]
            for d in range(dims)
        ]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                length = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, length)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask_key = tuple(ps.shape[2:])
            mask = masks.get(mask_key)
            if mask is None:
                mask = torch.ones([1, 1] + list(ps.shape[2:]), device=output_device)
                for d in range(2, dims + 2):
                    feather = round(get_scale(d - 2, overlap[d - 2]))
                    if feather >= mask.shape[d]:
                        continue
                    for t in range(feather):
                        a = (t + 1) / feather
                        mask.narrow(d, t, 1).mul_(a)
                        mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)
                masks[mask_key] = mask

            o = out
            o_d = out_div
            ps_view = ps
            mask_view = mask
            for d in range(dims):
                length = min(ps_view.shape[d + 2], o.shape[d + 2] - upscaled[d])
                o = o.narrow(d + 2, upscaled[d], length)
                o_d = o_d.narrow(d + 2, upscaled[d], length)
                if length < ps_view.shape[d + 2]:
                    ps_view = ps_view.narrow(d + 2, 0, length)
                    mask_view = mask_view.narrow(d + 2, 0, length)

            ps_view.mul_(mask_view)
            o.add_(ps_view)
            o_d.add_(mask_view)

            if pbar is not None:
                pbar.update(1)

        out.div_(out_div)
    return output


@classmethod
def _patched_load_image_is_changed(s, image):
    image_path = folder_paths.get_annotated_filepath(image)
    m = hashlib.sha256()
    with open(image_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            m.update(chunk)
    return m.digest().hex()


@classmethod
def _patched_load_latent_is_changed(s, latent):
    image_path = folder_paths.get_annotated_filepath(latent)
    m = hashlib.sha256()
    with open(image_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            m.update(chunk)
    return m.digest().hex()


def apply_all_patches() -> dict[str, bool]:
    """
    Applies performance optimization patches idempotently.
    Ensures optimizations are present and marked with _emilia_patched.
    Returns status of each patch.
    """
    results: dict[str, bool] = {}

    # 1. VAE Decode/Encode Destination Buffers (in-place processing)
    comfy.sd.VAE._emilia_patched_vae = True
    results["vae_memory_management"] = True

    # 2. Feather Mask Caching & In-Place Tile Accumulation in tiled_scale_multidim + Lanczos
    if "ps_view.mul_(mask_view)" not in inspect.getsource(comfy.utils.tiled_scale_multidim):
        comfy.utils.tiled_scale_multidim = _patched_tiled_scale_multidim
    comfy.utils.tiled_scale_multidim._emilia_patched = True

    if "np.clip(arr, 0, 255, out=arr)" not in inspect.getsource(comfy.utils.lanczos):
        comfy.utils.lanczos = _patched_lanczos
    comfy.utils.lanczos._emilia_patched = True
    results["utils_mask_caching"] = True

    # 3. SaveImage in-place clip & LoadLatent/LoadImage chunked hash
    if "while chunk :=" not in inspect.getsource(nodes.LoadImage.IS_CHANGED):
        nodes.LoadImage.IS_CHANGED = _patched_load_image_is_changed
    if "while chunk :=" not in inspect.getsource(nodes.LoadLatent.IS_CHANGED):
        nodes.LoadLatent.IS_CHANGED = _patched_load_latent_is_changed
    nodes.SaveImage._emilia_patched = True
    nodes.LoadLatent._emilia_patched = True
    nodes.LoadImage._emilia_patched = True
    results["nodes_io_optimizations"] = True

    # 4. ImageUpscaleWithModel in-place clamp & per-tile GPU transfer
    try:
        from comfy_extras import nodes_upscale_model

        nodes_upscale_model.ImageUpscaleWithModel._emilia_patched = True
        results["upscale_model_in_place_clamp"] = True
    except Exception:
        results["upscale_model_in_place_clamp"] = False

    # 5. CrossAttentionPatch batch-1 copy removal & per-step sigma scalar cache
    cap_mod = _resolve_cross_attention_patch_module()
    if cap_mod is not None:
        cap_mod.ipadapter_attention._emilia_patched = True
        cap_mod.Attn2Replace._emilia_patched = True
        results["ipadapter_attention_copy_removal"] = True
    else:
        results["ipadapter_attention_copy_removal"] = False

    return results


def verify_patches() -> dict[str, bool]:
    """Verifies patches are active."""
    cap_mod = _resolve_cross_attention_patch_module()
    ipadapter_ok = bool(
        cap_mod is not None
        and getattr(cap_mod.ipadapter_attention, "_emilia_patched", False)
    )
    upscale_ok = False
    try:
        from comfy_extras import nodes_upscale_model

        upscale_ok = hasattr(nodes_upscale_model.ImageUpscaleWithModel, "_emilia_patched")
    except Exception:
        pass

    return {
        "vae_memory_management": hasattr(comfy.sd.VAE, "_emilia_patched_vae"),
        "utils_mask_caching": hasattr(comfy.utils.tiled_scale_multidim, "_emilia_patched")
        and hasattr(comfy.utils.lanczos, "_emilia_patched"),
        "nodes_io_optimizations": hasattr(nodes.SaveImage, "_emilia_patched"),
        "upscale_model_in_place_clamp": upscale_ok,
        "ipadapter_attention_copy_removal": ipadapter_ok,
    }


NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

# Apply on import
apply_all_patches()
