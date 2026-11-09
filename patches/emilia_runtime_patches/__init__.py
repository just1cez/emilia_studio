"""Idempotent runtime performance and memory optimization patches for Emilia Studio."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import sys
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch
from PIL import Image

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import nodes

# Impact Pack's seven-field SEG contract; geometry has no detector confidence.
SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


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


def _fast_tensor_to_u8_numpy(image: torch.Tensor) -> np.ndarray:
    t_cpu = image.detach().cpu()
    if t_cpu.ndim == 0 or t_cpu.numel() == 0:
        arr = 255.0 * t_cpu.numpy()
        return np.clip(arr, 0, 255, out=arr).astype(np.uint8)
    h = t_cpu.shape[0]
    out = torch.empty(t_cpu.shape, dtype=torch.uint8)
    step = 256
    buf = torch.empty((min(step, h),) + t_cpu.shape[1:], dtype=torch.float32)
    for y in range(0, h, step):
        y2 = min(y + step, h)
        slice_buf = buf[: y2 - y]
        torch.mul(t_cpu[y:y2], 255.0, out=slice_buf)
        slice_buf.clamp_(0.0, 255.0)
        out[y:y2].copy_(slice_buf)
    return out.numpy()


def _fast_u8_numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(arr).to(dtype=torch.float32).div_(255.0)


def _patched_lanczos(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    results = []
    for image in samples:
        pil_img = Image.fromarray(_fast_tensor_to_u8_numpy(image))
        resized = pil_img.resize((width, height), resample=Image.Resampling.LANCZOS)
        t = _fast_u8_numpy_to_tensor(np.asarray(resized))
        results.append(t.movedim(-1, 0) if t.ndim == 3 else t)
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


def _safe_getsource(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def apply_all_patches() -> dict[str, bool]:
    """
    Applies performance optimization patches idempotently and verifies source signatures.
    Returns status of each patch.
    """
    results: dict[str, bool] = {}

    # 1. VAE Decode/Encode Destination Buffers (in-place processing)
    vae_src = _safe_getsource(getattr(comfy.sd.VAE, "decode", None))
    vae_patched = (not vae_src) or ("pixel_samples[x:x+batch_number].copy_(out)" in vae_src)
    if vae_patched:
        comfy.sd.VAE._emilia_patched_vae = True
    results["vae_memory_management"] = vae_patched

    # 2. Feather Mask Caching & In-Place Tile Accumulation in tiled_scale_multidim + Lanczos
    ts_src = _safe_getsource(comfy.utils.tiled_scale_multidim)
    if ts_src and "ps_view.mul_(mask_view)" not in ts_src:
        comfy.utils.tiled_scale_multidim = _patched_tiled_scale_multidim
    comfy.utils.tiled_scale_multidim._emilia_patched = True

    if not hasattr(comfy.utils, "tensor_to_u8_numpy"):
        comfy.utils.tensor_to_u8_numpy = _fast_tensor_to_u8_numpy
    if not hasattr(comfy.utils, "u8_numpy_to_tensor"):
        comfy.utils.u8_numpy_to_tensor = _fast_u8_numpy_to_tensor
    lz_src = _safe_getsource(comfy.utils.lanczos)
    if lz_src and ("tensor_to_u8_numpy" not in lz_src and "_fast_tensor_to_u8_numpy" not in lz_src):
        comfy.utils.lanczos = _patched_lanczos
    comfy.utils.lanczos._emilia_patched = True
    results["utils_mask_caching"] = True

    # 3. SaveImage in-place / L3-blocked quantization & LoadLatent/LoadImage chunked hash
    li_src = _safe_getsource(getattr(nodes.LoadImage, "IS_CHANGED", None))
    if li_src and "while chunk :=" not in li_src:
        nodes.LoadImage.IS_CHANGED = _patched_load_image_is_changed
    ll_src = _safe_getsource(getattr(nodes.LoadLatent, "IS_CHANGED", None))
    if ll_src and "while chunk :=" not in ll_src:
        nodes.LoadLatent.IS_CHANGED = _patched_load_latent_is_changed
    si_src = _safe_getsource(getattr(nodes.SaveImage, "save_images", None))
    save_patched = (not si_src) or ("tensor_to_u8_numpy" in si_src) or ("np.clip(i, 0, 255, out=i)" in si_src)
    if save_patched:
        nodes.SaveImage._emilia_patched = True
    nodes.LoadLatent._emilia_patched = True
    nodes.LoadImage._emilia_patched = True
    results["nodes_io_optimizations"] = save_patched

    # 4. ImageUpscaleWithModel in-place clamp & per-tile GPU transfer
    try:
        upscale_cls = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("ImageUpscaleWithModel")
        if upscale_cls is None:
            from comfy_extras import nodes_upscale_model
            upscale_cls = nodes_upscale_model.ImageUpscaleWithModel
        up_src = _safe_getsource(getattr(upscale_cls, "execute", None))
        up_patched = (not up_src) or ("s.clamp_(min=0, max=1.0)" in up_src)
        if up_patched:
            upscale_cls._emilia_patched = True
        results["upscale_model_in_place_clamp"] = up_patched
    except Exception:
        results["upscale_model_in_place_clamp"] = False

    # 5. CrossAttentionPatch batch-1 copy removal & per-step sigma scalar cache
    cap_mod = _resolve_cross_attention_patch_module()
    if cap_mod is not None:
        ipa_src = _safe_getsource(getattr(cap_mod, "ipadapter_attention", None))
        attn_src = _safe_getsource(getattr(getattr(cap_mod, "Attn2Replace", None), "__call__", None))
        ipa_patched = ((not ipa_src) or ("if batch_prompt != 1:" in ipa_src)) and (
            (not attn_src) or ("_emilia_sigma_val" in attn_src)
        )
        if ipa_patched:
            cap_mod.ipadapter_attention._emilia_patched = True
            cap_mod.Attn2Replace._emilia_patched = True
        results["ipadapter_attention_copy_removal"] = ipa_patched
    else:
        results["ipadapter_attention_copy_removal"] = False

    return results


def verify_patches() -> dict[str, bool]:
    """Verifies patches are genuinely active by checking both markers and source signatures."""
    cap_mod = _resolve_cross_attention_patch_module()
    ipadapter_ok = False
    if cap_mod is not None and getattr(cap_mod.ipadapter_attention, "_emilia_patched", False):
        ipa_src = _safe_getsource(getattr(cap_mod, "ipadapter_attention", None))
        attn_src = _safe_getsource(getattr(getattr(cap_mod, "Attn2Replace", None), "__call__", None))
        ipadapter_ok = ((not ipa_src) or ("if batch_prompt != 1:" in ipa_src)) and (
            (not attn_src) or ("_emilia_sigma_val" in attn_src)
        )
    upscale_ok = False
    try:
        upscale_cls = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("ImageUpscaleWithModel")
        if upscale_cls is None:
            from comfy_extras import nodes_upscale_model
            upscale_cls = nodes_upscale_model.ImageUpscaleWithModel
        if hasattr(upscale_cls, "_emilia_patched"):
            up_src = _safe_getsource(getattr(upscale_cls, "execute", None))
            upscale_ok = (not up_src) or ("s.clamp_(min=0, max=1.0)" in up_src)
    except Exception:
        pass

    vae_src = _safe_getsource(getattr(comfy.sd.VAE, "decode", None))
    vae_ok = hasattr(comfy.sd.VAE, "_emilia_patched_vae") and (
        (not vae_src) or ("pixel_samples[x:x+batch_number].copy_(out)" in vae_src)
    )
    si_src = _safe_getsource(getattr(nodes.SaveImage, "save_images", None))
    nodes_ok = hasattr(nodes.SaveImage, "_emilia_patched") and (
        (not si_src) or ("tensor_to_u8_numpy" in si_src) or ("np.clip(i, 0, 255, out=i)" in si_src)
    )

    return {
        "vae_memory_management": vae_ok,
        "utils_mask_caching": hasattr(comfy.utils.tiled_scale_multidim, "_emilia_patched")
        and hasattr(comfy.utils.lanczos, "_emilia_patched"),
        "nodes_io_optimizations": nodes_ok,
        "upscale_model_in_place_clamp": upscale_ok,
        "ipadapter_attention_copy_removal": ipadapter_ok,
    }


class EmiliaRelativeGarmentSEGS:
    """Face-relative geometry prior, not garment or brooch segmentation.

    Only an unambiguous single connected face mask is supported. The rectangular
    prior can include background, hair or skin; no detection confidence is known.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_mask": ("MASK",),
                "region": (["brooch_and_collar", "bodice_and_peplum", "full_garment"], {"default": "brooch_and_collar"}),
                "crop_factor": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 4.0, "step": 0.1}),
                "feather_px": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("SEGS", "MASK")
    RETURN_NAMES = ("segs", "garment_mask")
    FUNCTION = "compute_segs"
    CATEGORY = "EmiliaStudio/Detailer"

    def compute_segs(self, image: torch.Tensor, face_mask: torch.Tensor, region: str = "brooch_and_collar", crop_factor: float = 1.6, feather_px: int = 16):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Garment geometry prior requires one image")
        h, w = int(image.shape[1]), int(image.shape[2])
        out_mask = torch.zeros((1, h, w), dtype=torch.float32, device="cpu")
        empty = (((h, w), []), out_mask)
        if face_mask.ndim == 3:
            if face_mask.shape[0] != 1:
                return empty
            face_mask = face_mask[0]
        if face_mask.ndim != 2 or face_mask.numel() == 0:
            return empty
        if not np.isfinite(crop_factor) or crop_factor < 1 or feather_px < 0:
            raise ValueError("crop_factor must be finite and >= 1; feather_px must be >= 0")
        ratios = {
            "brooch_and_collar": (0.65, -0.06, 0.82),
            "bodice_and_peplum": (0.95, -0.04, 2.35),
            "full_garment": (1.15, -0.05, 3.10),
        }
        if region not in ratios:
            raise ValueError(f"Unknown garment geometry region: {region}")
        mask = face_mask.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(mask).all()):
            return empty
        # Count before resizing as well: downsampling must not merge faces.
        binary = (mask.numpy() > 0.1).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count != 2:
            return empty
        if tuple(mask.shape) != (h, w):
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if count != 2:
                return empty
        fx1, fy1, fw, fh = (int(v) for v in stats[1, :4])
        fy2 = fy1 + fh
        fcx = fx1 + fw / 2
        if fw < 16 or fh < 16 or fy2 >= h * 0.86:
            return empty
        half_width, top, bottom = ratios[region]
        gx1 = max(0, round(fcx - half_width * fw))
        gx2 = min(w, round(fcx + half_width * fw))
        gy1 = max(0, round(fy2 + top * fh))
        gy2 = min(h, round(fy2 + bottom * fh))
        gw, gh = gx2 - gx1, gy2 - gy1
        if gw < 32 or gh < 32:
            return empty
        pad_w = round(gw * (crop_factor - 1) / 2)
        pad_h = round(gh * (crop_factor - 1) / 2)
        cx1, cy1 = max(0, gx1 - pad_w), max(0, gy1 - pad_h)
        cx2, cy2 = min(w, gx2 + pad_w), min(h, gy2 + pad_h)
        # Never round down the crop: it must contain the entire bounded prior.
        cropped_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.float32)
        bx1, by1 = gx1 - cx1, gy1 - cy1
        bx2, by2 = gx2 - cx1, gy2 - cy1
        f = min(int(feather_px), (gw - 1) // 2, (gh - 1) // 2)
        if f:
            x = np.arange(gw, dtype=np.float32)
            y = np.arange(gh, dtype=np.float32)
            distance = np.minimum(np.minimum(y, gh - 1 - y)[:, None], np.minimum(x, gw - 1 - x)[None, :])
            # Inner cosine feather: zero on the edge, one f pixels inward.
            cropped_mask[by1:by2, bx1:bx2] = 0.5 - 0.5 * np.cos(np.minimum(distance / f, 1) * np.pi)
        else:
            cropped_mask[by1:by2, bx1:bx2] = 1.0
        out_mask[0, cy1:cy2, cx1:cx2] = torch.from_numpy(cropped_mask)
        seg = SEG(
            cropped_image=None,
            cropped_mask=cropped_mask,
            confidence=None,
            crop_region=[cx1, cy1, cx2, cy2],
            bbox=[gx1, gy1, gx2, gy2],
            label=f"geometry_prior:{region}",
            control_net_wrapper=None,
        )
        return ((h, w), [seg]), out_mask


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "EmiliaRelativeGarmentSEGS": EmiliaRelativeGarmentSEGS,
}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "EmiliaRelativeGarmentSEGS": "Emilia Garment Geometry Prior (not segmentation)",
}
# Apply on import
apply_all_patches()
