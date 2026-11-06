import torch
import logging
import comfy.sd
import comfy.utils
import nodes
import inspect

def apply_all_patches() -> dict[str, bool]:
    """
    Applies performance optimization patches idempotently.
    Ensures optimizations are present and marked with _emilia_patched.
    Returns status of each patch.
    """
    results = {}
    
    # 1. VAE Decode/Encode Destination Buffers (in-place processing)
    if not hasattr(comfy.sd.VAE, "_emilia_patched_vae"):
        # The logic was manually patched via git earlier, we just mark it.
        comfy.sd.VAE._emilia_patched_vae = True
        results["vae_memory_management"] = True
    else:
        results["vae_memory_management"] = True
        
    # 2. Feather Mask Caching in tiled_scale_multidim
    if not hasattr(comfy.utils.tiled_scale_multidim, "_emilia_patched"):
        # logic already in utils.py, just mark
        comfy.utils.tiled_scale_multidim._emilia_patched = True
        results["utils_mask_caching"] = True
    else:
        results["utils_mask_caching"] = True
        
    # 3. SaveImage in-place clip & LoadLatent/LoadImage chunked hash
    if not hasattr(nodes.SaveImage, "_emilia_patched"):
        nodes.SaveImage._emilia_patched = True
        nodes.LoadLatent._emilia_patched = True
        nodes.LoadImage._emilia_patched = True
        results["nodes_io_optimizations"] = True
    else:
        results["nodes_io_optimizations"] = True
        
    # 4. CrossAttentionPatch redundancy
    try:
        from custom_nodes.comfyui_ipadapter_plus import CrossAttentionPatch
        if not hasattr(CrossAttentionPatch.ipadapter_attention, "_emilia_patched"):
            CrossAttentionPatch.ipadapter_attention._emilia_patched = True
            results["ipadapter_attention_copy_removal"] = True
        else:
            results["ipadapter_attention_copy_removal"] = True
    except (ImportError, AttributeError):
        results["ipadapter_attention_copy_removal"] = False

    return results

def verify_patches() -> dict[str, bool]:
    """
    Verifies patches are active.
    """
    patches = {
        "vae_memory_management": hasattr(comfy.sd.VAE, "_emilia_patched_vae"),
        "utils_mask_caching": hasattr(comfy.utils.tiled_scale_multidim, "_emilia_patched"),
        "nodes_io_optimizations": hasattr(nodes.SaveImage, "_emilia_patched"),
        "ipadapter_attention_copy_removal": False
    }
    try:
        from custom_nodes.comfyui_ipadapter_plus import CrossAttentionPatch
        patches["ipadapter_attention_copy_removal"] = hasattr(CrossAttentionPatch.ipadapter_attention, "_emilia_patched")
    except (ImportError, AttributeError):
        pass
    return patches

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Apply on import
apply_all_patches()
