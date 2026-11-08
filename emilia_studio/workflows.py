"""Workflow construction and auditing for Emilia Studio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import StudioConfig, load_config

@dataclass
class WorkflowSpec:
    """Declarative specification for Emilia workflows."""
    graph: Dict[str, Any]
    visual_template: Optional[Path] = None

_RAW_WORKFLOW_CACHE: dict[str, str] = {}


class EmiliaWorkflowBuilder:
    """Builder for constructing standard Emilia API graphs."""
    
    def __init__(self, config: StudioConfig | None = None):
        self.config = config or load_config()

    def _load_api(self, name: str) -> Dict[str, Any]:
        path = self.config.workflows_dir / name
        key = str(path.resolve())
        raw = _RAW_WORKFLOW_CACHE.get(key)
        if raw is None:
            raw = path.read_text(encoding="utf-8")
            _RAW_WORKFLOW_CACHE[key] = raw
        return json.loads(raw)
    def build_highres_native(
        self,
        seed: int = 20261027,
        pag: bool = False,
        pag_scale: float = 1.0,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        width: int = 1152,
        height: int = 1728,
        steps: int = 36,
        cfg: float = 5.5,
        ipadapter_weight: float = 0.4,
        ipadapter_end_at: float = 0.75,
        filename_prefix: str | None = None
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_ipadapter_highres_production.api.json")
        graph["3"]["inputs"]["seed"] = seed
        graph["3"]["inputs"]["steps"] = steps
        graph["3"]["inputs"]["cfg"] = cfg
        graph["5"]["inputs"].update(width=width, height=height)
        graph["42"]["inputs"].update(weight=ipadapter_weight, end_at=ipadapter_end_at)
        if positive_prompt:
            graph["6"]["inputs"]["text"] = positive_prompt
        if negative_prompt:
            graph["7"]["inputs"]["text"] = negative_prompt
        if filename_prefix:
            graph["9"]["inputs"]["filename_prefix"] = filename_prefix
        
        if pag:
            graph["60"] = {
                "class_type": "PerturbedAttentionGuidance",
                "inputs": {"model": graph["3"]["inputs"]["model"], "scale": pag_scale}
            }
            graph["3"]["inputs"]["model"] = ["60", 0]
            if not filename_prefix:
                graph["9"]["inputs"]["filename_prefix"] = "emilia/community/quality_pag_native"
        
        return graph

    def build_highres_finish(
        self,
        image_path: str,
        seed: int = 20261027,
        pag: bool = False,
        face_denoise: float = 0.22,
        hand_denoise: float = 0.24,
        face_steps: int = 20,
        hand_steps: int = 20,
        upscale_factor: float = 2.0
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_ipadapter_highres_upscale.api.json")
        graph["1"]["inputs"]["image"] = image_path
        graph["21"]["inputs"]["seed"] = seed + 3
        graph["21"]["inputs"]["denoise"] = face_denoise
        graph["21"]["inputs"]["steps"] = face_steps
        graph["28"]["inputs"]["seed"] = seed + 4
        graph["28"]["inputs"]["denoise"] = hand_denoise
        graph["28"]["inputs"]["steps"] = hand_steps
        if upscale_factor != 2.0 and "15" in graph and "inputs" in graph["15"]:
            graph["15"]["inputs"]["scale_by"] = round(float(upscale_factor) / 4.0, 4)
        if pag:
            graph["29"]["inputs"]["filename_prefix"] = "emilia/community/quality_pag_refined"
            graph["16"]["inputs"]["filename_prefix"] = "emilia/community/quality_pag_finished"
            
        return graph

    def build_eye_repair(
        self,
        image_path: str,
        seed: int = 20260936,
        denoise: float = 0.42,
        steps: int = 16,
        guide_size: int = 640,
        upscale: bool = True
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_eye_repair.api.json")
        graph["1"]["inputs"]["image"] = image_path
        graph["21"]["inputs"].update({
            "seed": seed,
            "denoise": denoise,
            "steps": steps,
            "guide_size": guide_size
        })
        return graph

    def build_lora_native(
        self,
        positive_prompt: str,
        seed: int,
        width: int = 832,
        height: int = 1216,
        steps: int = 24,
        cfg: float = 5.5,
        filename_prefix: str = "ollama_animagine"
    ) -> Dict[str, Any]:
        graph = self._load_api("emilia_lora_native.api.json")
        graph["3"]["inputs"].update(seed=seed, steps=steps, cfg=cfg)
        graph["5"]["inputs"].update(width=width, height=height)
        graph["6"]["inputs"]["text"] = positive_prompt
        graph["9"]["inputs"]["filename_prefix"] = filename_prefix
        return graph


    def build_ultra_native(
        self,
        seed: int = 20261027,
        pag: bool = False,
        pag_scale: float = 0.8,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        base_width: int = 832,
        base_height: int = 1216,
        width: int = 1152,
        height: int = 1728,
        base_steps: int = 28,
        hires_steps: int = 18,
        cfg_base: float = 4.5,
        cfg_hires: float = 4.5,
        hires_denoise: float = 0.40,
        rescale_cfg_multiplier: float = 0.75,
        lora_name: str | None = None,
        lora_strength_base: float = 0.25,
        lora_strength_hires: float = 0.0,
        ipadapter_weight: float = 0.18,
        skirt_ipadapter_weight: float = 0.45,
        ipadapter_start_at: float = 0.15,
        ipadapter_end_at: float = 0.60,
        filename_prefix: str | None = None,
    ) -> Dict[str, Any]:
        """Constructs the multi-scale 832x1216 -> 1152x1728 Hires.fix native graph with spatial-masked official Skirt IPAdapter & RescaleCFG painterly lighting."""
        pos = positive_prompt or self.config.ultra_positive_prompt
        neg = negative_prompt or self.config.ultra_negative_prompt
        out_prefix = filename_prefix or (
            "emilia/ultra/01_pag_base" if pag else "emilia/ultra/01_base"
        )
        app_dir = self.config.app_dir
        if lora_name is None:
            if app_dir is not None and (app_dir / "models" / "loras" / "Emilia_Rezero_Q-000005.safetensors").is_file():
                lora_name = "Emilia_Rezero_Q-000005.safetensors"
            else:
                lora_name = "emilia-re-zero-re.safetensors"

        has_general_ipa = bool(
            app_dir is not None
            and (app_dir / "models" / "ipadapter" / "ip-adapter-plus_sdxl_vit-h.safetensors").is_file()
        )
        has_skirt_ref = bool(
            (app_dir is not None and (app_dir / "input" / "emilia_official_skirt_and_bodice.png").is_file())
            or (self.config.assets_dir / "official_references" / "emilia_official_skirt_and_bodice.png").is_file()
        )
        use_skirt_ipa = skirt_ipadapter_weight > 0 and has_general_ipa and has_skirt_ref

        graph: Dict[str, Any] = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "official_face_center_clean.png"},
                "_meta": {"title": "Official clean center face reference"},
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "Illustrious-XL-v0.1.safetensors"},
                "_meta": {"title": "Illustrious XL v0.1"},
            },
            "10": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength_base,
                    "strength_clip": lora_strength_base,
                },
                "_meta": {"title": "Emilia Canonical Outfit LoRA (Base Pass)"},
            },
            "11": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength_hires,
                    "strength_clip": lora_strength_hires,
                },
                "_meta": {"title": "Emilia LoRA (1152x1728 Hires Refinement Pass)"},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": pos, "clip": ["10", 1]},
                "_meta": {"title": "Pass 1: Canonical outfit + scene conditioning"},
            },
            "61": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": pos,
                    "clip": ["4", 1] if lora_strength_hires == 0 else ["11", 1],
                },
                "_meta": {"title": "Pass 2: Painterly Illustrious lighting & detail conditioning"},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": neg, "clip": ["4", 1]},
                "_meta": {"title": "Anatomy, footwear, cloak & overexposure guardrails"},
            },
            "40": {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"},
                "_meta": {"title": "IPAdapter Plus Face SDXL"},
            },
            "41": {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
                "_meta": {"title": "CLIP Vision ViT-H"},
            },
            "42": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["10", 0],
                    "ipadapter": ["40", 0],
                    "image": ["1", 0],
                    "weight": ipadapter_weight,
                    "weight_type": "ease in-out",
                    "combine_embeds": "average",
                    "start_at": ipadapter_start_at,
                    "end_at": ipadapter_end_at,
                    "embeds_scaling": "V only",
                    "clip_vision": ["41", 0],
                },
                "_meta": {"title": "Gentle V-only ease in-out face anchor"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": base_width, "height": base_height, "batch_size": 1},
                "_meta": {"title": "SDXL Native 832x1216 Proportion Canvas"},
            },
        }
        base_model_ref: list[Any] = ["42", 0] if ipadapter_weight > 0 else ["10", 0]

        if use_skirt_ipa:
            graph["2"] = {
                "class_type": "LoadImage",
                "inputs": {"image": "emilia_official_skirt_and_bodice.png"},
                "_meta": {"title": "Official Emilia bodice, 3-tier scalloped peplum & pleated skirt sheet"},
            }
            graph["103"] = {
                "class_type": "PrepImageForClipVision",
                "inputs": {
                    "image": ["2", 0],
                    "interpolation": "LANCZOS",
                    "crop_position": "center",
                    "sharpening": 0.1,
                },
                "_meta": {"title": "Prepare official skirt reference for CLIP Vision"},
            }
            graph["200"] = {
                "class_type": "SolidMask",
                "inputs": {"value": 0.0, "width": base_width, "height": base_height},
            }
            graph["201"] = {
                "class_type": "SolidMask",
                "inputs": {
                    "value": 1.0,
                    "width": round(base_width * 0.58),
                    "height": round(base_height * 0.53),
                },
            }
            graph["202"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["200", 0],
                    "source": ["201", 0],
                    "x": round(base_width * 0.21),
                    "y": round(base_height * 0.28),
                    "operation": "add",
                },
            }
            graph["203"] = {
                "class_type": "FeatherMask",
                "inputs": {
                    "mask": ["202", 0],
                    "left": 96,
                    "top": 96,
                    "right": 96,
                    "bottom": 96,
                },
                "_meta": {"title": "Spatial torso/skirt attention mask (protects background lighting)"},
            }
            graph["43"] = {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"},
                "_meta": {"title": "General SDXL IPAdapter Plus (Outfit & Skirt Geometry)"},
            }
            graph["44"] = {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": base_model_ref,
                    "ipadapter": ["43", 0],
                    "image": ["103", 0],
                    "weight": skirt_ipadapter_weight,
                    "weight_type": "ease in-out",
                    "combine_embeds": "norm average",
                    "start_at": 0.10,
                    "end_at": ipadapter_end_at,
                    "embeds_scaling": "V only",
                    "attn_mask": ["203", 0],
                    "clip_vision": ["41", 0],
                },
                "_meta": {"title": "Spatial-masked official skirt & bodice conditioning"},
            }
            base_model_ref = ["44", 0]

        if pag:
            graph["60"] = {
                "class_type": "PerturbedAttentionGuidance",
                "inputs": {"model": base_model_ref, "scale": pag_scale},
            }
            base_model_ref = ["60", 0]

        if rescale_cfg_multiplier > 0:
            graph["46"] = {
                "class_type": "RescaleCFG",
                "inputs": {"model": base_model_ref, "multiplier": rescale_cfg_multiplier},
                "_meta": {"title": "RescaleCFG (prevents highlight blowout & neon saturation)"},
            }
            base_model_ref = ["46", 0]

        graph["3"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": base_model_ref,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": base_steps,
                "cfg": cfg_base,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
            },
            "_meta": {"title": "Pass 1: 832x1216 Global Anatomy & Official Skirt Render"},
        }
        graph["50"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "FP32 Tiled Decode 832x1216"},
        }
        graph["51"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["50", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
            "_meta": {"title": "Lanczos Pixel Upscale to 1152x1728"},
        }
        graph["52"] = {
            "class_type": "VAEEncodeTiled",
            "inputs": {
                "pixels": ["51", 0],
                "vae": ["4", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "FP32 Tiled Encode 1152x1728"},
        }
        hires_model_ref: list[Any] = ["4", 0] if lora_strength_hires == 0 else ["11", 0]
        if rescale_cfg_multiplier > 0:
            graph["55"] = {
                "class_type": "RescaleCFG",
                "inputs": {"model": hires_model_ref, "multiplier": rescale_cfg_multiplier},
                "_meta": {"title": "Pass 2 RescaleCFG for luminous painterly lighting"},
            }
            hires_model_ref = ["55", 0]

        graph["53"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": hires_model_ref,
                "positive": ["61", 0],
                "negative": ["7", 0],
                "latent_image": ["52", 0],
                "seed": seed + 1,
                "steps": hires_steps,
                "cfg": cfg_hires,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": hires_denoise,
            },
            "_meta": {"title": "Pass 2: Full-Canvas 1152x1728 Painterly Lighting & Skirt Hires.fix"},
        }
        graph["8"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["53", 0],
                "vae": ["4", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "FP32 Tiled Decode 1152x1728"},
        }
        graph["9"] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": out_prefix,
            },
            "_meta": {"title": "Save 1152x1728 Multi-Scale Base"},
        }
        return graph

    def build_ultra_finish(
        self,
        image_path: str,
        seed: int = 20261027,
        pag: bool = False,
        lora_name: str | None = None,
        lora_strength: float = 0.25,
        face_denoise: float = 0.28,
        hand_denoise: float = 0.30,
        face_steps: int = 20,
        hand_steps: int = 20,
        face_guide_size: int = 768,
        face_max_size: int = 1024,
        hand_guide_size: int = 640,
        hand_max_size: int = 896,
        upscale_factor: float = 2.0,
    ) -> Dict[str, Any]:
        """Constructs the upgraded Stage 2 Face/Hand SAM finishing + 2x RealESRGAN graph."""
        app_dir = self.config.app_dir
        if lora_name is None:
            if app_dir is not None and (app_dir / "models" / "loras" / "Emilia_Rezero_Q-000005.safetensors").is_file():
                lora_name = "Emilia_Rezero_Q-000005.safetensors"
            else:
                lora_name = "emilia-re-zero-re.safetensors"
        graph = self.build_highres_finish(
            image_path=image_path,
            seed=seed,
            pag=pag,
            face_denoise=face_denoise,
            hand_denoise=hand_denoise,
            face_steps=face_steps,
            hand_steps=hand_steps,
            upscale_factor=upscale_factor,
        )
        graph["10"]["inputs"]["lora_name"] = lora_name
        graph["10"]["inputs"]["strength_model"] = lora_strength
        graph["10"]["inputs"]["strength_clip"] = lora_strength
        graph["17"]["inputs"]["text"] = self.config.ultra_face_positive_prompt
        graph["18"]["inputs"]["text"] = self.config.ultra_face_negative_prompt
        graph["21"]["inputs"]["guide_size"] = face_guide_size
        graph["21"]["inputs"]["max_size"] = face_max_size
        graph["21"]["inputs"]["cfg"] = 5.0
        graph["26"]["inputs"]["text"] = self.config.ultra_hand_positive_prompt
        graph["27"]["inputs"]["text"] = self.config.ultra_hand_negative_prompt
        graph["28"]["inputs"]["guide_size"] = hand_guide_size
        graph["28"]["inputs"]["max_size"] = hand_max_size
        graph["28"]["inputs"]["cfg"] = 5.0
        if not pag:
            graph["29"]["inputs"]["filename_prefix"] = "emilia/ultra/02_refined"
            graph["16"]["inputs"]["filename_prefix"] = "emilia/ultra/03_finished_2x"
        return graph
    def sync_visual_workflow(self, api_graph: Dict[str, Any], visual_template_path: Path) -> Dict[str, Any]:
        """Synchronizes API graph values into a visual template."""
        visual = json.loads(visual_template_path.read_text(encoding="utf-8"))
        for node in visual.get("nodes", []):
            node_id = str(node["id"])
            if node_id in api_graph:
                inputs = api_graph[node_id].get("inputs", {})
                wvn = node.get("widgets_values_named")
                if wvn is not None:
                    for key in wvn:
                        if key in inputs:
                            wvn[key] = inputs[key]
                    node["widgets_values"] = list(wvn.values())
        return visual

    def audit_workflow_directory(self) -> Dict[str, list[str]]:
        """Classifies all JSON files in workflows/api and workflows/reports (or app/emilia)."""
        results: Dict[str, list[str]] = {
            "canonical_workflows": [],
            "historical_workflows": [],
            "experiment_reports": [],
        }
        search_dirs = [self.config.workflows_dir]
        reports_dir = self.config.root_dir / "workflows" / "reports"
        if reports_dir.is_dir() and reports_dir not in search_dirs:
            search_dirs.append(reports_dir)
        seen: set[str] = set()
        for d in search_dirs:
            if d is None or not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                name = p.name
                if name in seen:
                    continue
                seen.add(name)
                if name.endswith(".api.json"):
                    results["canonical_workflows"].append(name)
                elif any(sub in name for sub in ["report", "test", "experiment", "evaluation", "run"]):
                    results["experiment_reports"].append(name)
                else:
                    results["historical_workflows"].append(name)
        return results
