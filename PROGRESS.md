# PROGRESS.md — Emilia Studio Engineering Progress & Experiment Ledger

This document tracks the end-to-end engineering progression of **Emilia Studio**, from initial 8 GiB VRAM SDXL bring-up through community workflow evaluations, low-level PyTorch/ComfyUI memory optimizations, 50-image bit-exact verification rounds, and the v2.0 declarative architecture overhaul.

---

## Executive Status Summary (Current: v2.0.0 — 2026-11-06)

| Subsystem | Current Production Baseline | Verification Status |
| :--- | :--- | :--- |
| **Architecture** | Standalone top-level `emilia_studio` package + isolated `app/` ComfyUI runtime (Option A) | ✅ Active |
| **Stage 1: Native Generation** | `Illustrious-XL-v0.1` + `emilia-re-zero-re` LoRA (0.8) + `IPAdapter Plus Face SDXL` (weight 0.40, end_at 0.75), `1152×1728`, 36 steps, `dpmpp_2m_sde` / `karras`, CFG 5.5, FP32 tiled VAE (`512` tile / `64` overlap) | ✅ Verified Bit-Exact |
| **Intermediate Quality Gate** | CPU-only YOLOv8n (`face`/`hand`) + HSV circular iris hue analyzer (`[250°, 302°]` violet band, `32°` heterochromia threshold) | ✅ Verified on 108+ runs |
| **Stage 2: Refinement & 2× Delivery** | `FaceDetailer` (guide 640, max 896, 20 steps, denoise 0.22) $\rightarrow$ `DetailerForEach` (`hand_yolov8n` + `sam_vit_b_01ec64`, guide 512, max 768, 20 steps, denoise 0.24) $\rightarrow$ `RealESRGAN_x4plus_anime_6B` (4×) $\rightarrow$ Lanczos 0.5× (`2304×3456`) | ✅ Verified Bit-Exact |
| **Ollama Bridge & Compiler** | `qwen3.5:9b` (`num_gpu=0, keep_alive=0`), dynamic framing & IPAdapter weight routing, integrated into 2-stage HighRes Quality pipeline | ✅ Verified |
| **Unit Test Suite** | 147 unit tests in `app/tests-unit/comfy_test/` (including 8 `test_emilia_studio.py` architecture tests and 22 memory/tile/hash regression tests) | ✅ 147 / 147 Passing (24.4s) |
| **SQLite Experiment Ledger** | `app/emilia/studio_ledger.sqlite3` indexing 108 historical `.run.json` receipts and 50 ten-seed benchmark records | ✅ Indexed & Queryable |

---

## Chronological Milestones

### Phase 1: Lightweight 8 GiB Baseline & Anatomy Refinement (2026-09 — 2026-10)
- **Safe Baseline (`emilia_lora_safe.api.json` / `Emilia_03_LoRA_8GB.json`)**:
  - Established `832×1216` native generation (Euler Ancestral, 24 steps, CFG 5.5) + `RealESRGAN x4` + `Lanczos 0.5x` (`1664×2432`).
  - Rejected whole-image 1.25× latent upscale (`emilia_lora_optimized.api.json`) as default because fixed-seed results were softer and less reliable than direct native-to-upscale.
- **Targeted Face & Hand Refinement (`emilia_lora_production.api.json`)**:
  - Integrated `ComfyUI-Impact-Pack` (8.28.3) and `ComfyUI-Impact-Subpack` (1.3.5) with `face_yolov8n.pt`, `hand_yolov8n.pt`, and `sam_vit_b_01ec64.pth`.
  - Confirmed SAM masking on detected hands protects adjacent bell sleeves and dress trim from unintended repainting.
- **Targeted Eye-Color Fallback (`emilia_eye_repair.api.json`)**:
  - Identified blue/pink heterochromia drift on certain seeds; built a targeted higher-denoise facial pass (`guide_size=640, denoise=0.42, steps=16`) that restores matching violet irises without altering full-body composition.

### Phase 2: Hardware Utilization & Kernel Evaluation on Turing CMP 40HX
- **Measured Hardware Profile**:
  - GPU: 8 GiB NVIDIA CMP 40HX (Turing, CC 7.5); CPU: Intel Xeon E5-2690 v2 (10C/20T, 10 PyTorch intra-op threads).
  - During diffusion sampling, GPU utilization reaches 97–100% with VRAM peaking at `7147 / 8192 MiB`.
- **Rejected Experimental Accelerations**:
  - `--fast autotune fp16_accumulation cublas_ops` regressed total runtime (`380s` vs `256s` baseline).
  - `xFormers 0.0.35` reproduced baseline pixels bit-for-bit without speed improvement and was removed.
  - Retained native PyTorch memory-efficient SDPA, `--reserve-vram 0.8`, `--fp32-vae`, and `--disable-pinned-memory`.

### Phase 3: Quality-First High-Resolution 2-Stage Pipeline (2026-10-23)
- **Process-Isolated `1152×1728` $\rightarrow$ `2304×3456` Architecture**:
  - Upgraded native resolution to `1152×1728` (36 steps, `dpmpp_2m_sde` / `karras`, CFG 5.5) conditioned on `official_face_center_clean.png` via `IPAdapterAdvanced` (`weight=0.40, end_at=0.75`).
  - Split execution into two sequential child processes (`native` then `finish`) so `CLIP-ViT-H` + `IPAdapter` models fully exit before `YOLO` + `SAM` + `RealESRGAN` load, preventing Windows system commit exhaustion.
  - Switched native VAE decode to explicit `VAEDecodeTiled` (`tile_size=512, overlap=64`) in FP32 to prevent full-frame decode OOM.

### Phase 4: Community Node Audits & GUI/API Parity Fix (2026-10-25)
- **Optional PAG Mode (`--pag`)**:
  - Validated ComfyUI's built-in `PerturbedAttentionGuidance` (`scale=1.0`) after IPAdapter (`Emilia_10_Community_PAG_Quality_8GB.json`) as an optional composition variant.
- **API vs. GUI Drift Discovery & Fix**:
  - Discovered that API workflows omitted `noise_mask_feather` (defaulting to `0` at runtime) while GUI workflows serialized `20`. Standardized both API and GUI workflows to explicit `noise_mask_feather: 0` while preserving paste-back `feather: 12`.
- **Audited & Disabled Global-Hook Nodes**:
  - Exercised `UltimateSDUpscale` and `TiledDiffusion` on full `2304×3456` runs; moved both out of `app/custom_nodes/` into `temp/community_quality/*-tested-disabled/` so their global sampler hooks do not affect production runs.

### Phase 5: Quality-Preserving Memory Optimizations & Five 10-Seed Benchmark Rounds (2026-11-03 — 2026-11-05)
Implemented five cumulative memory/allocation optimizations across ComfyUI and IPAdapter, verified across **5 rounds × 10 fixed seeds (`20261027`–`20261036`) = 50 complete `2304×3456` deliveries**:

| Round | Cumulative Engineering Change | Median Command (s) | 10-Image Total (s) | Native Pixel Match vs R1 | Final Pixel Match vs R1 |
| :---: | :--- | ---: | ---: | :---: | :---: |
| **1** | Implementation Baseline | 315.560 | 3305.424 | 10 / 10 | 10 / 10 |
| **2** | `VAE.decode` writes directly into owned destination slice (`comfy/sd.py`) | 314.586 | 3140.055 | 10 / 10 | 10 / 10 |
| **3** | `VAE.encode` pre-allocates destination & releases batch temporaries (`comfy/sd.py`) | 316.577 | 3152.022 | 10 / 10 | 10 / 10 |
| **4** | `VAE.decode_tiled_` caller-owned 3-pass buffer reuse + `tiled_scale_multidim` feather mask cache (`comfy/sd.py`, `comfy/utils.py`) | 317.588 | 3298.729 | 10 / 10 | 10 / 10 |
| **5** | 1 MiB streaming SHA-256 in `LoadImage`/`LoadLatent` + in-place `np.clip` in `SaveImage` (`nodes.py`) + `batch_prompt==1` copy skip in `CrossAttentionPatch.py` | 316.586 | 3166.081 | 10 / 10 | 10 / 10 |

- **Additional Safeguards Added**:
  - Explicit clearing of local tensor references (`pixel_samples = None`, `samples = None`, `pixels_in = None`, `out = None`) prior to `soft_empty_cache()` when `VAE.decode` or `VAE.encode` falls back to tiling on OOM.
  - Out-of-place first sum in `VAE.decode_tiled_` so callers outside `torch.inference_mode()` own a normal mutable tensor before in-place `+=` and `/= 3.0`.

### Phase 6: Emilia Studio v2.0 Declarative Overhaul & Option A Repository Standardization (2026-11-06)
- **Unified `emilia_studio` Package**:
  - Consolidated configuration (`config.py`), declarative workflow generation (`workflows.py`), CPU-only quality/iris inspection (`inspector.py`), Ollama prompt compilation (`compiler.py`), two-stage execution (`engine.py`), SQLite experiment tracking (`ledger.py`), and CLI tooling (`cli.py`).
- **Runtime Patch Plugin (`app/custom_nodes/emilia_runtime_patches/`)**:
  - Packaged the Phase 5 core and IPAdapter optimizations into an idempotent runtime module (`apply_all_patches()`, `verify_patches()`).
- **End-to-End Verification (`03_finished_2x_00060_.png`)**:
  - Executed full two-stage render on seed `20261027` via the new engine (`native`: `287.0s`, `finish`: `79.1s`).
  - Confirmed decoded PNG SHA-256 hashes match Round 1 bit-for-bit:
    - Native (`1152×1728`): `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`
    - Final (`2304×3456`): `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`
  - Confirmed `QualityInspector` automatically catches the historical seed `20261030` two-person hallucination (`face_count=2, multiple_subjects=True -> reject_seed`) and synthetic blue/pink heterochromia (`repair_eyes`) using zero GPU VRAM.
- **Option A GitHub Readiness**:
  - Added root `AGENTS.md`, `PROGRESS.md`, `README.md`, `LICENSE`, `.gitignore`, `config.example.toml`, `models_manifest.json` (with `emilia-studio setup` verification), and `.github/workflows/ci.yml`.
