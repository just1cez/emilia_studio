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

### Phase 7: Deep Runtime & Studio Hot-Path Optimizations (2026-11-07)
Implemented 5 targeted performance and memory-commit optimizations across `emilia_studio/` and the ComfyUI/IPAdapter/Impact-Pack runtime, verified bit-exact on full end-to-end render `03_finished_2x_00061_.png` (seed `20261027`, native `276.9s`, finish `73.2s`):

| # | Component | Optimization | Measured Impact |
| :- | :--- | :--- | :--- |
| **1** | **Engine Stage Handoff & Subprocess Startup** (`emilia_studio/engine.py`) | Replaced localhost HTTP multipart upload (`_upload_image` + $O(N)$ `app/input` collision SHA-256 scan) with zero-copy ComfyUI `[output]` annotated references (`emilia/quality/01_base_XXXXX_.png [output]`), switched ephemeral stage SQLite DB to `sqlite:///:memory:`, removed duplicate `Image.open()` checks, and tightened readiness polling (`0.5s` $\to$ `0.1s`). | Zero Stage 1 $\to$ Stage 2 file copy/upload IO; zero `app/input/` duplicate file accumulation; faster stage startup. |
| **2** | **CPU QualityInspector & YOLO Cache** (`emilia_studio/inspector.py`) | Added process-wide `_YOLO_MODEL_CACHE` across `QualityInspector` instances, eliminated `95.55 MiB` `arr.astype(np.float32)` copy (`arr.mean(dtype=np.float64)`), replaced `23.89 MiB` `im.tobytes()` with zero-copy `memoryview(arr)` SHA-256, and passed `rgb_array` directly to `analyze_iris_color`. | **~167 MiB peak RAM reduction** per `2304×3456` inspection; repeat instance inspection dropped from **`1261.6 ms` $\to$ `762.1 ms` (-39.6%)**. |
| **3** | **ComfyUI Core, Upscaler, IPAdapter & SAM Hot Paths** (`comfy/utils.py`, `nodes_upscale_model.py`, `CrossAttentionPatch.py`, `impact/core.py`, `patches/emilia_runtime_patches/`) | • `tiled_scale_multidim`: in-place `ps_view.mul_(mask_view); o.add_(ps_view)` (**-676.8 MiB** transient tile allocations).<br>• `ImageUpscaleWithModel`: CPU view `in_img` + in-place `s.clamp_(0, 1.0)` before `.movedim(-3, -1)` (**-382.2 MiB** peak CPU RAM, **-23.9 MiB** VRAM).<br>• `comfy.utils.lanczos`: in-place `np.clip(..., out=arr)`, `t /= 255.0`, and `unsqueeze(0)` for `batch=1` (**-692.8 MiB** CPU churn during `4608×6912` $\to$ `2304×3456` downsample; `1383.4 ms` $\to$ `1239.3 ms`).<br>• `CrossAttentionPatch.Attn2Replace`: cached scalar `sigma` on `sigmas` tensor per step, eliminating **2,484 synchronous GPU$\to$CPU `cudaMemcpyDtoH` stalls** across 36 steps, and fixed hyphenated `comfyui-ipadapter-plus` resolution in `emilia_runtime_patches`.<br>• `SAMWrapper.predict`: cached `SamPredictor.set_image` embedding per image reference, cutting SAM ViT-B `1024×1024` backbone passes by **50%** on two-hand refinement. | **>1.7 GiB cumulative CPU allocation churn eliminated**; Stage 2 runtime `73.2s`; 100% bit-exact pixel parity (`3b58d9daf2...` / `d7fe10c200...`). |
| **4** | **SQLite ExperimentLedger Batch & Incremental Indexing** (`emilia_studio/ledger.py`) | Replaced per-file `sqlite3.connect()` + individual fsync commits (~109 commits per `ingest_existing_artifacts()` call) with single-connection `executemany()` batching and `existing_paths` incremental skip, wrapped in `contextlib.closing`. | Cold batch ingest (109 files): **`12,233.4 ms` $\to$ `177.5 ms` (68.9× faster)**; incremental ingest: **`12,233.4 ms` $\to$ `82.4 ms` (148.5× faster)**. |
| **5** | **Workflow Template Cache & Prompt Compiler** (`emilia_studio/workflows.py`, `compiler.py`) | Added in-memory `_RAW_WORKFLOW_CACHE` for canonical `.api.json` templates, wired `upscale_factor` (`scale_by = upscale_factor / 4.0`) in `build_highres_finish`, eliminated 14× redundant `.lower()` allocations in `clean_generated_prompt`, and reduced Ollama startup poll interval (`2.0s` $\to$ `0.25s`). | Zero repeated disk reads on workflow builds; faster Ollama cold start. |

### Phase 8: Next-Gen Ultra Multi-Scale Hires.fix Workflow & Canonical Outfit Lock (2026-11-07)
Following a 10-image diverse scene audit (`03_finished_2x_00062_.png` – `00071_.png`), diagnosed and resolved three structural visual bottlenecks in the legacy direct-`1152×1728` pipeline:

1. **Root-Cause Diagnosis**:
   - **Body & Leg Distortion (Vertical Stretching & 3rd-Leg Hallucinations)**: Direct `1152×1728` (`1.99 MP`) single-pass sampling exceeds SDXL/Illustrious-XL's native `832×1216` (`1.01 MP`) training bucket by 97% in area (216 vs 152 vertical latent rows), causing elongated lower torsos, bowed calves, and a 3rd leg column on seed `20261106` (`00067_.png`). Moreover, Stage 2 refined only Face and Hands, leaving 90% of the canvas (body, dress, legs, boots) unrefined before 4× RealESRGAN upscaling.
   - **Outfit Drift (Cloak/Capelet & Gold Clasp Bleeding)**: Prompt tags previously included `white cape` / `white capelet` and omitted `detached white collar, bare shoulders, white and purple off-shoulder dress, purple corset waist, layered frilled white mini skirt, long split white coattails`. Combined with `strength_model=0.80` on `emilia-re-zero-re.safetensors` (a 1-epoch Civitai LoRA trained on mixed hooded-cloak and fanart outfits), the model bled white hooded cloaks and high mandarin collars into standard dress scenes.
   - **Facial Flattening & Concentric Iris Rings**: `IPAdapterAdvanced` (`weight=0.40–0.50`, `linear`, `start_at=0.0`, `K+V`) forced the flat front-facing TV screenshot (`official_face_center_clean.png`, `277×416`) from step 0, while Stage 2 `FaceDetailer` (`denoise=0.22`, `guide_size=640`) lacked the denoise budget to remodel facial geometry and etched concentric rings into portrait irises (`00066_.png`).
2. **New `ultra` Profile Architecture (`build_ultra_native` + `build_ultra_finish`)**:
   - **Stage 1 Multi-Scale Hires.fix (`emilia_ultra_multiscale_native.api.json`)**:
     - **Pass 1 (`832×1216`, 28 steps, `dpmpp_2m_sde` Karras, `CFG 5.0`)**: Generates global full-body anatomy inside Illustrious-XL's exact native 2:3 bucket with `LoRA=0.42` and gentle `IPAdapterAdvanced` (`weight=0.18–0.20`, `weight_type="ease in-out"`, `start_at=0.15`, `end_at=0.65`, `embeds_scaling="V only"`).
     - **Pass 2 (`832×1216` $\to$ Tiled FP32 Decode $\to$ Lanczos `1152×1728` $\to$ Tiled FP32 Encode $\to$ 16-step `dpmpp_2m` Karras Hires.fix at `denoise=0.34, CFG 4.5, LoRA=0.38`, IPAdapter detached)**: Refines the entire body, dress pleats, corset, legs, stockings, and boots at `1152×1728` before Stage 1 exits.
   - **Stage 2 High-Resolution Detailer + 2× Upscale (`emilia_ultra_multiscale_finish.api.json`)**:
     - Upgraded `FaceDetailer` (`guide_size=768, max_size=1024, denoise=0.28, CFG 5.0, LoRA=0.38`) and `DetailerForEach` Hand SAM refiner (`guide_size=640, max_size=896, denoise=0.30, CFG 5.0`), followed by `RealESRGAN_x4plus_anime_6B` 4× and Lanczos 0.5× to `2304×3456`.
3. **Controlled A/B Verification (`app/temp/ab_ultra_workflow_20261107/ultra_before_after_3scenes.jpg` & `app/output/emilia/ultra/03_finished_2x_00001_.png`)**:
   - **Faster & Safer on 8 GiB VRAM**: Stage 1 runtime dropped from **`181.2s` $\to$ `136.6s` (-44.6s faster)** because the 28-step base pass runs on a `832×1216` canvas (half the tokens of `1152×1728`). Total end-to-end CLI `--profile ultra` runtime is **`278.6s`** (`136.6s` Stage 1 + `85.8s` Stage 2).
   - **Visual Defects Eliminated**: Eliminated the 3rd-leg hallucination on seed `20261106`, eliminated concentric iris rings and high-neck cloak collars on portrait seed `20261105` (`heterochromia_score=1.02°`), and restored Emilia's exact off-shoulder dress, detached white collar, single green emerald crystal, purple corset, layered mini skirt, and white knee-high boots on seeds `20261104` and `20261027`.

### Phase 9: Multi-Angle Official Reference Pack, Spatial-Masked Skirt IPAdapter & Painterly RescaleCFG Lighting (2026-11-08)
Conducted a direct pixel-by-pixel audit against official Re:Zero TV anime character design turnarounds (`assets/official_references/`) to fix fine-grained dress/skirt geometry inaccuracies and flat anime lighting:

1. **Official Dress & Skirt Geometry Audit vs. Previous Generates**:
   - **Previous Inaccuracies**: Earlier renders produced a generic solid-purple waist corset and plain ruffled mini skirt, missing Emilia's canonical **white corset waist panel with six vertical purple dash markings**, **three-tiered white petal-scalloped peplum overskirt with lavender edges**, **white box-pleated miniskirt with a thin purple stripe near the hem**, **long split white coattails with purple/gold four-leaf clover trim**, and **white thighhigh stockings with purple zig-zag V-cuffs**.
   - **Multi-Reference Conditioning + Canonical Outfit LoRA**:
     - Added 8 curated multi-angle official reference crops (`assets/official_references/emilia_official_skirt_and_bodice.png`, `emilia_official_fullbody_front.png`, `emilia_official_action_angle.png`, etc.) synced via `emilia-studio setup --sync`.
     - Integrated general SDXL `ip-adapter-plus_sdxl_vit-h.safetensors` (`weight=0.45`, `ease in-out`, `embeds_scaling="V only"`, `combine_embeds="norm average"`) conditioned on `emilia_official_skirt_and_bodice.png` through a **feathered torso/skirt spatial `attn_mask` (`FeatherMask=96px`)** alongside `Emilia_Rezero_Q-000005.safetensors` (`strength=0.25` in Pass 1, `0.0` in Pass 2). Confining the Skirt IPAdapter to the central torso/skirt mask prevents white reference-sheet backgrounds from washing out scene lighting.
2. **Painterly Luminous Lighting Architecture**:
   - Added `RescaleCFG` (`multiplier=0.75`) on both Pass 1 (`832×1216`) and Pass 2 (`1152×1728`, `denoise=0.40`, pure Illustrious-XL `strength_hires=0.0`) to eliminate highlight blowout and flat cel-shaded overexposure while rendering volumetric rim light, subsurface scattering on silver hair, and ambient occlusion inside skirt pleats.
   - Updated `EmiliaRenderEngine` so `ultra` mode handles minor iris hue drift inside Stage 2's high-resolution `FaceDetailer` (`guide_size=768, denoise=0.36`) rather than double-inpainting with the legacy `guide_size=640` `emilia_eye_repair` graph, and tightened `QualityInspector._extract_iris_hues` (`65° <= hue <= 345°`) so warm golden sunset/eyelash pixels (`43°–58°`) no longer trigger false heterochromia alarms.
3. **Multi-Lighting `2304×3456` Showcase Verification**:
   - **Daylight Flower Garden (`seed 20261027`, `exp4_garden_20261027_final_2x_00001_.png`)**: SHA-256 `e4efa1fa7807336f...` (`167.2s` Stage 1 + `70.9s` Stage 2).
   - **Moonlit Castle Balcony (`seed 20261104`, `exp4_balcony_20261104_final_2x_00001_.png`)**: SHA-256 `d8623c2f4edfdfa0...` (`171.1s` Stage 1 + `81.1s` Stage 2, `heterochromia_score=4.07°`).
   - **Golden Sunset Terrace (`seed 20261108`, `output/emilia/ultra/03_finished_2x_00004_.png`)**: SHA-256 `84156d8839314126...` (`161.8s` Stage 1 + `69.2s` Stage 2).
   - **Crystal Forest Sanctuary (`seed 20261109`, `output/emilia/ultra/03_finished_2x_00005_.png`)**: SHA-256 `22902ab0e6ac71a0...` (`167.7s` Stage 1 + `80.0s` Stage 2).
