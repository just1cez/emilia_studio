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

### Phase 10: Portable CLI distribution

- Separated immutable bundled resources from writable workspace/output/temp/ledger paths. Removed workstation drive discovery and arbitrary working-directory config loading. Explicit config/environment, user config and source config have tested precedence; relative paths resolve beside TOML.
- Added `emilia` alias, interactive ASCII welcome/shell, config management, doctor, model pull, presets and gallery. Original JSON query commands remain directly parseable. Rendering and download progress use stderr, with terminal and plain-log modes.
- Downloads validate ranges/lengths and available authoritative SHA256 metadata before replacing files. Two unverified Civitai download IDs were removed; these LoRAs require manual installation rather than downloading guessed models.
- Built wheel and Windows console executable with canonical reference/workflow assets. Heavy inference dependencies remain external. Both clean-wheel and frozen-executable smoke runs from `D:/` loaded 29 bundled workflow graphs and used real external CPU YOLO on the fixed baseline image (decoded SHA256 `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`).
- Verified live downloads of face and hand YOLO models, interactive ConPTY shell entry, preset command, and clean exit. Doctor, setup sync and ledger stats executed successfully from the packaged application.
- Regression evidence: 51 CLI/Studio tests passed; required focused runtime suite passed and full runtime suite passed 152 tests. One anyio pytest deprecation warning remains. An earlier concurrent run timed out; the completed run above passed.
- Standalone `dist/emilia.exe`, launched from `D:/`, completed the fixed seed `20261027` quality pipeline using bundled workflows and separate output/temp/workspace directories. Native: `261.6086275 s`; finish: `69.5341173 s`; command wall time: `416.36 s` (includes external CPU inspection/startup). Both decoded hashes exactly match baseline: native `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`; final `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`. Receipt: `temp/cli-smoke-output/emilia/quality/03_finished_2x_00001_.run.json`.

### Phase 11: Generation Quality Investigation, Candidate Ablations & Creative CLI Suite (2026-11-09; local only)

- **Spatial Outfit Attention-Mask Fix (`emilia_studio/workflows.py`, `workflows/api/emilia_ultra_multiscale_native.api.json`)**:
  - Corrected `FeatherMask` ordering so the torso/skirt rectangle (`node 201`) is feathered (`node 203`) *before* `MaskComposite` (`node 202`) onto the full canvas (`node 200`). Executing the real ComfyUI mask nodes (`tests/test_generation_options.py::test_outfit_mask_softens_internal_edges`) verified that internal rectangle corners drop from `1.0` to `0.00010850695` while the center remains `1.0` and outside pixels remain `0.0`.
- **Opt-In Stage 1 & Stage 2 Pipeline Extensions (`emilia_studio/workflows.py`, `emilia_studio/engine.py`, `emilia_studio/inspector.py`, `patches/emilia_runtime_patches/__init__.py`)**:
  - Added opt-in paired verified front/three-quarter face embedding averaging (`reference_mode="paired"`), framing-aware outfit attention (`upper_body` vs `full_body`, disabled in `portrait`), `RealESRGAN_x4plus_anime_6B` Hires.fix bridge (`hires_bridge="anime"`), base-pass `PatchModelAddDownscale` (`downscale_factor`), face-relative cosine-feathered `EmiliaRelativeGarmentSEGS` (`brooch_and_collar` / `bodice_and_peplum` / `full_garment`), dimension-matched secondary upscaler blending (`secondary_upscaler_name` + `upscale_blend_strength`), and advisory `analyze_garment_colors` telemetry in `QualityInspector`.
  - Enforced strict safety boundaries (`tests/test_generation_safety.py`, `tests/test_generation_options.py`): rejected native or failed eye-repair stages never enter Stage 2 under `--auto-gate`, strict seed lock preserves baseline stage execution, and eye repair inherits selected `ultra` checkpoint/LoRA settings.
- **Controlled 14-Image `2304×3456` Ablation & Visual Review (`workflows/reports/generation_candidates_20261109.json`, `workflows/reports/generation_visual_review_20261109.json`)**:
  - **Garden Control vs. Paired+Anime Bridge (`seed 20261027`)**: `garden_single_lanczos_final_00001_.png` (`327.8s` native / `110.7s` finish, SHA256 `a6184ab1f241f2a3...`) vs `garden_paired_anime_final_00001_.png` (`254.0s` native / `61.6s` finish, SHA256 `56781f1b36838660...`). Paired+Anime improved facial exposure and six-dash corset clarity, but both remained seated/cropped on seed `20261027` and Paired+Anime added a corner watermark; kept opt-in.
  - **Portrait Control vs. Paired+Anime Bridge (`seed 20261105`)**: `portrait_single_lanczos_final_00001_.png` (`122.0s` / `123.6s`, SHA256 `e08337472262425b...`) vs `portrait_paired_anime_final_00001_.png` (`130.4s` / `147.4s`, SHA256 `58c59638480c1b4e...`). Portrait framing cleanly omits lower-body skirt/boot tags.
  - **Candidate Checkpoint `Illustrious-XL-v1.1.safetensors` (`SHA256 536863e9...`)**: Tested on garden (`seed 20261027`, `236.9s` / `61.9s`, final SHA256 `1cec6ffed53fbfd5...`) and portrait (`seed 20261105`, `120.5s` / `95.4s`, final SHA256 `d6e889bfdc4abb4d...`). Produced flatter cel shading and softer finger anatomy than `Illustrious-XL-v0.1` with our Emilia LoRAs; retained as opt-in `checkpoint_name`, rejected as default.
  - **Candidate Upscaler `4x-AnimeSharp.safetensors` (`SHA256 7fc60054...`)**: Dual-upscaler `0.35` blend tested on garden (`256.0s` finish, SHA256 `303de0592247a7ee...`) and portrait (`228.0s` finish, SHA256 `f930bc5ad488fbc7...`). More than doubled Stage 2 time (`+104%` to `+131%`) for minor line contrast differences; retained as opt-in `secondary_upscaler_name`, rejected as default.
  - **Garment Geometry-Prior Detailer (`brooch_and_collar`, `denoise=0.22, steps=14`)**: Tested on garden (`114.9s` finish, SHA256 `812044d26740ebc1...`) and portrait (`144.0s` finish, SHA256 `f64c3760bde06d44...`). Cleanly sharpens the green diamond pendant facets and gold clasp at low overhead; retained as opt-in `garment_region`.
  - **Multi-Outfit / Angle / Companion / Magic Showcase (`2304×3456`)**:
    - `cloak` (`seed 20261110`, `upper_body`, `side`, `puck`, `ice`): `142.5s` / `88.9s`, final SHA256 `80c4465166870337d1bce3eefdbbbd4f0ec27823265994d649575fa9c247882a` (`output/emilia/ultra/03_finished_2x_00006_.png`).
    - `frozen-bond` (`seed 20261111`, `full_body`, `three-quarter`, `blizzard`): `133.4s` / `86.2s`, final SHA256 `0e52afe6728c20bdc9da6c1fa7ffff1319b06d9aac26a709cf1e14ccd05177ca` (`03_finished_2x_00007_.png`).
    - `sanctuary` (`seed 20261112`, `full_body`, `low-angle`, `ice`): `163.9s` / `97.7s`, final SHA256 `f55e4fbb2daafbb8d50e647565112925f19d65e6c7b897e75e3a7ca9c36e33a0` (`03_finished_2x_00008_.png`, complete head-to-boot standing full-body with floating ice crystals).
    - `nightgown` (`seed 20261113`, `upper_body`, `front`): `136.1s` / `83.4s`, final SHA256 `3218c1ee129343c026e9a75028e48a8453c385f2baa911e7c0ae14297d543f3c` (`03_finished_2x_00009_.png`).
- **Expanded Interactive CLI Command Suite (`emilia_studio/commands.py`, `emilia_studio/cli.py`, `ollama_comfyui_bridge.py`)**:
  - Added creative flags `--outfit` (`royal` / `cloak` / `frozen-bond` / `sanctuary` / `nightgown`), `--angle` (`auto` / `front` / `three-quarter` / `side` / `high-angle` / `low-angle`), `--puck`, and `--magic` (`none` / `ice` / `blizzard`) to `emilia render`, `emilia idea`, and `ollama_comfyui_bridge.py`.
  - Added 5 new CLI subcommands: `outfits`, `compare`, `export`, `benchmark`, and `clean`.

### Phase 12: CPU Pose Completeness Gate, Zero-LoRA-Bleed Alternate Costume Routing & Same-Stage Batch Pool (2026-11-09; local only)

- **1. CPU Pose & Full-Body Completeness Gate (`yolov8n-pose.pt`, `emilia_studio/inspector.py`, `emilia_studio/inspection_worker.py`)**:
  - Added `yolov8n-pose.pt` (`6,832,633` bytes, SHA256 `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`) to `models_manifest.json` and `QualityInspector`.
  - Implemented `analyze_pose_completeness` combining face-height-to-canvas ratio (`fh / H <= 0.165`) and face-anchored CPU `yolov8n-pose` lower-body keypoints (`max_ankle_confidence >= 0.30`, `hip_y_ratio <= 0.60`, `ankle_y_ratio >= 0.68`).
  - Verified across 11 historical renders: accurately flags seated/cropped full-body failures (`garden_single_lanczos_final_00001_.png` `fh/H=0.182, ankle_conf=0.150`; `03_finished_2x_00007_.png` `ankle_y=0.613`) as `reject_seed` under `--auto-gate` while accepting standing full-body renders (`03_finished_2x_00008_.png`, `03_finished_2x_00013_.png` `fh/H=0.1451, ankle_conf=0.7275, hip_y=0.5211, ankle_y=0.8119`, final SHA256 `8107a1dd62de258e17f39e5ace3bdcfd60a32768e78bb14c8002760360d17aa9`).
- **2. Non-Canonical Outfit Zero-LoRA-Bleed Routing (`emilia_studio/config.py`, `emilia_studio/compiler.py`, `emilia_studio/cli.py`)**:
  - Decoupled Stage 1 `lora_strength_base` (`0.0` for `frozen-bond` and `nightgown`) from Stage 2 `lora_strength_finish` (`0.28` in `FaceDetailer`), paired with `reference_mode="paired"` (`ipadapter_weight=0.28`), `alternate costume` positive emphasis, removal of `floor-length gown, wedding dress` from negative prompts, and explicit negative suppression of `official costume, default outfit, corset, waist panel, vertical purple dash markings, juliet bell sleeves, split coattails, yellow hawk emblem`.
  - Live `2304×3456` verification:
    - `nightgown` (`seed 20261114`, `output/emilia/ultra/03_finished_2x_00011_.png`, SHA256 `a98be5bc1dd54f403f9029e4badd2d6fd77adade4301914f0b404c06a9efd891`): cleanly renders a soft white/lavender ribbon-bow morning dress without the six-dash corset panel.
    - `frozen-bond` (`seed 20261111`, `output/emilia/ultra/03_finished_2x_00014_.png`, `129.6s` native / `98.8s` finish, SHA256 `665b8f73dca66ea039c6c83a95367fa36e7854ddeba4a82d0ca99c56b0e34481`): completely eliminates the six-dash royal corset, juliet sleeves, and hawk coattails, rendering a pure white one-piece forest sundress with a lavender waist ribbon sash and white ankle boots on ice while preserving Emilia's face, violet eyes, and white rose hair ornament.
- **3. Multi-Image Same-Stage Process Pool (`render_batch_pipeline`, `emilia batch`)**:
  - Added `_stage_server_session` and `render_batch_pipeline` in `emilia_studio/engine.py` plus the `emilia batch` CLI subcommand (`--seeds` / `--seed` + `--count`). Executes all Stage 1 native renders on one persistent ComfyUI process, runs CPU quality/pose gates, exits Stage 1 cleanly to free all VRAM/commit memory, and runs all Stage 2 finish renders on one persistent Stage 2 process.
  - Live 2-seed `2304×3456` batch verification (`seeds 20261113, 20261114`): second image completed in `120.6s` native + `55.0s` finish (`175.6s` total GPU stage time, total 2-image batch wall time `441.4s`), eliminating redundant 6.94 GiB checkpoint cold-start overhead while respecting the 8 GiB VRAM isolation boundary.

### Phase 13: Top-10 Audit Optimizations — Coattail Deduplication, Watermark Suppression, Iris Inner Ring, Fast Eye Repair & Full Suite CI (2026-11-09; local only)

- **1. Bottom-Corner Watermark Suppression (`emilia_studio/config.py`)**:
  - Moved `text, watermark, signature, copyright, copyright name, logo, artist name, english text` into the first 77-token CLIP chunk of `DEFAULT_ULTRA_NEGATIVE_PROMPT` (unweighted tokens; parenthesized `:1.3` weights in negative prompts combined with `RescaleCFG(0.75)` were found to trigger green color-cast instability).
- **2. Coattail Hawk-Emblem Deduplication (`emilia_studio/workflows.py`, `emilia_studio/config.py`)**:
  - Narrowed and raised the `full_body` spatial torso/peplum mask (`node 201`/`202` in `build_ultra_native`) from `width=0.58W, height=0.53H (x=0.21W, y=0.28H..0.81H)` to `width=0.42W, height=0.34H (x=0.29W, y=0.25H..0.59H)` so the lateral sleeve tails in `emilia_official_skirt_and_bodice.png` are not projected onto the lower skirt coattails (`0.59H..0.81H`).
  - Disambiguated `plain white split dress coattails with thin purple border trim and pale lavender lining` from `pointed purple sleeve cuffs bearing yellow hawk emblem` and added `emblem on skirt, emblem on coattails, four purple tails` to `DEFAULT_ULTRA_NEGATIVE_PROMPT`.
- **3. Fast Stage-1.5 `eye_repair` Without Redundant 4× RealESRGAN (`emilia_studio/workflows.py`, `emilia_studio/engine.py`)**:
  - Wired `if not upscale:` in `EmiliaWorkflowBuilder.build_eye_repair` to prune nodes `13`, `14`, `15`, `16` (`UpscaleModelLoader`, `ImageUpscaleWithModel` 4×, `ImageScaleBy`, `SaveImage`) and passed `upscale=False` from `EmiliaRenderEngine` (`render_quality_pipeline` and `render_batch_pipeline`), plus `ultra` face prompts and `noise_mask_feather: 0`. Measured live `eye_repair` stage completion time dropped to **`22.1s`** while preserving `upscale=True` default parity with `emilia_eye_repair.api.json`.
- **4. SQLite Ledger `.rglob("*.run.json")` Coverage & WAL Concurrency (`emilia_studio/ledger.py`, `emilia_studio/engine.py`)**:
  - Updated `ExperimentLedger.ingest_existing_artifacts` to scan `output/emilia/**/*.run.json` (indexing all `ultra`, `quality`, and experiment receipts), safely handle `None` or dict-valued fields, preserve file `st_mtime` ISO timestamps and `profile`/`_pag` modes, and enable `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=5000`.
- **5. Root `pytest tests/` Collection Fix & Full CI Suite (`tests/conftest.py`, `.github/workflows/ci.yml`, `build_support.py`, `emilia_studio/cli.py`)**:
  - Added `tests/conftest.py` so `pytest tests/` from the repository root resolves `app/` and cleanly skips `tests/comfy_core/` only when `app/comfy` or `torch` is absent; updated `.github/workflows/ci.yml` to run `python -m pytest tests/`; added `workflows/WORKFLOW_DECISIONS.txt` to `build_support.py::RESOURCE_PATTERNS`; and updated `setup --sync` to sync all `tests/comfy_core/test_*.py` modules into `app/tests-unit/comfy_test/`.
- **6. Canonical Cyan-Blue Inner Iris Ring (`emilia_studio/config.py`)**:
  - Removed `"concentric iris rings, ringed eyes"` from `DEFAULT_ULTRA_NEGATIVE_PROMPT` and `DEFAULT_ULTRA_FACE_NEGATIVE_PROMPT` (replacing `"blue eyes"` with `"solid blue eyes"`) and specified `"symmetrical violet eyes, clear glossy purple outer irises with luminous cyan-blue inner ring around soft vertical oval pupils"` in `DEFAULT_ULTRA_FACE_POSITIVE_PROMPT`.
- **7. Visual Workflow Link-Overwriting Guard & Seed Control Fix (`emilia_studio/workflows.py`, `emilia_studio/cli.py`, `workflows/visual/`)**:
  - Guarded `sync_visual_workflow` with `not isinstance(inputs[key], list)` so linked API inputs never overwrite scalar GUI widgets; fixed `"control_after_generate": "fixed"` in `Emilia_08` and `Emilia_10`; and wired `emilia workflows --sync-visual`.
- **8. CPU Inspector Zero-Face Single-Pass & Framing-Conditional Pose Inference (`emilia_studio/inspector.py`)**:
  - Added `_detect_if_missing=False` when `inspect_image` calls `analyze_iris_color` (eliminating duplicate 3-model YOLO execution when `face_count == 0`), skipped `yolov8n-pose.pt` when `framing != "full_body"`, and guarded in-process YOLO `.predict()` with `_YOLO_LOCK`.
- **9 & 10. Runtime Patch Source-Signature Verification (`patches/emilia_runtime_patches/__init__.py`, `patches/comfyui_core.patch`)**:
  - Upgraded `apply_all_patches()` and `verify_patches()` to inspect live function source signatures (`pixel_samples[x:x+batch_number].copy_(out)`, `np.clip(i, 0, 255, out=i)`, `s.clamp_(min=0, max=1.0)`, `if batch_prompt != 1:`, `_emilia_sigma_val`) instead of unconditionally stamping `_emilia_patched = True`.
- **Live `2304×3456` Verification (`app/output/emilia/ultra/03_finished_2x_00015_.png`, `seed 20261027`, `preset=rose-garden`)**:
  - Stage 1 `165.7s`, Stage 2 `86.5s`, decoded final SHA256 `71f1e7deaea0c2f5e475e3e79886e8979d85ba7c8ef31405f6a2247787412e62`.
  - Visually verified against `03_finished_2x_00013_.png`: (1) lower skirt coattails are clean white/lavender with **zero** duplicated gold hawk emblems (only the two sleeve cuffs bear the gold hawk emblem); (2) bottom-right `©ENTLELS...` watermark is **100% eliminated**; (3) violet irises display a luminous cyan-blue inner pupil ring (`left_eye_hue_deg=263.88°, right_eye_hue_deg=266.54°, heterochromia_score=2.66°`); and (4) full-body standing pose from crown braid to white ankle boots passed CPU pose gate (`fh/H=0.0993, max_ankle_confidence=0.6576, hip_y=0.5059, ankle_y=0.8080`).

### Phase 14: Complete Resolution of All 17 Remaining Audit Findings (2026-11-09; local only)

- **CLI, TUI, Bridge & Config Completeness (`commands.py`, `cli.py`, `tui.py`, `config.py`, `ollama_comfyui_bridge.py`, `start-comfyui.ps1`)**:
  - Added `experimental_models` (`illustrious-xl-1.1`, `animesharp`) support to `emilia pull` and `emilia doctor`.
  - Updated `emilia compare` to reuse cached `.run.json` YOLO iris telemetry or live CPU YOLO when models are present.
  - Exposed `--reference-mode`, `--hires-bridge`, `--checkpoint`, `--garment-region`, `--upscaler`, `--secondary-upscaler`, `--upscale-blend`, and `--max-retries` across `render`, `batch`, and `idea`, and resolved `--preset` `default_seed` when `--seed` is omitted.
  - Added `workflows_dir`, `visual_workflows_dir`, and `resources_dir` to `PATH_KEYS` in `config set`; supported `help <subcommand>` in `emilia shell`.
  - Hardened `render_table` and `ProgressBar` against multiline cells, ANSI escape widths, and East Asian wide characters.
  - Added `-Listen` and `-Port` parameters to `start-comfyui.ps1` and forwarded configured host/port from `ollama_comfyui_bridge.py`.
  - Fixed `"gentle gentle smile"` typo in `DEFAULT_LORA_PRODUCTION_POSITIVE_PROMPT`, added `.expanduser()` to all path fields in `StudioConfig.__post_init__`, and validated port bounds (`1..65535`).
- **Prompt Compiler & Workflow Builder Hardening (`compiler.py`, `workflows.py`)**:
  - Hardened `clean_generated_prompt` to strip unclosed `<think>` or stray `</think>` reasoning chains; routed outfit-specific `ipadapter_weight` in `infer_composition_hints`; filtered Emilia-specific eye/cloak bans from non-Emilia negative prompts; replaced fragile substring `.replace()` with tag-set filtering in `build_emilia_ultra_negative`.
  - Added 64-bit integer `seed` validation in `build_ultra_native` and `build_ultra_finish`; deduplicated individual angle tags; stripped lower-body tags on direct `build_ultra_native(framing="portrait")` calls; and reused `LoraLoader` node `10` when `lora_strength_hires == lora_strength_base`.
- **Persistent External YOLO Worker, Corner Watermark Diagnostics, Benchmark Scripts & ComfyUI Cache Accounting (`inspection_worker.py`, `inspector.py`, `app/`)**:
  - Added `--server` persistent JSON-lines loop with `_MODEL_CACHE` and direct `image_path` loading to `inspection_worker.py` + `QualityInspector._detect_anatomy_external`, plus advisory `analyze_corner_artifacts` 4-corner stroke density diagnostics on `InspectionReport`.
  - Replaced `time.sleep(2)` with `child.wait(timeout=2.0)` and reused receipt `pixel_sha256` in `benchmark_architecture_iterations.py`; cleaned dead helpers and bare `assert`s in `render_emilia_quality.py` and `evaluate_architecture_iteration.py`.
  - Released `del cache_results` in `app/execution.py` and added `Mapping` (`dict`) recursion + `untyped_storage()` deduplication in `app/comfy_execution/caching.py` (captured in `patches/comfyui_core.patch`).

### Phase 15: Hardware-Ceiling Hot-Path & Orchestration Optimizations (2026-11-10)

- **Hardware Bottleneck Audit (`NVIDIA CMP 40HX` + `Xeon E5-2690 v2`)**:
  - Confirmed `PCIe 2.0 x16` bus (`5.58 GB/s` H2D / `6.23 GB/s` D2H pinned DMA; `3.76–4.64 GB/s` pageable H2D under `--disable-pinned-memory`, while NVML misreports `Device Max: 1` at idle) and `46.25 TFLOPS` FP16 TensorCore ceiling, plus `54.657s` non-execution overhead (`2,027.5ms` Windows Winsock `connect_ex` SYN retries on free ports, `9.72s` per-stage `comfy_extras` imports of 110 unused video/audio/3D modules, and `8.81s` cold CPU YOLO initialization blocking Stage 2).
- **1. Zero-Wait Orchestration, Stage-Minimal Boot & Background YOLO Pipelining (`emilia_studio/engine.py`, `inspector.py`, `inspection_worker.py`, `app/main.py`, `app/nodes.py`)**:
  - Replaced `sock.connect_ex` (`2,027.5ms` on free port) with `SO_EXCLUSIVEADDRUSE` `sock.bind` (`0.24ms`) and added `server.log` readiness detection (`25ms` poll before HTTP `/system_stats`) to eliminate Winsock SYN retry stalls.
  - Added stage-specific `--disable-all-custom-nodes --whitelist-custom-nodes` (`comfyui-ipadapter-plus` + `emilia_runtime_patches` in Stage 1; `comfyui-impact-pack` + `comfyui-impact-subpack` + `emilia_runtime_patches` in Stage 2), `--mmap-torch-files`, and `EMILIA_HEADLESS_FAST_BOOT=1` (loading only the 13 essential `comfy_extras` modules in `1.07s` vs `9.72s` and skipping ephemeral `:memory:` Alembic migrations).
  - Added `QualityInspector.warmup()` (and `"cmd": "warmup"` in `inspection_worker.py --server`) triggered asynchronously during Stage 1 KSampler execution, and overlapped CPU `inspect_image()` with `process.terminate() -> process.wait()` via `asyncio.gather` while strictly completing Stage 1 exit before Stage 2 launch.
- **2. GPU, PCIe & L3-Cache Hot-Path Optimizations (`patches/`)**:
  - **IPAdapter KV, Spatial Broadcast Mask & Zero-Uncond Caching (`patches/ipadapter_plus.patch`)**: Cached step-invariant `(ip_k, ip_v)` linear projections on `cond._emilia_kv_cache` (eliminating 15,000+ CUDA kernel launches per render), cached `[B, S, 1]` interpolated spatial masks on `mask._emilia_attn_mask_cache` without `.repeat(1, 1, C)` (`1280×` less mask bandwidth), and cached zero-image `CLIPVision` uncond hidden states on `clipvision._emilia_zero_uncond_cache`.
  - **L3-Cache-Blocked Multi-Threaded Quantization & Fast Static `LoadImage` (`patches/comfyui_core.patch`, `patches/emilia_runtime_patches/`)**: Added 256-row L3-blocked OpenMP `tensor_to_u8_numpy` (`324ms -> 97.5ms` on `4608×6912`, eliminating `382.2 MiB` temporary float32 allocation in `lanczos` and `91.1 MiB` in `SaveImage`), zero-copy `u8_numpy_to_tensor`, PyAV-free static RGB PNG `LoadImage` fast path (`365ms -> 293ms`), `torch.inference_mode()` guard in `ImageUpscaleWithModel`, and inference-mode in-place tile accumulation in `VAE.decode_tiled_`.
  - **`SAM ViT-B` Meta+MMap+Assign Load & Zero-Copy BBox Masks (`patches/impact_pack.patch`, `patches/impact_subpack.patch`)**: Loaded `sam_vit_b_01ec64.pth` onto `meta` device with `torch.load(..., mmap=True)` + `load_state_dict(..., assign=True)` (`1,159.7ms -> 193.0ms`, `6.0×` faster, `-750 MiB` peak host commit), and removed dead full-frame BGR/grayscale copies in `subcore.py::inference_bbox`.
  - **CPU Inspector, Workflow & Ledger Fast Paths (`inspector.py`, `workflows.py`, `ledger.py`)**: Replaced 5-pass `float64` image stats (`208.3ms`, `191 MiB` alloc) with C `rgb_im.histogram()` exact reduction (`40.1ms`, `0` alloc, `0.00e+00` diff), sliced 4 corner ROIs before `float32` luma conversion (`107.6ms -> 8.4ms`, `12.8×` faster), added `_PARSED_WORKFLOW_CACHE` structural cloning, and deduplicated per-process SQLite DDL initialization.
- **Live Bit-Exact Benchmark Verification (`seed 20261027`)**:
  - **`quality` profile (`03_finished_2x_00072_.run.json` before $\to$ `03_finished_2x_00073_.run.json` after)**:
    - Total wall time: **`314.083s` $\to$ `261.985s` (`-52.098s`, `-16.6%`)**; Stage 1 `190.962s` $\to$ `160.893s` (`3.78s/it` $\to$ `3.56s/it`); Stage 2 `68.464s` $\to$ `63.331s`; non-execution overhead `54.657s` $\to$ `37.761s` (`-30.9%`).
    - Decoded PNG SHA-256 **100% bit-exact**: Native `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`, Final `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`.
  - **`ultra` profile (`preset=rose-garden --auto-gate`, `03_finished_2x_00016_.run.json`)**:
    - Total wall time: **`299.006s`** (Stage 1 `176.659s`, Stage 2 `84.709s`, non-execution overhead `37.637s` including dual 3-model CPU pose/iris/watermark gates).
    - Decoded PNG SHA-256 **100% bit-exact**: Native `ba5a5412fd5af1cd5d7a679a62bae5404e62bbfc5e145a285fc01e73376f4238`, Final `71f1e7deaea0c2f5e475e3e79886e8979d85ba7c8ef31405f6a2247787412e62`.
- **3. Persistent GPU Tuning Profile (`gpu_profile.py`, `install-gpu-profile-task.ps1`)**:
  - Earlier brief GEMM/SDPA/Conv2d probes reported 48-51°C and 44.48 TFLOPS; these were not thermal-equilibrium or full-render tests. Both fan targets were 100%, but only one tachometer reported approximately 3603 RPM; the second reported zero. Shared tachometer wiring and normal operation of both physical fans were not established. A historical cumulative thermal counter does not establish when or at which temperature throttling occurred.
  - `gpu_profile.py apply|reset|status` handles NVML power/fans/locked clocks and NvAPI P0 offsets; the SYSTEM task `EmiliaGPUProfile` restores the requested settings. Successful configuration does not establish cooling adequacy or long-term overclock stability.

### Monitored GPU profile render (2026-11-10)

- Command: `python -m emilia_studio.cli render --profile quality --seed 20261027`; no PAG or auto-gate; no hardware settings changed during this investigation. Completed with exit 0 in 314.156 s command wall time; native execution 210.4 s, finish 64.4 s. Receipt: `app/output/emilia/quality/03_finished_2x_00074_.run.json`. This is slower than the earlier 261.985 s run, not evidence of an overclock speedup; uncontrolled single runs cannot establish the cause.
- 304 approximately one-second NVML samples saved under `temp/gpu_render_probe_20261110_215707/`: peak core temperature 88°C, peak sampled power 226.938 W, peak VRAM 7977.5 MiB. Among 55 samples with GPU utilization >=80%, core clocks were 1815-1980 MHz (mean 1922.7), memory 7500 MHz, temperature 71-88°C (mean 80.6). SW power-cap bit appeared in 56 total samples; no sampled SW/HW thermal slowdown, HW slowdown or HW power-brake bits. Sampled absence is not proof of no transient throttling.
- Post-render readback: 220 W limit, fan targets [100,100], core +105 MHz, memory +500 MHz. Actual fan feedback remained ambiguous (second channel zero); inspect physical rotation before prolonged unattended loads. No claim of full-load 50°C operation or guaranteed thermal stability.
- Native `01_base_00073_.png` (1152x1728) decoded SHA-256: `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`. Final `03_finished_2x_00074_.png` (2304x3456): `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`. Both exactly match baseline. This verifies this render only, not general overclock stability.

### Phase 16: Full 10-Point Storage-Free HDD, Batch Wave & Fast TensorCore Optimizations (2026-11-12)

- **1. Six Lossless (100% Bit-Exact) Optimizations (Zero `C:` SSD Usage)**:
  - **Windows `PrefetchVirtualMemory` Sequential HDD MMap Prefetch (`app/comfy/utils.py`, `patches/comfyui_core.patch`)**: Coalesces `ModelMMAP` page faults into sequential `512 MiB` read windows on `D:` HDD (`WDC WD10EZEX`), eliminating scattered 4 KB page-fault seek thrashing while keeping `C:` SSD untouched.
  - **Background Stage 2 Standby RAM Pre-Warming (`emilia_studio/engine.py`)**: Sequentially streams Stage 2 exclusive weights (`sam_vit_b_01ec64.pth`, `RealESRGAN_x4plus_anime_6B.pth`, `face_yolov8n.pt`, `hand_yolov8n.pt`, and optional `sdxl-vae-fp16-fix.safetensors`) into OS file cache during Stage 1 KSampler execution while the disk is idle.
  - **Unified Stage 2 Server Merging `Wave 1.5 (Eye Repair)` + `Wave 2 (Finish)` (`emilia_studio/engine.py`)**: Eliminates the separate `batch_eye_repair` ComfyUI server boot in `render_batch_pipeline`; `eye_repair` now executes as Sub-wave 2a on the shared `batch_finish` server before Sub-wave 2b `finish` (zero duplicate 6.5 GB checkpoint reload).
  - **Asynchronous Pipelined CPU Inspection in Batch Waves (`emilia_studio/engine.py`)**: Overlaps `QualityInspector.inspect_image` of item `i-1` on background threads while the GPU immediately executes item `i` across both Wave 1 and Wave 2.
  - **Persistent Ollama Prompt Cache + Physical Core `num_thread` (`emilia_studio/compiler.py`)**: Caches compiled tags by `(model, style, idea)` in memory and `temp/ollama_prompt_cache.json` (`<1 ms` hit vs `12–25s` cold CPU LLM load) and sets `"num_thread": min(10, cpu_count // 2)` alongside `"num_gpu": 0, "keep_alive": 0`.
  - **Zero-Waste Micro-Cleanups (`patches/impact_pack.patch`, `app/nodes.py`, `inspector.py`, `start-comfyui.ps1`)**: Short-circuits `make_sam_mask` / `make_sam_mask_segmented` before moving `358 MiB` SAM onto GPU when `len(segs[1]) == 0`; pre-downscales `>960px` (`2304×3456`) images to `640`-class resolution with exact bbox/keypoint coordinate rescaling in `QualityInspector`; uses lossless `compress_level=1` in headless `SaveImage`; adds `--mmap-torch-files` to `start-comfyui.ps1`.
- **2. Four Opt-In High-Speed Inference Optimizations (`--fast` / `--fp16-upscaler` / `--fp16-vae` / `--fast-tiled-vae` / `--steps`)**:
  - **FP16 TensorCore `RealESRGAN` (`app/comfy_extras/nodes_upscale_model.py`, `EMILIA_FP16_UPSCALER=1`)**: Isolated microbenchmark on `02_refined_00074_.png` (`1152×1728 -> 4x`): **`25.28s` (FP32) $\to$ `2.45s` (FP16) (`-22.84s`, `10.34×` faster)**; `91.11%` pixels identical, `99.99%` within `1/255` (`mean_abs=0.089/255`).
  - **Standalone `sdxl-vae-fp16-fix.safetensors` (`319 MiB` on `D:`, SHA-256 `235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1`)**: Registered in `models_manifest.json` (`experimental_models`), wired via `EmiliaWorkflowBuilder.inject_external_vae` + `--fp16-vae` when `--fp16-vae` or `--fast` is passed.
  - **Single-Pass Feathered Tiled VAE (`app/comfy/sd.py`, `EMILIA_FAST_TILED_VAE=1`)**: Replaces ComfyUI's 3-pass redundant tiled decode/encode with a single `overlap = max(overlap, tile // 4)` feathered pass.
  - **CLI & Bridge Step Overrides (`cli.py`, `ollama_comfyui_bridge.py`, `render-emilia-quality.ps1`)**: Exposed `--steps`, `--face-steps`, `--hand-steps`, `--fp16-upscaler`, `--fp16-vae`, `--fast-tiled-vae`, and `--fast` (`-Fast` in PS1, `--fast-finish` in bridge).
- **3. Live End-to-End Verification (`seed 20261027`, `158` Studio tests + `156` ComfyUI tests passing)**:
  - **Default Lossless `quality` (`03_finished_2x_00075_.run.json` cold `341.508s` / `03_finished_2x_00077_.run.json` warm `271.538s`: Stage 1 `167.326s`, Stage 2 `63.374s`)**:
    - **100% bit-exact decoded PNG SHA-256**: Native `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`, Final `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`.
  - **`--fast` Preset `quality` (`03_finished_2x_00076_.run.json`, 32 native / 16 hand steps + FP16 upscaler + FP16-fix VAE + 1-pass tiled VAE)**:
    - **Total wall time `213.686s` (`-57.85s` / `-21.3%` vs warm baseline; `-127.82s` / `-37.4%` vs cold run)**: Stage 1 `167.326s -> 137.168s` (`-30.16s`), Stage 2 `63.374s -> 35.624s` (`-27.75s`, **`1.78×` faster Stage 2**), quality gate `PASS` (`2304×3456`, violet irises `287.48° / 282.96°`).
