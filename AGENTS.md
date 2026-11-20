# AGENTS.md — Emilia Studio Engineering & Agent Guidelines

## 1. Repository Topology (Option A: Standalone Top-Level Studio + ComfyUI Runtime)

This repository uses an **Option A** two-tier layout separating our declarative studio orchestration (`emilia_studio/`) from the underlying ComfyUI execution engine (`app/`):

- **`emilia_studio/`** (Top-level Python package — Primary Ownership):
  - `_version.py` & `config.py`: Canonical package version, path resolution, environment variable overrides (`EMILIA_*`), `config.toml` loader, and canonical Emilia character/prompt constants.
  - `workflows.py`: `EmiliaWorkflowBuilder` generating API execution graphs (`ultra_native` and `lightning_preview` built in code; `ultra_finish` and `ultra_eye_repair` adapted from JSON templates; `highres_native`, `highres_finish`, `eye_repair`, `lora_native` loaded from JSON templates) and synchronized GUI visual workflows (`sync_visual_workflow`). A graph depends only on its arguments and configuration, never on which model files are installed.
  - `inspector.py` & `inspection_worker.py`: Zero-VRAM (CPU-enforced) `QualityInspector` and persistent JSON-lines YOLO worker using YOLOv8n (`face_yolov8n.pt`, `hand_yolov8n.pt`, `yolov8n-pose.pt`), COCO eye-landmark constrained HSV circular hue analysis, `ultra` full-body standing/ankle completeness gates, and advisory garment/corner-watermark diagnostics.
  - `compiler.py`: `OllamaPromptCompiler` translating Chinese natural-language ideas into Danbooru-style SDXL tags with persistent memory/disk tag caching, dynamic composition framing (`portrait` / `upper_body` / `full_body`), non-canonical outfit LoRA decoupling (`0.0 -> 0.28`), and IPAdapter weight routing.
  - `engine.py`: Two-stage isolated wave rendering engine (`EmiliaRenderEngine`) with per-graph total (`stage_timeout_s`) and prompt-event silence (`stall_timeout_s`) watchdogs. `render_quality_pipeline` is the single-seed form of `render_batch_pipeline`: one Stage 1 process for all seeds and retries, the CPU gate, then one Stage 2 process for eye repairs and finishing; `.run.json` receipts are written beside the image they describe.
  - `ledger.py`: WAL-mode SQLite experiment and render ledger (`ExperimentLedger`) indexing `.run.json` receipts under `output/emilia/` and 10-seed architecture benchmark rounds.
  - `cli.py`, `commands.py` & `tui.py`: Unified CLI/TUI entrypoint (`emilia` / `emilia-studio` / `python -m emilia_studio.cli`).
- **Top-Level Tracked Assets (`workflows/`, `patches/`, `assets/`, `tests/`, `scripts/`)**:
  - `workflows/api/`: 29 `.api.json` graphs — 5 canonical parity-verified graphs (`emilia_ipadapter_highres_production`, `emilia_ipadapter_highres_upscale`, `emilia_eye_repair`, `emilia_ultra_multiscale_native`, `emilia_ultra_multiscale_finish`), 1 single-stage bridge template (`emilia_lora_native`), and 23 historical experiment graphs.
  - `workflows/visual/`: 10 ComfyUI visual workflows (`Emilia_01..10`; `Emilia_08..10` re-synced via `emilia workflows --sync-visual`).
  - `workflows/reports/` & `workflows/WORKFLOW_DECISIONS.txt`: Historical benchmark/ablation JSON reports and architectural decision records.
  - `patches/`: `emilia_runtime_patches/` custom-node plugin (`EmiliaRelativeGarmentSEGS`, `EmiliaCheckpointConditioning`, and source-verified runtime hooks) plus `comfyui_core.patch` (7 core files), `ipadapter_plus.patch`, `impact_pack.patch`, and `impact_subpack.patch`. Base commits are pinned in `models_manifest.json`.
  - `assets/`: `official_face_center_clean.png` plus 10 crops in `assets/official_references/` (`official_face_center_clean.png`, `official_references/emilia_official_face_3q.png`, and `official_references/emilia_official_skirt_and_bodice.png` are wired into production graphs).
  - `tests/`: 9 standalone top-level test modules (`tests/test_*.py`), `tests/conftest.py`, and 4 ComfyUI operator unit modules (`tests/comfy_core/test_*.py`).
  - `scripts/`: 10-seed benchmark round runner (`benchmark_architecture_iterations.py`), evaluator (`evaluate_architecture_iteration.py`), and shared PowerShell Python resolver (`EmiliaPython.ps1`).
- **Root Launchers & Config**:
  - `render-emilia-quality.ps1`: PowerShell shortcut for `emilia render` (`-Seed`, `-Profile`, `-Pag`, `-AutoGate`, `-Fast`, `-Preview`).
  - `start-comfyui.ps1`: Interactive GUI server launcher (`-Listen`, `-Port`, default `127.0.0.1:8188`).
  - `ollama_comfyui_bridge.py`: CLI bridge driving either the 2-stage pipeline (`--mode ultra`, default, or `--mode quality`, with `--fast-finish` for TensorCore speedups) or single-stage interactive server queueing (`--mode fast`).
  - `config.example.toml` & `models_manifest.json`: Portable TOML configuration template; manifest of 11 required models, 4 experimental models, reference assets, and pinned commits/patches.
  - `gpu_profile.py` & `install-gpu-profile-task.ps1`: Optional CMP 40HX tuning profile and versioned, ACL-audited `%ProgramData%\EmiliaGPUProfile` SYSTEM scheduled task.
- **`app/`** (Git-Ignored External ComfyUI Runtime Subtree):
  - Upstream ComfyUI checkout (`Comfy-Org/ComfyUI`, retaining its own `app/.git`) plus custom nodes (`comfyui-ipadapter-plus`, `comfyui-impact-pack`, `comfyui-impact-subpack`), each pinned to a commit in `models_manifest.json` and modified by the matching `patches/*.patch`. `emilia doctor` verifies both.
  - Run `python -m emilia_studio.cli setup --sync` to sync top-level `assets/`, the patch plugin, `workflows/`, and runtime tests into `app/`.
  - When modifying anything inside `app/comfy/` or `app/nodes.py`, also follow `app/AGENTS.md` and regenerate `patches/comfyui_core.patch`.
---

## 2. Hardware, Memory, and Process Isolation Rules

Target workstation baseline: **8 GiB NVIDIA CMP 40HX (Turing, compute capability 7.5)** + **Intel Xeon E5-2690 v2 (10C/20T)** on Windows with bounded system commit limits.

1. **Mandatory Server Flags**:
   - Always launch headless ComfyUI stages with:
     `--reserve-vram 0.8 --disable-pinned-memory --mmap-torch-files --preview-method none --disable-api-nodes` plus `--fp32-vae` by default (`--fp16-vae` only when explicitly opted into `sdxl-vae-fp16-fix.safetensors`) and stage-specific `--disable-all-custom-nodes --whitelist-custom-nodes`.
   - Interactive GUI launcher `start-comfyui.ps1` retains `--fp32-vae --mmap-torch-files --preview-method latent2rgb`.
   - Do **not** enable pinned memory or static VRAM modes; historical trials exhausted the Windows commit limit regardless of free physical RAM.
2. **Two-Stage Process Isolation**:
   - **Stage 1 (`native`)** loads `Illustrious-XL` + Emilia LoRA + `CLIP-ViT-H` + `IPAdapter`:
     - `quality` profile: single-pass `1152×1728` (36 steps, `dpmpp_2m_sde` Karras, `emilia-re-zero-re` 0.80, Face IPA 0.40 `K+V`).
     - `ultra` profile: Pass 1 `832×1216` (28 steps, `Emilia_Rezero_Q-000005` 0.25, Face IPA `V only` + feathered torso-masked Skirt IPA `V only` + `RescaleCFG`) $\rightarrow$ tiled VAE bridge $\rightarrow$ Pass 2 `1152×1728` (`dpmpp_2m` Karras Hires.fix, IPAdapter detached).
   - **Stage 2 (`finish`)** loads `Illustrious-XL` + Emilia LoRA + YOLOv8n (`face`/`hand`) + SAM (`sam_vit_b_01ec64.pth`) + optional pre-finish `eye_repair` (`upscale=False`) + `FaceDetailer` + `DetailerForEach` + `RealESRGAN_x4plus_anime_6B.pth` (4× upscale followed by 0.5× Lanczos downsample to `2304×3456`).
   - **Preview (`--preview`)** runs a single-stage text-only layout scout (`sdxl_lightning_8step_unet.safetensors`, `832×1216`, 8 steps, Euler/`sgm_uniform`, CFG 1.0) using `EmiliaCheckpointConditioning` (CLIP/VAE only, without instantiating the checkpoint UNet, LoRA, IPAdapter, or Stage 2).
   - Never keep Stage 1 and Stage 2 models co-resident in the same process under the 8 GiB VRAM / Windows commit budget. Stage 1 child process must exit cleanly before Stage 2 starts.
3. **Zero-VRAM Pre-Flight & Inspection**:
   - **Ollama**: Must always be invoked with `"options": {"num_gpu": 0, ...}` and `"keep_alive": 0` so LLM weights never touch GPU VRAM.
   - **QualityInspector**: All YOLO inference (`face_yolov8n.pt`, `hand_yolov8n.pt`, `yolov8n-pose.pt`) in `inspection_worker.py` / `inspector.py` MUST explicitly run on `device="cpu"`.
4. **No Unsupported Kernel or CLIP-Skip Experiments**:
   - Do not install xFormers, Triton, SageAttention, or FP8/INT8 quantization kernels for this Turing GPU. PyTorch native memory-efficient SDPA is verified pixel-identical and optimal on CC 7.5.
   - Do not insert `CLIPSetLastLayer`: ComfyUI's `SDXLClipModel` (`app/comfy/sdxl_clip.py`) natively initializes both `clip_l` and `clip_g` at penultimate `layer_idx = -2` (WebUI CLIP Skip 2), matching Illustrious-XL v0.1's exact training setup (`CLIPSetLastLayer(-1)` shifts `clip_l` to final layer norm and degrades conditioning).
   - Do not re-enable `UltimateSDUpscale` or `TiledDiffusion` in `app/custom_nodes/`; their audited disabled copies reside in `temp/community_quality/*-tested-disabled/` so global sampler hooks stay clean.

---

## 3. Workflow & Quality Invariants

1. **Single Source of Truth for Workflows**:
   - Do not proliferate ad-hoc static JSON copies for minor seed, weight, or resolution tweaks. Use `EmiliaWorkflowBuilder` in `emilia_studio/workflows.py`.
   - Default calls to `build_highres_native(seed=20261027, pag=False)`, `build_highres_finish(image_path=..., seed=20261027, pag=False)`, `build_eye_repair(...)`, `build_ultra_native(seed=20261027)`, and `build_ultra_finish(...)` **must remain 100% key-and-value identical** to the canonical `.api.json` definitions in `workflows/api/` (synced to `app/emilia/` by `setup --sync`). An intentional default change regenerates the canonical file from the builder in the same commit and is recorded in `PROGRESS.md`; `emilia workflows` exits `2` on any other drift.
   - Every detailer sets `noise_mask_feather` explicitly and keeps paste-back `feather: 12`. `quality` API and GUI graphs use `0` on `FaceDetailer` (node `21`) and `DetailerForEach` (node `28`); `ultra` finishing uses `0` on node `21` and `8` on the hand `DetailerForEach` (DifferentialDiffusion soft mask, adopted in Phase 19 from a same-native A/B). Use `builder.sync_visual_workflow()` when updating GUI JSONs in `app/user/default/workflows/`.
2. **Bit-Exact Pixel Reproducibility**:
   - For fixed seed `20261027` (`pag=False`, `strict_seed_lock=True`), any performance or architectural change must preserve exact decoded PNG SHA-256 hashes (`hashlib.sha256(Image.open(p).tobytes()).hexdigest()`):
     - Native (`1152×1728`): `3b58d9daf2770d0ad7e87b971912b854e1e7495f9e99a09f348af9bf258c9425`
     - Final (`2304×3456`): `d7fe10c200535d8656ab633248ca7cc651ee6d2e60bd309c23c43fa1ce5af63e`
   - If a hash differs, first rule out the hardware: `python gpu_profile.py status` shows whether the clock offsets are active, and `gpu_profile.py reset` (elevated) removes them for a control run.
3. **Quality Gate Operating Rule**:
   - Never upscale malformed anatomy or unaccepted eye repairs in unattended creative mode (`--auto-gate`).
   - `QualityInspector` flags `multiple_subjects` (`face_count > 1` on solo prompts, e.g., seed `20261030`), missing faces, and (on `ultra` `full_body` renders under `--auto-gate` or `inspect --require-full-body-pose`) seated, kneeling or cropped-foot poses (`yolov8n-pose.pt`; face-height body proportions on the best-supported person box) as `reject_seed`. Lying flat seen from above is not detectable from 2D keypoints.
   - It flags heterochromia (`heterochromia_score > 32.0°`) or non-violet irises (`hue` outside `[250°, 302°]`) as `repair_eyes`, routing through `build_eye_repair(..., upscale=False)` (`quality`) or `build_ultra_eye_repair(...)` (`ultra`, `lora_strength >= 0.25`) at `denoise=0.42` inside Stage 2 and re-inspecting before finishing. An eye with fewer than 8 iris pixels is unmeasured (`eye_color_verified=false`), never reported as violet and never repaired.
   - In benchmark / default fixed-seed mode (`strict_seed_lock=True`), `QualityInspector` records full diagnostics into the `.run.json` receipt and SQLite ledger without altering execution flow or pixel hashes.
---

## 4. Verification & Testing Requirements

Before completing any code or workflow change, run the unit test suite using the project Conda environment:

```powershell
# 1. Run the top-level Emilia Studio unit suite
D:\conda-data\envs\comfyui\python.exe -m pytest tests/

# 2. Run the full ComfyUI runtime unit suite in app/tests-unit/comfy_test/ (after `setup --sync`)
D:\conda-data\envs\comfyui\python.exe -m pytest app/tests-unit/comfy_test/

# 3. Verify CLI health (doctor, workflows, and SQLite ledger)
D:\conda-data\envs\comfyui\python.exe -m emilia_studio.cli doctor
D:\conda-data\envs\comfyui\python.exe -m emilia_studio.cli workflows
D:\conda-data\envs\comfyui\python.exe -m emilia_studio.cli ledger stats
```

## 5. Git & Repository Hygiene

- Never commit model weights (`*.safetensors`, `*.pth`, `*.pt`, `*.task`), local Conda environments (`env/`), runtime caches (`cache/`, `temp/`), SQLite databases (`*.sqlite3`, `*.db`), or generated image outputs (`app/output/`).
- Keep `config.toml` local-only (ignored by Git); commit schema changes to `config.example.toml`.
- Document every workflow, quality, or performance finding in `PROGRESS.md` and `workflows/WORKFLOW_DECISIONS.txt` with exact seed numbers, wall times, and decoded pixel SHA-256 hashes.
