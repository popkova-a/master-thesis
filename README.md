# What Can Large Language Models Offer to Event-Based Vision?

Master's thesis at the University of Bern, conducted in collaboration with the **Robotics and Perception Group** at the University of Zurich, supervised by **Prof. Dr. Davide Scaramuzza** and **Prof. Dr. Andrea Agazzi**. The project investigates how large vision-language models (VLMs) can be leveraged to understand event-based camera data (a fundamentally asynchronous, sparse, and high-dynamic-range sensing modality) by training a three-stage pipeline: self-supervised contrastive pretraining of an event encoder, CLIP-style event-text alignment, and event-conditioned language generation.

---

## Repository Structure

```
.
├── construct_dataset/
│   ├── scrape_video.py     # YouTube video scraper (InternVid-10M dataset)
│   ├── v2e_wrapper.py      # Video-to-events simulator wrapper (v2e)
│   ├── vid2e_wrapper.py    # Video-to-events simulator wrapper (vid2e / rpg_vid2e)
│   └── __init__.py
│
└── vlm/
    ├── configs/
    │   ├── pretrain_config.yaml        # SimCLR pretraining on multiple event datasets
    │   ├── pretrain_CIFAR_config.yaml  # SimCLR pretraining on CIFAR10-DVS only
    │   ├── clip_config.yaml            # CLIP-style event-text alignment
    │   ├── train_config.yaml           # VLM finetuning on N-InternVid
    │   └── train_CIFAR_config.yaml     # VLM finetuning on CIFAR10-DVS
    │
    ├── models/
    │   ├── event_clr.py                # EventCLR — SimCLR-style contrastive model
    │   ├── event_clip.py               # EventCLIP — event-text alignment model
    │   ├── event_vlm.py                # EventVLM — event-conditioned LLM (DeepSeek-VL)
    │   ├── event_blip.py               # EventVLM — BLIP-based captioning model
    │   └── __init__.py
    │
    ├── data/
    │   ├── dataset.py                  # Dataset classes for all event datasets
    │   ├── augmenter.py                # Spatiotemporal event augmentation pipeline
    │   ├── event_tokenizer.py          # GET-style event-to-token converter
    │   ├── processor.py                # Event-language processor for VLM inputs
    │   ├── pretrain_dataloader.py      # Dataloader for contrastive pretraining
    │   ├── clip_dataloader.py          # Dataloader for CLIP-style training
    │   ├── train_dataloader.py         # Dataloader for VLM finetuning
    │   └── __init__.py
    │
    ├── train/
    │   ├── trainer.py                  # Trainer for VLM finetuning
    │   ├── clip_trainer.py             # Trainer for CLIP-style alignment
    │   ├── objectives.py               # NT-Xent and classification loss functions
    │   ├── scheduler.py                # Cosine annealing with warmup and restarts
    │   ├── amp_scaler.py               # Mixed-precision gradient scaler with norm tracking
    │   ├── checkpointer.py             # Checkpoint save/load manager
    │   ├── logger.py                   # WandB logger with video visualization
    │   └── __init__.py
    │
    ├── pretrain/
    │   ├── pretrainer.py               # Pretrainer for SimCLR-style contrastive learning
    │   ├── evaluator.py                # Linear probe evaluator for frozen representations
    │   └── __init__.py
    │
    ├── utils/
    │   ├── utils.py                    # Distributed training setup, seeding, parameter groups
    │   ├── config_parser.py            # YAML config parser
    │   └── __init__.py
    │
    ├── pretrain_main.py                # Entry point — SimCLR pretraining
    ├── clip_main.py                    # Entry point — CLIP-style event-text alignment
    ├── train_main.py                   # Entry point — VLM finetuning
    ├── pretrain_vis.py                 # t-SNE visualization of frozen encoder features
    └── __init__.py                     # Experiment notes (experiments 5–8)
```

---

## Training Pipeline

The project follows a three-stage training pipeline, each with its own entry point and config file.

### Stage 1 — Self-supervised Pretraining (`pretrain_main.py`)

The GET (Group Event Transformer) event encoder is pretrained with a **SimCLR-style contrastive objective** (NT-Xent loss) on event streams from multiple datasets (N-Caltech101, N-Cars, N-ImageNet, CIFAR10-DVS, DailyDVS-200). Each sample produces an anchor and a positive view via spatiotemporal augmentation. The encoder learns to map augmented views of the same scene to nearby embeddings in a shared projection space.

Configs: `pretrain_config.yaml`, `pretrain_CIFAR_config.yaml`

### Stage 2 — Event-Text Alignment (`clip_main.py`)

The pretrained event encoder is combined with CLIP's frozen text transformer to perform **event-text contrastive alignment**. Event embeddings and class-label text embeddings are projected into a shared space and aligned via a classification loss with a learnable temperature. The first three stages of the encoder backbone are frozen during this phase. Learnable soft prompts with an optional gating mechanism are introduced into the text branch.

Config: `clip_config.yaml`

### Stage 3 — VLM Finetuning (`train_main.py`)
The pretrained GET encoder is connected to BLIP's frozen text encoder and a LoRA-finetuned text decoder. A custom MLP aligner projects event embeddings into BLIP's 768-dimensional space. Training jointly optimizes a contrastive image-text loss (between pooled visual and CLS text embeddings) and a cross-entropy captioning loss (teacher-forced decoding conditioned on the visual context). Only the aligner and LoRA adapter weights are updated; the event encoder and text encoder remain frozen.

Configs: `train_config.yaml`, `train_CIFAR_config.yaml`

---

## Modules

### `models/`

#### `EventCLR` (`event_clr.py`)
SimCLR-style contrastive model wrapping a GET backbone with an adaptive average pooling layer and a linear projection head. Accepts anchor-positive pairs and returns their projected embeddings for NT-Xent loss computation. Exposes `extract_features()` for linear probe evaluation.

#### `EventCLIP` (`event_clip.py`)
Event-text alignment model combining a GET event encoder with CLIP's frozen text transformer. Introduces learnable soft prompts and an optional sigmoid gate into the text branch. Encodes event streams and text descriptions into a shared normalized embedding space and returns a similarity matrix scaled by a learnable temperature. Exposes `extract_features()` for downstream evaluation.

#### `EventVLM` (`event_vlm.py`)
Event-conditioned language model combining a GET event encoder, DeepSeek-VL's aligner, and DeepSeek-VL's frozen language model. Event tokens are encoded, projected through the aligner, and inserted into the LLM's input embedding sequence at placeholder positions defined by `event_seq_mask` and `event_emb_mask`. Supports both training (forward with labels) and inference (`generate()`).

#### `EventVLM` — BLIP variant (`event_blip.py`)
BLIP-based captioning model. Uses a GET encoder with a custom MLP aligner projecting into BLIP's 768-dimensional space. Combines a contrastive image-text loss with a LoRA-finetuned BLIP text decoder captioning loss. The encoder and text encoder are frozen; only the aligner and LoRA adapter weights are trained.

---

### `data/`

#### `dataset.py`
Dataset classes for all supported event camera datasets: N-Caltech101, N-Cars, N-ImageNet, CIFAR10-DVS, DailyDVS-200, and N-InternVid (simulated events). Each class loads event files (`.npy`, `.npz`, `.h5`, `.aedat4`) and returns `(events, label)` pairs in `(t, x, y, p)` format. Factory functions `build_dataset`, `build_contrastive_dataset`, and `build_clip_dataset` select the appropriate dataset class from the config data path.

#### `augmenter.py` — `EventAugmenter`
Configurable spatiotemporal augmentation pipeline for event streams. Supported transforms: spatial shift, horizontal flip, temporal flip (with polarity inversion), spatial centering, temporal crop, temporal drop, area drop, point drop, and Gaussian coordinate noise. Augmentations can be selectively enabled via `include_augmentations`. Ensures the output always contains at least `min_event_num` events.

#### `event_tokenizer.py` — `EventTokenizer`
Converts a raw event stream `(t, x, y, p)` into a fixed-size token representation following the GET paper. Events are dynamically rescaled to a reference resolution, spatially binned into a patch grid, temporally divided into bins, and encoded as polarity-separated histograms. Output shape: `[1, C, H, W]` where `C = 2 × embed_split × patch_size²` and `H = W = ref_resolution // patch_size`. Also provides `E2IMG` for converting events to RGB visualization frames.

#### `processor.py` — `EventVLProcessor`
Wraps DeepSeek-VL's `VLChatProcessor` to handle event-language inputs. Formats prompts using the SFT conversation template, tokenizes text with event placeholder tokens, and produces `EventVLProcessorOutput` / `BatchedEventVLProcessorOutput` objects containing `input_ids`, `event_representations`, `attention_mask`, `event_seq_mask`, and `event_emb_mask`.

#### Dataloaders
Three specialized dataloaders wrap PyTorch `DataLoader` with in-loop tokenization and device management, all supporting `DistributedSampler`:

- `pretrain_dataloader.py` — yields `TokenizedBatch` with anchor and positive tokenized pairs for SimCLR.
- `clip_dataloader.py` — yields `TokenizedBatch` with event representations and class labels for CLIP.
- `train_dataloader.py` — yields `ProcessedBatch` with tokenized event-text pairs, attention masks, and label IDs for VLM training.

---

### `train/`

#### `objectives.py`
Two loss modules: `NTXentLoss` (NT-Xent / InfoNCE with distributed tensor synchronization via `SyncFunction`) and `ClassificationLoss` (cross-entropy with label smoothing and configurable ignore index). Both expose `compute_metrics()` returning positive/negative cosine similarities, top-1/top-5 accuracy, and mean rank.

#### `scheduler.py` — `CosineAnnealingWarmupRestarts`
Custom `_LRScheduler` implementing cosine annealing with linear warmup and periodic restarts. Supports per-parameter-group learning rates (separate rates for backbone and projection head) and a gamma decay factor applied to the max learning rate at each restart.

#### `amp_scaler.py` — `GradScalerWithNormTracking`
Wraps PyTorch's `GradScaler` with gradient norm computation and clipping before the optimizer step. Returns the gradient norm for logging.

#### `checkpointer.py` — `Checkpointer`
Saves and loads `.pth.tar` checkpoints containing epoch, model state dict (excluding frozen language model and CLIP weights), optimizer state dict, and optionally scheduler state dict. Validates model config consistency on load.

#### `logger.py` — `WandBLogger`
WandB-based logger supporting scalar metrics, structured prediction tables (ground truth vs. generated captions), and event stream video visualization (renders positive/negative polarity events as red/blue frames at 10 fps).

#### `trainer.py` / `clip_trainer.py` / `pretrain/pretrainer.py`
Three trainer classes with a shared structure — `run_epoch()` for a single train/val pass, `train()` / `pretrain()` for the full loop with checkpointing, linear probe evaluation, and WandB logging. All support DistributedDataParallel (DDP) and mixed-precision training (AMP). The CLIP trainer additionally freezes the first three encoder stages and constructs zero-shot text prompts from class names.

---

### `pretrain/evaluator.py` — `LinearProbeEvaluator`
Evaluates frozen encoder representations by extracting features from training and validation sets and fitting a logistic regression classifier with cross-validated regularization strength. Supports distributed feature extraction via all-gather.

---

### `utils/`

#### `utils.py`
Distributed training utilities: `setup_distributed_environment()` (NCCL process group init, seeding), `cleanup_distributed_environment()`, `get_parameter_groups()` (splits parameters into backbone/head groups with per-component learning rates and weight decay), and `load_pretrained_checkpoint()` (loads and adapts checkpoint state dicts across model variants with backbone name remapping).

#### `config_parser.py` — `ConfigParser`
Parses YAML configuration files and returns a dictionary with `event_vision_model`, `vision_language_model`, `data`, and `train` sections.

---


### `construct_dataset/`

This folder contains the data collection and event simulation pipeline used to build the N-InternVid dataset — a large-scale paired event-text dataset created by scraping video clips from YouTube and converting them to DVS events.

#### `scrape_video.py` — `YoutubeScraper`
Downloads short video segments from YouTube based on the [InternVid-10M](https://huggingface.co/datasets/OpenGVLab/InternVid) dataset. For each entry, it checks that the clip duration does not exceed a configurable limit (default 10 s), downloads the segment using `yt-dlp` with `ffmpeg` at up to 360p resolution, and skips clips that are already present in the output folder. Downloads are parallelized across multiple processes. Successfully downloaded clips and their metadata are saved to a `dataset.csv` file.

#### `v2e_wrapper.py` — `VideoToEvents` (v2e)
Converts downloaded video files to DVS event streams using the [v2e](https://github.com/SensorsINI/v2e) simulator. Processes videos in parallel across multiple GPUs, skipping files that already have a corresponding `.h5` output. Produces `.h5` event files and optional `.avi` DVS visualizations. Exposes `simulate()` for batch conversion and `check_dataset()` to identify mismatches between input videos and output event files. Configurable DVS parameters include resolution, positive/negative thresholds, noise rates, and slow-motion settings.

#### `vid2e_wrapper.py` — `VideoToEvents` (vid2e)
Alternative event simulator using [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e), which applies frame-level super-resolution upsampling before event simulation for higher temporal fidelity. The pipeline runs in three sequential stages — resize (via `ffmpeg`), upsample (via `upsample.py`), and simulate (via `generate_events.py`) — each parallelized across GPUs. Intermediate results are stored in `resized_`, `upsampled_`, and `events_` subdirectories. Stages are automatically skipped if their output already exists.

---

## Datasets

| Dataset | Modality | Task |
|---------|----------|------|
| N-Caltech101 | DVS events | Object recognition (101 classes) |
| N-Cars | DVS events | Binary car/background classification |
| N-ImageNet | DVS events | Large-scale object recognition |
| CIFAR10-DVS | DVS events | Object recognition (10 classes) |
| DailyDVS-200 | DVS events | Daily activity recognition (200 classes) |
| N-InternVid | Simulated (v2e) | Event-text alignment |

---

## Dependencies

```
torch
torchvision
transformers
peft
deepseek_vl
clip
timm
einops
wandb
sklearn
tqdm
numpy
pandas
h5py
aedat
pyyaml
```

---

## Running

All entry points are designed for `torchrun` (multi-GPU DDP):

```bash
# Stage 1 — SimCLR pretraining
torchrun --nproc_per_node=NUM_GPUS pretrain_main.py

# Stage 2 — CLIP-style event-text alignment
torchrun --nproc_per_node=NUM_GPUS clip_main.py

# Stage 3 — VLM finetuning
torchrun --nproc_per_node=NUM_GPUS train_main.py

# t-SNE visualization of frozen encoder features
torchrun --nproc_per_node=1 pretrain_vis.py
```

Set `resume=True` in the `main()` call to resume from the latest checkpoint. Configs are loaded from `configs/` relative to the working directory. Checkpoints are saved to `checkpoint/` and logs to `logs/`.
