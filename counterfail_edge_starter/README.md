# CounterFail-Edge: Lightweight Before–After–Instruction Robotic Failure Detector

> **CounterFail-Edge** trains a compact neural network that takes a *before* image, an *after* image, and a natural-language instruction, then predicts whether the robotic execution **succeeded** or **failed**. It uses counterfactual data augmentation for robust cross-domain generalization and is designed for real-time edge deployment.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
   - [Step 0 – Environment Setup](#step-0--environment-setup)
   - [Step 1 – Download Datasets](#step-1--download-datasets)
   - [Step 2 – Prepare Manifests](#step-2--prepare-manifests)
   - [Step 3 – Train](#step-3--train)
   - [Step 4 – Evaluate](#step-4--evaluate-cross-dataset)
   - [Step 5 – Export ONNX](#step-5--export-onnx-for-edge-deployment)
4. [Project Structure](#project-structure)
5. [Dataset Details](#dataset-details)
6. [Counterfactual Augmentation](#counterfactual-augmentation)
7. [Troubleshooting](#troubleshooting)
8. [Citation](#citation)

---

## Architecture Overview

```
┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
│ Before Image│   │ After Image │   │  Instruction     │
└──────┬──────┘   └──────┬──────┘   │  (natural lang.) │
       │                 │          └────────┬─────────┘
       ▼                 ▼                   ▼
  ┌────────────────────────┐        ┌───────────────┐
  │ Shared Image Encoder   │        │ Text Encoder  │
  │ (MobileNetV3-Large/    │        │ (BiGRU /      │
  │  Small / EfficientNet) │        │  MeanPool)    │
  └───────────┬────────────┘        └───────┬───────┘
              │                             │
              ▼                             │
  ┌───────────────────────┐                 │
  │ Visual Difference     │                 │
  │ [fb, fa, fa-fb, |Δ|]  │                 │
  └───────────┬───────────┘                 │
              │                             │
              ▼                             ▼
        ┌────────────────────────────────────────┐
        │   Multimodal Fusion + Classifier       │
        │   [v, t, v⊙t, |v-t|] → MLP → logit   │
        └────────────────────┬───────────────────┘
                             │
                      ┌──────▼──────┐
                      │ Success / ✗ │
                      │ Failure     │
                      └─────────────┘
```

The model also produces normalized pair and text embeddings for an auxiliary **symmetric InfoNCE contrastive loss**, ensuring that visual outcomes and instructions are aligned in embedding space.

---

## Requirements

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.2
- **GPU** recommended (CUDA) but CPU training works for small experiments
- **Disk space**: ~2 GB for `guardian_light` preset, ~90 GB for `bridge_original`

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `pillow`, `numpy`, `scikit-learn`, `tqdm`, `huggingface_hub`, `pandas`, `matplotlib`

---

## Quick Start

### Step 0 – Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

For Hugging Face private/gated datasets (if needed):

```bash
huggingface-cli login
```

### Step 1 – Download Datasets

**Fast preset** (recommended, ~2 GB): BridgeDataV2-Fail train/val/test + RLBench-Fail test + UR5-Fail test.

```bash
python -m src.counterfail.download --preset guardian_light --out_dir data/raw
```

**Full preset** (all Guardian splits for all datasets):

```bash
python -m src.counterfail.download --preset guardian_full --out_dir data/raw
```

**Optional**: Huge original BridgeData2 LeRobot v3 package (~90 GB):

```bash
python -m src.counterfail.download --preset bridge_original --out_dir data/raw
```

> **Note**: The download script automatically extracts `records.tar.gz` archives in each dataset folder. If you want to skip extraction (e.g., already extracted), add `--no_extract`.

### Step 2 – Prepare Manifests

Train from successful BridgeDataV2-Fail execution samples, then generate synthetic counterfactual negatives:

```bash
python -m src.counterfail.prepare_guardian \
  --raw_root data/raw \
  --out_root data/processed \
  --train_source bdv2fail \
  --success_only_counterfactual \
  --neg_per_pos 5 \
  --seed 42
```

**Windows PowerShell** – use backtick `` ` `` instead of backslash `\` for line continuation:

```powershell
python -m src.counterfail.prepare_guardian `
  --raw_root data/raw `
  --out_root data/processed `
  --train_source bdv2fail `
  --success_only_counterfactual `
  --neg_per_pos 5 `
  --seed 42
```

**Generated files:**

```
data/processed/
├── train.jsonl              # Training set (success + counterfactual failures)
├── val_bdv2fail.jsonl       # Validation set (BridgeDataV2-Fail)
├── test_bdv2fail.jsonl      # Test set (BridgeDataV2-Fail)
├── test_rlbenchfail.jsonl   # Test set (RLBench-Fail, cross-domain)
├── test_ur5fail.jsonl       # Test set (UR5-Fail, cross-domain)
└── vocab.json               # Token vocabulary for text encoder
```

### Step 3 – Train

```bash
python -m src.counterfail.train \
  --train_jsonl data/processed/train.jsonl \
  --val_jsonl data/processed/val_bdv2fail.jsonl \
  --vocab data/processed/vocab.json \
  --out_dir runs/counterfail_mbv3large \
  --epochs 12 \
  --batch_size 24 \
  --img_size 224 \
  --encoder mobilenet_v3_large \
  --text_encoder bigru \
  --pretrained \
  --lr 3e-4 \
  --backbone_lr_mult 0.25 \
  --contrastive_weight 0.03 \
  --balanced_sampler \
  --pos_weight none \
  --select_metric group_hmean_recall \
  --seed 42 \
  --amp
```

**Windows PowerShell:**

```powershell
python -m src.counterfail.train `
  --train_jsonl data/processed/train.jsonl `
  --val_jsonl data/processed/val_bdv2fail.jsonl `
  --vocab data/processed/vocab.json `
  --out_dir runs/counterfail_mbv3large `
  --epochs 12 `
  --batch_size 24 `
  --img_size 224 `
  --encoder mobilenet_v3_large `
  --text_encoder bigru `
  --pretrained `
  --contrastive_weight 0.03 `
  --balanced_sampler `
  --pos_weight none `
  --select_metric group_hmean_recall `
  --amp
```

**Tips:**
- If GPU memory is tight, set `--batch_size 16`
- If pretrained ImageNet weights cannot download, omit `--pretrained`
- Add `--amp` for mixed-precision training (faster on NVIDIA GPUs with Tensor Cores)
- Outputs: `runs/counterfail_mbv3/best.pt` (best checkpoint) + `history.json` (training curves)

### Step 4 – Evaluate Cross-Dataset

```bash
# Same-domain evaluation
python -m src.counterfail.eval \
  --ckpt runs/counterfail_mbv3_v2/best.pt \
  --jsonl data/processed/test_bdv2fail.jsonl \
  --vocab data/processed/vocab.json \
  --use_ckpt_threshold \
  --save_preds runs/counterfail_mbv3_v2/preds_bdv2.jsonl

# Cross-domain: RLBench-Fail
python -m src.counterfail.eval \
  --ckpt runs/counterfail_mbv3/best.pt \
  --jsonl data/processed/test_rlbenchfail.jsonl \
  --vocab data/processed/vocab.json

# Cross-domain: UR5-Fail
python -m src.counterfail.eval \
  --ckpt runs/counterfail_mbv3/best.pt \
  --jsonl data/processed/test_ur5fail.jsonl \
  --vocab data/processed/vocab.json
```

**Evaluation metrics**: success_f1, failure_f1, macro_f1, success_recall, failure_recall, balanced_acc, group_hmean_recall, AUROC, AUPRC, ECE (Expected Calibration Error), Risk@Coverage, per-type recall.

### Step 5 – Export ONNX for Edge Deployment

```bash
python -m src.counterfail.export_onnx \
  --ckpt runs/counterfail_mbv3/best.pt \
  --vocab data/processed/vocab.json \
  --out runs/counterfail_mbv3/counterfail_edge.onnx
```

---

## Project Structure

```
counterfail_edge_starter/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                           # Downloaded HuggingFace datasets
│   │   ├── paulpacaud__bdv2fail_train_dataset/
│   │   │   ├── metadata_execution.jsonl
│   │   │   ├── records.tar.gz         # Images (auto-extracted on download)
│   │   │   └── data/failure_forge/... # Extracted image directories
│   │   ├── paulpacaud__bdv2fail_val_dataset/
│   │   └── ...
│   └── processed/                     # Generated manifests (after prepare step)
│       ├── train.jsonl
│       ├── val_bdv2fail.jsonl
│       ├── test_*.jsonl
│       └── vocab.json
├── runs/                              # Training outputs
│   └── counterfail_mbv3/
│       ├── best.pt                    # Best model checkpoint
│       └── history.json               # Training history
└── src/counterfail/
    ├── __init__.py
    ├── data.py                        # Dataset, transforms, data augmentation
    ├── download.py                    # HuggingFace dataset downloader + auto-extract
    ├── eval.py                        # Evaluation script with per-type recall
    ├── export_onnx.py                 # ONNX export for edge inference
    ├── latency.py                     # Latency benchmarking
    ├── metrics.py                     # Binary metrics, ECE, risk-coverage
    ├── model.py                       # CounterFailNet architecture
    ├── paths.py                       # Path resolution utilities
    ├── prepare_guardian.py            # Manifest generation + counterfactual augmentation
    └── train.py                       # Training loop with contrastive loss
```

---

## Dataset Details

| Dataset | Source | Train | Val | Test | Domain |
|---------|--------|-------|-----|------|--------|
| BridgeDataV2-Fail | `paulpacaud/bdv2fail_*` | ✅ | ✅ | ✅ | Tabletop manipulation |
| RLBench-Fail | `paulpacaud/rlbenchfail_*` | ✅ | ✅ | ✅ | Simulated manipulation |
| UR5-Fail | `paulpacaud/ur5fail_*` | ✅ | ✅ | ✅ | UR5 robot tasks |

Each sample in `metadata_execution.jsonl` contains:
- **images**: Paths to before/after execution images
- **task_instruction**: Natural language task description
- **execution_reward**: Binary label (1 = success, 0 = failure)
- **failure_mode**: Type of failure (e.g., `ground_truth`, `wrong_object`, etc.)
- **taskvar**: Task variant identifier

---

## Counterfactual Augmentation

When using `--success_only_counterfactual`, the pipeline generates 6 types of semantic-hard synthetic negatives from each success sample:

| Type | Strategy | Rationale |
|------|----------|-----------|
| `no_progress` | Replace *after* with *before* image | Robot did nothing |
| `temporal_reverse` | Swap *before* ↔ *after* | Undo instead of do |
| `instruction_mismatch_hard` | Replace instruction with a hard-negative task's | Wrong task (high token overlap) |
| `endpoint_mismatch_hard` | Replace *after* with a hard-negative task's *after* | Wrong outcome (semantically similar) |
| `wrong_object_like` | Replace instruction with same-action/different-object task | Simulates wrong object manipulation |
| `wrong_state_or_placement_like` | Replace *after* with same-object/different-location task's *after* | Simulates wrong placement/state |

This creates a balanced training set where the model must verify **all three modalities** (before, after, instruction) agree — not just check individual signals.

---

## Troubleshooting

### `train.jsonl` is empty (0 samples)

**Cause**: The `records.tar.gz` archives were not extracted, so image files don't exist and all samples are filtered out.

**Fix**: Re-run the download with the latest `download.py` (auto-extracts), or manually extract:

```bash
# For each dataset directory
tar -xzf data/raw/<dataset>/records.tar.gz -C data/raw/<dataset>/
```

Then re-run `prepare_guardian`.

### `ModuleNotFoundError: No module named 'src'`

**Cause**: Running from the wrong working directory.

**Fix**: Always `cd` into the `counterfail_edge_starter/` directory before running commands:

```bash
cd counterfail_edge_starter
python -m src.counterfail.train ...
```

### `ImportError: attempted relative import with no known parent package`

**Cause**: Running a module file directly with `python file.py` instead of `python -m`.

**Fix**: Always use `python -m src.counterfail.<module>` syntax, never run `.py` files directly.

### GPU out of memory

**Fix**: Reduce batch size: `--batch_size 8` or `--batch_size 4`.

### Cannot download pretrained weights

**Fix**: Omit `--pretrained` to train from scratch, or set `HTTP_PROXY`/`HTTPS_PROXY` environment variables.

---

## Citation

If you use this codebase in your research, please cite the relevant Guardian / Failure Forge papers and datasets.
