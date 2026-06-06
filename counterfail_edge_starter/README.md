# CounterFail-Edge starter

Starter code for a lightweight before-after-instruction robotic failure detector.

## 0. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Hugging Face private/gated datasets if needed:

```bash
huggingface-cli login
```

## 1. Download datasets

Fast one-week preset: BridgeDataV2-Fail train/val/test + RLBench-Fail test + UR5-Fail test.

```bash
python -m src.counterfail.download --preset guardian_light --out_dir data/raw
```

Full preset, including train/val/test for all Guardian datasets:

```bash
python -m src.counterfail.download --preset guardian_full --out_dir data/raw
```

Optional huge original BridgeData2 LeRobot v3 package (~90GB):

```bash
python -m src.counterfail.download --preset bridge_original --out_dir data/raw
```

## 2. Prepare manifests

Train from successful BridgeDataV2-Fail execution samples only, then generate synthetic counterfactual failures.

```bash
python -m src.counterfail.prepare_guardian \
  --raw_root data/raw \
  --out_root data/processed \
  --train_source bdv2fail \
  --success_only_counterfactual \
  --neg_per_pos 4 \
  --seed 42
```

Generated files:

```text
data/processed/train.jsonl
data/processed/val_bdv2fail.jsonl
data/processed/test_bdv2fail.jsonl
data/processed/test_rlbenchfail.jsonl
data/processed/test_ur5fail.jsonl
data/processed/vocab.json
```

## 3. Train

```bash
python -m src.counterfail.train \
  --train_jsonl data/processed/train.jsonl \
  --val_jsonl data/processed/val_bdv2fail.jsonl \
  --vocab data/processed/vocab.json \
  --out_dir runs/counterfail_mbv3 \
  --epochs 12 \
  --batch_size 32 \
  --img_size 224 \
  --encoder mobilenet_v3_small \
  --pretrained \
  --contrastive_weight 0.05
```

If GPU memory is tight, set `--batch_size 16`. If pretrained ImageNet weights cannot download, omit `--pretrained`.

## 4. Evaluate cross-dataset

```bash
python -m src.counterfail.eval \
  --ckpt runs/counterfail_mbv3/best.pt \
  --jsonl data/processed/test_bdv2fail.jsonl \
  --vocab data/processed/vocab.json

python -m src.counterfail.eval \
  --ckpt runs/counterfail_mbv3/best.pt \
  --jsonl data/processed/test_rlbenchfail.jsonl \
  --vocab data/processed/vocab.json

python -m src.counterfail.eval \
  --ckpt runs/counterfail_mbv3/best.pt \
  --jsonl data/processed/test_ur5fail.jsonl \
  --vocab data/processed/vocab.json
```

## 5. Export ONNX for edge/latency table

```bash
python -m src.counterfail.export_onnx \
  --ckpt runs/counterfail_mbv3/best.pt \
  --vocab data/processed/vocab.json \
  --out runs/counterfail_mbv3/counterfail_edge.onnx
```
