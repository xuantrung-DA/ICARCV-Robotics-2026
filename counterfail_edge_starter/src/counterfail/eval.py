import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairInstructionDataset, collate_fn, load_vocab
from .metrics import binary_metrics, expected_calibration_error, find_best_threshold, risk_coverage
from .model import CounterFailNet
from .paths import resolve_input_path, resolve_output_path


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    y_true, y_prob, failure_types, sources = [], [], [], []
    for batch in loader:
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_lens = batch["text_lens"].to(device)
        logits, _, _ = model(before, after, text_ids, text_lens)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        y_prob.extend(probs)
        y_true.extend(batch["label"].cpu().numpy().tolist())
        failure_types.extend(batch["failure_type"])
        sources.extend(batch["source"])
    return np.asarray(y_true), np.asarray(y_prob), failure_types, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_ckpt_threshold", action="store_true", help="Use validation-selected threshold stored in the checkpoint.")
    parser.add_argument("--tune_threshold", action="store_true", help="Report the best threshold on this eval split. Use only for analysis, not final test claims.")
    parser.add_argument("--threshold_metric", choices=["f1", "balanced_acc", "auroc", "auprc"], default="f1")
    parser.add_argument("--save_preds", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_input_path(args.ckpt)
    jsonl_path = resolve_input_path(args.jsonl)
    vocab_path = resolve_input_path(args.vocab)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab not found: {vocab_path}")

    vocab = load_vocab(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt.get("args", {})

    model = CounterFailNet(
        vocab_size=len(vocab),
        encoder=train_args.get("encoder", "mobilenet_v3_small"),
        pretrained=False,
        freeze_image=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ds = PairInstructionDataset(jsonl_path, vocab, img_size=args.img_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    y_true, y_prob, failure_types, sources = run_eval(model, loader, device)

    threshold = float(ckpt.get("best_threshold", args.threshold)) if args.use_ckpt_threshold else args.threshold
    if args.tune_threshold:
        metrics = find_best_threshold(y_true, y_prob, metric=args.threshold_metric)
        threshold = metrics["threshold"]
    else:
        metrics = binary_metrics(y_true, y_prob, threshold=threshold)
        metrics["threshold"] = threshold
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    metrics.update(risk_coverage(y_true, y_prob, threshold=threshold))
    metrics["n"] = int(len(y_true))
    metrics["jsonl"] = str(jsonl_path)
    print(json.dumps(metrics, indent=2))

    # Per-failure-type recall on failure samples. Useful for paper table.
    pred = (y_prob >= threshold).astype(int)
    per_type = {}
    for ft in sorted(set(failure_types)):
        idx = np.array([x == ft for x in failure_types])
        if idx.sum() == 0:
            continue
        # For success type: success recall. For failure types: failure recall = predicted failure.
        if ft == "success":
            val = (pred[idx] == 1).mean()
        else:
            val = (pred[idx] == 0).mean()
        per_type[ft] = {"n": int(idx.sum()), "recall": float(val)}
    print("\nPer-type recall:")
    print(json.dumps(per_type, indent=2))

    if args.save_preds:
        rows = []
        for r, yt, yp, ft, src in zip(ds.rows, y_true.tolist(), y_prob.tolist(), failure_types, sources):
            rows.append({**r, "y_true": int(yt), "p_success": float(yp), "pred": int(yp >= threshold), "threshold": float(threshold), "failure_type": ft, "source": src})
        out = resolve_output_path(args.save_preds)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {out}")


if __name__ == "__main__":
    main()
