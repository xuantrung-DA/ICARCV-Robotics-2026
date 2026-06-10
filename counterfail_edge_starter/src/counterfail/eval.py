"""CounterFail-Edge evaluation script."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairInstructionDataset, collate_fn, load_vocab
from .metrics import (
    binary_metrics,
    expected_calibration_error,
    find_best_threshold,
    group_balanced_type_recall,
    group_hmean_recall,
    per_type_mean_recall,
    per_type_recall,
    risk_coverage,
)
from .model import CODE_VERSION, CounterFailNet
from .paths import resolve_input_path, resolve_output_path


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    failure_types, sources = [], []
    instructions, befores, afters = [], [], []
    taskvars, ctypes, synthetics = [], [], []
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
        instructions.extend(batch.get("instruction", [""] * len(probs)))
        befores.extend(batch.get("before_path", [""] * len(probs)))
        afters.extend(batch.get("after_path", [""] * len(probs)))
        taskvars.extend(batch.get("taskvar", [""] * len(probs)))
        ctypes.extend(batch.get("counterfactual_type", [""] * len(probs)))
        synthetics.extend(batch.get("synthetic", [False] * len(probs)))
    return {
        "y_true": np.asarray(y_true),
        "y_prob": np.asarray(y_prob),
        "failure_types": failure_types,
        "sources": sources,
        "instructions": instructions,
        "befores": befores,
        "afters": afters,
        "taskvars": taskvars,
        "ctypes": ctypes,
        "synthetics": synthetics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_ckpt_threshold", action="store_true",
                        help="Use validation-selected threshold stored in the checkpoint.")
    parser.add_argument("--tune_threshold", action="store_true",
                        help="Report the best threshold on this eval split. Use only for analysis, not final test claims.")
    parser.add_argument(
        "--threshold_metric",
        choices=["macro_f1", "balanced_acc", "failure_f1", "failure_recall",
                 "success_f1", "per_type_mean_recall", "group_hmean_recall",
                 "group_balanced_type_recall"],
        default="macro_f1",
    )
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--confidence_mode", choices=["margin", "prob"], default="margin")
    parser.add_argument("--save_preds", type=str, default=None)
    parser.add_argument("--save_metrics", type=str, default=None)
    args = parser.parse_args()

    # ---- Startup info ----
    print(f"[eval.py] __file__={__file__}")
    print(f"[eval.py] CODE_VERSION={CODE_VERSION}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_input_path(args.ckpt)
    jsonl_path = resolve_input_path(args.jsonl)
    vocab_path = resolve_input_path(args.vocab)
    for name, path in [("ckpt", ckpt_path), ("jsonl", jsonl_path), ("vocab", vocab_path)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    print(f"[eval.py] ckpt={ckpt_path}")
    print(f"[eval.py] jsonl={jsonl_path}")
    print(f"[eval.py] vocab={vocab_path}")

    vocab = load_vocab(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    encoder = train_args.get("encoder", "mobilenet_v3_large")
    text_encoder_type = train_args.get("text_encoder", "mean")
    text_dim = train_args.get("text_dim", 256)
    hidden_dim = train_args.get("hidden_dim", 256)
    print(f"[eval.py] encoder={encoder}  text_encoder_type={text_encoder_type}")

    model = CounterFailNet(
        vocab_size=len(vocab),
        encoder=encoder,
        pretrained=False,
        text_encoder_type=text_encoder_type,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        freeze_image=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ds = PairInstructionDataset(jsonl_path, vocab, img_size=args.img_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    results = run_eval(model, loader, device)
    y_true = results["y_true"]
    y_prob = results["y_prob"]
    failure_types = results["failure_types"]
    sources = results["sources"]

    # ---- Threshold ----
    threshold = float(ckpt.get("best_threshold", args.threshold)) if args.use_ckpt_threshold else args.threshold
    if args.tune_threshold:
        print("WARNING: threshold tuned on this eval split; use only for analysis, not final claims.",
              file=sys.stderr)
        metrics = find_best_threshold(
            y_true, y_prob,
            metric=args.threshold_metric,
            lo=args.threshold_min,
            hi=args.threshold_max,
            failure_types=failure_types,
        )
        threshold = metrics["threshold"]
    else:
        metrics = binary_metrics(y_true, y_prob, threshold=threshold)
    print(f"[eval.py] selected threshold={threshold:.4f}")

    # ---- Per-type recall ----
    pt = per_type_recall(y_true, y_prob, failure_types, threshold)
    ptmr = per_type_mean_recall(y_true, y_prob, failure_types, threshold)
    ghr = group_hmean_recall(y_true, y_prob, failure_types, threshold)
    gbtr = group_balanced_type_recall(y_true, y_prob, failure_types, threshold)
    metrics["per_type_recall"] = pt
    metrics["per_type_mean_recall"] = ptmr
    metrics["group_hmean_recall"] = ghr
    metrics["group_balanced_type_recall"] = gbtr

    # ---- Calibration & risk ----
    metrics["ece_prob"] = expected_calibration_error(y_true, y_prob, threshold=threshold)
    metrics.update(risk_coverage(y_true, y_prob, threshold=threshold,
                                 confidence_mode=args.confidence_mode))

    # ---- Metadata ----
    metrics["n"] = int(len(y_true))
    metrics["jsonl"] = str(jsonl_path)
    metrics["encoder"] = encoder
    metrics["text_encoder_type"] = text_encoder_type
    metrics["code_version"] = CODE_VERSION

    print(json.dumps(metrics, indent=2, default=str))

    print("\nPer-type recall:")
    print(json.dumps(pt, indent=2))
    print(f"\nPer-type mean recall: {ptmr:.4f}")

    # ---- Save metrics ----
    if args.save_metrics:
        out = resolve_output_path(args.save_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Saved metrics to {out}")

    # ---- Save preds ----
    if args.save_preds:
        rows = []
        for i, (yt, yp) in enumerate(zip(y_true.tolist(), y_prob.tolist())):
            r = ds.rows[i] if i < len(ds.rows) else {}
            rows.append({
                "y_true": int(yt),
                "p_success": float(yp),
                "p_failure": float(1.0 - yp),
                "pred": int(yp >= threshold),
                "threshold": float(threshold),
                "correct": int((yp >= threshold) == yt),
                "failure_type": failure_types[i] if i < len(failure_types) else "",
                "source": sources[i] if i < len(sources) else "",
                "instruction": r.get("instruction", ""),
                "before": r.get("before", ""),
                "after": r.get("after", ""),
                "taskvar": r.get("taskvar", ""),
                "episode_id": r.get("episode_id", ""),
                "counterfactual_type": r.get("counterfactual_type", ""),
                "synthetic": r.get("synthetic", False),
            })
        out = resolve_output_path(args.save_preds)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {out}")


if __name__ == "__main__":
    main()
