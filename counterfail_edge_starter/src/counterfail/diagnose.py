"""CounterFail-Edge diagnostics tool.

Usage — data diagnostics:
  python -m src.counterfail.diagnose --jsonl data/processed_semhard/train.jsonl --vocab data/processed_semhard/vocab.json

Usage — predictions analysis:
  python -m src.counterfail.diagnose --preds runs/.../preds_bdv2.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .model import CODE_VERSION


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str):
    import re
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def diagnose_data(jsonl_path: str, vocab_path: str = None):
    """Print label/source/type counts and OOV rate for a data JSONL."""
    rows = read_jsonl(jsonl_path)
    print(f"\n=== Data Diagnostics: {jsonl_path} ===")
    print(f"Total rows: {len(rows)}")

    labels = Counter(r.get("label", -1) for r in rows)
    print(f"Label counts: {dict(labels)}")

    sources = Counter(r.get("source", "") for r in rows)
    print(f"Source counts: {dict(sources)}")

    ft = Counter(r.get("failure_type", "") for r in rows)
    print(f"Failure type counts:")
    for k, v in ft.most_common():
        print(f"  {k:40s} {v}")

    ct = Counter(r.get("counterfactual_type", "") for r in rows)
    print(f"Counterfactual type counts:")
    for k, v in ct.most_common():
        print(f"  {k:40s} {v}")

    syn = Counter(r.get("synthetic", "unknown") for r in rows)
    print(f"Synthetic counts: {dict(syn)}")

    if vocab_path:
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        total, unk = 0, 0
        unk_tokens: Counter = Counter()
        unk_id = vocab.get("<unk>", 1)
        for r in rows:
            for tok in tokenize(r.get("instruction", "")):
                total += 1
                if vocab.get(tok, unk_id) == unk_id:
                    unk += 1
                    unk_tokens[tok] += 1
        print(f"\nOOV rate: {unk}/{total} = {unk/max(total,1):.4f}")
        if unk_tokens:
            print(f"Top UNK tokens: {unk_tokens.most_common(15)}")

    # Show examples per counterfactual_type
    by_ct = defaultdict(list)
    for r in rows:
        by_ct[r.get("counterfactual_type", "unknown")].append(r)
    print("\n--- 5 examples per counterfactual_type ---")
    for ct_name, ct_rows in sorted(by_ct.items()):
        print(f"\n[{ct_name}] ({len(ct_rows)} total)")
        for ex in ct_rows[:5]:
            instr = ex.get("instruction", "")[:80]
            print(f"  label={ex.get('label')} ft={ex.get('failure_type','')} instr={instr}")


def diagnose_preds(preds_path: str):
    """Analyze prediction JSONL: per-type recall, false positives/negatives."""
    rows = read_jsonl(preds_path)
    print(f"\n=== Predictions Diagnostics: {preds_path} ===")
    print(f"Total predictions: {len(rows)}")

    by_ft = defaultdict(list)
    for r in rows:
        by_ft[r.get("failure_type", "unknown")].append(r)

    print(f"\n{'failure_type':45s} {'n':>5s} {'recall':>7s} {'mean_p':>7s} {'med_p':>7s}")
    print("-" * 80)
    for ft_name in sorted(by_ft.keys()):
        ft_rows = by_ft[ft_name]
        n = len(ft_rows)
        probs = [r.get("p_success", 0.5) for r in ft_rows]
        preds = [r.get("pred", 1) for r in ft_rows]
        y_true = [r.get("y_true", 0) for r in ft_rows]

        if ft_name == "success":
            recall = sum(1 for p in preds if p == 1) / max(n, 1)
        else:
            recall = sum(1 for p in preds if p == 0) / max(n, 1)

        mean_p = np.mean(probs)
        med_p = np.median(probs)
        print(f"  {ft_name:43s} {n:5d} {recall:7.3f} {mean_p:7.3f} {med_p:7.3f}")

    # False positives for failure types (true failure predicted success)
    print("\n--- False positives (true failure predicted success) ---")
    fp_count = 0
    for ft_name, ft_rows in sorted(by_ft.items()):
        if ft_name == "success":
            continue
        fps = [r for r in ft_rows if r.get("pred", 1) == 1]
        if fps:
            fp_count += len(fps)
            print(f"  {ft_name}: {len(fps)}/{len(ft_rows)} false positives")
            for ex in fps[:3]:
                print(f"    p_success={ex.get('p_success',0):.3f} instr={ex.get('instruction','')[:60]}")

    # False negatives for success (true success predicted failure)
    print("\n--- False negatives (true success predicted failure) ---")
    if "success" in by_ft:
        fns = [r for r in by_ft["success"] if r.get("pred", 1) == 0]
        print(f"  success: {len(fns)}/{len(by_ft['success'])} false negatives")
        for ex in fns[:5]:
            print(f"    p_success={ex.get('p_success',0):.3f} instr={ex.get('instruction','')[:60]}")


def main():
    parser = argparse.ArgumentParser(description="CounterFail-Edge diagnostics")
    parser.add_argument("--jsonl", type=str, default=None, help="Data JSONL for data diagnostics")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab JSON for OOV analysis")
    parser.add_argument("--preds", type=str, default=None, help="Predictions JSONL for pred analysis")
    args = parser.parse_args()

    print(f"[diagnose.py] CODE_VERSION={CODE_VERSION}")

    if args.jsonl:
        diagnose_data(args.jsonl, args.vocab)
    if args.preds:
        diagnose_preds(args.preds)
    if not args.jsonl and not args.preds:
        parser.print_help()


if __name__ == "__main__":
    main()
