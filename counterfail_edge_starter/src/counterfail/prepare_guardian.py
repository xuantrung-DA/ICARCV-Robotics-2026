import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from .paths import resolve_input_path, resolve_output_path


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse {path}:{line_no}: {e}")


def repo_source_name(repo_dir: Path) -> str:
    name = repo_dir.name.lower()
    if "bdv2fail" in name:
        return "bdv2fail"
    if "rlbenchfail" in name:
        return "rlbenchfail"
    if "ur5fail" in name:
        return "ur5fail"
    return name


def repo_split_name(repo_dir: Path) -> str:
    name = repo_dir.name.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return "unknown"


def find_metadata_execution(repo_dir: Path) -> Optional[Path]:
    candidates = list(repo_dir.rglob("metadata_execution.jsonl"))
    if candidates:
        return candidates[0]
    # Some hub snapshots may preserve slightly different names.
    candidates = list(repo_dir.rglob("*metadata*execution*.jsonl"))
    return candidates[0] if candidates else None


def _view_score(path: str, preferred_view: str) -> Tuple[int, str]:
    p = path.lower()
    prefs = [preferred_view.lower(), "viewpoint_0", "front", "global", "left", "right", "wrist"]
    for i, key in enumerate(prefs):
        if key and key in p:
            return (i, p)
    return (999, p)


def choose_before_after(image_paths: List[str], preferred_view: str = "viewpoint_0") -> Tuple[Optional[str], Optional[str]]:
    starts = [p for p in image_paths if "start" in Path(p).name.lower()]
    ends = [p for p in image_paths if "end" in Path(p).name.lower()]
    if not starts or not ends:
        # Fallback: assume first half starts, second half ends.
        n = len(image_paths)
        if n >= 2:
            starts, ends = image_paths[: n // 2], image_paths[n // 2 :]
    if not starts or not ends:
        return None, None
    starts = sorted(starts, key=lambda p: _view_score(p, preferred_view))
    ends = sorted(ends, key=lambda p: _view_score(p, preferred_view))
    return starts[0], ends[0]


def _resolve_image_path(repo_dir: Path, rel_path: str) -> Optional[Path]:
    """Resolve an image path from metadata, handling prefix mismatches.

    Metadata may store paths like 'data/failure_forge/data/.../records/...'
    but after tar extraction, images land directly under 'records/' in repo_dir.
    Try the literal path first, then strip the prefix up to 'records/'.
    """
    candidate = repo_dir / rel_path
    if candidate.exists():
        return candidate
    # Strip everything before 'records/' and retry.
    parts = rel_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "records":
            short = "/".join(parts[i:])
            candidate = repo_dir / short
            if candidate.exists():
                return candidate
            break
    return None


def normalize_sample(raw: dict, repo_dir: Path, source: str, split: str, preferred_view: str) -> Optional[dict]:
    images = raw.get("images") or []
    before_rel, after_rel = choose_before_after(images, preferred_view=preferred_view)
    if not before_rel or not after_rel:
        return None

    before = _resolve_image_path(repo_dir, before_rel)
    after = _resolve_image_path(repo_dir, after_rel)
    if not before or not after:
        return None
    task_instruction = raw.get("task_instruction") or ""
    subtask = raw.get("detailed_subtask_name") or ""
    instruction = task_instruction.strip()
    if subtask:
        instruction = f"{instruction} [SUBTASK] {subtask}".strip()

    reward = raw.get("execution_reward", raw.get("reward", 0))
    try:
        label = int(float(reward))
    except (TypeError, ValueError):
        label = 0

    failure_mode = raw.get("failure_mode")
    if label == 1:
        failure_mode = "success"
    elif not failure_mode:
        failure_mode = "failure"

    return {
        "before": str(before.resolve()),
        "after": str(after.resolve()),
        "instruction": instruction,
        "label": label,
        "failure_type": str(failure_mode),
        "source": source,
        "split": split,
        "taskvar": str(raw.get("taskvar", "")),
        "episode_id": raw.get("episode_id", None),
        "synthetic": False,
        "counterfactual_type": "real_or_ground_truth",
    }


def load_all(raw_root: Path, preferred_view: str) -> List[dict]:
    rows: List[dict] = []
    for repo_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        meta_path = find_metadata_execution(repo_dir)
        if meta_path is None:
            print(f"[WARN] No metadata_execution.jsonl found in {repo_dir}")
            continue
        source = repo_source_name(repo_dir)
        split = repo_split_name(repo_dir)
        print(f"Reading {meta_path} source={source} split={split}")
        for raw in tqdm(read_jsonl(meta_path)):
            row = normalize_sample(raw, repo_dir, source, split, preferred_view)
            if row is not None:
                rows.append(row)
    return rows


def make_counterfactuals(success_rows: List[dict], neg_per_pos: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    rows: List[dict] = []
    types = ["no_progress", "temporal_reverse", "instruction_mismatch", "after_mismatch"]

    by_task = defaultdict(list)
    for r in success_rows:
        by_task[r.get("taskvar", "")].append(r)

    for r in success_rows:
        pos = dict(r)
        pos["label"] = 1
        pos["failure_type"] = "success"
        pos["synthetic"] = False
        pos["counterfactual_type"] = "positive_success"
        rows.append(pos)

        selected = rng.sample(types, k=min(neg_per_pos, len(types)))
        for t in selected:
            neg = dict(r)
            neg["label"] = 0
            neg["failure_type"] = t
            neg["synthetic"] = True
            neg["counterfactual_type"] = t

            if t == "no_progress":
                neg["after"] = r["before"]
            elif t == "temporal_reverse":
                neg["before"], neg["after"] = r["after"], r["before"]
            elif t == "instruction_mismatch":
                candidates = [x for x in success_rows if x.get("taskvar") != r.get("taskvar")]
                other = rng.choice(candidates or success_rows)
                neg["instruction"] = other["instruction"]
            elif t == "after_mismatch":
                candidates = [x for x in success_rows if x.get("taskvar") != r.get("taskvar")]
                other = rng.choice(candidates or success_rows)
                neg["after"] = other["after"]
            rows.append(neg)
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def build_vocab(rows: List[dict], min_freq: int = 1, max_size: int = 20000) -> Dict[str, int]:
    counter = Counter()
    for r in rows:
        counter.update(tokenize(r.get("instruction", "")))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common(max_size - len(vocab)):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/raw")
    parser.add_argument("--out_root", type=str, default="data/processed")
    parser.add_argument("--train_source", type=str, default="bdv2fail", choices=["bdv2fail", "rlbenchfail", "ur5fail", "all"])
    parser.add_argument("--success_only_counterfactual", action="store_true")
    parser.add_argument("--neg_per_pos", type=int, default=4)
    parser.add_argument("--preferred_view", type=str, default="viewpoint_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_word_freq", type=int, default=1)
    args = parser.parse_args()

    raw_root = resolve_input_path(args.raw_root)
    out_root = resolve_output_path(args.out_root)
    random.seed(args.seed)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"raw_root not found: {raw_root}. Run download first or pass --raw_root with the dataset location."
        )

    all_rows = load_all(raw_root, preferred_view=args.preferred_view)
    print(f"Loaded normalized execution rows: {len(all_rows)}")

    train_candidates = [r for r in all_rows if r["split"] == "train"]
    if args.train_source != "all":
        train_candidates = [r for r in train_candidates if r["source"] == args.train_source]

    if args.success_only_counterfactual:
        success_rows = [r for r in train_candidates if r["label"] == 1]
        train_rows = make_counterfactuals(success_rows, neg_per_pos=args.neg_per_pos, seed=args.seed)
    else:
        train_rows = train_candidates
        random.shuffle(train_rows)

    write_jsonl(out_root / "train.jsonl", train_rows)

    # Save each val/test source separately for cross-domain evaluation.
    for split in ["val", "test"]:
        for source in ["bdv2fail", "rlbenchfail", "ur5fail"]:
            rows = [r for r in all_rows if r["split"] == split and r["source"] == source]
            if rows:
                write_jsonl(out_root / f"{split}_{source}.jsonl", rows)

    vocab = build_vocab(train_rows, min_freq=args.min_word_freq)
    with (out_root / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Report stats.
    print("\n=== Manifest stats ===")
    for name, rows in [("train", train_rows)] + [
        (p.stem, list(read_jsonl(p))) for p in sorted(out_root.glob("val_*.jsonl"))
    ] + [
        (p.stem, list(read_jsonl(p))) for p in sorted(out_root.glob("test_*.jsonl"))
    ]:
        labels = Counter(r["label"] for r in rows)
        types = Counter(r.get("failure_type", "") for r in rows)
        print(f"{name:20s} n={len(rows):6d} labels={dict(labels)} top_types={types.most_common(6)}")
    print(f"Vocab size: {len(vocab)} -> {out_root / 'vocab.json'}")


if __name__ == "__main__":
    main()
