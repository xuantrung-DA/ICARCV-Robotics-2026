"""CounterFail-Edge data preparation with semantic-hard counterfactual generation."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from .paths import resolve_input_path, resolve_output_path

CODE_VERSION = "counterfail_semhard_mbv3large_v1"

# ---------------------------------------------------------------------------
# Stopwords for content/action extraction
# ---------------------------------------------------------------------------
STOPWORDS = frozenset({
    "the", "a", "an", "to", "in", "on", "into", "from", "of", "and", "or",
    "with", "without", "robot", "arm", "object", "item", "thing", "task",
    "subtask", "put", "place", "move", "pick", "up", "grasp", "manipulate",
    "use", "is", "it", "its", "this", "that", "be", "was", "were", "been",
})

ACTION_WORDS = frozenset({
    "open", "close", "push", "pull", "slide", "rotate", "turn", "lift",
    "drop", "press", "flip", "fold", "unfold", "pour", "screw", "unscrew",
    "insert", "remove", "stack", "unstack", "sweep", "wipe", "cut", "peel",
    "pick", "place", "put", "move", "grasp", "release", "reach", "drag",
})

LOCATION_WORDS = frozenset({
    "left", "right", "top", "bottom", "front", "back", "center", "middle",
    "above", "below", "near", "far", "inside", "outside", "corner",
    "edge", "side", "drawer", "shelf", "table", "bin", "box", "tray",
    "plate", "bowl", "cup", "container", "slot", "rack", "hook",
})


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
        n = len(image_paths)
        if n >= 2:
            starts, ends = image_paths[: n // 2], image_paths[n // 2 :]
    if not starts or not ends:
        return None, None
    starts = sorted(starts, key=lambda p: _view_score(p, preferred_view))
    ends = sorted(ends, key=lambda p: _view_score(p, preferred_view))
    return starts[0], ends[0]


def _resolve_image_path(repo_dir: Path, rel_path: str) -> Optional[Path]:
    """Resolve an image path from metadata, handling prefix mismatches."""
    candidate = repo_dir / rel_path
    if candidate.exists():
        return candidate
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


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def tokenize_instruction(text: str) -> List[str]:
    return tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def extract_content_tokens(row: dict) -> set:
    tokens = tokenize_instruction(row.get("instruction", ""))
    return set(remove_stopwords(tokens))


def extract_action_tokens(row: dict) -> set:
    tokens = tokenize_instruction(row.get("instruction", ""))
    return set(t for t in tokens if t in ACTION_WORDS)


def extract_object_target_tokens(row: dict) -> set:
    tokens = tokenize_instruction(row.get("instruction", ""))
    return set(t for t in tokens if t not in STOPWORDS and t not in ACTION_WORDS)


def extract_object_only_tokens(row: dict) -> set:
    """Extract tokens likely referring to objects (not locations/states)."""
    tokens = tokenize_instruction(row.get("instruction", ""))
    return set(t for t in tokens if t not in STOPWORDS and t not in ACTION_WORDS and t not in LOCATION_WORDS)


def extract_location_state_tokens(row: dict) -> set:
    """Extract tokens likely referring to locations or states."""
    tokens = tokenize_instruction(row.get("instruction", ""))
    return set(t for t in tokens if t in LOCATION_WORDS)


# ---------------------------------------------------------------------------
# Hard candidate scoring
# ---------------------------------------------------------------------------
def hard_candidates(row: dict, success_rows: List[dict], mode: str = "semantic_hard",
                    top_k: int = 50, same_source: bool = True) -> List[dict]:
    """Score and rank candidates for hard negative generation.

    Scoring uses overlap ratios for finer-grained ranking.
    """
    row_src = row.get("source", "")
    row_taskvar = row.get("taskvar", "")
    row_episode = row.get("episode_id")
    row_action = extract_action_tokens(row)
    row_content = extract_content_tokens(row)
    row_objects = extract_object_only_tokens(row)
    row_locations = extract_location_state_tokens(row)
    row_instr = row.get("instruction", "")
    row_before = row.get("before", "")
    row_after = row.get("after", "")

    scored = []
    for c in success_rows:
        if c.get("episode_id") == row_episode and row_episode is not None:
            continue
        if c.get("before") == row_before and c.get("after") == row_after and c.get("instruction") == row_instr:
            continue
        s = 0.0
        if same_source and c.get("source", "") == row_src:
            s += 3.0
        c_action = extract_action_tokens(c)
        c_content = extract_content_tokens(c)
        c_objects = extract_object_only_tokens(c)
        c_locations = extract_location_state_tokens(c)

        # Action overlap ratio
        action_union = row_action | c_action
        if action_union:
            action_overlap = len(row_action & c_action) / len(action_union)
            s += action_overlap * 3.0  # 0-3 continuous

        # Content overlap ratio
        content_union = row_content | c_content
        if content_union:
            content_overlap = len(row_content & c_content) / len(content_union)
            s += content_overlap * 2.0  # 0-2 continuous

        c_taskvar = c.get("taskvar", "")
        if row_taskvar and c_taskvar:
            if row_taskvar.split("+")[0] == c_taskvar.split("+")[0]:
                s += 2.0  # Same task family
            if row_taskvar == c_taskvar:
                s -= 1.0  # Same exact taskvar less useful
        if c.get("instruction", "") == row_instr:
            s -= 3.0  # Identical instruction not useful
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in scored[:max(1, top_k)]]
    return result if result else success_rows[:1]


def _pick_wrong_object_candidate(row: dict, candidates: List[dict], rng: random.Random) -> Optional[dict]:
    """Pick candidate with action overlap HIGH but object overlap LOW.

    This simulates 'robot did the right action on the wrong object'.
    """
    row_action = extract_action_tokens(row)
    row_objects = extract_object_only_tokens(row)
    good = []
    acceptable = []
    for c in candidates:
        c_action = extract_action_tokens(c)
        c_objects = extract_object_only_tokens(c)
        # Action overlap ratio
        action_union = row_action | c_action
        action_overlap = len(row_action & c_action) / max(len(action_union), 1)
        # Object overlap ratio
        object_union = row_objects | c_objects
        object_overlap = len(row_objects & c_objects) / max(len(object_union), 1)

        if action_overlap >= 0.5 and object_overlap <= 0.3 and c_objects:
            good.append(c)  # Best: same action, different object
        elif action_overlap > 0 and object_overlap < 0.8 and row_objects != c_objects:
            acceptable.append(c)  # Fallback: some action overlap, not identical objects

    if good:
        return rng.choice(good)
    if acceptable:
        return rng.choice(acceptable)
    return rng.choice(candidates) if candidates else None


def _pick_wrong_placement_candidate(row: dict, candidates: List[dict], rng: random.Random) -> Optional[dict]:
    """Pick candidate with overlapping object tokens but different location/state.

    This simulates 'robot manipulated the right object but to the wrong place/state'.
    """
    row_objects = extract_object_only_tokens(row)
    row_locations = extract_location_state_tokens(row)
    good = []
    acceptable = []
    for c in candidates:
        c_objects = extract_object_only_tokens(c)
        c_locations = extract_location_state_tokens(c)
        # Object overlap ratio
        object_union = row_objects | c_objects
        object_overlap = len(row_objects & c_objects) / max(len(object_union), 1)

        if object_overlap >= 0.5 and row_locations != c_locations:
            good.append(c)  # Best: same objects, different location/state
        elif object_overlap > 0.3 and row_objects != c_objects:
            acceptable.append(c)  # Fallback: high object overlap, some difference

    if good:
        return rng.choice(good)
    if acceptable:
        return rng.choice(acceptable)
    return rng.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Counterfactual generation
# ---------------------------------------------------------------------------
ALL_CF_TYPES = [
    "no_progress", "temporal_reverse", "instruction_mismatch_hard",
    "endpoint_mismatch_hard", "wrong_object_like", "wrong_state_or_placement_like",
]

DEFAULT_CF_TYPES = "no_progress,temporal_reverse,instruction_mismatch_hard,endpoint_mismatch_hard,wrong_object_like,wrong_state_or_placement_like"


def make_counterfactuals(
    success_rows: List[dict],
    neg_per_pos: int,
    seed: int,
    counterfactual_mode: str = "semantic_hard",
    counterfactual_types: List[str] = None,
    max_no_progress_frac: float = 0.20,
    hard_top_k: int = 50,
    same_source_hard: bool = True,
    allow_visual_corruption: bool = False,
) -> List[dict]:
    rng = random.Random(seed)
    if counterfactual_types is None:
        counterfactual_types = list(ALL_CF_TYPES)

    rows: List[dict] = []
    no_progress_count = 0
    total_neg_count = 0
    max_no_progress = int(len(success_rows) * neg_per_pos * max_no_progress_frac)

    for r in tqdm(success_rows, desc="Generating counterfactuals"):
        # Add positive
        pos = dict(r)
        pos["label"] = 1
        pos["failure_type"] = "success"
        pos["synthetic"] = False
        pos["counterfactual_type"] = "positive_success"
        rows.append(pos)

        # Get hard candidates once per positive
        cands = hard_candidates(r, success_rows, mode=counterfactual_mode,
                                top_k=hard_top_k, same_source=same_source_hard)

        available_types = list(counterfactual_types)
        # Cap no_progress
        if no_progress_count >= max_no_progress and "no_progress" in available_types:
            available_types = [t for t in available_types if t != "no_progress"]
        if not available_types:
            available_types = ["endpoint_mismatch_hard"]

        selected = rng.sample(available_types, k=min(neg_per_pos, len(available_types)))
        # If need more, allow repeats of non-no_progress types
        while len(selected) < neg_per_pos:
            extra_pool = [t for t in available_types if t != "no_progress"]
            if not extra_pool:
                extra_pool = available_types
            selected.append(rng.choice(extra_pool))

        for t in selected:
            neg = dict(r)
            neg["label"] = 0
            neg["synthetic"] = True

            if t == "no_progress":
                neg["after"] = r["before"]
                neg["failure_type"] = "no_progress_synth"
                neg["counterfactual_type"] = "no_progress"
                no_progress_count += 1

            elif t == "temporal_reverse":
                neg["before"], neg["after"] = r["after"], r["before"]
                neg["failure_type"] = "temporal_reverse_synth"
                neg["counterfactual_type"] = "temporal_reverse"

            elif t == "instruction_mismatch_hard":
                other = rng.choice(cands)
                neg["instruction"] = other["instruction"]
                neg["failure_type"] = "instruction_mismatch_hard"
                neg["counterfactual_type"] = "instruction_mismatch_hard"

            elif t == "endpoint_mismatch_hard":
                other = rng.choice(cands)
                neg["after"] = other["after"]
                neg["failure_type"] = "endpoint_mismatch_hard"
                neg["counterfactual_type"] = "endpoint_mismatch_hard"

            elif t == "wrong_object_like":
                other = _pick_wrong_object_candidate(r, cands, rng)
                if other:
                    neg["instruction"] = other["instruction"]
                    neg["failure_type"] = "wrong object manipulated"
                    neg["counterfactual_type"] = "wrong_object_like"
                else:
                    other = rng.choice(cands)
                    neg["instruction"] = other["instruction"]
                    neg["failure_type"] = "instruction_mismatch_hard"
                    neg["counterfactual_type"] = "instruction_mismatch_hard"

            elif t == "wrong_state_or_placement_like":
                other = _pick_wrong_placement_candidate(r, cands, rng)
                if other:
                    neg["after"] = other["after"]
                    neg["failure_type"] = "wrong object state or placement"
                    neg["counterfactual_type"] = "wrong_state_or_placement_like"
                else:
                    other = rng.choice(cands)
                    neg["after"] = other["after"]
                    neg["failure_type"] = "endpoint_mismatch_hard"
                    neg["counterfactual_type"] = "endpoint_mismatch_hard"

            else:
                continue

            total_neg_count += 1
            rows.append(neg)

    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_vocab(rows: List[dict], min_freq: int = 1, max_size: int = 20000) -> Dict[str, int]:
    counter = Counter()
    for r in rows:
        counter.update(tokenize(r.get("instruction", "")))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common(max_size - len(vocab)):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def compute_oov_rate(rows: List[dict], vocab: Dict[str, int]) -> dict:
    total, unk = 0, 0
    unk_tokens: Counter = Counter()
    unk_id = vocab.get("<unk>", 1)
    for r in rows:
        for tok in tokenize(r.get("instruction", "")):
            total += 1
            if vocab.get(tok, unk_id) == unk_id:
                unk += 1
                unk_tokens[tok] += 1
    return {
        "total_tokens": total,
        "unk_tokens": unk,
        "unk_rate": unk / max(total, 1),
        "top_unk": unk_tokens.most_common(20),
    }


def _print_manifest_stats(name: str, rows: List[dict]):
    labels = Counter(r["label"] for r in rows)
    types = Counter(r.get("failure_type", "") for r in rows)
    ct = Counter(r.get("counterfactual_type", "") for r in rows)
    syn = Counter(r.get("synthetic", False) for r in rows)
    print(f"\n{name:20s} n={len(rows):6d}")
    print(f"  labels: {dict(labels)}")
    print(f"  failure_types: {dict(types.most_common(10))}")
    print(f"  counterfactual_types: {dict(ct.most_common(10))}")
    print(f"  synthetic: {dict(syn)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/raw")
    parser.add_argument("--out_root", type=str, default="data/processed")
    parser.add_argument("--train_source", type=str, default="bdv2fail",
                        choices=["bdv2fail", "rlbenchfail", "ur5fail", "all"])
    parser.add_argument("--success_only_counterfactual", action="store_true")
    parser.add_argument("--counterfactual_mode", choices=["basic", "semantic_hard"], default="semantic_hard")
    parser.add_argument("--counterfactual_types", type=str, default=DEFAULT_CF_TYPES,
                        help="Comma-separated counterfactual types.")
    parser.add_argument("--neg_per_pos", type=int, default=3)
    parser.add_argument("--max_no_progress_frac", type=float, default=0.15)
    parser.add_argument("--hard_top_k", type=int, default=50)
    parser.add_argument("--same_source_hard_negatives", action="store_true", default=True)
    parser.add_argument("--allow_visual_corruption_negative", action="store_true", default=False)
    parser.add_argument("--real_failure_mix", type=float, default=0.0)
    parser.add_argument("--real_failure_max_per_type", type=int, default=100000)
    parser.add_argument("--preferred_view", type=str, default="viewpoint_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_word_freq", type=int, default=1)
    parser.add_argument("--write_diagnostics", action="store_true", default=False)
    args = parser.parse_args()

    print(f"[prepare_guardian.py] __file__={__file__}")
    print(f"[prepare_guardian.py] CODE_VERSION={CODE_VERSION}")

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

    cf_types = [t.strip() for t in args.counterfactual_types.split(",") if t.strip()]

    if args.success_only_counterfactual:
        success_rows = [r for r in train_candidates if r["label"] == 1]
        print(f"Success-only counterfactual mode: {len(success_rows)} success rows")
        train_rows = make_counterfactuals(
            success_rows,
            neg_per_pos=args.neg_per_pos,
            seed=args.seed,
            counterfactual_mode=args.counterfactual_mode,
            counterfactual_types=cf_types,
            max_no_progress_frac=args.max_no_progress_frac,
            hard_top_k=args.hard_top_k,
            same_source_hard=args.same_source_hard_negatives,
            allow_visual_corruption=args.allow_visual_corruption_negative,
        )

        # Hybrid mode: mix real failures
        if args.real_failure_mix > 0:
            real_failures = [r for r in train_candidates if r["label"] == 0]
            if not real_failures:
                print("WARNING: training source lacks real failures but real_failure_mix > 0. Continuing without.")
            else:
                n_real = int(len(train_rows) * args.real_failure_mix)
                by_type: Dict[str, List[dict]] = {}
                for r in real_failures:
                    ft = r.get("failure_type", "failure")
                    by_type.setdefault(ft, []).append(r)
                mixed = []
                for ft, ft_rows in by_type.items():
                    random.shuffle(ft_rows)
                    cap = min(len(ft_rows), args.real_failure_max_per_type)
                    mixed.extend(ft_rows[:cap])
                random.shuffle(mixed)
                mixed = mixed[:n_real]
                for r in mixed:
                    r["counterfactual_type"] = "real_failure_mix"
                train_rows.extend(mixed)
                random.shuffle(train_rows)
                print(f"Added {len(mixed)} real failure rows (hybrid mode)")
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

    vocab = build_vocab(all_rows, min_freq=args.min_word_freq)
    with (out_root / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # ---- Manifest stats ----
    print("\n=== Manifest stats ===")
    _print_manifest_stats("train", train_rows)

    # Synthetic negative analysis
    synth_negs = [r for r in train_rows if r.get("synthetic") and r["label"] == 0]
    if synth_negs:
        np_count = sum(1 for r in synth_negs if r.get("counterfactual_type") == "no_progress")
        print(f"\nno_progress fraction among synthetic negatives: {np_count}/{len(synth_negs)} = {np_count/len(synth_negs):.3f}")

    for split in ["val", "test"]:
        for source in ["bdv2fail", "rlbenchfail", "ur5fail"]:
            p = out_root / f"{split}_{source}.jsonl"
            if p.exists():
                split_rows = list(read_jsonl(p))
                _print_manifest_stats(f"{split}_{source}", split_rows)

    print(f"\nVocab size: {len(vocab)} -> {out_root / 'vocab.json'}")

    # OOV report
    for split in ["val", "test"]:
        for source in ["bdv2fail", "rlbenchfail", "ur5fail"]:
            p = out_root / f"{split}_{source}.jsonl"
            if p.exists():
                split_rows = list(read_jsonl(p))
                oov = compute_oov_rate(split_rows, vocab)
                print(f"OOV {split}_{source}: rate={oov['unk_rate']:.4f} ({oov['unk_tokens']}/{oov['total_tokens']})")

    # Write diagnostics
    if args.write_diagnostics:
        stats = {
            "train_n": len(train_rows),
            "labels": dict(Counter(r["label"] for r in train_rows)),
            "failure_types": dict(Counter(r.get("failure_type", "") for r in train_rows).most_common()),
            "counterfactual_types": dict(Counter(r.get("counterfactual_type", "") for r in train_rows).most_common()),
            "synthetic": dict(Counter(r.get("synthetic", False) for r in train_rows)),
            "vocab_size": len(vocab),
            "code_version": CODE_VERSION,
        }
        with (out_root / "manifest_stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

        # Sample examples per counterfactual type
        by_ct: Dict[str, List[dict]] = {}
        for r in train_rows:
            ct = r.get("counterfactual_type", "unknown")
            by_ct.setdefault(ct, []).append(r)
        examples = []
        for ct, ct_rows in by_ct.items():
            for ex in ct_rows[:3]:
                examples.append({k: v for k, v in ex.items()})
        with (out_root / "counterfactual_examples.jsonl").open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # OOV report
        oov_report = {}
        for split in ["val", "test"]:
            for source in ["bdv2fail", "rlbenchfail", "ur5fail"]:
                p = out_root / f"{split}_{source}.jsonl"
                if p.exists():
                    oov_report[f"{split}_{source}"] = compute_oov_rate(list(read_jsonl(p)), vocab)
        with (out_root / "oov_report.json").open("w", encoding="utf-8") as f:
            json.dump(oov_report, f, indent=2, default=str)

        print(f"\nDiagnostics written to {out_root}")


if __name__ == "__main__":
    main()
