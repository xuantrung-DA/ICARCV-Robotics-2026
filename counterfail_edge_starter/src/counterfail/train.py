import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import PairInstructionDataset, collate_fn, load_vocab
from .metrics import binary_metrics, expected_calibration_error, find_best_threshold
from .model import CounterFailNet, contrastive_loss
from .paths import resolve_input_path, resolve_output_path


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_pos_weight(value: str, rows: list[dict]) -> float | None:
    if value.lower() == "none":
        return None
    labels = np.asarray([float(r["label"]) for r in rows])
    pos = float((labels > 0.5).sum())
    neg = float((labels <= 0.5).sum())
    if value.lower() == "auto":
        return neg / max(pos, 1.0)
    parsed = float(value)
    return parsed if parsed > 0 else None


def make_balanced_sampler(rows: list[dict]) -> WeightedRandomSampler:
    labels = np.asarray([int(float(r["label"]) > 0.5) for r in rows])
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = np.asarray([1.0 / max(counts[label], 1.0) for label in labels], dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def bce_focal_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, prob, 1.0 - prob)
    return ((1.0 - pt).pow(gamma) * bce).mean()


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, tune_threshold: bool = False, threshold_metric: str = "f1"):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_lens = batch["text_lens"].to(device)
        labels = batch["label"].cpu().numpy().tolist()
        logits, _, _ = model(before, after, text_ids, text_lens)
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        y_true.extend(labels)
        y_prob.extend(probs)
    if tune_threshold:
        metrics = find_best_threshold(y_true, y_prob, metric=threshold_metric)
    else:
        metrics = binary_metrics(y_true, y_prob, threshold=threshold)
        metrics["threshold"] = float(threshold)
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out_dir", default="runs/counterfail")
    parser.add_argument("--encoder", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_image", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--contrastive_weight", type=float, default=0.05)
    parser.add_argument("--pos_weight", type=str, default="auto", help="Use 'auto', 'none', or a positive float for BCE positive-class weight.")
    parser.add_argument("--balanced_sampler", action="store_true", help="Sample success/failure classes with equal probability during training.")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--no_paired_aug", action="store_true", help="Use independent before/after augmentations instead of pair-consistent training augmentations.")
    parser.add_argument("--select_metric", choices=["f1", "balanced_acc", "auroc", "auprc"], default="f1")
    parser.add_argument("--fixed_threshold", type=float, default=None, help="If set, validate with this threshold instead of tuning on validation each epoch.")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_jsonl = resolve_input_path(args.train_jsonl)
    val_jsonl = resolve_input_path(args.val_jsonl)
    vocab_path = resolve_input_path(args.vocab)
    out_dir = resolve_output_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path_name, path in [("train_jsonl", train_jsonl), ("val_jsonl", val_jsonl), ("vocab", vocab_path)]:
        if not path.exists():
            raise FileNotFoundError(f"{path_name} not found: {path}")

    vocab = load_vocab(vocab_path)
    train_ds = PairInstructionDataset(train_jsonl, vocab, img_size=args.img_size, train=True, paired_aug=not args.no_paired_aug)
    val_ds = PairInstructionDataset(val_jsonl, vocab, img_size=args.img_size, train=False)
    sampler = make_balanced_sampler(train_ds.rows) if args.balanced_sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CounterFailNet(
        vocab_size=len(vocab),
        encoder=args.encoder,
        pretrained=args.pretrained,
        freeze_image=args.freeze_image,
    ).to(device)

    backbone_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("image_encoder.")]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_mult, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr, "name": "head"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    pos_weight_value = parse_pos_weight(args.pos_weight, train_ds.rows)
    pos_weight = None if pos_weight_value is None else torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    labels_np = np.asarray([float(r["label"]) for r in train_ds.rows])
    print(json.dumps({
        "train_samples": int(len(labels_np)),
        "train_success": int((labels_np > 0.5).sum()),
        "train_failure": int((labels_np <= 0.5).sum()),
        "pos_weight": None if pos_weight_value is None else float(pos_weight_value),
        "balanced_sampler": bool(args.balanced_sampler),
        "paired_aug": bool(not args.no_paired_aug),
    }, indent=2))

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_con = 0.0
        n_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            before = batch["before"].to(device, non_blocking=True)
            after = batch["after"].to(device, non_blocking=True)
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_lens = batch["text_lens"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits, pair_z, text_z = model(before, after, text_ids, text_lens)
                if args.loss == "focal":
                    bce = bce_focal_loss(logits, labels, pos_weight=pos_weight, gamma=args.focal_gamma)
                else:
                    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                con = contrastive_loss(pair_z, text_z, labels)
                loss = bce + args.contrastive_weight * con
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            total_bce += float(bce.detach().cpu())
            total_con += float(con.detach().cpu())
            n_steps += 1
            pbar.set_postfix(loss=total_loss / n_steps, bce=total_bce / n_steps, con=total_con / n_steps)

        scheduler.step()
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=0.5 if args.fixed_threshold is None else args.fixed_threshold,
            tune_threshold=args.fixed_threshold is None,
            threshold_metric=args.select_metric,
        )
        lrs = scheduler.get_last_lr()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_steps),
            "lr_backbone": lrs[0] if len(lrs) > 1 else None,
            "lr_head": lrs[-1],
            **val_metrics,
        }
        history.append(row)
        print("VAL", json.dumps(row, indent=2))

        score = val_metrics[args.select_metric]
        if score > best_score:
            best_score = score
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "vocab_size": len(vocab),
                "best_score": best_score,
                "best_metric": args.select_metric,
                "best_threshold": val_metrics.get("threshold", 0.5),
                "best_val_metrics": val_metrics,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best checkpoint to {out_dir / 'best.pt'}")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
