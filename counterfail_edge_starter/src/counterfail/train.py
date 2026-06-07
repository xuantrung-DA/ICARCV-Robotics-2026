import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PairInstructionDataset, collate_fn, load_vocab
from .metrics import binary_metrics, expected_calibration_error
from .model import CounterFailNet, contrastive_loss
from .paths import resolve_input_path, resolve_output_path


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
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
    metrics = binary_metrics(y_true, y_prob)
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
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--contrastive_weight", type=float, default=0.05)
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
    train_ds = PairInstructionDataset(train_jsonl, vocab, img_size=args.img_size, train=True)
    val_ds = PairInstructionDataset(val_jsonl, vocab, img_size=args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CounterFailNet(
        vocab_size=len(vocab),
        encoder=args.encoder,
        pretrained=args.pretrained,
        freeze_image=args.freeze_image,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    best_f1 = -1.0
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
                bce = F.binary_cross_entropy_with_logits(logits, labels)
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

        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, n_steps), **val_metrics}
        history.append(row)
        print("VAL", json.dumps(row, indent=2))

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "vocab_size": len(vocab),
                "best_f1": best_f1,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best checkpoint to {out_dir / 'best.pt'}")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
