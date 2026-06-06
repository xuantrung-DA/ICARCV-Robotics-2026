import argparse
import time

import torch

from .data import load_vocab
from .model import CounterFailNet


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.vocab)
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get("args", {})
    model = CounterFailNet(vocab_size=len(vocab), encoder=train_args.get("encoder", "mobilenet_v3_small"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    before = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    after = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    text_ids = torch.ones(args.batch_size, 32, dtype=torch.long, device=device)
    text_lens = torch.full((args.batch_size,), 32, dtype=torch.long, device=device)

    for _ in range(args.warmup):
        model(before, after, text_ids, text_lens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        model(before, after, text_ids, text_lens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters
    params = sum(p.numel() for p in model.parameters())
    print({
        "device": str(device),
        "batch_size": args.batch_size,
        "latency_ms_per_batch": dt * 1000,
        "latency_ms_per_sample": dt * 1000 / args.batch_size,
        "fps": args.batch_size / dt,
        "params_million": params / 1e6,
    })


if __name__ == "__main__":
    main()
