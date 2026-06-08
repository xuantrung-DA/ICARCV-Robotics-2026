"""CounterFail-Edge latency benchmark."""

import argparse
import time

import torch

from .data import load_vocab
from .model import CODE_VERSION, CounterFailNet


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

    print(f"[latency.py] CODE_VERSION={CODE_VERSION}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.vocab)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    encoder = train_args.get("encoder", "mobilenet_v3_large")
    text_encoder_type = train_args.get("text_encoder", "mean")
    text_dim = train_args.get("text_dim", 256)
    hidden_dim = train_args.get("hidden_dim", 256)

    model = CounterFailNet(
        vocab_size=len(vocab),
        encoder=encoder,
        pretrained=False,
        text_encoder_type=text_encoder_type,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
    ).to(device)
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
        "encoder": encoder,
        "text_encoder": text_encoder_type,
        "batch_size": args.batch_size,
        "latency_ms_per_batch": dt * 1000,
        "latency_ms_per_sample": dt * 1000 / args.batch_size,
        "fps": args.batch_size / dt,
        "params_million": params / 1e6,
    })


if __name__ == "__main__":
    main()
