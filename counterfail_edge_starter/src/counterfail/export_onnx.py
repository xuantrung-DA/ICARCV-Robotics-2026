import argparse
import torch

from .data import load_vocab
from .model import CounterFailNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=64)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})
    model = CounterFailNet(
        vocab_size=len(vocab),
        encoder=train_args.get("encoder", "mobilenet_v3_small"),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    before = torch.randn(1, 3, args.img_size, args.img_size)
    after = torch.randn(1, 3, args.img_size, args.img_size)
    text_ids = torch.ones(1, args.max_text_len, dtype=torch.long)
    text_lens = torch.tensor([args.max_text_len], dtype=torch.long)

    torch.onnx.export(
        model,
        (before, after, text_ids, text_lens),
        args.out,
        input_names=["before", "after", "text_ids", "text_lens"],
        output_names=["logit", "pair_z", "text_z"],
        dynamic_axes={
            "before": {0: "batch"},
            "after": {0: "batch"},
            "text_ids": {0: "batch", 1: "seq"},
            "text_lens": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX to {args.out}")


if __name__ == "__main__":
    main()
