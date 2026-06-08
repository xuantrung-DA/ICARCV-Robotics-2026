"""CounterFail-Edge dataset and transforms."""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 64) -> torch.Tensor:
    ids = [vocab.get(tok, vocab.get("<unk>", 1)) for tok in tokenize(text)[:max_len]]
    if not ids:
        ids = [vocab.get("<unk>", 1)]
    return torch.tensor(ids, dtype=torch.long)


def compute_oov_rate(rows: List[dict], vocab: Dict[str, int]) -> dict:
    """Report total tokens, unk tokens, unk rate, top unknown tokens."""
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


class RandomOcclusion:
    def __init__(self, p: float = 0.25, min_frac: float = 0.08, max_frac: float = 0.25):
        self.p = p
        self.min_frac = min_frac
        self.max_frac = max_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img = img.copy()
        w, h = img.size
        occ_w = int(w * random.uniform(self.min_frac, self.max_frac))
        occ_h = int(h * random.uniform(self.min_frac, self.max_frac))
        x0 = random.randint(0, max(0, w - occ_w))
        y0 = random.randint(0, max(0, h - occ_h))
        patch = Image.new("RGB", (occ_w, occ_h), (127, 127, 127))
        img.paste(patch, (x0, y0))
        return img


class RandomBlur:
    def __init__(self, p: float = 0.15, radius: float = 1.2):
        self.p = p
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class StrongCorruption:
    """Strong visual corruption — only applied when counterfactual_type=='visual_corruption'."""

    def __init__(self, min_frac: float = 0.35, max_frac: float = 0.60, blur_radius: float = 2.5):
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.blur_radius = blur_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.copy().filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        w, h = img.size
        occ_w = int(w * random.uniform(self.min_frac, self.max_frac))
        occ_h = int(h * random.uniform(self.min_frac, self.max_frac))
        x0 = random.randint(0, max(0, w - occ_w))
        y0 = random.randint(0, max(0, h - occ_h))
        img.paste(Image.new("RGB", (occ_w, occ_h), (127, 127, 127)), (x0, y0))
        return img


def make_transforms(img_size: int, train: bool):
    aug = []
    if train:
        aug += [
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
            RandomOcclusion(p=0.20),
            RandomBlur(p=0.10),
        ]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _sample_jitter(value: float) -> float:
    return random.uniform(max(0.0, 1.0 - value), 1.0 + value)


class PairTrainTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.occlusion = RandomOcclusion(p=0.16)
        self.blur = RandomBlur(p=0.08)

    def _color_jitter(self, img: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img

    def __call__(self, before: Image.Image, after: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        before = transforms.functional.resize(before, (self.img_size, self.img_size))
        after = transforms.functional.resize(after, (self.img_size, self.img_size))
        brightness = _sample_jitter(0.12)
        contrast = _sample_jitter(0.12)
        saturation = _sample_jitter(0.06)
        before = self._color_jitter(before, brightness, contrast, saturation)
        after = self._color_jitter(after, brightness, contrast, saturation)
        before = self.blur(self.occlusion(before))
        after = self.blur(self.occlusion(after))
        return self.to_tensor(before), self.to_tensor(after)


class PairInstructionDataset(Dataset):
    def __init__(self, jsonl: str, vocab: Dict[str, int], img_size: int = 224,
                 train: bool = False, max_text_len: int = 64, paired_aug: bool = True):
        self.rows = read_jsonl(jsonl)
        self.vocab = vocab
        self.tf = make_transforms(img_size, train=train)
        self.pair_tf = PairTrainTransform(img_size) if train and paired_aug else None
        self.strong_corruption = StrongCorruption()
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.tf(img)

    def _load_pil(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        before_img = self._load_pil(r["before"])
        after_img = self._load_pil(r["after"])

        # Only apply StrongCorruption if explicitly marked as visual_corruption
        if r.get("counterfactual_type") == "visual_corruption":
            if r.get("corrupt_target", "after") == "before":
                before_img = self.strong_corruption(before_img)
            else:
                after_img = self.strong_corruption(after_img)

        if self.pair_tf is None:
            before = self.tf(before_img)
            after = self.tf(after_img)
        else:
            before, after = self.pair_tf(before_img, after_img)
        text_ids = encode_text(r.get("instruction", ""), self.vocab, max_len=self.max_text_len)
        label = torch.tensor(float(r["label"]), dtype=torch.float32)
        return {
            "before": before,
            "after": after,
            "text_ids": text_ids,
            "label": label,
            "failure_type": r.get("failure_type", ""),
            "source": r.get("source", ""),
            "taskvar": r.get("taskvar", ""),
            "synthetic": r.get("synthetic", False),
            "counterfactual_type": r.get("counterfactual_type", ""),
            "instruction": r.get("instruction", ""),
            "before_path": r.get("before", ""),
            "after_path": r.get("after", ""),
        }


def collate_fn(batch: List[dict]) -> dict:
    before = torch.stack([b["before"] for b in batch], dim=0)
    after = torch.stack([b["after"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    text_lens = torch.tensor([len(b["text_ids"]) for b in batch], dtype=torch.long)
    text_ids = pad_sequence([b["text_ids"] for b in batch], batch_first=True, padding_value=0)
    return {
        "before": before,
        "after": after,
        "text_ids": text_ids,
        "text_lens": text_lens,
        "label": labels,
        "failure_type": [b["failure_type"] for b in batch],
        "source": [b["source"] for b in batch],
        "taskvar": [b["taskvar"] for b in batch],
        "synthetic": [b["synthetic"] for b in batch],
        "counterfactual_type": [b["counterfactual_type"] for b in batch],
        "instruction": [b["instruction"] for b in batch],
        "before_path": [b["before_path"] for b in batch],
        "after_path": [b["after_path"] for b in batch],
    }
