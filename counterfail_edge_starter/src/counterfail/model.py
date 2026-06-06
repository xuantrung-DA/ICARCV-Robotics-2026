from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MeanTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, out_dim: int = 256, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, text_ids: torch.Tensor, text_lens: torch.Tensor) -> torch.Tensor:
        x = self.emb(text_ids)  # B, T, E
        mask = (text_ids != 0).float().unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = summed / denom
        return self.proj(mean)


def build_image_encoder(name: str = "mobilenet_v3_small", pretrained: bool = True) -> Tuple[nn.Module, int]:
    name = name.lower()
    weights = None
    if name == "mobilenet_v3_small":
        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None
        m = models.mobilenet_v3_small(weights=weights)
        encoder = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        feat_dim = m.classifier[0].in_features
    elif name == "resnet18":
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        m = models.resnet18(weights=weights)
        encoder = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        feat_dim = m.fc.in_features
    else:
        raise ValueError(f"Unknown encoder: {name}")
    return encoder, feat_dim


class CounterFailNet(nn.Module):
    """Tiny before-after-instruction verifier.

    Output logit > 0 means success; logit < 0 means failure.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: str = "mobilenet_v3_small",
        pretrained: bool = True,
        text_dim: int = 256,
        hidden_dim: int = 256,
        freeze_image: bool = False,
    ):
        super().__init__()
        self.image_encoder, feat_dim = build_image_encoder(encoder, pretrained=pretrained)
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.text_encoder = MeanTextEncoder(vocab_size=vocab_size, out_dim=text_dim)
        visual_in = feat_dim * 4  # before, after, after-before, abs(after-before)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def encode_pair(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        fb = self.image_encoder(before)
        fa = self.image_encoder(after)
        v = torch.cat([fb, fa, fa - fb, torch.abs(fa - fb)], dim=1)
        return self.visual_proj(v)

    def forward(self, before: torch.Tensor, after: torch.Tensor, text_ids: torch.Tensor, text_lens: torch.Tensor):
        v = self.encode_pair(before, after)
        t = self.text_proj(self.text_encoder(text_ids, text_lens))
        fusion = torch.cat([v, t, v * t, torch.abs(v - t)], dim=1)
        logit = self.classifier(fusion).squeeze(1)
        pair_z = F.normalize(self.pair_head(v), dim=1)
        text_z = F.normalize(self.text_head(t), dim=1)
        return logit, pair_z, text_z


def contrastive_loss(pair_z: torch.Tensor, text_z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE over successful pairs only.

    For positives, pair_z[i] should match text_z[i]. For a batch with <2 positives,
    returns zero to avoid noisy contrastive updates.
    """
    idx = torch.where(labels > 0.5)[0]
    if idx.numel() < 2:
        return pair_z.sum() * 0.0
    p = pair_z[idx]
    t = text_z[idx]
    logits = (p @ t.t()) / temperature
    target = torch.arange(idx.numel(), device=pair_z.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
