"""CounterFail-Edge model with pluggable visual and text encoders."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

CODE_VERSION = "counterfail_semhard_mbv3large_v1"


# ---------------------------------------------------------------------------
# Text encoders
# ---------------------------------------------------------------------------

class MeanTextEncoder(nn.Module):
    """Bag-of-embeddings text encoder (baseline)."""

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


class BiGRUTextEncoder(nn.Module):
    """Bidirectional GRU text encoder — captures word order and object/target semantics."""

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_size: int = 128,
                 out_dim: int = 256, pad_idx: int = 0, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        # bidirectional → 2 * hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_ids: torch.Tensor, text_lens: torch.Tensor) -> torch.Tensor:
        B = text_ids.size(0)
        x = self.emb(text_ids)  # B, T, E

        # Clamp lengths to be at least 1 and at most seq_len for pack safety
        seq_len = text_ids.size(1)
        lengths = text_lens.clamp(min=1, max=seq_len).cpu()

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: (2, B, H)
        # Concatenate forward and backward final hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # B, 2*H
        return self.proj(hidden)


def _build_text_encoder(text_encoder_type: str, vocab_size: int, emb_dim: int = 128,
                        out_dim: int = 256) -> nn.Module:
    """Factory for text encoder modules."""
    text_encoder_type = text_encoder_type.lower()
    if text_encoder_type == "mean":
        return MeanTextEncoder(vocab_size=vocab_size, emb_dim=emb_dim, out_dim=out_dim)
    elif text_encoder_type == "bigru":
        return BiGRUTextEncoder(vocab_size=vocab_size, emb_dim=emb_dim,
                                hidden_size=emb_dim, out_dim=out_dim)
    else:
        raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}. Choose 'mean' or 'bigru'.")


# ---------------------------------------------------------------------------
# Image encoder factory
# ---------------------------------------------------------------------------

def build_image_encoder(name: str = "mobilenet_v3_large", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Build a visual backbone.

    Supported:
      - mobilenet_v3_small
      - mobilenet_v3_large  (default)
      - efficientnet_b0
      - resnet18
      - shufflenet_v2_x1_0
    """
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
    elif name == "mobilenet_v3_large":
        if pretrained:
            try:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
            except Exception:
                weights = None
        m = models.mobilenet_v3_large(weights=weights)
        encoder = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        feat_dim = m.classifier[0].in_features
    elif name == "efficientnet_b0":
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            except Exception:
                weights = None
        m = models.efficientnet_b0(weights=weights)
        encoder = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        feat_dim = m.classifier[1].in_features
    elif name == "resnet18":
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        m = models.resnet18(weights=weights)
        encoder = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        feat_dim = m.fc.in_features
    elif name == "shufflenet_v2_x1_0":
        if pretrained:
            try:
                weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            except Exception:
                weights = None
        m = models.shufflenet_v2_x1_0(weights=weights)
        # Remove the final FC layer
        encoder = nn.Sequential(
            m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5,
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        feat_dim = m.fc.in_features
    else:
        raise ValueError(f"Unknown encoder: {name}. Choose from: "
                         "mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0, resnet18, shufflenet_v2_x1_0")
    return encoder, feat_dim


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class CounterFailNet(nn.Module):
    """Tiny before-after-instruction verifier.

    Output logit > 0 means success; logit < 0 means failure.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: str = "mobilenet_v3_large",
        pretrained: bool = True,
        text_encoder_type: str = "mean",
        text_dim: int = 256,
        hidden_dim: int = 256,
        freeze_image: bool = False,
    ):
        super().__init__()
        self.image_encoder, feat_dim = build_image_encoder(encoder, pretrained=pretrained)
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.text_encoder = _build_text_encoder(text_encoder_type, vocab_size=vocab_size, out_dim=text_dim)
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

    Contrastive alignment is applied only to successful pairs;
    counterfactual negatives are supervised by BCE/focal loss.

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
