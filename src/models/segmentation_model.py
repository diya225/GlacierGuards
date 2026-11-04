"""U-Net + EfficientNet-B3 lake segmentation model and BCE+Dice+Focal loss."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


def build_model(encoder: str = "efficientnet-b3", in_channels: int = 10, pretrained: bool = True) -> nn.Module:
    """U-Net with a first conv adapted to `in_channels` (smp re-initialises/tiles ImageNet weights).

    Falls back to random init if ImageNet weights cannot be downloaded.
    """
    weights = "imagenet" if pretrained else None
    try:
        return smp.Unet(encoder, encoder_weights=weights, in_channels=in_channels, classes=1)
    except Exception as exc:  # offline / download failure
        logger.warning("Pretrained weights unavailable (%s); using random init.", exc)
        return smp.Unet(encoder, encoder_weights=None, in_channels=in_channels, classes=1)


class CompositeLoss(nn.Module):
    """BCE + Dice + Focal, summed (weights from config), masked by a per-pixel validity map."""

    def __init__(self, bce_w=1.0, dice_w=1.0, focal_w=1.0, gamma=2.0, alpha=0.25):
        super().__init__()
        self.w = (bce_w, dice_w, focal_w)
        self.gamma, self.alpha = gamma, alpha

    def components(self, logits, target, valid=None):
        valid = torch.ones_like(target) if valid is None else valid
        n = valid.sum().clamp(min=1.0)
        bce_px = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce_px * valid).sum() / n
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        a = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = ((a * (1 - pt) ** self.gamma * bce_px) * valid).sum() / n
        pv, tv = p * valid, target * valid
        inter = (pv * tv).sum()
        dice = 1 - (2 * inter + 1) / (pv.sum() + tv.sum() + 1)
        return bce, dice, focal

    def forward(self, logits, target, valid=None):
        bce, dice, focal = self.components(logits, target, valid)
        total = self.w[0] * bce + self.w[1] * dice + self.w[2] * focal
        return total, {"bce": bce.item(), "dice": dice.item(), "focal": focal.item()}


@torch.no_grad()
def binary_metrics(logits, target, valid=None, thr: float = 0.5) -> dict[str, float]:
    valid = torch.ones_like(target) if valid is None else valid
    pred = (torch.sigmoid(logits) > thr).float() * valid
    t = target * valid
    tp = (pred * t).sum().item()
    fp = (pred * (1 - t) * valid).sum().item()
    fn = ((1 - pred) * t * valid).sum().item()
    tn = ((1 - pred) * (1 - t) * valid).sum().item()
    eps = 1e-9
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def summarize(c: dict[str, float]) -> dict[str, float]:
    eps = 1e-9
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }
