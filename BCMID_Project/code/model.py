from __future__ import annotations

from typing import Iterable

import torch.nn as nn


SUPPORTED_BACKBONES = (
    "efficientnet_b0",
    "convnext_small",
    "vit_base_patch16_224",
)


def create_single_modality_model(
    backbone: str,
    pretrained: bool = True,
    num_classes: int = 1,
) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        valid = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of: {valid}")
    import timm

    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def supported_backbones() -> Iterable[str]:
    return SUPPORTED_BACKBONES
