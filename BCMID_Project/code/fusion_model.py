from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model import supported_backbones


SUPPORTED_FUSION_METHODS = ("late", "concat", "gated")


def create_feature_extractor(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone not in supported_backbones():
        valid = ", ".join(supported_backbones())
        raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of: {valid}")
    import timm

    model = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
    feature_dim = int(model.num_features)
    return model, feature_dim


class BCMIDFusionModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        fusion_method: str = "late",
        pretrained: bool = True,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion_method not in SUPPORTED_FUSION_METHODS:
            valid = ", ".join(SUPPORTED_FUSION_METHODS)
            raise ValueError(f"Unsupported fusion method '{fusion_method}'. Expected one of: {valid}")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")

        self.fusion_method = fusion_method
        self.modality_dropout = modality_dropout

        self.mammogram_encoder, mammogram_dim = create_feature_extractor(backbone, pretrained)
        self.ultrasound_encoder, ultrasound_dim = create_feature_extractor(backbone, pretrained)

        self.mammogram_proj = nn.Sequential(
            nn.Linear(mammogram_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.ultrasound_proj = nn.Sequential(
            nn.Linear(ultrasound_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.mammogram_head = nn.Linear(fusion_dim, 1)
        self.ultrasound_head = nn.Linear(fusion_dim, 1)
        self.concat_head = nn.Sequential(
            nn.Linear(fusion_dim * 2 + 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2 + 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )
        self.gated_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(
        self,
        mammogram: torch.Tensor,
        ultrasound: torch.Tensor,
        mammogram_mask: torch.Tensor,
        ultrasound_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mammogram_mask = mammogram_mask.float().view(-1, 1)
        ultrasound_mask = ultrasound_mask.float().view(-1, 1)
        mammogram_mask, ultrasound_mask = self._apply_modality_dropout(mammogram_mask, ultrasound_mask)

        mammogram_feat = self.mammogram_proj(self.mammogram_encoder(mammogram)) * mammogram_mask
        ultrasound_feat = self.ultrasound_proj(self.ultrasound_encoder(ultrasound)) * ultrasound_mask
        masks = torch.cat([mammogram_mask, ultrasound_mask], dim=1)

        if self.fusion_method == "late":
            mammogram_logit = self.mammogram_head(mammogram_feat)
            ultrasound_logit = self.ultrasound_head(ultrasound_feat)
            denom = torch.clamp(mammogram_mask + ultrasound_mask, min=1.0)
            logits = (mammogram_logit * mammogram_mask + ultrasound_logit * ultrasound_mask) / denom
        elif self.fusion_method == "concat":
            logits = self.concat_head(torch.cat([mammogram_feat, ultrasound_feat, masks], dim=1))
        else:
            gate_logits = self.gate(torch.cat([mammogram_feat, ultrasound_feat, masks], dim=1))
            gate_logits = gate_logits.masked_fill(masks <= 0.0, -1e4)
            gates = torch.softmax(gate_logits, dim=1)
            fused = gates[:, 0:1] * mammogram_feat + gates[:, 1:2] * ultrasound_feat
            logits = self.gated_head(fused)

        return {
            "logits": logits.view(-1, 1),
            "mammogram_available": mammogram_mask,
            "ultrasound_available": ultrasound_mask,
        }

    def _apply_modality_dropout(
        self,
        mammogram_mask: torch.Tensor,
        ultrasound_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0.0:
            return mammogram_mask, ultrasound_mask

        both_available = (mammogram_mask > 0.0) & (ultrasound_mask > 0.0)
        drop = torch.rand_like(mammogram_mask) < self.modality_dropout
        choose_mammogram = torch.rand_like(mammogram_mask) < 0.5
        drop_mammogram = both_available & drop & choose_mammogram
        drop_ultrasound = both_available & drop & (~choose_mammogram)
        mammogram_mask = mammogram_mask.masked_fill(drop_mammogram, 0.0)
        ultrasound_mask = ultrasound_mask.masked_fill(drop_ultrasound, 0.0)
        return mammogram_mask, ultrasound_mask
