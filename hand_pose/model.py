from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings

from .data import CAMERA_MASK_SLICE, CAMERA_VECTOR_SIZE


def _build_vit_b16(image_size: int, pretrained: bool, weights_name: str) -> nn.Module:
    backbone = vit_b_16(weights=None, image_size=image_size)
    if pretrained:
        try:
            weights = getattr(ViT_B_16_Weights, weights_name)
        except AttributeError as error:
            available = [weight.name for weight in ViT_B_16_Weights]
            raise ValueError(f"Unknown ViT weights {weights_name!r}; available: {available}") from error
        state = weights.get_state_dict(progress=True, check_hash=True)
        state = interpolate_embeddings(
            image_size=image_size,
            patch_size=16,
            model_state=state,
            reset_heads=True,
        )
        backbone.load_state_dict(state, strict=False)
    backbone.heads = nn.Identity()
    return backbone


class Exo2EgoHandPoseModel(nn.Module):
    """EgoWorld-style ViT + MLP model for ego 2-D hand coordinates."""

    def __init__(
        self,
        image_size: int = 384,
        pretrained: bool = True,
        weights: str = "IMAGENET1K_SWAG_E2E_V1",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_camera: bool = True,
        camera_dim: int = CAMERA_VECTOR_SIZE,
        camera_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.use_camera = bool(use_camera)
        self.camera_dim = int(camera_dim)
        self.backbone = _build_vit_b16(self.image_size, pretrained=pretrained, weights_name=weights)
        feature_dim = 768
        output_dim = 2 * 21 * 2
        self.rgb_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        if self.use_camera:
            self.camera_encoder = nn.Sequential(
                nn.Linear(self.camera_dim, camera_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(camera_hidden_dim),
                nn.Linear(camera_hidden_dim, camera_hidden_dim),
                nn.GELU(),
            )
            self.camera_head = nn.Sequential(
                nn.Linear(feature_dim + camera_hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            # Stage-2 camera fine-tuning starts exactly from the RGB prediction.
            nn.init.zeros_(self.camera_head[-1].weight)
            nn.init.zeros_(self.camera_head[-1].bias)
        else:
            self.camera_encoder = None
            self.camera_head = None

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone.requires_grad_(trainable)

    def forward(self, image: torch.Tensor, camera_vector: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        rgb_logits = self.rgb_head(features).reshape(-1, 2, 21, 2)
        rgb_prediction = torch.sigmoid(rgb_logits)

        residual_logits = torch.zeros_like(rgb_logits)
        camera_presence = torch.zeros(image.shape[0], device=image.device, dtype=rgb_logits.dtype)
        if self.use_camera and camera_vector is not None:
            if camera_vector.ndim != 2 or camera_vector.shape[1] != self.camera_dim:
                raise ValueError(
                    f"camera_vector must be [B,{self.camera_dim}], got {tuple(camera_vector.shape)}"
                )
            camera_vector = camera_vector.to(dtype=features.dtype)
            camera_features = self.camera_encoder(camera_vector)
            residual_logits = self.camera_head(torch.cat([features, camera_features], dim=-1)).reshape(-1, 2, 21, 2)
            camera_presence = camera_vector[:, CAMERA_MASK_SLICE].amax(dim=-1).to(dtype=rgb_logits.dtype)

        full_logits = rgb_logits + camera_presence[:, None, None, None] * residual_logits
        return {
            "features": features,
            "rgb_logits": rgb_logits,
            "camera_residual_logits": residual_logits,
            "keypoints_rgb_norm": rgb_prediction,
            "keypoints_full_norm": torch.sigmoid(full_logits),
            "camera_presence": camera_presence,
        }


def apply_camera_modality_dropout(
    camera_vector: torch.Tensor,
    rgb_only_probability: float = 0.5,
    intrinsics_probability: float = 0.2,
) -> torch.Tensor:
    """Apply per-sample camera modality dropout.

    Default states are 50% RGB-only, 20% RGB+intrinsics, and 30% full
    camera conditioning. Physical availability masks from the dataset are
    always respected.
    """

    if camera_vector.ndim != 2 or camera_vector.shape[1] != CAMERA_VECTOR_SIZE:
        raise ValueError(f"Expected camera_vector [B,{CAMERA_VECTOR_SIZE}]")
    if rgb_only_probability < 0 or intrinsics_probability < 0:
        raise ValueError("Camera dropout probabilities must be non-negative")
    if rgb_only_probability + intrinsics_probability > 1:
        raise ValueError("RGB-only and intrinsics probabilities must sum to <= 1")

    output = camera_vector.clone()
    draw = torch.rand(output.shape[0], device=output.device)
    rgb_only = draw < rgb_only_probability
    intrinsics_only = (draw >= rgb_only_probability) & (
        draw < rgb_only_probability + intrinsics_probability
    )
    output[rgb_only] = 0.0
    if intrinsics_only.any():
        output[intrinsics_only, 8:17] = 0.0
        output[intrinsics_only, 19] = 0.0
    return output
