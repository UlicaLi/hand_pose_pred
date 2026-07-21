from __future__ import annotations

import math

import torch


@torch.no_grad()
def match_pose_slots(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match two predicted slots to two teacher slots by weighted 2-D error."""

    error_00 = torch.linalg.vector_norm(prediction[:, 0] - target[:, 0], dim=-1)
    error_11 = torch.linalg.vector_norm(prediction[:, 1] - target[:, 1], dim=-1)
    error_01 = torch.linalg.vector_norm(prediction[:, 0] - target[:, 1], dim=-1)
    error_10 = torch.linalg.vector_norm(prediction[:, 1] - target[:, 0], dim=-1)
    identity = (error_00 * weights[:, 0] + error_11 * weights[:, 1]).sum(dim=-1)
    swapped = (error_01 * weights[:, 1] + error_10 * weights[:, 0]).sum(dim=-1)
    use_swap = swapped < identity
    matched = torch.where(use_swap[:, None, None, None], prediction.flip(1), prediction)
    return matched, use_swap


class PoseMetricAccumulator:
    GROUPS = {
        "wrist": (0,),
        "palm": (0, 1, 5, 9, 13, 17),
        "fingertip": (4, 8, 12, 16, 20),
    }

    def __init__(self) -> None:
        self.count = 0
        self.pixel_error_sum = 0.0
        self.normalized_error_sum = 0.0
        self.pck05_count = 0
        self.pck10_count = 0
        self.auc10_sum = 0.0
        self.group_count = {name: 0 for name in self.GROUPS}
        self.group_pixel_error_sum = {name: 0.0 for name in self.GROUPS}

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        image_size: torch.Tensor,
        matching_weights: torch.Tensor | None = None,
    ) -> None:
        matching_weights = valid.float() if matching_weights is None else matching_weights
        prediction, _ = match_pose_slots(prediction, target, matching_weights)
        scale = image_size[:, None, None, :].to(prediction)
        pixel_error = torch.linalg.vector_norm((prediction - target) * scale, dim=-1)
        diagonal = torch.linalg.vector_norm(image_size.to(prediction), dim=-1)[:, None, None]
        normalized_error = pixel_error / diagonal.clamp_min(1.0e-8)

        selected_pixel = pixel_error[valid]
        selected_normalized = normalized_error[valid]
        self.count += int(valid.sum().item())
        self.pixel_error_sum += float(selected_pixel.sum().item())
        self.normalized_error_sum += float(selected_normalized.sum().item())
        self.pck05_count += int((selected_normalized <= 0.05).sum().item())
        self.pck10_count += int((selected_normalized <= 0.10).sum().item())
        self.auc10_sum += float((1.0 - selected_normalized / 0.10).clamp(0.0, 1.0).sum().item())

        for name, indices in self.GROUPS.items():
            group_valid = valid[:, :, indices]
            group_error = pixel_error[:, :, indices]
            self.group_count[name] += int(group_valid.sum().item())
            self.group_pixel_error_sum[name] += float(group_error[group_valid].sum().item())

    def compute(self) -> dict[str, float | int]:
        denominator = max(self.count, 1)
        output: dict[str, float | int] = {
            "valid_joints": self.count,
            "mpe_px": self.pixel_error_sum / denominator,
            "nme_diagonal": self.normalized_error_sum / denominator,
            "pck_0.05": self.pck05_count / denominator,
            "pck_0.10": self.pck10_count / denominator,
            "auc_0.10": self.auc10_sum / denominator,
        }
        if self.count == 0:
            for key in ("mpe_px", "nme_diagonal", "pck_0.05", "pck_0.10", "auc_0.10"):
                output[key] = math.nan
        for name in self.GROUPS:
            count = self.group_count[name]
            output[f"mpe_px_{name}"] = (
                self.group_pixel_error_sum[name] / count if count > 0 else math.nan
            )
        return output
