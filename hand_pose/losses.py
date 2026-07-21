from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def teacher_confidence_weights(
    target_valid: torch.Tensor,
    target_score: torch.Tensor,
    tau: float = 0.2,
    gamma: float = 1.0,
) -> torch.Tensor:
    confidence = ((target_score - tau) / max(1.0 - tau, 1.0e-6)).clamp(0.0, 1.0).pow(gamma)
    return target_valid.to(dtype=target_score.dtype) * confidence


def _assignment_numerators(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    def point_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, gt, reduction="none", beta=beta).mean(dim=-1)

    error_00 = point_error(prediction[:, 0], target[:, 0])
    error_11 = point_error(prediction[:, 1], target[:, 1])
    error_01 = point_error(prediction[:, 0], target[:, 1])
    error_10 = point_error(prediction[:, 1], target[:, 0])
    identity = (error_00 * weights[:, 0] + error_11 * weights[:, 1]).sum(dim=-1)
    swapped = (error_01 * weights[:, 1] + error_10 * weights[:, 0]).sum(dim=-1)
    return identity, swapped


def permutation_invariant_pose_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 0.01,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Confidence-weighted SmoothL1 after exact matching of two hand slots."""

    if prediction.shape != target.shape or prediction.shape[1:] != (2, 21, 2):
        raise ValueError(f"Expected prediction and target [B,2,21,2], got {prediction.shape} and {target.shape}")
    if weights.shape != target.shape[:-1]:
        raise ValueError(f"Expected weights [B,2,21], got {weights.shape}")
    if sample_mask is not None:
        weights = weights * sample_mask.to(weights.dtype)[:, None, None]

    identity, swapped = _assignment_numerators(prediction, target, weights, beta=beta)
    use_swap = swapped < identity
    denominator = weights.sum(dim=(1, 2))
    per_sample = torch.minimum(identity, swapped) / denominator.clamp_min(1.0e-8)
    supervised = denominator > 0
    if supervised.any():
        loss = per_sample[supervised].mean()
    else:
        loss = prediction.sum() * 0.0
    matched_prediction = torch.where(
        use_swap[:, None, None, None],
        prediction.flip(dims=(1,)),
        prediction,
    )
    return {
        "loss": loss,
        "per_sample": per_sample,
        "use_swap": use_swap,
        "matched_prediction": matched_prediction,
        "num_supervised_samples": supervised.sum(),
    }


def compute_training_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    stage: str,
    confidence_tau: float = 0.2,
    confidence_gamma: float = 1.0,
    smooth_l1_beta: float = 0.01,
    lambda_camera: float = 1.0,
    lambda_residual: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    target = batch["target_xy"]
    weights = teacher_confidence_weights(
        batch["target_valid"],
        batch["target_score"],
        tau=confidence_tau,
        gamma=confidence_gamma,
    )
    rgb = permutation_invariant_pose_loss(
        outputs["keypoints_rgb_norm"], target, weights, beta=smooth_l1_beta
    )
    zero = rgb["loss"].new_zeros(())
    camera_loss = zero
    residual_loss = zero
    if stage == "camera":
        camera = permutation_invariant_pose_loss(
            outputs["keypoints_full_norm"],
            target,
            weights,
            beta=smooth_l1_beta,
            sample_mask=outputs["camera_presence"] > 0,
        )
        camera_loss = camera["loss"]
        residual_loss = outputs["camera_residual_logits"].square().mean()
    elif stage != "rgb":
        raise ValueError(f"Unknown training stage {stage!r}; expected 'rgb' or 'camera'")

    total = rgb["loss"] + lambda_camera * camera_loss + lambda_residual * residual_loss
    return {
        "loss": total,
        "loss_rgb": rgb["loss"],
        "loss_camera": camera_loss,
        "loss_residual": residual_loss,
        "num_supervised_samples": rgb["num_supervised_samples"],
    }
