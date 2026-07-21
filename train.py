from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from hand_pose.data import ExoEgoHandPoseDataset
from hand_pose.losses import compute_training_loss, teacher_confidence_weights
from hand_pose.metrics import PoseMetricAccumulator
from hand_pose.model import Exo2EgoHandPoseModel, apply_camera_modality_dropout
from hand_pose.utils import append_jsonl, atomic_torch_save, load_config, move_to_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exo RGB -> ego 2-D hand pose")
    parser.add_argument("--config", default="configs/rgb.yaml")
    parser.add_argument("--resume", help="Resume a complete training checkpoint")
    parser.add_argument("--init-checkpoint", help="Initialize model weights only (use RGB best.pt for camera stage)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", help="Override train.output_dir")
    parser.add_argument("--epochs", type=int, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size and val_batch_size")
    parser.add_argument("--num-workers", type=int, help="Override train.num_workers")
    parser.add_argument("--max-train-samples", type=int, help="Debug subset")
    parser.add_argument("--max-val-samples", type=int, help="Debug subset")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/load ImageNet ViT weights")
    return parser.parse_args()


def make_dataset(
    config: dict[str, Any],
    split: str,
    use_camera: bool,
    augment: bool,
    max_samples: int | None,
) -> ExoEgoHandPoseDataset:
    data = config["data"]
    return ExoEgoHandPoseDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=split,
        image_size=int(data.get("image_size", 384)),
        augment=augment,
        augmentation=data.get("augmentation"),
        use_camera=use_camera,
        exo_calibration_size=tuple(data.get("exo_calibration_size", (3840, 2160))),
        ego_calibration_size=tuple(data.get("ego_calibration_size", (512, 512))),
        label_cache_size=int(data.get("label_cache_size", 800)),
        max_samples=max_samples,
    )


def make_loader(dataset: ExoEgoHandPoseDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def make_optimizer(model: Exo2EgoHandPoseModel, train_cfg: dict[str, Any]) -> AdamW:
    groups: list[dict[str, Any]] = [
        {"params": list(model.backbone.parameters()), "lr": float(train_cfg["backbone_lr"]), "name": "backbone"},
        {"params": list(model.rgb_head.parameters()), "lr": float(train_cfg["rgb_head_lr"]), "name": "rgb_head"},
    ]
    if model.use_camera:
        camera_parameters = list(model.camera_encoder.parameters()) + list(model.camera_head.parameters())
        groups.append({"params": camera_parameters, "lr": float(train_cfg["camera_lr"]), "name": "camera"})
    return AdamW(groups, weight_decay=float(train_cfg.get("weight_decay", 0.05)))


def make_scheduler(
    optimizer: AdamW,
    updates_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(0, updates_per_epoch * warmup_epochs)

    def schedule(step: int) -> float:
        if warmup_updates > 0 and step < warmup_updates:
            return max(1.0e-8, (step + 1) / warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=schedule)


def load_model_state(model: Exo2EgoHandPoseModel, checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return payload


def amp_settings(train_cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    name = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    if name not in {"float16", "bfloat16"}:
        raise ValueError("train.amp_dtype must be float16 or bfloat16")
    return enabled, torch.float16 if name == "float16" else torch.bfloat16


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: Exo2EgoHandPoseModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
    device: torch.device,
    stage: str,
    loss_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    camera_dropout_cfg: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    backbone_frozen: bool,
) -> dict[str, float]:
    model.train()
    if backbone_frozen:
        model.backbone.eval()
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    totals = {"loss": 0.0, "loss_rgb": 0.0, "loss_camera": 0.0, "loss_residual": 0.0}
    sample_count = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", dynamic_ncols=True)
    for batch_index, batch in enumerate(progress):
        batch = move_to_device(batch, device)
        camera_vector = None
        if stage == "camera":
            camera_vector = apply_camera_modality_dropout(batch["camera_vector"], **camera_dropout_cfg)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(batch["image"], camera_vector)
            losses = compute_training_loss(outputs, batch, stage=stage, **loss_cfg)
            scaled_loss = losses["loss"] / accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_size = int(batch["image"].shape[0])
        sample_count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * batch_size
        progress.set_postfix(loss=f"{totals['loss'] / sample_count:.4f}")
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: Exo2EgoHandPoseModel,
    loader: DataLoader,
    device: torch.device,
    stage: str,
    loss_cfg: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    model.eval()
    metric = PoseMetricAccumulator()
    loss_sum = 0.0
    sample_count = 0
    for batch in tqdm(loader, desc="val", dynamic_ncols=True):
        batch = move_to_device(batch, device)
        camera_vector = batch["camera_vector"] if stage == "camera" else None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(batch["image"], camera_vector)
            losses = compute_training_loss(outputs, batch, stage=stage, **loss_cfg)
        prediction = outputs["keypoints_full_norm"] if stage == "camera" else outputs["keypoints_rgb_norm"]
        matching_weights = teacher_confidence_weights(
            batch["target_valid"],
            batch["target_score"],
            tau=float(loss_cfg.get("confidence_tau", 0.2)),
            gamma=float(loss_cfg.get("confidence_gamma", 1.0)),
        )
        metric.update(
            prediction.float(),
            batch["target_xy"],
            batch["target_valid"],
            batch["target_image_size"],
            matching_weights=matching_weights,
        )
        batch_size = int(batch["image"].shape[0])
        sample_count += batch_size
        loss_sum += float(losses["loss"].item()) * batch_size
    result = metric.compute()
    result["loss"] = loss_sum / max(sample_count, 1)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    stage = str(train_cfg.get("stage", "rgb"))
    if stage not in {"rgb", "camera"}:
        raise ValueError("train.stage must be rgb or camera")
    if stage == "camera" and not bool(config["model"].get("use_camera", True)):
        raise ValueError("Camera stage requires model.use_camera=true")
    if int(config["data"].get("image_size", 384)) != int(config["model"].get("image_size", 384)):
        raise ValueError("data.image_size and model.image_size must match")
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
        train_cfg["val_batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers

    seed_everything(int(config.get("seed", 12580)))
    device = torch.device(args.device)
    use_camera_data = stage == "camera"
    train_dataset = make_dataset(config, "train", use_camera_data, True, args.max_train_samples)
    val_dataset = make_dataset(config, "val", use_camera_data, False, args.max_val_samples)
    workers = int(train_cfg.get("num_workers", 8))
    train_loader = make_loader(train_dataset, int(train_cfg["batch_size"]), workers, shuffle=True)
    val_loader = make_loader(val_dataset, int(train_cfg.get("val_batch_size", train_cfg["batch_size"])), workers, shuffle=False)
    print("train dataset:", json.dumps(train_dataset.stats, ensure_ascii=False))
    print("val dataset:", json.dumps(val_dataset.stats, ensure_ascii=False))

    model_cfg = dict(config["model"])
    if args.no_pretrained or args.resume or args.init_checkpoint:
        model_cfg["pretrained"] = False
    model = Exo2EgoHandPoseModel(**model_cfg).to(device)
    if args.init_checkpoint:
        load_model_state(model, args.init_checkpoint)
        print(f"Initialized model from {args.init_checkpoint}")

    optimizer = make_optimizer(model, train_cfg)
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    epochs = int(train_cfg["epochs"])
    scheduler = make_scheduler(
        optimizer,
        updates_per_epoch=updates_per_epoch,
        epochs=epochs,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 0)),
    )
    amp_enabled, amp_dtype = amp_settings(train_cfg, device)
    scaler = make_grad_scaler(enabled=amp_enabled and amp_dtype == torch.float16)

    start_epoch = 0
    best_nme = math.inf
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = load_model_state(model, args.resume)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_nme = float(checkpoint.get("best_nme", best_nme))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    output_dir = Path(args.output_dir or train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    loss_cfg = config.get("loss", {})
    camera_dropout_cfg = config.get("camera_dropout", {})

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        backbone_frozen = epoch < freeze_epochs
        model.set_backbone_trainable(not backbone_frozen)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            stage=stage,
            loss_cfg=loss_cfg,
            train_cfg=train_cfg,
            camera_dropout_cfg=camera_dropout_cfg,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            backbone_frozen=backbone_frozen,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            stage=stage,
            loss_cfg=loss_cfg,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        current_nme = float(val_metrics["nme_diagonal"])
        is_best = current_nme < best_nme
        if current_nme < best_nme - early_stopping_min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        best_nme = min(best_nme, current_nme)
        record = {
            "epoch": epoch,
            "stage": stage,
            "backbone_frozen": backbone_frozen,
            "epochs_without_improvement": epochs_without_improvement,
            "seconds": time.time() - epoch_start,
            "lr": {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)},
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=True))

        checkpoint = {
            "epoch": epoch,
            "best_nme": best_nme,
            "epochs_without_improvement": epochs_without_improvement,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        if is_best:
            atomic_torch_save(checkpoint, output_dir / "best.pt")
            print(f"New best NME: {best_nme:.6f}")
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs without "
                f"an NME improvement greater than {early_stopping_min_delta:.6g}."
            )
            break


if __name__ == "__main__":
    main()
