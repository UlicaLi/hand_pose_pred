from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hand_pose.data import ExoEgoHandPoseDataset
from hand_pose.losses import teacher_confidence_weights
from hand_pose.metrics import PoseMetricAccumulator
from hand_pose.model import Exo2EgoHandPoseModel
from hand_pose.utils import load_config, move_to_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exo RGB -> ego 2-D hand pose")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional YAML override; by default use checkpoint config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--rgb-only", action="store_true", help="Evaluate RGB head even for a camera-stage checkpoint")
    parser.add_argument("--predictions", help="Optional raw prediction JSONL output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: dict[str, Any] = load_config(args.config) if args.config else checkpoint["config"]
    seed_everything(int(config.get("seed", 12580)))
    checkpoint_stage = str(config["train"].get("stage", "rgb"))
    use_camera = checkpoint_stage == "camera" and not args.rgb_only
    data = config["data"]
    dataset = ExoEgoHandPoseDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=args.split,
        image_size=int(data.get("image_size", 384)),
        augment=False,
        use_camera=use_camera,
        exo_calibration_size=tuple(data.get("exo_calibration_size", (3840, 2160))),
        ego_calibration_size=tuple(data.get("ego_calibration_size", (512, 512))),
        label_cache_size=int(data.get("label_cache_size", 800)),
        max_samples=args.max_samples,
    )
    train_cfg = config["train"]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(train_cfg.get("val_batch_size", 16)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 8)),
        pin_memory=torch.cuda.is_available(),
    )
    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    device = torch.device(args.device)
    model = Exo2EgoHandPoseModel(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    metrics = {
        "all": PoseMetricAccumulator(),
        "score_ge_0.5": PoseMetricAccumulator(),
        "score_ge_0.7": PoseMetricAccumulator(),
    }
    output_handle = None
    if args.predictions:
        prediction_path = Path(args.predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = prediction_path.open("w", encoding="utf-8")
    loss_cfg = config.get("loss", {})

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=args.split, dynamic_ncols=True):
                batch = move_to_device(batch, device)
                outputs = model(batch["image"], batch["camera_vector"] if use_camera else None)
                prediction = outputs["keypoints_full_norm"] if use_camera else outputs["keypoints_rgb_norm"]
                prediction = prediction.float()
                confidence = teacher_confidence_weights(
                    batch["target_valid"],
                    batch["target_score"],
                    tau=float(loss_cfg.get("confidence_tau", 0.2)),
                    gamma=float(loss_cfg.get("confidence_gamma", 1.0)),
                )
                validity = {
                    "all": batch["target_valid"],
                    "score_ge_0.5": batch["target_valid"] & (batch["target_score"] >= 0.5),
                    "score_ge_0.7": batch["target_valid"] & (batch["target_score"] >= 0.7),
                }
                for name, accumulator in metrics.items():
                    accumulator.update(
                        prediction,
                        batch["target_xy"],
                        validity[name],
                        batch["target_image_size"],
                        matching_weights=confidence * validity[name].float(),
                    )

                if output_handle is not None:
                    pixel_prediction = prediction * batch["target_image_size"][:, None, None, :]
                    for index in range(prediction.shape[0]):
                        row = {
                            "take_name": batch["take_name"][index],
                            "frame_idx": int(batch["frame_idx"][index].item()),
                            "cam_name": batch["cam_name"][index],
                            "exo_path": batch["exo_path"][index],
                            "keypoints_norm": prediction[index].cpu().tolist(),
                            "keypoints_px": pixel_prediction[index].cpu().tolist(),
                        }
                        output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mode": "camera" if use_camera else "rgb",
        "dataset": dataset.stats,
        "metrics": {name: accumulator.compute() for name, accumulator in metrics.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
