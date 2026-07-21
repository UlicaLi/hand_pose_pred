from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from hand_pose.data import ExoImageTransform, camera_vector_from_pose, letterbox_exo_intrinsics
from hand_pose.model import Exo2EgoHandPoseModel


HAND_EDGES = tuple(
    (start, start + 1)
    for finger_start in (1, 5, 9, 13, 17)
    for start in range(finger_start, finger_start + 3)
) + tuple((0, finger_start) for finger_start in (1, 5, 9, 13, 17))
SLOT_COLORS = ((0, 220, 255), (255, 130, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ego 2-D hand pose from one exo image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exo-image", required=True)
    parser.add_argument("--pose-json", help="Optional dataset pose/<take>.json")
    parser.add_argument("--exo-camera", help="Required with --pose-json, e.g. cam03")
    parser.add_argument("--ego-camera", default="aria")
    parser.add_argument("--frame-idx", type=int, help="Required with --pose-json")
    parser.add_argument("--ego-width", type=int, default=448)
    parser.add_argument("--ego-height", type=int, default=448)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--visualization", help="Optional output image with the predicted ego hand skeleton")
    parser.add_argument(
        "--ego-image",
        help=(
            "Optional ego image used as visualization background. If omitted, infer the sibling "
            "<ego-camera>/<frame name> dataset path when available, otherwise use a blank canvas"
        ),
    )
    return parser.parse_args()


def resolve_ego_image(exo_image: str | Path, ego_image: str | None, ego_camera: str) -> Path | None:
    if ego_image:
        return Path(ego_image)
    exo_path = Path(exo_image)
    candidate = exo_path.parent.parent / ego_camera / exo_path.name
    return candidate if candidate.is_file() else None


def draw_pose(image: Image.Image, keypoints_norm: np.ndarray) -> Image.Image:
    if keypoints_norm.shape != (2, 21, 2):
        raise ValueError(f"Expected keypoints [2,21,2], got {keypoints_norm.shape}")
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    keypoints = keypoints_norm * np.asarray([width, height], dtype=np.float32)
    line_width = max(2, int(round(min(width, height) / 150)))
    point_radius = max(3, int(round(min(width, height) / 110)))

    for slot, color in enumerate(SLOT_COLORS):
        points = keypoints[slot]
        for start, end in HAND_EDGES:
            segment = [tuple(points[start]), tuple(points[end])]
            draw.line(segment, fill=(0, 0, 0), width=line_width + 2)
            draw.line(segment, fill=color, width=line_width)
        for x, y in points:
            box = (x - point_radius, y - point_radius, x + point_radius, y + point_radius)
            draw.ellipse(box, fill=color, outline=(0, 0, 0), width=1)

        label_y = 8 + slot * 18
        draw.rectangle((7, label_y - 1, 18, label_y + 10), fill=color, outline=(0, 0, 0))
        draw.text((23, label_y - 2), f"slot {slot} (unordered)", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def main() -> None:
    args = parse_args()
    if args.pose_json and (not args.exo_camera or args.frame_idx is None):
        raise ValueError("--pose-json also requires --exo-camera and --frame-idx")
    if args.ego_image and not args.visualization:
        raise ValueError("--ego-image requires --visualization")
    if args.ego_width <= 0 or args.ego_height <= 0:
        raise ValueError("--ego-width and --ego-height must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_cfg = config["data"]
    checkpoint_uses_camera = str(config["train"].get("stage", "rgb")) == "camera"
    image_size = int(data_cfg.get("image_size", 384))
    transform = ExoImageTransform(image_size=image_size, augment=False)
    with Image.open(args.exo_image) as image:
        image_tensor, geometry = transform(image)

    camera_vector = np.zeros(20, dtype=np.float32)
    if args.pose_json and checkpoint_uses_camera:
        with Path(args.pose_json).open("r", encoding="utf-8") as handle:
            pose_data = json.load(handle)
        camera_vector = camera_vector_from_pose(
            pose_data=pose_data,
            exo_camera_name=args.exo_camera,
            ego_camera_name=args.ego_camera,
            frame_idx=args.frame_idx,
            exo_calibration_size=tuple(data_cfg.get("exo_calibration_size", (3840, 2160))),
            ego_calibration_size=tuple(data_cfg.get("ego_calibration_size", (512, 512))),
        )
        camera_vector = letterbox_exo_intrinsics(
            camera_vector,
            original_size=(int(geometry["original_width"]), int(geometry["original_height"])),
            output_size=image_size,
            scale=float(geometry["scale"]),
            pad_left=int(geometry["pad_left"]),
            pad_top=int(geometry["pad_top"]),
        )

    model_cfg = dict(config["model"])
    model_cfg["pretrained"] = False
    device = torch.device(args.device)
    model = Exo2EgoHandPoseModel(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    camera_tensor = torch.from_numpy(camera_vector).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device), camera_tensor)
    camera_used = checkpoint_uses_camera and bool(outputs["camera_presence"][0].item() > 0)
    prediction = outputs["keypoints_full_norm"] if camera_used else outputs["keypoints_rgb_norm"]
    normalized = prediction[0].float().cpu()

    visualization_path = None
    ego_image_path = None
    ego_width, ego_height = args.ego_width, args.ego_height
    if args.visualization:
        ego_image_path = resolve_ego_image(args.exo_image, args.ego_image, args.ego_camera)
        if ego_image_path is not None:
            with Image.open(ego_image_path) as ego_image:
                canvas = ego_image.convert("RGB")
            ego_width, ego_height = canvas.size
        else:
            canvas = Image.new("RGB", (ego_width, ego_height), (32, 32, 32))
        visualization_path = Path(args.visualization)
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
        draw_pose(canvas, normalized.numpy()).save(visualization_path)

    scale = torch.tensor([ego_width, ego_height], dtype=torch.float32)
    result = {
        "exo_image": str(Path(args.exo_image).resolve()),
        "ego_image": str(ego_image_path.resolve()) if ego_image_path is not None else None,
        "camera_used": camera_used,
        "slot_semantics": "unordered",
        "ego_image_size": [ego_width, ego_height],
        "keypoints_norm": normalized.tolist(),
        "keypoints_px": (normalized * scale).tolist(),
        "visualization": str(visualization_path.resolve()) if visualization_path is not None else None,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
