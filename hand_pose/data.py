from __future__ import annotations

import json
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CAMERA_VECTOR_SIZE = 20
CAMERA_MASK_SLICE = slice(17, 20)


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _as_homogeneous(extrinsics: Any) -> np.ndarray | None:
    if extrinsics is None or isinstance(extrinsics, dict):
        return None
    matrix = np.asarray(extrinsics, dtype=np.float32)
    if matrix.shape == (3, 4):
        matrix = np.concatenate(
            [matrix, np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)],
            axis=0,
        )
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def _frame_extrinsics(camera: dict[str, Any] | None, frame_idx: int) -> np.ndarray | None:
    if not camera:
        return None
    extrinsics = camera.get("camera_extrinsics")
    if isinstance(extrinsics, dict):
        extrinsics = extrinsics.get(str(int(frame_idx)), extrinsics.get(int(frame_idx)))
    return _as_homogeneous(extrinsics)


def _intrinsics(camera: dict[str, Any] | None) -> np.ndarray | None:
    if not camera:
        return None
    matrix = np.asarray(camera.get("camera_intrinsics"), dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    return matrix


def resolve_ego_camera_name(pose_data: dict[str, Any], requested: str = "aria") -> str | None:
    if requested in pose_data:
        return requested
    matches = sorted(key for key in pose_data if key.startswith(requested.rstrip("*")))
    return matches[0] if matches else None


def camera_vector_from_pose(
    pose_data: dict[str, Any],
    exo_camera_name: str,
    ego_camera_name: str,
    frame_idx: int,
    exo_calibration_size: tuple[int, int] = (3840, 2160),
    ego_calibration_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Build the 20-D camera vector before exo-image letterboxing.

    Layout: exo K (4), ego K (4), relative R6D (6), relative t (3),
    and availability flags for exo K / ego K / relative pose (3).

    Pose files in this dataset use the same convention as the existing
    Exo2Ego code: the relative matrix is inv(T_ego) @ T_exo.
    """

    vector = np.zeros(CAMERA_VECTOR_SIZE, dtype=np.float32)
    ego_name = resolve_ego_camera_name(pose_data, ego_camera_name)
    exo_camera = pose_data.get(exo_camera_name)
    ego_camera = pose_data.get(ego_name) if ego_name else None

    exo_k = _intrinsics(exo_camera)
    if exo_k is not None:
        width, height = exo_calibration_size
        vector[0:4] = [
            exo_k[0, 0] / width,
            exo_k[1, 1] / height,
            exo_k[0, 2] / width,
            exo_k[1, 2] / height,
        ]
        vector[17] = 1.0

    ego_k = _intrinsics(ego_camera)
    if ego_k is not None:
        width, height = ego_calibration_size
        vector[4:8] = [
            ego_k[0, 0] / width,
            ego_k[1, 1] / height,
            ego_k[0, 2] / width,
            ego_k[1, 2] / height,
        ]
        vector[18] = 1.0

    exo_pose = _frame_extrinsics(exo_camera, frame_idx)
    ego_pose = _frame_extrinsics(ego_camera, frame_idx)
    if exo_pose is not None and ego_pose is not None:
        try:
            relative = np.linalg.inv(ego_pose) @ exo_pose
        except np.linalg.LinAlgError:
            relative = None
        if relative is not None and np.isfinite(relative).all():
            rotation = relative[:3, :3]
            # Zhou et al. 6-D rotation representation: first two columns.
            vector[8:14] = rotation[:, :2].T.reshape(-1)
            vector[14:17] = relative[:3, 3]
            vector[19] = 1.0

    return vector


def letterbox_exo_intrinsics(
    vector: np.ndarray,
    original_size: tuple[int, int],
    output_size: int,
    scale: float,
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    vector = vector.copy()
    if vector[17] <= 0:
        return vector
    width, height = original_size
    vector[0] = vector[0] * width * scale / output_size
    vector[1] = vector[1] * height * scale / output_size
    vector[2] = (vector[2] * width * scale + pad_left) / output_size
    vector[3] = (vector[3] * height * scale + pad_top) / output_size
    return vector


class ExoImageTransform:
    """Aspect-preserving exo transform with non-geometric augmentation only."""

    def __init__(self, image_size: int = 384, augment: bool = False, cfg: dict[str, Any] | None = None):
        if image_size <= 0 or image_size % 16 != 0:
            raise ValueError("image_size must be a positive multiple of the ViT patch size (16)")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.cfg = cfg or {}
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def _photometric(self, image: Image.Image) -> Image.Image:
        jitter_probability = float(self.cfg.get("color_jitter_probability", 0.8))
        if random.random() < jitter_probability:
            brightness = float(self.cfg.get("brightness", 0.2))
            contrast = float(self.cfg.get("contrast", 0.2))
            saturation = float(self.cfg.get("saturation", 0.15))
            operations = [
                (ImageEnhance.Brightness, random.uniform(1.0 - brightness, 1.0 + brightness)),
                (ImageEnhance.Contrast, random.uniform(1.0 - contrast, 1.0 + contrast)),
                (ImageEnhance.Color, random.uniform(1.0 - saturation, 1.0 + saturation)),
            ]
            random.shuffle(operations)
            for enhancer, factor in operations:
                image = enhancer(image).enhance(factor)
        if random.random() < float(self.cfg.get("blur_probability", 0.1)):
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))
        return image

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, dict[str, float | int]]:
        image = image.convert("RGB")
        if self.augment:
            image = self._photometric(image)

        original_width, original_height = image.size
        scale = min(self.image_size / original_width, self.image_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        pad_left = (self.image_size - resized_width) // 2
        pad_top = (self.image_size - resized_height) // 2
        fill = tuple(int(round(value * 255.0)) for value in IMAGENET_MEAN)
        canvas = Image.new("RGB", (self.image_size, self.image_size), fill)
        canvas.paste(resized, (pad_left, pad_top))

        array = np.array(canvas, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.augment and random.random() < float(self.cfg.get("noise_probability", 0.15)):
            sigma = random.uniform(0.0, float(self.cfg.get("noise_std", 0.02)))
            tensor = (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)
        tensor = (tensor - self.mean) / self.std

        if self.augment and random.random() < float(self.cfg.get("erase_probability", 0.1)):
            area = self.image_size * self.image_size
            erase_area = random.uniform(0.02, float(self.cfg.get("erase_max_area", 0.08))) * area
            aspect = random.uniform(0.5, 2.0)
            erase_height = min(self.image_size, max(1, int(round(math.sqrt(erase_area / aspect)))))
            erase_width = min(self.image_size, max(1, int(round(math.sqrt(erase_area * aspect)))))
            top = random.randint(0, self.image_size - erase_height)
            left = random.randint(0, self.image_size - erase_width)
            tensor[:, top : top + erase_height, left : left + erase_width] = 0.0

        geometry: dict[str, float | int] = {
            "original_width": original_width,
            "original_height": original_height,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }
        return tensor, geometry


class ExoEgoHandPoseDataset(Dataset):
    """Single-frame exo RGB -> ego 2-D hand pose dataset.

    Video windows in ``exo2ego_manifest.json`` are flattened and de-duplicated.
    Frames without the required exo image or teacher labels are filtered.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        split: str,
        image_size: int = 384,
        augment: bool = False,
        augmentation: dict[str, Any] | None = None,
        use_camera: bool = False,
        exo_calibration_size: tuple[int, int] = (3840, 2160),
        ego_calibration_size: tuple[int, int] = (512, 512),
        label_cache_size: int = 800,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.use_camera = bool(use_camera)
        self.exo_calibration_size = tuple(int(x) for x in exo_calibration_size)
        self.ego_calibration_size = tuple(int(x) for x in ego_calibration_size)
        self.label_cache_size = max(1, int(label_cache_size))
        self.transform = ExoImageTransform(image_size=image_size, augment=augment, cfg=augmentation)
        self._label_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._zero_camera_vector = np.zeros(CAMERA_VECTOR_SIZE, dtype=np.float32)

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if split not in manifest.get("splits", {}):
            raise KeyError(f"Unknown split {split!r}; available splits: {list(manifest.get('splits', {}))}")

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        skipped_missing_exo = 0
        skipped_missing_label_file = 0
        for window in manifest["splits"][split]:
            take_name = str(window["take_name"])
            label_path = self.dataset_root / "hand_pose" / "takes" / f"{take_name}.npz"
            for frame in window.get("frames", []):
                exo_path = resolve_path(self.dataset_root, frame.get("exo_path"))
                if exo_path is None or not exo_path.is_file():
                    skipped_missing_exo += 1
                    continue
                if not label_path.is_file():
                    skipped_missing_label_file += 1
                    continue
                frame_idx = int(frame["frame_idx"])
                key = (take_name, frame_idx, str(exo_path))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "take_name": take_name,
                        "frame_idx": frame_idx,
                        "exo_path": exo_path,
                        "cam_name": str(frame.get("cam_name") or ""),
                        "ego_camera_name": str(window.get("ego_camera_name") or "aria"),
                        "pose_path": resolve_path(self.dataset_root, window.get("pose_path")),
                        "label_path": label_path,
                    }
                )

        valid_frames_by_take: dict[str, set[int]] = {}
        for record in candidates:
            take_name = record["take_name"]
            if take_name not in valid_frames_by_take:
                with np.load(record["label_path"], allow_pickle=False) as labels:
                    valid_frames_by_take[take_name] = {int(x) for x in labels["frame_idx"]}
        records = [record for record in candidates if record["frame_idx"] in valid_frames_by_take[record["take_name"]]]
        skipped_missing_label_frame = len(candidates) - len(records)
        if max_samples is not None:
            records = records[: int(max_samples)]
        if not records:
            raise RuntimeError(f"No usable samples for split {split!r}")

        if self.use_camera:
            self._precompute_camera_vectors(records)
        self.records = records
        self.stats = {
            "split": split,
            "samples": len(records),
            "skipped_missing_exo": skipped_missing_exo,
            "skipped_missing_label_file": skipped_missing_label_file,
            "skipped_missing_label_frame": skipped_missing_label_frame,
            "camera_enabled": self.use_camera,
        }

    def _precompute_camera_vectors(self, records: list[dict[str, Any]]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["take_name"]].append(record)
        for take_records in grouped.values():
            pose_path = take_records[0]["pose_path"]
            pose_data: dict[str, Any] = {}
            if pose_path is not None and pose_path.is_file():
                with pose_path.open("r", encoding="utf-8") as handle:
                    pose_data = json.load(handle)
            for record in take_records:
                record["camera_vector"] = camera_vector_from_pose(
                    pose_data=pose_data,
                    exo_camera_name=record["cam_name"],
                    ego_camera_name=record["ego_camera_name"],
                    frame_idx=record["frame_idx"],
                    exo_calibration_size=self.exo_calibration_size,
                    ego_calibration_size=self.ego_calibration_size,
                )

    def _load_take_labels(self, take_name: str, path: Path) -> dict[str, Any]:
        cached = self._label_cache.pop(take_name, None)
        if cached is not None:
            self._label_cache[take_name] = cached
            return cached
        with np.load(path, allow_pickle=False) as labels:
            frame_indices = np.asarray(labels["frame_idx"], dtype=np.int64)
            data = {
                "frame_to_row": {int(frame): row for row, frame in enumerate(frame_indices)},
                "keypoints_norm": np.asarray(labels["keypoints_norm"], dtype=np.float32),
                "keypoint_scores": np.asarray(labels["keypoint_scores"], dtype=np.float32),
                "keypoint_valid": np.asarray(labels["keypoint_valid"], dtype=bool),
                "hand_valid": np.asarray(labels["hand_valid"], dtype=bool),
                "image_size": np.asarray(labels["image_size"], dtype=np.float32),
            }
        self._label_cache[take_name] = data
        while len(self._label_cache) > self.label_cache_size:
            self._label_cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record["exo_path"]) as image:
            image_tensor, geometry = self.transform(image)

        labels = self._load_take_labels(record["take_name"], record["label_path"])
        row = labels["frame_to_row"][record["frame_idx"]]
        target_xy = labels["keypoints_norm"][row].copy()
        target_score = labels["keypoint_scores"][row].copy()
        keypoint_valid = labels["keypoint_valid"][row].copy()
        hand_valid = labels["hand_valid"][row].copy()
        finite = np.isfinite(target_xy).all(axis=-1) & np.isfinite(target_score)
        in_bounds = ((target_xy >= 0.0) & (target_xy <= 1.0)).all(axis=-1)
        target_valid = keypoint_valid & hand_valid[:, None] & finite & in_bounds
        target_xy[~target_valid] = 0.0
        target_score = np.where(target_valid, target_score, 0.0).astype(np.float32)

        image_size = labels["image_size"].reshape(-1)
        if image_size.size == 1:
            image_size = np.repeat(image_size, 2)
        if image_size.size < 2:
            image_size = np.asarray([448.0, 448.0], dtype=np.float32)

        camera_vector = letterbox_exo_intrinsics(
            record.get("camera_vector", self._zero_camera_vector),
            original_size=(int(geometry["original_width"]), int(geometry["original_height"])),
            output_size=self.transform.image_size,
            scale=float(geometry["scale"]),
            pad_left=int(geometry["pad_left"]),
            pad_top=int(geometry["pad_top"]),
        )
        return {
            "image": image_tensor,
            "target_xy": torch.from_numpy(target_xy),
            "target_valid": torch.from_numpy(target_valid),
            "target_score": torch.from_numpy(target_score),
            "hand_valid": torch.from_numpy(hand_valid),
            "target_image_size": torch.from_numpy(image_size[:2].copy()),
            "camera_vector": torch.from_numpy(camera_vector),
            "take_name": record["take_name"],
            "cam_name": record["cam_name"],
            "frame_idx": torch.tensor(record["frame_idx"], dtype=torch.long),
            "exo_path": str(record["exo_path"]),
        }
