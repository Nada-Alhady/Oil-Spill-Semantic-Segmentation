"""Dataset pairing, RGB-mask conversion, preprocessing, and paired augmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import CLASS_COLORS, NUM_CLASSES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_COLOR_TO_ID = {color: index for index, color in enumerate(CLASS_COLORS)}


def pair_files(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    """Pair images and masks by relative path and filename stem."""
    image_root, mask_root = Path(image_dir), Path(mask_dir)
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Expected image and mask directories: {image_root}, {mask_root}")

    def indexed_files(root: Path) -> dict[str, Path]:
        indexed: dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                key = path.relative_to(root).with_suffix("").as_posix().casefold()
                if key in indexed:
                    raise ValueError(f"Duplicate file stem under {root}: {key}")
                indexed[key] = path
        return indexed

    images, masks = indexed_files(image_root), indexed_files(mask_root)
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Image/mask file mismatch. Missing masks for {missing_masks[:5]}; "
            f"missing images for {missing_images[:5]}"
        )
    if not images:
        raise ValueError(f"No supported image files found in {image_root}")
    return [(images[key], masks[key]) for key in sorted(images)]


def rgb_mask_to_labels(
    mask: np.ndarray, path: str | Path = "<mask>", tolerance: int = 0
) -> np.ndarray:
    """Convert an RGB color mask into class IDs.

    tolerance == 0: only exact palette colors are accepted (anything else raises).
    tolerance  > 0: a pixel is snapped to its nearest palette color if every channel is
                    within ``tolerance`` of it; otherwise it raises.
    tolerance  < 0: every pixel is snapped to the nearest palette color (Euclidean RGB
                    distance). Use this when masks contain blended edge colors from
                    anti-aliasing or lossy saving, e.g. (182, 15, 15) between red and black.
    """
    if mask.ndim != 3 or mask.shape[2] < 3:
        raise ValueError(f"Expected an RGB mask at {path}; got shape {mask.shape}")
    rgb = np.asarray(mask[..., :3], dtype=np.uint8)
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    if tolerance == 0:
        for color, class_id in _COLOR_TO_ID.items():
            labels[np.all(rgb == color, axis=-1)] = class_id
    else:
        palette = np.asarray(CLASS_COLORS, dtype=np.int32)
        diff = rgb.astype(np.int32)[..., None, :] - palette
        nearest = (diff**2).sum(axis=-1).argmin(axis=-1)
        if tolerance > 0:
            worst_channel = np.abs(diff).max(axis=-1)
            within = np.take_along_axis(worst_channel, nearest[..., None], axis=-1)[..., 0] <= tolerance
        else:
            within = np.ones(nearest.shape, dtype=bool)
        labels[within] = nearest[within].astype(np.uint8)
    unknown = np.unique(rgb[labels == 255], axis=0)
    if unknown.size:
        examples = [tuple(int(channel) for channel in color) for color in unknown[:8]]
        raise ValueError(
            f"Unrecognized RGB colors in {path}: {examples}. Check the dataset's exact mask colors."
        )
    return labels


def split_pairs(
    pairs: Sequence[tuple[Path, Path]], validation_fraction: float = 0.15, seed: int = 42
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Create deterministic train/validation subsets from official training files."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required for a train/validation split")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(len(pairs) - 1, round(len(pairs) * validation_fraction)))
    validation_ids = set(indices[:validation_count])
    train = [pair for index, pair in enumerate(pairs) if index not in validation_ids]
    validation = [pair for index, pair in enumerate(pairs) if index in validation_ids]
    return train, validation


class OilSpillDataset(Dataset):
    """Load aligned image/mask pairs and return normalized image and integer labels."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        image_size: int = 256,
        augment: bool = False,
        color_tolerance: int = 0,
    ) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16 for the four-level U-Net")
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment
        self.color_tolerance = color_tolerance

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("RGB")
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask dimensions differ for {image_path.name}: {image.size} vs {mask.size}"
            )

        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        label_array = rgb_mask_to_labels(np.asarray(mask), mask_path, self.color_tolerance)

        image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1)
        label_tensor = torch.from_numpy(label_array.astype(np.int64, copy=False))
        if self.augment:
            if random.random() < 0.5:
                image_tensor = image_tensor.flip(-1)
                label_tensor = label_tensor.flip(-1)
            if random.random() < 0.5:
                image_tensor = image_tensor.flip(-2)
                label_tensor = label_tensor.flip(-2)
            turns = random.randrange(4)
            if turns:
                image_tensor = torch.rot90(image_tensor, turns, (-2, -1))
                label_tensor = torch.rot90(label_tensor, turns, (-2, -1))
        return image_tensor, label_tensor


def class_pixel_counts(
    pairs: Sequence[tuple[Path, Path]], color_tolerance: int = 0
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Count mask pixels by class and return original dimensions per pair."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    dimensions: dict[tuple[int, int], int] = {}
    for image_path, mask_path in pairs:
        with Image.open(image_path) as source:
            image_size = source.size
        with Image.open(mask_path) as source:
            mask = source.convert("RGB")
        if image_size != mask.size:
            raise ValueError(
                f"Image/mask dimensions differ for {image_path.name}: {image_size} vs {mask.size}"
            )
        dimensions[mask.size] = dimensions.get(mask.size, 0) + 1
        labels = rgb_mask_to_labels(np.asarray(mask), mask_path, color_tolerance)
        counts += np.bincount(labels.reshape(-1), minlength=NUM_CLASSES)
    return counts, dimensions


def find_unknown_colors(pairs: Sequence[tuple[Path, Path]], limit: int = 20) -> dict[tuple[int, int, int], int]:
    """Return non-palette mask colors and how many pixels use them (for dataset inspection)."""
    palette = set(CLASS_COLORS)
    found: dict[tuple[int, int, int], int] = {}
    for _, mask_path in pairs:
        with Image.open(mask_path) as source:
            rgb = np.asarray(source.convert("RGB")).reshape(-1, 3)
        colors, counts = np.unique(rgb, axis=0, return_counts=True)
        for color, count in zip(colors, counts):
            key = tuple(int(channel) for channel in color)
            if key not in palette:
                found[key] = found.get(key, 0) + int(count)
    return dict(sorted(found.items(), key=lambda item: -item[1])[:limit])
