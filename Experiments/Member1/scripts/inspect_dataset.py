"""Inspect paired samples, verify mask colors, summarize class balance, and save previews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilspill import CLASS_COLORS, CLASS_NAMES
from oilspill.data import class_pixel_counts, find_unknown_colors, pair_files, rgb_mask_to_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview.png"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--color-tolerance", type=int, default=0,
                        help="0 = exact palette only; N>0 = snap colors within N per channel; -1 = snap every pixel to the nearest class color")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")

    pairs = pair_files(args.images, args.masks)
    unknown = find_unknown_colors(pairs)
    if unknown:
        print("WARNING: mask colors outside the assignment palette (color: pixel count):")
        for color, count in unknown.items():
            print(f"  {color}: {count:,d}")
        if args.color_tolerance == 0:
            print("These look like blended/compressed edge colors. Use --color-tolerance -1 to snap each "
                  "pixel to the nearest class color (the training/evaluation scripts need the same flag).")
    else:
        print("All mask pixels use the exact assignment palette.")
    counts, dimensions = class_pixel_counts(pairs, args.color_tolerance)
    total = int(counts.sum())
    print(f"Validated {len(pairs)} image/mask pairs")
    print(f"Mask dimensions and image counts: {dimensions}")
    print("Class pixel distribution:")
    for class_id, (name, color, count) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, counts)):
        fraction = 100.0 * int(count) / max(total, 1)
        print(f"  {class_id}: {name:12s} RGB={color} pixels={int(count):>12,d} ({fraction:6.3f}%)")

    preview_pairs = pairs[: min(args.samples, len(pairs))]
    fig, axes = plt.subplots(len(preview_pairs), 2, figsize=(8, 3.2 * len(preview_pairs)), squeeze=False)
    palette = np.asarray(CLASS_COLORS, dtype=np.uint8)
    for row, (image_path, mask_path) in enumerate(preview_pairs):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            raw_mask = np.asarray(source.convert("RGB"))
        labels = rgb_mask_to_labels(raw_mask, mask_path, args.color_tolerance)
        color_mask = palette[labels]
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image: {image_path.name} ({image.width}x{image.height})")
        axes[row, 1].imshow(color_mask)
        axes[row, 1].set_title("Ground-truth mask")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Preview saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
