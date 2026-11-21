"""Final evaluation of a trained U-Net checkpoint on a held-out split (normally the official test set).

Writes metrics.json, per_class_metrics.csv, confusion_matrix.png, predictions.png
(representative examples), and failure_cases.png (oil/look-alike confusion and missed ships).
Run this only after model selection on the validation set is finished.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES
from .data import OilSpillDataset, pair_files
from .metrics import metrics_from_confusion, update_confusion
from .unet import UNet

OIL, LOOK_ALIKE, SHIP = 1, 2, 3
PALETTE = np.asarray(CLASS_COLORS, dtype=np.uint8)


def plot_confusion(confusion: np.ndarray, path: Path, title: str) -> None:
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=float), where=row_totals > 0)
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    axis.set(xlabel="Predicted class", ylabel="True class", title=title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                      color="white" if normalized[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(image, ax=axis, label="Fraction of true-class pixels")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_examples(samples: list[dict], path: Path, title: str) -> None:
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 3.3 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["image"])
        axes[row, 0].set_title(f"SAR input: {sample['name']}", fontsize=9)
        axes[row, 1].imshow(PALETTE[sample["label"]])
        axes[row, 1].set_title("Ground truth", fontsize=9)
        axes[row, 2].imshow(PALETTE[sample["prediction"]])
        axes[row, 2].set_title(f"Prediction ({sample['note']})", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.asarray(c) / 255, ec="grey") for c in CLASS_COLORS]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=NUM_CLASSES)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Held-out image directory (official test set)")
    parser.add_argument("--masks", type=Path, required=True, help="Held-out RGB mask directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--color-tolerance", type=int, default=0)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    pairs = pair_files(args.images, args.masks)
    dataset = OilSpillDataset(pairs, state["image_size"], augment=False, color_tolerance=args.color_tolerance)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.output.mkdir(parents=True, exist_ok=True)

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    per_image: list[dict] = []
    index = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            update_confusion(confusion, labels, predictions)
            for label, prediction in zip(labels, predictions):
                oil_lookalike = int(((label == OIL) & (prediction == LOOK_ALIKE)).sum()
                                    + ((label == LOOK_ALIKE) & (prediction == OIL)).sum())
                ship_pixels = int((label == SHIP).sum())
                ship_hits = int(((label == SHIP) & (prediction == SHIP)).sum())
                per_image.append({
                    "index": index, "name": pairs[index][0].name,
                    "pixel_accuracy": float((label == prediction).float().mean()),
                    "oil_lookalike_confusion": oil_lookalike,
                    "ship_pixels": ship_pixels,
                    "ship_recall": ship_hits / ship_pixels if ship_pixels else None,
                })
                index += 1

    metrics = metrics_from_confusion(confusion)
    confusion_np = confusion.cpu().numpy()
    result = {"split": args.split_name, "checkpoint": str(args.checkpoint), "checkpoint_epoch": state["epoch"],
              "image_count": len(pairs), **metrics, "confusion_matrix": confusion_np.tolist(),
              "class_names": list(CLASS_NAMES)}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class", "iou", "dice_f1", "gt_pixels", "predicted_pixels"])
        for class_id, name in enumerate(CLASS_NAMES):
            writer.writerow([class_id, name, metrics["class_iou"][class_id], metrics["class_dice"][class_id],
                             int(confusion_np[class_id].sum()), int(confusion_np[:, class_id].sum())])
    plot_confusion(confusion_np, args.output / "confusion_matrix.png",
                   f"U-Net CE - {args.split_name} confusion (row-normalized)")

    def sample(entry: dict, note: str) -> dict:
        image, label = dataset[entry["index"]]
        with torch.inference_mode():
            prediction = model(image[None].to(device)).argmax(dim=1)[0].cpu().numpy()
        return {"image": image.permute(1, 2, 0).numpy(), "label": label.numpy(),
                "prediction": prediction, "name": entry["name"], "note": note}

    ranked = sorted(per_image, key=lambda e: e["pixel_accuracy"])
    picks = np.linspace(0, len(ranked) - 1, num=min(args.examples, len(ranked))).round().astype(int)
    representative = [sample(ranked[i], f"pixel acc {ranked[i]['pixel_accuracy']:.3f}") for i in picks]
    plot_examples(representative, args.output / "predictions.png",
                  f"Representative {args.split_name} predictions (worst to best pixel accuracy)")

    confused = sorted((e for e in per_image if e["oil_lookalike_confusion"] > 0),
                      key=lambda e: -e["oil_lookalike_confusion"])[:2]
    missed_ships = sorted((e for e in per_image if e["ship_pixels"] > 0),
                          key=lambda e: e["ship_recall"])[:2]
    failures = ([sample(e, f"oil<->look-alike px {e['oil_lookalike_confusion']}") for e in confused]
                + [sample(e, f"ship recall {e['ship_recall']:.2f}") for e in missed_ships])
    plot_examples(failures, args.output / "failure_cases.png",
                  "Failure cases: oil/look-alike confusion and missed ships")
    (args.output / "per_image.json").write_text(json.dumps(per_image, indent=2), encoding="utf-8")

    print(f"{args.split_name}: {len(pairs)} images | mIoU {metrics['miou']:.4f} | "
          f"mean Dice {metrics['mean_dice']:.4f} | pixel accuracy {metrics['pixel_accuracy']:.4f}")
    for name, iou, dice in zip(CLASS_NAMES, metrics["class_iou"], metrics["class_dice"]):
        fmt = lambda v: "  n/a " if v is None else f"{v:.4f}"
        print(f"  {name:12s} IoU {fmt(iou)}  Dice {fmt(dice)}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
