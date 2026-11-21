"""Train U-Net Experiment 1 with unweighted pixel-wise cross-entropy loss."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES
from .data import OilSpillDataset, pair_files, split_pairs
from .metrics import metrics_from_confusion, update_confusion
from .unet import UNet

HISTORY_FIELDS = [
    "epoch", "train_loss", "train_pixel_accuracy", "train_miou", "train_mean_dice",
    "val_loss", "val_pixel_accuracy", "val_miou", "val_mean_dice",
    *[f"val_iou_{name}" for name in CLASS_NAMES],
    "learning_rate", "seconds",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_pixels = 0.0, 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_pixels += labels.numel()
            update_confusion(confusion, labels, logits.argmax(dim=1))
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(total_pixels, 1)
    return metrics


def write_history(history: list[dict[str, object]], output_dir: Path, best_epoch: int) -> None:
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in HISTORY_FIELDS})
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["train_miou"] for row in history], label="Train mIoU")
    axes[1].plot(epochs, [row["val_miou"] for row in history], label="Validation mIoU")
    axes[1].plot(epochs, [row["train_mean_dice"] for row in history], "--", label="Train mean Dice")
    axes[1].plot(epochs, [row["val_mean_dice"] for row in history], "--", label="Validation mean Dice")
    axes[1].set(title="Class-sensitive metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for name in CLASS_NAMES:
        values = [np.nan if row[f"val_iou_{name}"] is None else row[f"val_iou_{name}"] for row in history]
        axes[2].plot(epochs, values, label=name)
    axes[2].set(title="Validation IoU per class", xlabel="Epoch", ylabel="IoU", ylim=(0, 1))
    for axis in axes:
        axis.axvline(best_epoch, color="grey", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"U-Net Exp. 1 (cross-entropy) - best epoch {best_epoch} by validation mIoU")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=Path, required=True, help="Official training image directory")
    parser.add_argument("--train-masks", type=Path, required=True, help="Official training mask directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/unet_ce"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--color-tolerance", type=int, default=0,
                        help="0 = exact palette; N>0 = snap within N per channel; -1 = snap to nearest class color")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <output>/last_checkpoint.pt if it exists")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")

    seed_everything(args.seed)
    pairs = pair_files(args.train_images, args.train_masks)
    train_pairs, validation_pairs = split_pairs(pairs, args.validation_fraction, args.seed)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                     "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        OilSpillDataset(train_pairs, args.image_size, augment=True, color_tolerance=args.color_tolerance),
        shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kwargs,
    )
    validation_loader = DataLoader(
        OilSpillDataset(validation_pairs, args.image_size, augment=False, color_tolerance=args.color_tolerance),
        shuffle=False, **loader_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "experiment": "U-Net from scratch - unweighted cross-entropy",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss (unweighted)",
        "normalization": "RGB scaled to [0, 1]; no mean/std normalization (no pretrained weights)",
        "augmentation": "train only: random horizontal/vertical flips and 90-degree rotations",
        "checkpoint_rule": "highest validation mIoU",
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "color_tolerance": args.color_tolerance,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "classes": [
            {"id": class_id, "name": name, "rgb": list(color)}
            for class_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
        ],
        "train_files": [image.relative_to(args.train_images).as_posix() for image, _ in train_pairs],
        "validation_files": [image.relative_to(args.train_images).as_posix() for image, _ in validation_pairs],
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
    }
    (args.output / "config.json").write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")

    best_miou, best_epoch, history, start_epoch = -1.0, 0, [], 1
    last_path = args.output / "last_checkpoint.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_miou, best_epoch, history = state["best_miou"], state["best_epoch"], state["history"]
        random.setstate(state["python_random_state"])
        # map_location=device moves every tensor to the GPU, but set_rng_state needs a CPU ByteTensor
        torch.set_rng_state(state["torch_rng_state"].cpu())
        start_epoch = state["epoch"] + 1
        print(f"Resumed from epoch {state['epoch']} (best val mIoU {best_miou:.4f} at epoch {best_epoch})")

    print(f"Device: {experiment_config['device']}; training pairs: {len(train_pairs)}; "
          f"validation pairs: {len(validation_pairs)}")
    print(f"Input: {args.image_size}x{args.image_size}; loss: unweighted CrossEntropyLoss; seed: {args.seed}")
    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, validation_loader, criterion, device)
        elapsed = time.perf_counter() - started
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "train_miou": train_metrics["miou"],
            "train_mean_dice": train_metrics["mean_dice"],
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_miou": val_metrics["miou"],
            "val_mean_dice": val_metrics["mean_dice"],
            **{f"val_iou_{name}": value for name, value in zip(CLASS_NAMES, val_metrics["class_iou"])},
            "val_class_dice": val_metrics["class_dice"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(elapsed, 2),
        }
        history.append(row)
        if float(val_metrics["miou"]) > best_miou:
            best_miou, best_epoch = float(val_metrics["miou"]), epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_miou": best_miou,
                 "class_names": CLASS_NAMES, "image_size": args.image_size, "seed": args.seed},
                args.output / "best_model.pt",
            )
        write_history(history, args.output, best_epoch)
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "best_miou": best_miou,
             "best_epoch": best_epoch, "history": history,
             "python_random_state": random.getstate(), "torch_rng_state": torch.get_rng_state()},
            last_path,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_metrics['loss']:.4f} "
            f"mIoU {train_metrics['miou']:.4f} | val loss {val_metrics['loss']:.4f} "
            f"mIoU {val_metrics['miou']:.4f} Dice {val_metrics['mean_dice']:.4f} | {elapsed:.1f}s"
        )
    total_seconds = sum(float(row["seconds"]) for row in history)
    summary = {"best_epoch": best_epoch, "best_val_miou": best_miou,
               "selection_rule": "highest validation mIoU", "epochs_run": len(history),
               "total_training_seconds": round(total_seconds, 1),
               "this_session_seconds": round(time.perf_counter() - run_started, 1)}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best validation mIoU: {best_miou:.4f} at epoch {best_epoch}; "
          f"training time {total_seconds / 60:.1f} min; artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
