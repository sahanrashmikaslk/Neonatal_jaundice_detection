"""
Plot training and validation accuracy over epochs from a saved metrics CSV.

Usage:
    python plot_accuracy.py path/to/metrics.csv --out accuracy_plot.png

The CSV is expected to have at least these columns:
    epoch, train_acc, val_acc
Additional columns (train_loss, val_loss, val_sens, val_spec, etc.) are ignored.
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def latest_csv_in(folder: Path) -> Path:
    """Return the most recently modified CSV in a folder, or raise if none."""
    candidates = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return candidates[0]


def plot_accuracy(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    required_cols = {"epoch", "train_acc", "val_acc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    epochs = df["epoch"]
    train_acc = df["train_acc"]
    val_acc = df["val_acc"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Accuracy vs Epoch")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved accuracy plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot train/val accuracy vs epoch from metrics CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to metrics CSV. If omitted, uses latest CSV in training_metrics/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("performance_evaluation/accuracy_plot_latest.png"),
        help="Output path for the PNG plot",
    )
    args = parser.parse_args()

    if args.csv is None:
        args.csv = latest_csv_in(Path("training_metrics"))
        print(f"No CSV provided, using latest: {args.csv}")

    plot_accuracy(args.csv, args.out)


if __name__ == "__main__":
    main()
