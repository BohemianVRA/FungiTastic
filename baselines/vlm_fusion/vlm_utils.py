"""Shared helpers for vision–language fusion baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Filenames with missing or corrupt Molmo captions (train/val/test).
INVALID_CAPTIONS = frozenset({
    "1-3861325341.JPG",
    "2-4100092812.JPG",
    "0-2238480458.JPG",
    "0-2238152962.JPG",
    "1-2238552443.JPG",
    "0-2597564674.JPG",
    "1-2238554704.JPG",
    "1-2238483885.JPG",
    "1-2238523717.JPG",
    "1-2237914372.JPG",
    "0-2238496151.JPG",
    "0-2238127565.JPG",
    "0-2238365191.JPG",
    "1-4169763613.JPG",
    "0-4465865676.JPG",
    "0-4465900600.JPG",
})

VARIANTS = {
    "mini": {
        "train_csv": "metadata/FungiTastic-Mini/FungiTastic-Mini-Train.csv",
        "val_csv": "metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Val.csv",
        "test_csv": "metadata/FungiTastic-Mini/FungiTastic-Mini-ClosedSet-Test.csv",
        "test_dev_csv": "metadata/FungiTastic/FungiTastic-test-DEV.csv",
        "test_images": "FungiTastic-Mini/test/500p",
        "beit_prefix": "mini-500p",
    },
    "full": {
        "train_csv": "metadata/FungiTastic/FungiTastic-Train.csv",
        "val_csv": "metadata/FungiTastic/FungiTastic-ClosedSet-Val.csv",
        "test_csv": "metadata/FungiTastic/FungiTastic-ClosedSet-Test.csv",
        "test_dev_csv": "metadata/FungiTastic/FungiTastic-test-DEV.csv",
        "test_images": "FungiTastic/test/500p",
        "beit_prefix": "full-500p",
    },
}


def metadata_path(dataset_dir: str | Path, variant: str, split: str) -> Path:
    cfg = VARIANTS[variant]
    rel = {
        "train": cfg["train_csv"],
        "val": cfg["val_csv"],
        "test": cfg["test_csv"],
        "test_dev": cfg["test_dev_csv"],
    }[split]
    return Path(dataset_dir) / rel


def load_split_df(dataset_dir: str | Path, variant: str, split: str) -> pd.DataFrame:
    """Load a metadata CSV; for test, merge closed-set filenames with DEV labels."""
    dataset_dir = Path(dataset_dir)
    if split == "test":
        df_test = pd.read_csv(metadata_path(dataset_dir, variant, "test"))
        df_dev = pd.read_csv(metadata_path(dataset_dir, variant, "test_dev"))
        df_dev = df_dev[df_dev["category_id"] != -1]
        return pd.merge(
            df_test,
            df_dev[["filename", "category_id"]],
            how="left",
            on="filename",
        )
    return pd.read_csv(metadata_path(dataset_dir, variant, split))


def read_caption(caption_dir: Path, filename: str) -> str:
    if filename in INVALID_CAPTIONS:
        return ""
    path = caption_dir / f"{filename}.json"
    with path.open() as f:
        text = "".join(line.rstrip() for line in f)
    return (
        text.replace("\\n", "\n")
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace('"', "")
    )


def load_captions(caption_dir: str | Path, df: pd.DataFrame) -> list[str]:
    caption_dir = Path(caption_dir)
    return [read_caption(caption_dir, row["filename"]) for _, row in df.iterrows()]


def align_logits_by_filename(
    feature_bundle: dict,
    filenames: Iterable[str],
) -> torch.Tensor:
    """Reorder saved image logits to match a metadata CSV row order."""
    files = np.asarray(feature_bundle["files"])
    logits = feature_bundle["logits"]
    rows = []
    for name in filenames:
        idx = int(np.where(files == name)[0][0])
        rows.append(logits[idx])
    return torch.stack(rows)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor | np.ndarray) -> dict[str, float]:
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    hits = labels == logits.argmax(1)
    top1 = hits.float().mean().item() * 100

    _, idx = logits.topk(3, dim=1)
    top3_hits = [gt.item() in idx[i].tolist() for i, gt in enumerate(labels)]
    top3 = float(np.mean(top3_hits)) * 100

    f1_macro = f1_score(labels.numpy(), logits.argmax(1).numpy(), average="macro") * 100
    return {"top1-acc": top1, "top3-acc": top3, "f1_macro": f1_macro}


def print_metrics_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    header = f"{'Model':<22} {'Top1':>8} {'Top3':>8} {'F1_m':>8}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<22} {m['top1-acc']:8.1f} {m['top3-acc']:8.1f} {m['f1_macro']:8.1f}")
