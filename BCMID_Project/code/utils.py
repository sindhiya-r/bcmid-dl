from __future__ import annotations

import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BCMID_ROOT = Path(r"E:\Multimodal_attention_DeepLearning\BCMID")
KAGGLE_BCMID_ROOT = Path("/kaggle/input/datasets/cs24m1005sindhiyar/bcmid-dataset/BCMID")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODALITY_FOLDERS = {
    "mammogram": "Mammogram",
    "ultrasound": "Ultrasound",
}


def infer_data_root(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if explicit.exists():
            return explicit.resolve()
        detected = find_bcmid_root()
        if detected is not None:
            return detected
        return explicit.resolve()

    env_path = os.environ.get("BCMID_DATA_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    if KAGGLE_BCMID_ROOT.exists():
        return KAGGLE_BCMID_ROOT
    if LOCAL_BCMID_ROOT.exists():
        return LOCAL_BCMID_ROOT
    detected = find_bcmid_root()
    if detected is not None:
        return detected
    return PROJECT_ROOT / "data" / "BCMID"


def find_bcmid_root() -> Optional[Path]:
    search_roots = [Path("/kaggle/input"), PROJECT_ROOT / "data"]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for label_path in search_root.rglob("BCMID_labels.csv"):
            return label_path.parent.resolve()
    return None


def default_results_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/results")
    return PROJECT_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_file: Path) -> logging.Logger:
    ensure_dir(log_file.parent)
    logger = logging.getLogger("bcmid")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_corrupt_image(path: Path, reason: str, log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{path}\t{reason}\n")


def normalize_modality(modality: str) -> str:
    value = modality.strip().lower()
    if value not in MODALITY_FOLDERS:
        valid = ", ".join(sorted(MODALITY_FOLDERS))
        raise ValueError(f"Unsupported modality '{modality}'. Expected one of: {valid}")
    return value


def read_labels_csv(data_root: Path) -> pd.DataFrame:
    label_path = data_root / "BCMID_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels file: {label_path}")

    raw = pd.read_csv(label_path, header=None)
    if raw.shape[1] >= 3:
        candidate = raw.iloc[:, :3].copy()
        candidate.columns = ["patient_id", "birads", "label"]
        candidate["label"] = pd.to_numeric(candidate["label"], errors="coerce")
        if candidate["label"].notna().all():
            df = candidate
        else:
            df = _read_headered_labels(label_path)
    else:
        df = _read_headered_labels(label_path)

    df["patient_id"] = df["patient_id"].astype(str)
    df["birads"] = df["birads"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    df = df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    bad_labels = sorted(set(df["label"].tolist()) - {0, 1})
    if bad_labels:
        raise ValueError(f"Labels must be binary 0/1. Found: {bad_labels}")
    return df


def _read_headered_labels(label_path: Path) -> pd.DataFrame:
    headered = pd.read_csv(label_path)
    lowered = {str(c).strip().lower(): c for c in headered.columns}
    patient_col = lowered.get("patient_id") or lowered.get("patient") or lowered.get("id")
    birads_col = lowered.get("birads") or lowered.get("bi-rads")
    label_col = lowered.get("label") or lowered.get("target") or lowered.get("malignant")
    if not patient_col or not label_col:
        raise ValueError(f"Could not infer patient and label columns from {label_path}")
    return pd.DataFrame(
        {
            "patient_id": headered[patient_col],
            "birads": headered[birads_col] if birads_col else "",
            "label": pd.to_numeric(headered[label_col], errors="raise"),
        }
    )


def create_patient_split(
    labels_df: pd.DataFrame,
    val_size: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    train_indices: List[int] = []

    for _, group in labels_df.groupby("label", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices.tolist())
            continue
        val_count = int(round(len(indices) * val_size))
        val_count = max(1, min(val_count, len(indices) - 1))
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    if not val_indices:
        all_indices = labels_df.index.to_numpy()
        rng.shuffle(all_indices)
        val_count = max(1, int(round(len(all_indices) * val_size)))
        val_count = min(val_count, len(all_indices) - 1)
        val_indices = all_indices[:val_count].tolist()
        train_indices = all_indices[val_count:].tolist()

    train_df = labels_df.loc[train_indices].copy()
    val_df = labels_df.loc[val_indices].copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    split_df = pd.concat([train_df, val_df], ignore_index=True)
    split_df = split_df[["patient_id", "birads", "label", "split"]]
    return split_df.sort_values(["split", "patient_id"]).reset_index(drop=True)


def load_or_create_split(
    data_root: Path,
    split_csv: Path,
    val_size: float,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if split_csv.exists():
        split_df = pd.read_csv(split_csv)
    else:
        labels_df = read_labels_csv(data_root)
        split_df = create_patient_split(labels_df, val_size=val_size, seed=seed)
        ensure_dir(split_csv.parent)
        split_df.to_csv(split_csv, index=False)
        if logger:
            logger.info("Created patient split: %s", split_csv)

    required = {"patient_id", "label", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")
    split_df["patient_id"] = split_df["patient_id"].astype(str)
    split_df["label"] = split_df["label"].astype(int)
    split_df["split"] = split_df["split"].astype(str).str.lower()
    return split_df


def list_modality_images(data_root: Path, patient_id: str, modality: str) -> List[Path]:
    modality = normalize_modality(modality)
    folder = data_root / patient_id / MODALITY_FOLDERS[modality]
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def compute_binary_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float64)
    y_pred_arr = (y_prob_arr >= threshold).astype(np.int64)

    if y_true_arr.size == 0:
        return {
            "auc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }

    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    auc = binary_auc(y_true_arr, y_prob_arr)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    return {
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def threshold_grid(start: float = 0.05, stop: float = 0.95, step: float = 0.01) -> List[float]:
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(count)]


def find_best_f1_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    thresholds: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    y_true_list = list(y_true)
    y_prob_list = list(y_prob)
    candidates = list(thresholds) if thresholds is not None else threshold_grid()
    best_threshold = candidates[0]
    best_metrics = compute_binary_metrics(y_true_list, y_prob_list, threshold=best_threshold)
    best_score = (
        _nan_to_score(best_metrics["f1"]),
        _nan_to_score(best_metrics["sensitivity"]),
        _nan_to_score(best_metrics["specificity"]),
    )

    for threshold in candidates[1:]:
        metrics = compute_binary_metrics(y_true_list, y_prob_list, threshold=threshold)
        score = (
            _nan_to_score(metrics["f1"]),
            _nan_to_score(metrics["sensitivity"]),
            _nan_to_score(metrics["specificity"]),
        )
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    return {
        "threshold": float(best_threshold),
        **best_metrics,
    }


def find_best_youden_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    thresholds: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    y_true_list = list(y_true)
    y_prob_list = list(y_prob)
    candidates = list(thresholds) if thresholds is not None else threshold_grid()
    best_threshold = candidates[0]
    best_metrics = compute_binary_metrics(y_true_list, y_prob_list, threshold=best_threshold)
    best_youden = _youden_score(best_metrics)
    best_score = (
        best_youden,
        _nan_to_score(best_metrics["sensitivity"]),
        _nan_to_score(best_metrics["specificity"]),
        _nan_to_score(best_metrics["f1"]),
    )

    for threshold in candidates[1:]:
        metrics = compute_binary_metrics(y_true_list, y_prob_list, threshold=threshold)
        youden = _youden_score(metrics)
        score = (
            youden,
            _nan_to_score(metrics["sensitivity"]),
            _nan_to_score(metrics["specificity"]),
            _nan_to_score(metrics["f1"]),
        )
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_youden = youden
            best_score = score

    return {
        "threshold": float(best_threshold),
        "youden": float(best_youden),
        **best_metrics,
    }


def compute_patient_metrics_with_thresholds(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    fixed_threshold: float = 0.5,
) -> Dict[str, float]:
    y_true_list = list(y_true)
    y_prob_list = list(y_prob)
    fixed = compute_binary_metrics(y_true_list, y_prob_list, threshold=fixed_threshold)
    best_f1 = find_best_f1_threshold(y_true_list, y_prob_list)
    best_youden = find_best_youden_threshold(y_true_list, y_prob_list)
    return {
        "auc": fixed["auc"],
        "accuracy": fixed["accuracy"],
        "f1": fixed["f1"],
        "sensitivity": fixed["sensitivity"],
        "specificity": fixed["specificity"],
        "best_f1": best_f1["f1"],
        "best_threshold": best_f1["threshold"],
        "best_threshold_sensitivity": best_f1["sensitivity"],
        "best_threshold_specificity": best_f1["specificity"],
        "youden": best_youden["youden"],
        "youden_threshold": best_youden["threshold"],
        "youden_sensitivity": best_youden["sensitivity"],
        "youden_specificity": best_youden["specificity"],
        "youden_f1": best_youden["f1"],
    }


def _nan_to_score(value: float) -> float:
    return -1.0 if np.isnan(value) else float(value)


def _youden_score(metrics: Dict[str, float]) -> float:
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    if np.isnan(sensitivity) or np.isnan(specificity):
        return -1.0
    return float(sensitivity + specificity - 1.0)


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        average_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def safe_torch_load(path: Path, map_location: str | Any = "cpu") -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint_path(output_dir: Path, resume: Optional[str], auto_resume: bool) -> Optional[Path]:
    if resume:
        ckpt = Path(resume).expanduser()
        return ckpt.resolve() if ckpt.exists() else ckpt
    if auto_resume:
        last = output_dir / "checkpoints" / "last.pt"
        if last.exists():
            return last
    return None


def relative_to_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
