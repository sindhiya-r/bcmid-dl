from __future__ import annotations

import argparse
import contextlib
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dataset import build_transforms, verify_image
from fusion_model import BCMIDFusionModel, SUPPORTED_FUSION_METHODS
from model import supported_backbones
from utils import (
    PROJECT_ROOT,
    append_csv_row,
    compute_patient_metrics_with_thresholds,
    default_results_dir,
    ensure_dir,
    infer_data_root,
    list_modality_images,
    load_or_create_split,
    log_corrupt_image,
    now_stamp,
    resolve_checkpoint_path,
    safe_torch_load,
    save_json,
    set_seed,
    setup_logger,
)


@dataclass(frozen=True)
class FusionRecord:
    patient_id: str
    label: int
    mammogram_path: Optional[Path]
    ultrasound_path: Optional[Path]


class BCMIDFusionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_df: pd.DataFrame,
        split: str,
        img_size: int,
        corrupt_log_path: Path,
        max_train_images_per_patient: int = 4,
        seed: int = 42,
        verify_images: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.img_size = img_size
        self.transform = build_transforms(img_size=img_size, split=self.split)
        self.corrupt_log_path = Path(corrupt_log_path)
        self.max_train_images_per_patient = max_train_images_per_patient
        self.seed = seed
        self.train_pools: Dict[str, Tuple[List[Path], List[Path]]] = {}

        if self.split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        patient_df = split_df[split_df["split"].astype(str).str.lower() == self.split].copy()
        if patient_df.empty:
            raise ValueError(f"No patients found for split '{self.split}'")

        self.records = self._build_records(patient_df, verify_images=verify_images)
        self.patient_labels = {record.patient_id: record.label for record in self.records}
        if not self.records:
            raise RuntimeError(f"No usable fusion records found for split '{self.split}' under {self.data_root}")

    def _build_records(self, patient_df: pd.DataFrame, verify_images: bool) -> List[FusionRecord]:
        records: List[FusionRecord] = []
        for row in patient_df.itertuples(index=False):
            patient_id = str(row.patient_id)
            label = int(row.label)
            mammograms = list_modality_images(self.data_root, patient_id, "mammogram")
            ultrasounds = list_modality_images(self.data_root, patient_id, "ultrasound")

            if verify_images:
                mammograms = self._filter_valid_images(mammograms)
                ultrasounds = self._filter_valid_images(ultrasounds)

            if self.split == "train":
                rng = random.Random(f"{self.seed}:{patient_id}:fusion")
                mammograms = self._cap_train_paths(mammograms, rng)
                ultrasounds = self._cap_train_paths(ultrasounds, rng)
                if mammograms or ultrasounds:
                    self.train_pools[patient_id] = (mammograms, ultrasounds)
                    records.append(
                        FusionRecord(
                            patient_id=patient_id,
                            label=label,
                            mammogram_path=None,
                            ultrasound_path=None,
                        )
                    )
                continue

            records.extend(self._validation_records(patient_id, label, mammograms, ultrasounds))
        return records

    def _filter_valid_images(self, paths: List[Path]) -> List[Path]:
        valid: List[Path] = []
        for path in paths:
            reason = verify_image(path)
            if reason is None:
                valid.append(path)
            else:
                log_corrupt_image(path, reason, self.corrupt_log_path)
        return valid

    def _cap_train_paths(self, paths: List[Path], rng: random.Random) -> List[Path]:
        paths = paths.copy()
        rng.shuffle(paths)
        return sorted(paths[: self.max_train_images_per_patient])

    def _validation_records(
        self,
        patient_id: str,
        label: int,
        mammograms: List[Path],
        ultrasounds: List[Path],
    ) -> List[FusionRecord]:
        if mammograms and ultrasounds:
            return [
                FusionRecord(patient_id, label, mammogram_path, ultrasound_path)
                for mammogram_path in mammograms
                for ultrasound_path in ultrasounds
            ]
        if mammograms:
            return [FusionRecord(patient_id, label, mammogram_path, None) for mammogram_path in mammograms]
        if ultrasounds:
            return [FusionRecord(patient_id, label, None, ultrasound_path) for ultrasound_path in ultrasounds]
        return []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Optional[Dict[str, object]]:
        record = self.records[index]
        if self.split == "train":
            record = self._sample_train_record(record)

        mammogram, mammogram_mask = self._load_or_zero(record.mammogram_path)
        ultrasound, ultrasound_mask = self._load_or_zero(record.ultrasound_path)
        if mammogram_mask == 0.0 and ultrasound_mask == 0.0:
            return None

        return {
            "mammogram": mammogram,
            "ultrasound": ultrasound,
            "mammogram_mask": torch.tensor(mammogram_mask, dtype=torch.float32),
            "ultrasound_mask": torch.tensor(ultrasound_mask, dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "patient_id": record.patient_id,
        }

    def _sample_train_record(self, record: FusionRecord) -> FusionRecord:
        rng = random.Random(f"{self.seed}:{record.patient_id}:{random.random()}")
        mammograms, ultrasounds = self.train_pools.get(record.patient_id, ([], []))
        mammogram_path = rng.choice(mammograms) if mammograms else None
        ultrasound_path = rng.choice(ultrasounds) if ultrasounds else None
        return FusionRecord(record.patient_id, record.label, mammogram_path, ultrasound_path)

    def _load_or_zero(self, path: Optional[Path]) -> Tuple[torch.Tensor, float]:
        if path is None:
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32), 0.0
        try:
            with Image.open(path) as img:
                return self.transform(img.convert("RGB")), 1.0
        except Exception as exc:
            log_corrupt_image(path, f"{type(exc).__name__}: {exc}", self.corrupt_log_path)
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32), 0.0

    def patient_label_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for label in self.patient_labels.values():
            counts[int(label)] += 1
        return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BCMID multimodal fusion binary classifier.")
    parser.add_argument("--fusion-method", "--fusion_method", dest="fusion_method", default="gated", choices=list(SUPPORTED_FUSION_METHODS))
    parser.add_argument("--backbone", default="efficientnet_b0", choices=list(supported_backbones()))
    parser.add_argument("--data-root", "--data_dir", "--data-dir", dest="data_root", default=None)
    parser.add_argument("--split-csv", "--split_csv", dest="split_csv", default=str(PROJECT_ROOT / "configs" / "patient_split.csv"))
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(default_results_dir()))
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--val-size", "--val_size", dest="val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--weighted-bce", "--weighted_bce", dest="weighted_bce", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--early-stopping-patience", "--early_stopping_patience", dest="early_stopping_patience", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fusion-dim", "--fusion_dim", dest="fusion_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--modality-dropout", "--modality_dropout", dest="modality_dropout", type=float, default=0.15)

    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", "--auto_resume", dest="auto_resume", action="store_true")
    parser.add_argument("--no-amp", "--no_amp", dest="no_amp", action="store_true")
    parser.add_argument("--no-pretrained", "--no_pretrained", dest="no_pretrained", action="store_true")
    parser.add_argument("--no-verify-images", "--no_verify_images", dest="no_verify_images", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace, results_root: Path) -> Path:
    if args.resume:
        ckpt = Path(args.resume).expanduser()
        if ckpt.name.endswith(".pt") and ckpt.parent.name == "checkpoints":
            return ckpt.parent.parent.resolve()
    if args.auto_resume and (results_root / "checkpoints" / "last.pt").exists():
        return results_root
    run_name = args.run_name or f"fusion_{args.fusion_method}_{args.backbone}_{now_stamp()}"
    return ensure_dir(results_root / run_name)


def make_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def collate_batch(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "mammogram": torch.stack([item["mammogram"] for item in batch], dim=0),
        "ultrasound": torch.stack([item["ultrasound"] for item in batch], dim=0),
        "mammogram_mask": torch.stack([item["mammogram_mask"] for item in batch], dim=0),
        "ultrasound_mask": torch.stack([item["ultrasound_mask"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "patient_id": [str(item["patient_id"]) for item in batch],
    }


def write_patient_predictions(path: Path, predictions: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["patient_id", "label", "probability", "prediction_at_0_5"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def forward_batch(model: nn.Module, batch: Dict[str, object], device: torch.device, use_amp: bool) -> torch.Tensor:
    mammogram = batch["mammogram"].to(device, non_blocking=True)
    ultrasound = batch["ultrasound"].to(device, non_blocking=True)
    mammogram_mask = batch["mammogram_mask"].to(device, non_blocking=True)
    ultrasound_mask = batch["ultrasound_mask"].to(device, non_blocking=True)
    with autocast_context(device, use_amp):
        return model(mammogram, ultrasound, mammogram_mask, ultrasound_mask)["logits"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for batch in progress:
        if batch is None:
            continue
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_batch(model, batch, device, use_amp)
        loss = criterion(logits.view(-1, 1), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        progress.set_postfix(loss=running_loss / max(seen, 1))
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    epoch: int,
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    running_loss = 0.0
    seen = 0
    patient_probs = defaultdict(list)
    patient_labels = {}
    progress = tqdm(loader, desc=f"Epoch {epoch} val", leave=False)

    for batch in progress:
        if batch is None:
            continue
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)
        logits = forward_batch(model, batch, device, use_amp)
        loss = criterion(logits.view(-1, 1), labels)
        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        labels_np = labels.view(-1).detach().cpu().numpy().astype(int)

        for patient_id, label, prob in zip(batch["patient_id"], labels_np, probs):
            patient_probs[patient_id].append(float(prob))
            patient_labels[patient_id] = int(label)

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        progress.set_postfix(loss=running_loss / max(seen, 1))

    y_true: List[int] = []
    y_prob: List[float] = []
    patient_predictions: List[Dict[str, object]] = []
    for patient_id in sorted(patient_probs):
        label = patient_labels[patient_id]
        probability = float(np.mean(patient_probs[patient_id]))
        y_true.append(label)
        y_prob.append(probability)
        patient_predictions.append(
            {
                "patient_id": patient_id,
                "label": label,
                "probability": probability,
                "prediction_at_0_5": int(probability >= 0.5),
            }
        )

    metrics = compute_patient_metrics_with_thresholds(y_true, y_prob, fixed_threshold=threshold)
    metrics["patients"] = int(len(y_true))
    metrics["samples"] = int(seen)
    return running_loss / max(seen, 1), metrics, patient_predictions


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    epoch: int,
    best_auc: float,
    args: argparse.Namespace,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    best_epoch_metrics: Dict[str, float],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_auc": best_auc,
            "args": vars(args),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "best_epoch_metrics": best_epoch_metrics,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = infer_data_root(args.data_root)
    split_csv = Path(args.split_csv).expanduser().resolve()
    results_root = ensure_dir(Path(args.output_dir).expanduser().resolve())
    run_dir = make_run_dir(args, results_root)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    logger = setup_logger(run_dir / "train.log")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but no CUDA device is available.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    logger.info("Data root: %s", data_root)
    logger.info("Results root: %s", results_root)
    logger.info("Run dir: %s", run_dir)
    logger.info("Device: %s | AMP: %s", device, use_amp)
    logger.info("Fusion method: %s | Backbone: %s", args.fusion_method, args.backbone)

    split_df = load_or_create_split(data_root, split_csv, args.val_size, args.seed, logger)
    corrupt_log = results_root / "corrupt_images.txt"
    train_ds = BCMIDFusionDataset(
        data_root=data_root,
        split_df=split_df,
        split="train",
        img_size=args.img_size,
        corrupt_log_path=corrupt_log,
        max_train_images_per_patient=4,
        seed=args.seed,
        verify_images=not args.no_verify_images,
    )
    val_ds = BCMIDFusionDataset(
        data_root=data_root,
        split_df=split_df,
        split="val",
        img_size=args.img_size,
        corrupt_log_path=corrupt_log,
        max_train_images_per_patient=4,
        seed=args.seed,
        verify_images=not args.no_verify_images,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        persistent_workers=args.num_workers > 0,
    )

    logger.info("Train patients: %d | Val samples: %d", len(train_ds.patient_labels), len(val_ds))
    logger.info("Train patient label counts: %s", train_ds.patient_label_counts())

    model = BCMIDFusionModel(
        backbone=args.backbone,
        fusion_method=args.fusion_method,
        pretrained=not args.no_pretrained,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        modality_dropout=args.modality_dropout,
    ).to(device)

    pos_weight = None
    if args.weighted_bce:
        counts = train_ds.patient_label_counts()
        pos_weight_value = counts[0] / max(counts[1], 1)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        logger.info("Using BCE pos_weight=%.6f", pos_weight_value)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == "cosine"
        else None
    )
    scaler = make_grad_scaler(device, enabled=use_amp)

    start_epoch = 1
    best_auc = -math.inf
    best_epoch_metrics: Dict[str, float] = {}
    no_improve_epochs = 0
    resume_path = resolve_checkpoint_path(run_dir, args.resume, args.auto_resume)
    if resume_path is not None and resume_path.exists():
        checkpoint = safe_torch_load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint.get("best_auc", best_auc))
        best_epoch_metrics = checkpoint.get("best_epoch_metrics", {})
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)
    elif resume_path is not None:
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    save_json(
        {
            **vars(args),
            "data_root": str(data_root),
            "split_csv": str(split_csv),
            "results_root": str(results_root),
            "run_dir": str(run_dir),
            "device": str(device),
            "amp": use_amp,
        },
        run_dir / "run_config.json",
    )

    history_fields = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "auc",
        "accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "best_f1",
        "best_threshold",
        "best_threshold_sensitivity",
        "best_threshold_specificity",
        "youden",
        "youden_threshold",
        "youden_sensitivity",
        "youden_specificity",
        "youden_f1",
        "patients",
        "samples",
    ]

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch)
        val_loss, val_metrics, patient_predictions = evaluate(
            model, val_loader, criterion, device, use_amp, args.threshold, epoch
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        append_csv_row(run_dir / "metrics_history.csv", row, history_fields)

        val_auc = float(val_metrics["auc"])
        improved = not math.isnan(val_auc) and val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_epoch_metrics = dict(val_metrics)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # save_checkpoint(
        #     ckpt_dir / f"epoch_{epoch:03d}.pt",
        #     model,
        #     optimizer,
        #     scaler,
        #     scheduler,
        #     epoch,
        #     best_auc,
        #     args,
        #     train_loss,
        #     val_loss,
        #     val_metrics,
        #     best_epoch_metrics,
        # )
        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            scaler,
            scheduler,
            epoch,
            best_auc,
            args,
            train_loss,
            val_loss,
            val_metrics,
            best_epoch_metrics,
        )
        if improved:
            write_patient_predictions(run_dir / "best_val_patient_predictions.csv", patient_predictions)
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                best_auc,
                args,
                train_loss,
                val_loss,
                val_metrics,
                best_epoch_metrics,
            )

        logger.info(
            "Epoch %03d | train_loss=%.5f | val_loss=%.5f | auc=%.5f | f1@%.2f=%.5f | sens@%.2f=%.5f | spec@%.2f=%.5f | best_f1=%.5f @ threshold=%.2f | youden=%.5f @ threshold=%.2f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["auc"],
            args.threshold,
            val_metrics["f1"],
            args.threshold,
            val_metrics["sensitivity"],
            args.threshold,
            val_metrics["specificity"],
            val_metrics["best_f1"],
            val_metrics["best_threshold"],
            val_metrics["youden"],
            val_metrics["youden_threshold"],
        )

        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without AUC improvement.", no_improve_epochs)
            break

    save_json(
        {
            "best_auc": best_auc,
            "best_f1": best_epoch_metrics.get("best_f1", float("nan")),
            "best_threshold": best_epoch_metrics.get("best_threshold", float("nan")),
            "best_threshold_sensitivity": best_epoch_metrics.get("best_threshold_sensitivity", float("nan")),
            "best_threshold_specificity": best_epoch_metrics.get("best_threshold_specificity", float("nan")),
            "youden": best_epoch_metrics.get("youden", float("nan")),
            "youden_threshold": best_epoch_metrics.get("youden_threshold", float("nan")),
            "youden_sensitivity": best_epoch_metrics.get("youden_sensitivity", float("nan")),
            "youden_specificity": best_epoch_metrics.get("youden_specificity", float("nan")),
            "early_stopping_metric": "auc",
            "last_epoch_metrics": val_metrics if "val_metrics" in locals() else {},
        },
        run_dir / "summary.json",
    )
    logger.info("Finished fusion training. Best patient-level AUC: %.5f", best_auc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
