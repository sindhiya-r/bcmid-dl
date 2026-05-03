from __future__ import annotations

import argparse
import contextlib
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import BCMIDSingleModalityDataset
from model import create_single_modality_model, supported_backbones
from utils import (
    PROJECT_ROOT,
    append_csv_row,
    compute_patient_metrics_with_thresholds,
    default_results_dir,
    ensure_dir,
    infer_data_root,
    load_or_create_split,
    now_stamp,
    resolve_checkpoint_path,
    safe_torch_load,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BCMID single-modality binary classifier.")
    parser.add_argument("--modality", required=True, choices=["mammogram", "ultrasound"])
    parser.add_argument("--backbone", default="efficientnet_b0", choices=list(supported_backbones()))
    parser.add_argument("--data-root", "--data_dir", "--data-dir", dest="data_root", default=None, help="BCMID dataset root. Auto-detected if omitted.")
    parser.add_argument("--split-csv", "--split_csv", dest="split_csv", default=str(PROJECT_ROOT / "configs" / "patient_split.csv"))
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(default_results_dir()))
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--val-size", "--val_size", dest="val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--weighted-bce", "--weighted_bce", dest="weighted_bce", action="store_true", help="Use auto-computed BCE pos_weight.")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--early-stopping-patience", "--early_stopping_patience", dest="early_stopping_patience", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--auto-resume",
        "--auto_resume",
        dest="auto_resume",
        action="store_true",
        help="Resume from output-dir/checkpoints/last.pt if output-dir is an existing run directory.",
    )
    parser.add_argument("--no-amp", "--no_amp", dest="no_amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--no-pretrained", "--no_pretrained", dest="no_pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--no-verify-images", "--no_verify_images", dest="no_verify_images", action="store_true", help="Skip startup corrupt-image verification.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace, results_root: Path) -> Path:
    if args.resume:
        ckpt = Path(args.resume).expanduser()
        if ckpt.name.endswith(".pt") and ckpt.parent.name == "checkpoints":
            return ckpt.parent.parent.resolve()
    if args.auto_resume and (results_root / "checkpoints" / "last.pt").exists():
        return results_root
    run_name = args.run_name or f"{args.modality}_{args.backbone}_{now_stamp()}"
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
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    patient_ids = [str(item["patient_id"]) for item in batch]
    paths = [str(item["path"]) for item in batch]
    return {"image": images, "label": labels, "patient_id": patient_ids, "path": paths}


def write_patient_predictions(path: Path, predictions: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["patient_id", "label", "probability", "prediction_at_0_5"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits.view(-1, 1), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
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
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits.view(-1, 1), labels)

        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        labels_np = labels.view(-1).detach().cpu().numpy().astype(int)

        for patient_id, label, prob in zip(batch["patient_id"], labels_np, probs):
            patient_probs[patient_id].append(float(prob))
            patient_labels[patient_id] = int(label)

        batch_size = images.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        progress.set_postfix(loss=running_loss / max(seen, 1))

    y_true = []
    y_prob = []
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
    metrics["images"] = int(seen)
    return running_loss / max(seen, 1), metrics, patient_predictions


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
            "best_score": best_auc,
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
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"

    logger.info("Data root: %s", data_root)
    logger.info("Results root: %s", results_root)
    logger.info("Run dir: %s", run_dir)
    logger.info("Device: %s | AMP: %s", device, use_amp)

    split_df = load_or_create_split(
        data_root=data_root,
        split_csv=split_csv,
        val_size=args.val_size,
        seed=args.seed,
        logger=logger,
    )
    corrupt_log = results_root / "corrupt_images.txt"

    train_ds = BCMIDSingleModalityDataset(
        data_root=data_root,
        split_df=split_df,
        split="train",
        modality=args.modality,
        img_size=args.img_size,
        corrupt_log_path=corrupt_log,
        max_train_images_per_patient=4,
        seed=args.seed,
        verify_images=not args.no_verify_images,
    )
    val_ds = BCMIDSingleModalityDataset(
        data_root=data_root,
        split_df=split_df,
        split="val",
        modality=args.modality,
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

    logger.info("Train images: %d | Val images: %d", len(train_ds), len(val_ds))
    logger.info("Train image-level label counts: %s", train_ds.label_counts())

    model = create_single_modality_model(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        num_classes=1,
    ).to(device)

    pos_weight = None
    if args.weighted_bce:
        counts = train_ds.label_counts()
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
        best_auc = float(checkpoint.get("best_auc", checkpoint.get("best_score", best_auc)))
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
        "images",
    ]

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch)
        val_loss, val_metrics, patient_predictions = evaluate(model, val_loader, criterion, device, use_amp, args.threshold, epoch)
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        append_csv_row(run_dir / "metrics_history.csv", row, history_fields)

        monitor_value = float(val_metrics["auc"])
        improved = not math.isnan(monitor_value) and monitor_value > best_auc
        if improved:
            best_auc = monitor_value
            best_epoch_metrics = dict(val_metrics)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
        # save_checkpoint(
        #     epoch_ckpt,
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
            "Epoch %03d | train_loss=%.5f | val_loss=%.5f | auc=%.5f | acc@%.2f=%.5f | f1@%.2f=%.5f | sens@%.2f=%.5f | spec@%.2f=%.5f | best_f1=%.5f @ threshold=%.2f | chosen_sens=%.5f | chosen_spec=%.5f | youden=%.5f @ threshold=%.2f | youden_sens=%.5f | youden_spec=%.5f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["auc"],
            args.threshold,
            val_metrics["accuracy"],
            args.threshold,
            val_metrics["f1"],
            args.threshold,
            val_metrics["sensitivity"],
            args.threshold,
            val_metrics["specificity"],
            val_metrics["best_f1"],
            val_metrics["best_threshold"],
            val_metrics["best_threshold_sensitivity"],
            val_metrics["best_threshold_specificity"],
            val_metrics["youden"],
            val_metrics["youden_threshold"],
            val_metrics["youden_sensitivity"],
            val_metrics["youden_specificity"],
        )

        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs without AUC improvement.", no_improve_epochs)
            break

    summary = {
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
    }
    save_json(summary, run_dir / "summary.json")
    logger.info("Finished training. Best patient-level AUC: %.5f", best_auc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
