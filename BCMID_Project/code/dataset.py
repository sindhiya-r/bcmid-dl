from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (
    list_modality_images,
    log_corrupt_image,
    normalize_modality,
    relative_to_or_absolute,
)


ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass(frozen=True)
class ImageRecord:
    patient_id: str
    label: int
    path: Path


def build_transforms(img_size: int, split: str):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def verify_image(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception as exc:  # PIL raises several decoder-specific exception types.
        return f"{type(exc).__name__}: {exc}"


class BCMIDSingleModalityDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_df: pd.DataFrame,
        split: str,
        modality: str,
        img_size: int,
        corrupt_log_path: Path,
        max_train_images_per_patient: int = 4,
        seed: int = 42,
        verify_images: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.modality = normalize_modality(modality)
        self.transform = build_transforms(img_size=img_size, split=self.split)
        self.corrupt_log_path = Path(corrupt_log_path)
        self.max_train_images_per_patient = max_train_images_per_patient
        self.seed = seed

        if self.split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        patient_df = split_df[split_df["split"].astype(str).str.lower() == self.split].copy()
        if patient_df.empty:
            raise ValueError(f"No patients found for split '{self.split}'")

        self.records = self._build_records(patient_df)
        if verify_images:
            self.records = self._filter_corrupt_records(self.records)

        if not self.records:
            raise RuntimeError(
                f"No usable {self.modality} images found for split '{self.split}' under {self.data_root}"
            )

    def _build_records(self, patient_df: pd.DataFrame) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        for row in patient_df.itertuples(index=False):
            patient_id = str(row.patient_id)
            label = int(row.label)
            image_paths = list_modality_images(self.data_root, patient_id, self.modality)

            if self.split == "train":
                rng = random.Random(f"{self.seed}:{patient_id}:{self.modality}")
                image_paths = image_paths.copy()
                rng.shuffle(image_paths)
                image_paths = sorted(image_paths[: self.max_train_images_per_patient])

            records.extend(ImageRecord(patient_id=patient_id, label=label, path=path) for path in image_paths)
        return records

    def _filter_corrupt_records(self, records: List[ImageRecord]) -> List[ImageRecord]:
        valid: List[ImageRecord] = []
        for record in records:
            reason = verify_image(record.path)
            if reason is None:
                valid.append(record)
            else:
                log_corrupt_image(record.path, reason, self.corrupt_log_path)
        return valid

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Optional[Dict[str, object]]:
        record = self.records[index]
        try:
            with Image.open(record.path) as img:
                image = self.transform(img.convert("RGB"))
        except Exception as exc:
            log_corrupt_image(record.path, f"{type(exc).__name__}: {exc}", self.corrupt_log_path)
            return None

        return {
            "image": image,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "patient_id": record.patient_id,
            "path": relative_to_or_absolute(record.path, self.data_root),
        }

    def label_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for record in self.records:
            counts[int(record.label)] += 1
        return counts
