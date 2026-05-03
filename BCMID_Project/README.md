# BCMID Phase 1 Single-Modality Training

This project supports local development and Kaggle GPU training for BCMID paired mammogram + ultrasound binary classification.

Labels:

- `0`: benign
- `1`: malignant

Phase 1 baselines:

- Mammogram only
- Ultrasound only

## Dataset Paths

Defaults are built in:

- Local: `E:\Multimodal_attention_DeepLearning\BCMID`
- Kaggle: `/kaggle/input/datasets/cs24m1005sindhiyar/bcmid-dataset/BCMID`

You can override with `--data-root` or `BCMID_DATA_ROOT`.

Expected dataset layout:

```text
BCMID/
  BCMID_labels.csv
  patient_id/
    Mammogram/
      image files
    Ultrasound/
      image files
```

`BCMID_labels.csv` may be headerless as:

```text
patient_id,BIRADS,label
```

## Local Setup

```bash
cd BCMID_Project
python -m pip install -r requirements.txt
```

If local PyTorch import fails on Windows with duplicate `libiomp5md.dll`, fix the Python environment first when possible. For a temporary local smoke test only, run PowerShell commands with:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

Create a reproducible patient-level split:

```bash
python scripts/create_split.py --data-root "E:\Multimodal_attention_DeepLearning\BCMID"
```

Train ultrasound baseline:

```bash
python code/train_single.py --modality ultrasound --backbone efficientnet_b0 --weighted-bce
```

Train mammogram baseline:

```bash
python code/train_single.py --modality mammogram --backbone efficientnet_b0 --weighted-bce
```

Supported backbones:

- `efficientnet_b0`
- `convnext_small`
- `vit_base_patch16_224`

## Outputs

Training writes to `results/` locally and `/kaggle/working/results/` on Kaggle.

Each run writes:

- `train.log`
- `run_config.json`
- `metrics_history.csv`
- `checkpoints/epoch_XXX.pt`
- `checkpoints/last.pt`
- `checkpoints/best.pt`

Corrupt images are skipped and logged to:

```text
results/corrupt_images.txt
```

Validation metrics are patient-level. Image probabilities are averaged per patient before computing AUC, accuracy, F1, sensitivity, and specificity.

## Resume Training

Resume explicitly:

```bash
python code/train_single.py --modality ultrasound --backbone efficientnet_b0 --resume results/<run>/checkpoints/last.pt
```

Resume automatically from an existing run directory:

```bash
python code/train_single.py --modality ultrasound --backbone efficientnet_b0 --output-dir results/<run> --auto-resume
```

## Kaggle Packaging

Create upload artifacts:

```bash
python scripts/package_for_kaggle.py
```

This creates:

- `backups/kaggle_upload_<timestamp>/code.zip`
- `backups/kaggle_upload_<timestamp>/configs.zip`
- `backups/kaggle_upload_<timestamp>/BCMID_Project_kaggle_bundle_<timestamp>.zip`
- `backups/BCMID_Project_kaggle_bundle_latest.zip`

Upload the bundle ZIP as a Kaggle dataset or notebook input, then run `notebooks/kaggle_train.ipynb`.
