from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from utils import create_patient_split, ensure_dir, infer_data_root, read_labels_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible patient-wise BCMID train/val split.")
    parser.add_argument("--data-root", default=None, help="BCMID dataset root. Auto-detected if omitted.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "configs" / "patient_split.csv"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = infer_data_root(args.data_root)
    output = Path(args.output).expanduser()

    labels_df = read_labels_csv(data_root)
    split_df = create_patient_split(labels_df, val_size=args.val_size, seed=args.seed)
    ensure_dir(output.parent)
    split_df.to_csv(output, index=False)

    summary = split_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Data root: {data_root}")
    print(f"Wrote split: {output}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
