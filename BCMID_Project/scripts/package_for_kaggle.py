from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and should_include(path):
            yield path


def zip_directory(source_dir: Path, zip_path: Path, arc_prefix: str | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        files = list(iter_files(source_dir))
        if not files:
            prefix = arc_prefix or source_dir.name
            zf.writestr(f"{prefix}/", "")
        for file_path in files:
            relative = file_path.relative_to(source_dir)
            arcname = Path(arc_prefix or source_dir.name) / relative
            zf.write(file_path, arcname.as_posix())


def create_bundle(project_root: Path, bundle_path: Path) -> None:
    include_paths = [
        project_root / "code",
        project_root / "configs",
        project_root / "notebooks" / "kaggle_train.ipynb",
        project_root / "requirements.txt",
        project_root / "README.md",
    ]
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include_paths:
            if path.is_dir():
                files = list(iter_files(path))
                if not files:
                    zf.writestr(f"{path.name}/", "")
                for file_path in files:
                    zf.write(file_path, file_path.relative_to(project_root).as_posix())
            elif path.exists() and should_include(path):
                zf.write(path, path.relative_to(project_root).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package BCMID project code for Kaggle upload.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root / "backups" / f"kaggle_upload_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    code_zip = output_dir / "code.zip"
    configs_zip = output_dir / "configs.zip"
    bundle_zip = output_dir / f"BCMID_Project_kaggle_bundle_{timestamp}.zip"

    zip_directory(project_root / "code", code_zip, arc_prefix="code")
    zip_directory(project_root / "configs", configs_zip, arc_prefix="configs")
    create_bundle(project_root, bundle_zip)

    latest_bundle = project_root / "backups" / "BCMID_Project_kaggle_bundle_latest.zip"
    latest_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_zip, latest_bundle)

    print(f"Wrote: {code_zip}")
    print(f"Wrote: {configs_zip}")
    print(f"Wrote: {bundle_zip}")
    print(f"Updated: {latest_bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
