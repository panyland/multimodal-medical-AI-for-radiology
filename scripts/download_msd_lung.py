"""Download and extract MSD Task06 Lung, then check file counts against dataset.json.

Usage:
    uv run python scripts/download_msd_lung.py [--root data/raw]
"""

import argparse
import json
from pathlib import Path

from monai.apps import DecathlonDataset, download_and_extract

TASK = "Task06_Lung"


def count_volumes(folder: Path) -> int:
    """Count .nii.gz files, skipping hidden macOS '._*' metadata files."""
    return sum(1 for p in folder.glob("*.nii.gz") if not p.name.startswith("."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    tar_path = args.root / f"{TASK}.tar"

    # Skips the download if the tar already exists and its MD5 matches.
    download_and_extract(
        url=DecathlonDataset.resource[TASK],
        filepath=tar_path,
        output_dir=args.root,
        hash_val=DecathlonDataset.md5[TASK],
        hash_type="md5",
    )

    task_dir = args.root / TASK
    meta = json.loads((task_dir / "dataset.json").read_text())
    checks = [
        ("imagesTr", meta["numTraining"]),
        ("labelsTr", meta["numTraining"]),
        ("imagesTs", meta["numTest"]),
    ]

    ok = True
    for sub, expected in checks:
        found = count_volumes(task_dir / sub)
        status = "OK" if found == expected else "MISMATCH"
        ok &= found == expected
        print(f"{sub:9s} found {found:3d} / expected {expected:3d}  {status}")

    print("All counts match." if ok else "Count mismatch: check the extraction.")


if __name__ == "__main__":
    main()
