"""Summarize MSD Task06 Lung training scans and plot one case with its tumour mask.

Usage:
    uv run python scripts/inspect_msd_lung.py [--root data/raw/Task06_Lung] [--case lung_001]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

LUNG_WINDOW = (-1350, 150)  # HU display window (level -600, width 1500)


def inspect_case(image_path: Path, label_path: Path) -> dict:
    """Load one image/label pair and return its key properties."""
    img = nib.load(image_path)
    lbl = nib.load(label_path)
    data = np.asanyarray(img.dataobj)  # raw voxel values (HU)
    mask = np.asanyarray(lbl.dataobj)
    spacing = img.header.get_zooms()[:3]  # mm per voxel along each axis

    tumour_voxels = int((mask > 0).sum())
    tumour_slices = int((mask > 0).any(axis=(0, 1)).sum())  # axial slices with tumour
    return {
        "case": image_path.name.removesuffix(".nii.gz"),
        "shape": data.shape,
        "spacing": tuple(round(float(s), 3) for s in spacing),
        "orient": "".join(nib.aff2axcodes(img.affine)),
        "hu_min": int(data.min()),
        "hu_max": int(data.max()),
        "labels": tuple(int(v) for v in np.unique(mask)),
        "same_shape": mask.shape == data.shape,
        "tumour_vox": tumour_voxels,
        "tumour_ml": tumour_voxels * float(np.prod(spacing)) / 1000.0,
        "tumour_frac": tumour_voxels / mask.size,
        "tumour_slices": tumour_slices,
    }


def print_summary(rows: list[dict]) -> None:
    """Print a per-case table, then min/median/max across cases."""
    print(f"{'case':9s} {'shape':16s} {'spacing (mm)':20s} {'orient':6s} "
          f"{'HU min':>7s} {'HU max':>7s} {'tum mL':>8s} {'tum %':>7s} {'tum sl':>6s}")
    for r in rows:
        print(f"{r['case']:9s} {str(r['shape']):16s} {str(r['spacing']):20s} {r['orient']:6s} "
              f"{r['hu_min']:7d} {r['hu_max']:7d} {r['tumour_ml']:8.2f} "
              f"{100 * r['tumour_frac']:7.4f} {r['tumour_slices']:6d}")

    def stats(name: str, values) -> None:
        v = np.asarray(values, dtype=float)
        print(f"{name:22s} min {v.min():10.3f}  median {np.median(v):10.3f}  max {v.max():10.3f}")

    print(f"\nAcross {len(rows)} training cases:")
    for axis in range(3):
        stats(f"shape[{axis}] (voxels)", [r["shape"][axis] for r in rows])
    for axis in range(3):
        stats(f"spacing[{axis}] (mm)", [r["spacing"][axis] for r in rows])
    stats("HU min", [r["hu_min"] for r in rows])
    stats("HU max", [r["hu_max"] for r in rows])
    stats("tumour volume (mL)", [r["tumour_ml"] for r in rows])
    stats("tumour fraction (%)", [100 * r["tumour_frac"] for r in rows])
    stats("tumour axial slices", [r["tumour_slices"] for r in rows])

    print("\nOrientations:", sorted({r["orient"] for r in rows}))
    print("Label values:", sorted({v for r in rows for v in r["labels"]}))
    print("Image/label shapes match:", all(r["same_shape"] for r in rows))


def plot_case(image_path: Path, label_path: Path, out_path: Path) -> None:
    """Plot axial/coronal/sagittal slices through the tumour centre with mask overlay."""
    img = nib.load(image_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    mask = np.asanyarray(nib.load(label_path).dataobj) > 0
    sx, sy, sz = img.header.get_zooms()[:3]

    # Tumour centre in voxel indices (x, y, z); fall back to volume centre if no tumour.
    cx, cy, cz = (np.argwhere(mask).mean(axis=0) if mask.any()
                  else np.array(data.shape) / 2).astype(int)

    # Arrays are indexed (x, y, z). Transposing puts the second index on the vertical axis;
    # origin="lower" keeps it from being flipped. aspect corrects for non-cubic voxels.
    views = [
        (f"axial z={cz}", data[:, :, cz].T, mask[:, :, cz].T, sy / sx),
        (f"coronal y={cy}", data[:, cy, :].T, mask[:, cy, :].T, sz / sx),
        (f"sagittal x={cx}", data[cx, :, :].T, mask[cx, :, :].T, sz / sy),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, image, m, aspect) in zip(axes, views):
        ax.imshow(image, cmap="gray", vmin=LUNG_WINDOW[0], vmax=LUNG_WINDOW[1],
                  origin="lower", aspect=aspect)
        ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.5,
                  origin="lower", aspect=aspect)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{image_path.name}  (lung window {LUNG_WINDOW} HU, tumour in red)")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    print(f"\nSaved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/Task06_Lung"))
    parser.add_argument("--case", default="lung_001")
    parser.add_argument("--out", type=Path, default=Path("outputs/inspect_lung.png"))
    args = parser.parse_args()

    image_paths = sorted(p for p in (args.root / "imagesTr").glob("*.nii.gz")
                         if not p.name.startswith("."))
    rows = [inspect_case(p, args.root / "labelsTr" / p.name) for p in image_paths]
    print_summary(rows)

    plot_case(args.root / "imagesTr" / f"{args.case}.nii.gz",
              args.root / "labelsTr" / f"{args.case}.nii.gz", args.out)


if __name__ == "__main__":
    main()
