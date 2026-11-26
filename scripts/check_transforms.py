"""Visibly check the preprocessing pipeline on one MSD Lung case.

Usage:
    uv run python scripts/check_transforms.py [--case lung_001] [--samples 8]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from monai.utils import set_determinism

from ctvlm.data.transforms import get_preprocess_transforms, get_train_transforms


def _fmt_spacing(spacing) -> tuple:
    return tuple(round(float(s), 3) for s in spacing)


def check_preprocessing(item: dict) -> dict:
    """Run preprocessing, print shape/spacing/range/tumour-volume checks, return output."""
    raw = nib.load(item["image"])
    print(f"raw          shape {raw.shape}  spacing {_fmt_spacing(raw.header.get_zooms())}")

    out = get_preprocess_transforms()(item)
    img, lbl = out["image"], out["label"]
    print(f"preprocessed shape {tuple(img.shape)}  spacing {_fmt_spacing(img.pixdim)}")
    print(f"image range  [{img.min():.3f}, {img.max():.3f}]   label values {sorted(lbl.unique().tolist())}")
    print(f"tumour voxels raw {int((np.asanyarray(nib.load(item['label']).dataobj) > 0).sum())}"
          f" -> preprocessed {int((lbl > 0).sum())}")
    return out


def save_nifti(out: dict, out_dir: Path, case: str) -> None:
    """Write preprocessed image + label as NIfTI for a 3D viewer.

    The affine maps voxel indices to mm in scanner space, so viewers show true geometry.
    vol[0] drops the channel dim: NIfTI stores plain (X, Y, Z) volumes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = out["image"].affine.numpy()
    for name, dtype in [("image", np.float32), ("label", np.uint8)]:
        path = out_dir / f"{case}_preprocessed_{name}.nii.gz"
        nib.save(nib.Nifti1Image(out[name][0].numpy().astype(dtype), affine), path)
        print(f"Saved {path}")


def check_crops(item: dict, num_samples: int) -> list[dict]:
    """Run train transforms, print each crop's shape and tumour voxel count."""
    crops = get_train_transforms(num_samples=num_samples)(item)
    print(f"\n{len(crops)} train crops:")
    for i, c in enumerate(crops):
        print(f"  crop {i}: image {tuple(c['image'].shape)}  label {tuple(c['label'].shape)}"
              f"  tumour voxels {int((c['label'] > 0).sum())}")
    return crops


def _tumour_centre_z(mask: np.ndarray) -> int:
    """Axial index through the tumour centre, or the middle slice if there is no tumour."""
    return int(np.argwhere(mask)[:, 2].mean()) if mask.any() else mask.shape[2] // 2


def plot_overlays(out: dict, crops: list[dict], out_path: Path) -> None:
    """Axial slice through tumour (full volume) + positive crop, mask in red."""
    img = out["image"]
    m = out["label"][0].numpy() > 0
    z = _tumour_centre_z(m)
    pos = next((c for c in crops if (c["label"] > 0).any()), crops[0])
    pm = pos["label"][0].numpy() > 0
    pz = _tumour_centre_z(pm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (title, im, mk) in zip(axes, [
        (f"preprocessed axial z={z}  {tuple(img.shape[1:])}", img[0, :, :, z].numpy(), m[:, :, z]),
        (f"positive 96^3 crop, z={pz}", pos["image"][0, :, :, pz].numpy(), pm[:, :, pz]),
    ]):
        # RAS arrays are (x->Right, y->Anterior). Transpose to (y, x), then flip both axes
        # so anterior is up and patient right is on screen left (radiological view).
        ax.imshow(np.flip(im.T), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.flip(np.ma.masked_where(~mk, mk).T), cmap="autumn", alpha=0.5)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    print(f"\nSaved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/Task06_Lung"))
    parser.add_argument("--case", default="lung_001")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/check_transforms.png"))
    args = parser.parse_args()
    set_determinism(seed=0)

    item = {
        "image": str(args.root / "imagesTr" / f"{args.case}.nii.gz"),
        "label": str(args.root / "labelsTr" / f"{args.case}.nii.gz"),
    }
    out = check_preprocessing(item)
    save_nifti(out, args.out.parent, args.case)
    crops = check_crops(item, args.samples)
    plot_overlays(out, crops, args.out)


if __name__ == "__main__":
    main()
