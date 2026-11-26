"""MONAI preprocessing pipelines for CT image/label pairs.

Order: load -> channel-first -> RAS orientation -> resample to TARGET_SPACING
-> clip HU window and scale to [0, 1] -> crop to body -> (train only) 96^3 crops.
"""

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)

TARGET_SPACING = (1.5, 1.5, 1.5)  # mm per voxel, isotropic
HU_WINDOW = (-1000.0, 1000.0)  # clipped, then mapped linearly to [0, 1]
BODY_HU = -500.0  # voxels above this count as body for the foreground crop
CROP_SIZE = (96, 96, 96)


def _is_body(x):
    """Foreground selector for CropForegroundd, applied to the already-scaled image."""
    lo, hi = HU_WINDOW
    return x > (BODY_HU - lo) / (hi - lo)


def get_preprocess_transforms(keys=("image", "label")) -> Compose:
    """Deterministic preprocessing. Output: (1, X, Y, Z) tensors at TARGET_SPACING."""
    has_label = "label" in keys
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),  # (X, Y, Z) -> (1, X, Y, Z)
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=TARGET_SPACING,
            # image: smooth interpolation; label: nearest so it stays 0/1
            mode=("bilinear", "nearest") if has_label else "bilinear",
        ),
        ScaleIntensityRanged(
            keys="image", a_min=HU_WINDOW[0], a_max=HU_WINDOW[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # Bounding box of the body; drops surrounding air (table is mostly kept).
        CropForegroundd(keys=keys, source_key="image", select_fn=_is_body),
    ])


def get_train_transforms(num_samples: int = 2) -> Compose:
    """Preprocessing + random CROP_SIZE patches, half centred on tumour, half elsewhere.

    Returns a list of `num_samples` dicts per input volume.
    """
    return Compose([
        get_preprocess_transforms(),
        SpatialPadd(keys=("image", "label"), spatial_size=CROP_SIZE),  # in case a dim < 96
        RandCropByPosNegLabeld(
            keys=("image", "label"), label_key="label", spatial_size=CROP_SIZE,
            pos=1, neg=1, num_samples=num_samples, image_key="image",
        ),
    ])
