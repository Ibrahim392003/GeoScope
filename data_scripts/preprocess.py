
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (IMAGENET_MEAN, IMAGENET_STD, LABEL_MAP, MANIFEST, SEED, SPLITS,
                    ensure_dirs, find_image_root, update_stats)

try:                                   # torch is only needed for training-time use
    from torch.utils.data import Dataset
except ImportError:                    # lets this file run on a torch-free machine
    Dataset = object


# ----------------------------------------------------------------------------
# Encoding + splitting
# ----------------------------------------------------------------------------
def encode_labels(df):
    """Integer-encode class names (sorted -> identical to torchvision's EuroSAT order)."""
    classes = sorted(df["class_name"].unique())
    mapping = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["label"] = df["class_name"].map(mapping)
    return df, mapping


def stratified_split(df, ratios=(0.70, 0.15, 0.15), seed=SEED):
    """Per-class shuffle & cut so every split keeps the class proportions."""
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=df.index)
    for _, idx in df.groupby("label").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_tr = int(round(ratios[0] * len(idx)))
        n_va = int(round(ratios[1] * len(idx)))
        split.loc[idx[:n_tr]] = "train"
        split.loc[idx[n_tr:n_tr + n_va]] = "val"
        split.loc[idx[n_tr + n_va:]] = "test"
    df = df.copy()
    df["split"] = split
    return df


def train_channel_stats(df_train):
    """Exact per-channel mean/std on the [0,1] scale from per-image statistics.

    All images have the same pixel count, so
        var_total = mean(var_i) + var(mean_i)
    """
    mean = np.array([df_train[f"mean_{c}"].mean() for c in "rgb"]) / 255.0
    var = np.array([(df_train[f"std_{c}"] ** 2).mean() + df_train[f"mean_{c}"].var(ddof=0)
                    for c in "rgb"]) / 255.0 ** 2
    return mean, np.sqrt(var)


# ----------------------------------------------------------------------------
# Scaling / normalization (used by the training phase)
# ----------------------------------------------------------------------------
def get_transforms(train=False, image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """64x64 -> image_size, scale to [0,1], normalize.

    * Resize: ResNet-18 / EfficientNet / ViT-B/16 were pretrained on 224x224 ImageNet.
    * ToTensor: uint8 [0,255] -> float [0,1] (scaling).
    * Normalize: standardise each channel with the statistics the pretrained
      weights were trained with (use the EuroSAT stats instead if training from scratch).
    * Augmentation (train only): flips - land-use patches have no canonical orientation.
    """
    from torchvision import transforms as T
    ops = [T.Resize((image_size, image_size))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class GeoScopeDataset(Dataset):
    """Reads data/splits.csv, returns (tensor, label) for one split."""

    def __init__(self, split, transform=None, splits_csv=SPLITS, image_root=None):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root = Path(image_root) if image_root else find_image_root()
        self.transform = transform or get_transforms(train=(split == "train"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


# ----------------------------------------------------------------------------
def main():
    ensure_dirs()
    if not MANIFEST.exists():
        raise SystemExit("Run data_scripts/clean_data.py first (manifest not found).")
    df = pd.read_csv(MANIFEST)
    df = df[df["keep"]].reset_index(drop=True)

    df, mapping = encode_labels(df)
    df = stratified_split(df)

    LABEL_MAP.write_text(json.dumps({"class_to_id": mapping,
                                     "id_to_class": {v: k for k, v in mapping.items()}}, indent=2))
    df[["path", "class_name", "label", "split"]].to_csv(SPLITS, index=False)

    mean, std = train_channel_stats(df[df["split"] == "train"])
    sizes = df["split"].value_counts()
    update_stats(
        Train=int(sizes["train"]), Val=int(sizes["val"]), Test=int(sizes["test"]),
        DatasetMean=", ".join(f"{m:.4f}" for m in mean),
        DatasetStd=", ".join(f"{s:.4f}" for s in std),
    )

    print("Label encoding:", mapping)
    print("\nSplit sizes:", sizes.to_dict())
    print("\nImages per class per split:")
    print(pd.crosstab(df["class_name"], df["split"]))
    print(f"\nEuroSAT train mean (RGB): {np.round(mean, 4)}")
    print(f"EuroSAT train std  (RGB): {np.round(std, 4)}")
    print(f"ImageNet mean/std used for pretrained models: {IMAGENET_MEAN} / {IMAGENET_STD}")
    print(f"\nSaved {SPLITS} and {LABEL_MAP}")


if __name__ == "__main__":
    main()
