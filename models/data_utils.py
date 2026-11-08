"""Shared data code for train.py and evaluate.py.

Reads the split file made in Phase 2 (data/splits.csv). Paths are built from the
location of this file, so the scripts work no matter which folder you run them from.
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]          # repo root
SPLITS_CSV = ROOT / "data" / "splits.csv"
LABEL_MAP = ROOT / "data" / "label_map.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_image_root():
    """Folder that holds one sub-folder per class (torchvision's EuroSAT layout)."""
    for cand in (ROOT / "data" / "eurosat" / "2750", ROOT / "data" / "2750", ROOT / "data" / "EuroSAT"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError("EuroSAT images not found under data/. Run data_scripts/download_data.py first.")


def load_class_names():
    """['AnnualCrop', 'Forest', ...] in label order (0..9), from data/label_map.json."""
    id_to_class = json.loads(LABEL_MAP.read_text())["id_to_class"]
    return [id_to_class[str(i)] for i in range(len(id_to_class))]


def get_transform(train: bool, image_size: int = 224):
    """Resize 64->224, (flips for training only), scale to [0,1], ImageNet normalization."""
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(ops)


class EuroSATDataset(Dataset):
    """Returns (image_tensor, label) for one split: 'train', 'val' or 'test'."""

    def __init__(self, split, transform=None, splits_csv=SPLITS_CSV, image_root=None):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_root = Path(image_root) if image_root else find_image_root()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_root / row["path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label
