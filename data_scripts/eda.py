"""Phase 2 - Exploratory data analysis. Saves all charts to figures/.

Run after clean_data.py:   python data_scripts/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import (FIG_DIR, IMAGENET_MEAN, IMAGENET_STD, MANIFEST, SEED, ensure_dirs,
                    find_image_root)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load_manifest():
    if not MANIFEST.exists():
        raise SystemExit("Run data_scripts/clean_data.py first (manifest not found).")
    df = pd.read_csv(MANIFEST)
    return df, df[df["keep"]].reset_index(drop=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def class_distribution(raw, clean):
    classes = sorted(raw["class_name"].unique())
    r = raw["class_name"].value_counts().reindex(classes)
    c = clean["class_name"].value_counts().reindex(classes).fillna(0)
    x, w = np.arange(len(classes)), 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, r.values, w, label="raw", color="skyblue")
    ax.bar(x + w / 2, c.values, w, label="after cleaning", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title(f"EuroSAT class distribution (imbalance ratio max/min = {r.max() / r.min():.2f})")
    ax.legend()
    save(fig, "class_distribution.png")


def sample_grid(clean, root, per_class=6):
    rng = np.random.default_rng(SEED)
    classes = sorted(clean["class_name"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.3, len(classes) * 1.3))
    for i, cls in enumerate(classes):
        paths = clean.loc[clean["class_name"] == cls, "path"].to_numpy()
        pick = rng.choice(paths, size=min(per_class, len(paths)), replace=False)
        for j in range(per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(Image.open(root / pick[j]).convert("RGB"))
            if j == 0:
                ax.text(-0.08, 0.5, cls, transform=ax.transAxes, ha="right", va="center", fontsize=8)
    fig.subplots_adjust(left=0.2, top=0.95, wspace=0.05, hspace=0.05)
    fig.suptitle("Random samples per class (64x64 Sentinel-2 RGB patches)", y=0.985)
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG_DIR / "sample_grid.png")


def sample_pixels(clean, root, per_class=100):
    rng = np.random.default_rng(SEED)
    chunks = []
    for cls, g in clean.groupby("class_name"):
        pick = rng.choice(g["path"].to_numpy(), size=min(per_class, len(g)), replace=False)
        chunks += [np.asarray(Image.open(root / p).convert("RGB"), dtype=np.float32) / 255.0 for p in pick]
    return np.stack(chunks)                       # (N, 64, 64, 3) in [0,1]


def channel_histograms(px):
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (name, col) in enumerate(zip("RGB", ("red", "green", "blue"))):
        ax.hist(px[..., k].ravel() * 255, bins=64, range=(0, 255), histtype="step",
                color=col, label=name, density=True)
    ax.set_xlabel("Pixel value (0-255)")
    ax.set_ylabel("Density")
    ax.set_title("Per-channel pixel intensity distribution (sampled images)")
    ax.legend()
    save(fig, "channel_histograms.png")


def brightness_by_class(clean):
    classes = sorted(clean["class_name"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, feat, title in zip(axes, ("brightness", "contrast"),
                               ("Mean brightness per class", "Contrast (gray std) per class")):
        ax.boxplot([clean.loc[clean["class_name"] == c, feat] for c in classes],
                   showfliers=True, flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels(classes, rotation=60, ha="right")
        ax.set_title(title)
    save(fig, "brightness_by_class.png")


def feature_correlation(clean):
    cols = ["mean_r", "mean_g", "mean_b", "brightness", "contrast", "sharpness"]
    corr = clean[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation of image-level features")
    save(fig, "feature_correlation.png")


def normalization_effect(px):
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    normed = (px - mean) / std
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in ((axes[0], px, "After scaling to [0, 1]"),
                            (axes[1], normed, "After ImageNet normalization")):
        for k, col in enumerate(("red", "green", "blue")):
            ax.hist(data[..., k].ravel(), bins=64, histtype="step", color=col, density=True)
        ax.set_title(title)
        ax.set_xlabel("Value")
    axes[0].set_ylabel("Density")
    save(fig, "normalization_effect.png")


def main():
    ensure_dirs()
    raw, clean = load_manifest()
    root = find_image_root()
    class_distribution(raw, clean)
    sample_grid(clean, root)
    px = sample_pixels(clean, root)
    channel_histograms(px)
    brightness_by_class(clean)
    feature_correlation(clean)
    normalization_effect(px)
    print("\nPer-class mean brightness / contrast:")
    print(clean.groupby("class_name")[["brightness", "contrast", "sharpness"]].mean().round(1))


if __name__ == "__main__":
    main()
