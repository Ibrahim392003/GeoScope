"""Phase 2 - Data cleaning for image data.

Raw images are NEVER modified or deleted. Instead we build a manifest
(data/clean_manifest.csv) with a `status` and a `keep` flag per image, so every
cleaning decision is reproducible and auditable.

Checks (image-data equivalents of the usual tabular checks):
  Missing values   -> unreadable/corrupt files, wrong mode/size, missing features,
                      empty class folders (missing labels)
  Duplicates       -> exact pixel-level duplicates (removed), images that appear
                      under two different labels (removed: label conflict),
                      near-duplicates via perceptual dHash (flagged only)
  Outliers         -> blank no-data patches (flat and pure black/white -> removed;
                      flat patches of normal brightness such as calm water are KEPT);
                      statistical outliers per class, 3*IQR fence on brightness,
                      contrast and log-sharpness (flagged, kept by default because
                      they are usually legitimate scene variation)

Usage:  python data_scripts/clean_data.py [--workers 4] [--drop-outliers] [--drop-near-dups]
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import (DATA_DIR, FIG_DIR, MANIFEST, REPORT_DIR, ensure_dirs,
                    find_image_root, list_images, update_stats)

EXPECTED_SIZE = (64, 64)
FEATURES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
            "brightness", "contrast", "sharpness"]
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ----------------------------------------------------------------------------
# Per-image analysis
# ----------------------------------------------------------------------------
def dhash(arr_uint8):
    """64-bit difference hash: robust to small brightness/compression changes."""
    small = np.asarray(Image.fromarray(arr_uint8).convert("L").resize((9, 8), Image.LANCZOS))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int.from_bytes(np.packbits(bits).tobytes(), 'big'):016x}"


def analyse(args):
    cls, path, rel = args
    rec = {"path": rel, "class_name": cls, "status": "ok"}
    try:
        with Image.open(path) as im:
            im.verify()                      # detects truncated / corrupt files
        with Image.open(path) as im:
            im.load()
            rec["mode"], (rec["width"], rec["height"]) = im.mode, im.size
            if im.mode != "RGB":
                rec["status"] = "bad_format"
                return rec
            if im.size != EXPECTED_SIZE:
                rec["status"] = "bad_format"
                return rec
            arr = np.asarray(im, dtype=np.uint8)
    except Exception as exc:                 # any read failure = corrupt file
        rec["status"] = "corrupt"
        rec["error"] = type(exc).__name__
        return rec

    f = arr.astype(np.float32)
    ch_mean, ch_std = f.mean(axis=(0, 1)), f.std(axis=(0, 1))
    gray = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    rec.update(
        mean_r=float(ch_mean[0]), mean_g=float(ch_mean[1]), mean_b=float(ch_mean[2]),
        std_r=float(ch_std[0]), std_g=float(ch_std[1]), std_b=float(ch_std[2]),
        brightness=float(gray.mean()), contrast=float(gray.std()),
        sharpness=float(lap.var()),
        pix_md5=hashlib.md5(arr.tobytes()).hexdigest(),
        dhash=dhash(arr),
    )
    return rec


# ----------------------------------------------------------------------------
# Dataset-level checks
# ----------------------------------------------------------------------------
def flag_near_duplicates(hashes_hex, threshold, chunk=256):
    """Chunked pairwise Hamming distance on 64-bit hashes (no O(n^2) memory)."""
    h = np.array([int(x, 16) for x in hashes_hex], dtype=np.uint64)
    n = len(h)
    flags = np.zeros(n, dtype=bool)
    pairs = 0
    cols = np.arange(n)
    for s in range(0, n, chunk):
        blk = h[s:s + chunk]
        x = np.bitwise_xor(blk[:, None], h[None, :])
        dist = POPCOUNT[x.view(np.uint8).reshape(x.shape[0], n, 8)].sum(axis=2)
        mask = (dist <= threshold) & (cols[None, :] > (s + np.arange(len(blk)))[:, None])
        pairs += int(mask.sum())
        flags[s:s + len(blk)] |= mask.any(axis=1)
        flags |= mask.any(axis=0)
    return flags, pairs


def flag_outliers(df, k=3.0):
    """Per-class Tukey fences (k*IQR) on brightness, contrast and log-sharpness."""
    out = pd.Series(False, index=df.index)
    reason = pd.Series("", index=df.index)
    for feat in ("brightness", "contrast", "sharpness"):
        v = np.log1p(df[feat]) if feat == "sharpness" else df[feat]
        g = v.groupby(df["class_name"])
        q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        bad = (v < q1 - k * iqr) | (v > q3 + k * iqr)
        out |= bad
        reason[bad] = reason[bad] + feat + ";"
    return out, reason


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--near-dup-threshold", type=int, default=3, help="max Hamming distance (of 64 bits)")
    ap.add_argument("--drop-outliers", action="store_true")
    ap.add_argument("--drop-near-dups", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    root = find_image_root(args.data_dir)
    images = list_images(root)
    class_dirs = sorted(d.name for d in Path(root).iterdir() if d.is_dir())
    empty_classes = sorted(set(class_dirs) - {c for c, _ in images})
    print(f"Image root: {root}\nFound {len(images)} files in {len(class_dirs)} class folders")

    work = [(c, str(p), f"{c}/{p.name}") for c, p in images]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            records = pool.map(analyse, work, chunksize=256)
    else:
        records = [analyse(w) for w in work]
    df = pd.DataFrame(records)
    total_raw = len(df)

    # --- missing values (features) -------------------------------------------
    ok = df["status"] == "ok"
    df.loc[ok & df[FEATURES].isna().any(axis=1), "status"] = "missing_features"

    # --- label conflicts: identical pixels under different labels -------------
    ok = df["status"] == "ok"
    n_labels = df[ok].groupby("pix_md5")["class_name"].nunique()
    conflict = df["pix_md5"].isin(n_labels[n_labels > 1].index) & ok
    df.loc[conflict, "status"] = "conflicting_label"

    # --- exact duplicates (keep first) ----------------------------------------
    ok = df["status"] == "ok"
    dup = df[ok].duplicated("pix_md5", keep="first")
    df.loc[dup[dup].index, "status"] = "exact_duplicate"

    # --- degenerate (blank / saturated) images ---------------------------------
    ok = df["status"] == "ok"
    # Only truly blank "no-data" patches: flat AND pure black/white.
    # Flat patches with normal brightness (calm water, dense forest) are valid samples.
    degenerate = ok & (df["contrast"] < 1.0) & ((df["brightness"] < 5) | (df["brightness"] > 250))
    df.loc[degenerate, "status"] = "degenerate"

    # --- near duplicates (flag only) ------------------------------------------
    ok = df["status"] == "ok"
    df["near_duplicate"] = False
    nd_flags, nd_pairs = flag_near_duplicates(df.loc[ok, "dhash"].tolist(), args.near_dup_threshold)
    df.loc[ok, "near_duplicate"] = nd_flags

    # --- statistical outliers (flag, optionally drop) -------------------------
    df["outlier"], df["outlier_reason"] = False, ""
    o, r = flag_outliers(df[ok])
    df.loc[ok, "outlier"], df.loc[ok, "outlier_reason"] = o, r

    df["keep"] = df["status"] == "ok"
    if args.drop_outliers:
        df.loc[df["outlier"], "keep"] = False
    if args.drop_near_dups:
        df.loc[df["near_duplicate"], "keep"] = False

    # --- outputs -----------------------------------------------------------------
    df.to_csv(MANIFEST, index=False)
    counts = df["status"].value_counts().to_dict()
    summary = {
        "image_root": str(root),
        "total_raw": total_raw,
        "status_counts": counts,
        "empty_class_folders": empty_classes,
        "near_duplicate_pairs": nd_pairs,
        "near_duplicate_images_flagged": int(df["near_duplicate"].sum()),
        "outliers_flagged": int(df["outlier"].sum()),
        "total_kept": int(df["keep"].sum()),
        "policy": {"drop_outliers": args.drop_outliers, "drop_near_dups": args.drop_near_dups,
                   "near_dup_threshold": args.near_dup_threshold, "outlier_rule": "3*IQR per class"},
    }
    (REPORT_DIR / "cleaning_report.json").write_text(json.dumps(summary, indent=2))
    update_stats(
        TotalRaw=total_raw,
        Corrupt=int(counts.get("corrupt", 0)),
        BadFormat=int(counts.get("bad_format", 0)),
        ConflictLabel=int(counts.get("conflicting_label", 0)),
        ExactDup=int(counts.get("exact_duplicate", 0)),
        Degenerate=int(counts.get("degenerate", 0)),
        NearDupFlagged=int(df["near_duplicate"].sum()),
        OutlierFlagged=int(df["outlier"].sum()),
        TotalClean=int(df["keep"].sum()),
    )
    make_figures(df)

    print("\n=== Cleaning summary ===")
    for k, v in counts.items():
        print(f"  {k:<20}{v:>8}")
    print(f"  empty class folders: {empty_classes or 'none'}")
    print(f"  near-duplicate images flagged (kept): {summary['near_duplicate_images_flagged']}")
    print(f"  outliers flagged (kept): {summary['outliers_flagged']}")
    print(f"  FINAL kept: {summary['total_kept']} / {total_raw}")
    print(f"Manifest -> {MANIFEST}")


def make_figures(df):
    # 1) what was removed and why
    removed = df.loc[df["status"] != "ok", "status"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if len(removed):
        ax.barh(removed.index, removed.values, color="#c0504d")
        for i, v in enumerate(removed.values):
            ax.text(v, i, f" {v}", va="center")
    else:
        ax.text(0.5, 0.5, "No images removed", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_xlabel("Number of images")
    ax.set_title("Images removed by the cleaning pipeline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cleaning_summary.png", dpi=200)
    plt.close(fig)

    # 2) outlier detection view
    d = df[df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(d.loc[~d["outlier"], "brightness"], d.loc[~d["outlier"], "contrast"],
               s=4, alpha=0.35, label="inlier")
    ax.scatter(d.loc[d["outlier"], "brightness"], d.loc[d["outlier"], "contrast"],
               s=14, color="red", label="outlier (3*IQR, per class)")
    ax.set_xlabel("Mean brightness (0-255)")
    ax.set_ylabel("Contrast (std of gray, 0-255)")
    ax.set_title("Outlier detection on image statistics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "outlier_detection.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()