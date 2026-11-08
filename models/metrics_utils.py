"""Metrics, error analysis, bias check and plots used by evaluate.py.

This file does not import torch, so it can be tested on its own with plain numpy arrays.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def top_confusions(cm, class_names, k=5):
    """The k most common mistakes as {true, pred, count}, from the confusion matrix."""
    n = len(class_names)
    pairs = [{"true": class_names[i], "pred": class_names[j], "count": int(cm[i, j])}
             for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda d: -d["count"])[:k]


def compute_metrics(y_true, y_pred, probs, class_names, train_counts=None):
    """Accuracy, precision, recall, F1 and AUC-ROC (overall and per class).

    y_true, y_pred : integer class ids, shape (N,)
    probs          : softmax probabilities, shape (N, n_classes), rows sum to 1
    Returns (summary dict, per-class DataFrame, confusion matrix).
    """
    n = len(class_names)
    labels = list(range(n))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # AUC-ROC one-vs-rest: "is it class c or not?" for every class, then averaged
    per_class_auc = [roc_auc_score((y_true == c).astype(int), probs[:, c]) for c in labels]
    macro_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)

    per_class = pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f1,
                              "auc": per_class_auc, "support": support})
    if train_counts is not None:
        per_class["train_images"] = [int(train_counts.get(c, 0)) for c in class_names]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    worst, best = per_class.loc[per_class["f1"].idxmin()], per_class.loc[per_class["f1"].idxmax()]
    summary = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(weighted_f1),
        "macro_auc_ovr": float(macro_auc),
        "worst_class": {"name": str(worst["class"]), "f1": float(worst["f1"])},
        "best_class": {"name": str(best["class"]), "f1": float(best["f1"])},
        "top_confusions": top_confusions(cm, class_names),
    }
    return summary, per_class, cm


# ----------------------------------------------------------------------------
# Bias / robustness check
# ----------------------------------------------------------------------------
def bias_check(test_df, correct, manifest_path):
    """Accuracy for different subgroups of the test set.

    EuroSAT has no demographic or location information, so "subgroups" here are:
      * each land-use class
      * images that are darker / brighter than usual FOR THEIR OWN CLASS
        (judged inside each class, so it is not just 'SeaLake is dark')
      * images with low / high contrast for their class
      * images flagged as statistical outliers or near-duplicates in Phase 2
    test_df must have columns 'path' and 'class_name'; correct is a True/False array.
    Returns a DataFrame with columns group_type, group, n, accuracy (or None).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"[bias check skipped] {manifest_path} not found. Run data_scripts/clean_data.py first.")
        return None

    meta = pd.read_csv(manifest_path, usecols=["path", "brightness", "contrast", "outlier", "near_duplicate"])
    df = test_df[["path", "class_name"]].merge(meta, on="path", how="left").reset_index(drop=True)
    df["correct"] = np.asarray(correct, dtype=bool)

    rows = []

    def add(group_type, group, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() > 0:
            rows.append({"group_type": group_type, "group": group, "n": int(mask.sum()),
                         "accuracy": float(df.loc[mask, "correct"].mean())})

    for cls in sorted(df["class_name"].unique()):
        add("class", cls, df["class_name"] == cls)

    for feat, names in (("brightness", ["darker third", "middle third", "brighter third"]),
                        ("contrast", ["lower third", "middle third", "higher third"])):
        pct = df.groupby("class_name")[feat].rank(pct=True)      # position inside the own class
        bins = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0], labels=names, include_lowest=True)
        for name in names:
            add(f"{feat} (within class)", name, (bins == name).to_numpy())

    for flag in ("outlier", "near_duplicate"):
        col = df[flag].fillna(False).astype(bool).to_numpy()
        add(flag, "flagged", col)
        add(flag, "not flagged", ~col)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def plot_confusion(cm, class_names, path, title):
    n = len(class_names)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)       # each row sums to 1 (= recall)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    _save(fig, path)


def plot_roc(y_true, probs, class_names, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), probs[:, c])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score((y_true == c).astype(int), probs[:, c]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


def plot_curves(history, path, title):
    """Loss and accuracy per epoch (train vs validation) from results/<run>_history.json."""
    ep = history["epochs"]
    x = [e["epoch"] for e in ep]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, [e["train_loss"] for e in ep], marker="o", label="train")
    axes[0].plot(x, [e["val_loss"] for e in ep], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(x, [e["train_acc"] for e in ep], marker="o", label="train")
    axes[1].plot(x, [e["val_acc"] for e in ep], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle(title)
    _save(fig, path)


def plot_errors(errors, image_root, path, title, max_images=12, cols=4):
    """Grid of the most confident wrong predictions. errors needs: path, true_name, pred_name, confidence."""
    errors = errors.head(max_images)
    if errors.empty:
        print("no misclassified images to plot")
        return False
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, e) in zip(axes.ravel(), errors.iterrows()):
        ax.imshow(Image.open(Path(image_root) / e["path"]).convert("RGB"))
        ax.set_title(f"true: {e['true_name']}\npred: {e['pred_name']} ({e['confidence']:.0%})", fontsize=7)
    fig.suptitle(title)
    _save(fig, path)
    return True


# ----------------------------------------------------------------------------
# Comparison table
# ----------------------------------------------------------------------------
def comparison_to_latex(df):
    """Model comparison as a LaTeX table (written by hand so jinja2 is not needed)."""
    def esc(s):
        return str(s).replace("_", r"\_")
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{Params (M)} & \textbf{Test acc} & \textbf{Macro P} & "
             r"\textbf{Macro R} & \textbf{Macro F1} & \textbf{AUC-ROC} & \textbf{ms/img}\\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{esc(r['run'])} & {r['params_millions']:.1f} & {r['accuracy']:.4f} & "
                     f"{r['macro_precision']:.4f} & {r['macro_recall']:.4f} & {r['macro_f1']:.4f} & "
                     f"{r['macro_auc_ovr']:.4f} & {r['latency_ms_per_image']:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
