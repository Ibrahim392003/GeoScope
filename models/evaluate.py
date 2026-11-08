"""Phase 4 - Evaluate trained models on the TEST split (never used during training).

Run from the repo root after training:
    python models/evaluate.py --runs resnet18 efficientnet vit

For every run it needs models/checkpoints/<run>_best.pth. It writes:
    results/<run>_metrics.json      accuracy, precision, recall, F1, AUC-ROC, top confusions
    results/<run>_per_class.csv     the same metrics for each of the 10 classes
    results/<run>_errors.csv        every misclassified test image with its confidence
    results/<run>_bias.csv          accuracy by subgroup (bias / robustness check)
    results/model_comparison.csv/.tex   one row per model (for the report)
    figures/cm_<run>.png, roc_<run>.png, curves_<run>.png, errors_<run>.png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from architectures import count_parameters, get_model
from data_utils import (ROOT, SPLITS_CSV, EuroSATDataset, find_image_root, get_transform,
                        load_class_names)
from metrics_utils import (bias_check, comparison_to_latex, compute_metrics, plot_confusion,
                           plot_curves, plot_errors, plot_roc)

KNOWN_MODELS = ("resnet18", "efficientnet", "vit")
FIG_DIR = ROOT / "figures"


def resolve_config(run, results_dir, default_dropout):
    """Find out which architecture/dropout a run used (from its history file if it exists)."""
    hist_path = results_dir / f"{run}_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        cfg = history["config"]
        return cfg["model"], cfg.get("dropout", 0.0), history
    for name in KNOWN_MODELS:                    # no history file: guess from the run name
        if run.startswith(name):
            return name, default_dropout, None
    raise ValueError(f"Cannot tell which model '{run}' is. Name it like 'resnet18' or 'vit_lr1e-5'.")


@torch.no_grad()
def predict(model, loader, device):
    """Softmax probabilities and true labels for every image in the loader."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    probs = np.concatenate(probs).astype(np.float64)
    probs /= probs.sum(axis=1, keepdims=True)    # make rows sum to exactly 1 (needed for AUC)
    return np.concatenate(labels), probs


@torch.no_grad()
def measure_latency(model, device, batch=32, warmup=3, iters=10):
    """Average inference time in milliseconds per image on random 224x224 input."""
    model.eval()
    x = torch.randn(batch, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()                 # wait for the GPU before starting the clock
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / (iters * batch) * 1000


def evaluate_run(run, args, device, class_names):
    ckpt_dir, results_dir = Path(args.ckpt_dir), Path(args.results_dir)
    ckpt = ckpt_dir / f"{run}_best.pth"
    if not ckpt.exists():
        print(f"[skip] {ckpt} not found")
        return None

    print(f"\n=== Evaluating {run} ===")
    model_name, dropout, history = resolve_config(run, results_dir, args.dropout)
    model = get_model(model_name, num_classes=len(class_names), pretrained=False, dropout=dropout)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    test_set = EuroSATDataset("test", transform=get_transform(train=False))
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        pin_memory=device.type == "cuda")
    y_true, probs = predict(model, loader, device)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    splits = pd.read_csv(SPLITS_CSV)
    train_counts = splits[splits["split"] == "train"]["class_name"].value_counts().to_dict()
    summary, per_class, cm = compute_metrics(y_true, y_pred, probs, class_names, train_counts)
    summary.update(run=run, model=model_name, dropout=dropout, test_images=int(len(y_true)),
                   num_parameters=count_parameters(model),
                   checkpoint_mb=round(ckpt.stat().st_size / 1e6, 1),
                   latency_ms_per_image=round(measure_latency(model, device), 3))
    if history:
        summary["best_val_acc"] = history.get("best_val_acc")
        summary["best_epoch"] = history.get("best_epoch")
        summary["train_time_sec"] = history.get("train_time_sec")
    results_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Error analysis: every wrong prediction, most confident mistakes first
    test_df = test_set.df
    wrong = y_pred != y_true
    errors = pd.DataFrame({"path": test_df["path"], "true_name": [class_names[i] for i in y_true],
                           "pred_name": [class_names[i] for i in y_pred], "confidence": confidence})[wrong]
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv(results_dir / f"{run}_errors.csv", index=False)

    bias = bias_check(test_df, ~wrong, ROOT / "data" / "clean_manifest.csv")
    if bias is not None:
        bias.to_csv(results_dir / f"{run}_bias.csv", index=False)

    per_class.to_csv(results_dir / f"{run}_per_class.csv", index=False)
    (results_dir / f"{run}_metrics.json").write_text(json.dumps(summary, indent=2))

    plot_confusion(cm, class_names, FIG_DIR / f"cm_{run}.png", f"Confusion matrix - {run} (test set)")
    plot_roc(y_true, probs, class_names, FIG_DIR / f"roc_{run}.png", f"ROC curves (one-vs-rest) - {run}")
    plot_errors(errors, find_image_root(), FIG_DIR / f"errors_{run}.png",
                f"Most confident mistakes - {run}")
    if history:
        plot_curves(history, FIG_DIR / f"curves_{run}.png", f"Training curves - {run}")

    print(f"Test accuracy {summary['accuracy']:.4f} | macro F1 {summary['macro_f1']:.4f} | "
          f"AUC-ROC {summary['macro_auc_ovr']:.4f} | {summary['latency_ms_per_image']} ms/img")
    print(f"Wrong predictions: {int(wrong.sum())} of {len(y_true)}")
    print("Most common mistakes:")
    for c in summary["top_confusions"]:
        print(f"   {c['true']} -> {c['pred']}: {c['count']}")
    print(f"Weakest class: {summary['worst_class']['name']} (F1 {summary['worst_class']['f1']:.3f})")
    if bias is not None:
        print("\nBias / robustness check (accuracy by subgroup):")
        print(bias.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained GeoScope models on the test set")
    ap.add_argument("--runs", nargs="+", default=list(KNOWN_MODELS),
                    help="run names = checkpoint file names without '_best.pth'")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="only used for runs that have no history file")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--ckpt_dir", default=str(ROOT / "models" / "checkpoints"))
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    class_names = load_class_names()

    summaries = [s for s in (evaluate_run(r, args, device, class_names) for r in args.runs) if s]
    if not summaries:
        raise SystemExit("No checkpoints were found - nothing evaluated.")

    table = pd.DataFrame([{
        "run": s["run"], "params_millions": s["num_parameters"] / 1e6, "best_val_acc": s.get("best_val_acc"),
        "accuracy": s["accuracy"], "macro_precision": s["macro_precision"],
        "macro_recall": s["macro_recall"], "macro_f1": s["macro_f1"],
        "macro_auc_ovr": s["macro_auc_ovr"], "latency_ms_per_image": s["latency_ms_per_image"],
        "checkpoint_mb": s["checkpoint_mb"]} for s in summaries])
    results_dir = Path(args.results_dir)
    table.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "model_comparison.tex").write_text(comparison_to_latex(table))
    print("\n=== Model comparison (test set) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {results_dir / 'model_comparison.csv'} and model_comparison.tex")


if __name__ == "__main__":
    main()
