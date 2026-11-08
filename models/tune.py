"""Phase 3 - Hyperparameter search.

Trains one model for every combination of the settings below and writes the results to
results/hparam_results_<model>.csv (this becomes the hyperparameter table in the report).
The CSV is rewritten after every run, so a Colab disconnect does not lose finished runs.

Example (run from the repo root, ~20 minutes for ResNet-18 on a T4 GPU):
    python models/tune.py --model resnet18 --lrs 1e-3 1e-4 1e-5 --dropouts 0 0.3 --epochs 3
"""
import argparse
import csv
import itertools

from data_utils import ROOT
from train import train_model

FIELDS = ["run_name", "model", "lr", "batch_size", "dropout", "weight_decay", "epochs",
          "best_val_acc", "best_val_loss", "best_epoch", "train_time_sec"]


def main():
    ap = argparse.ArgumentParser(description="Grid search over training hyperparameters")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet", "vit"])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[32])
    ap.add_argument("--dropouts", type=float, nargs="+", default=[0.0, 0.3])
    ap.add_argument("--weight_decays", type=float, nargs="+", default=[0.01])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", default=str(ROOT / "models" / "checkpoints" / "tuning"))
    ap.add_argument("--results_dir", default=str(ROOT / "results" / "tuning"))
    args = ap.parse_args()

    out_csv = ROOT / "results" / f"hparam_results_{args.model}.csv"   # one table per model
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.lrs, args.batch_sizes, args.dropouts, args.weight_decays))
    print(f"Grid search: {len(grid)} runs for {args.model}, {args.epochs} epochs each")
    rows = []
    for i, (lr, bs, do, wd) in enumerate(grid, start=1):
        run_name = f"{args.model}_lr{lr:g}_bs{bs}_do{do:g}_wd{wd:g}"
        print(f"\n=== Run {i}/{len(grid)}: {run_name} ===")
        rows.append(train_model(args.model, batch_size=bs, epochs=args.epochs, lr=lr, dropout=do,
                                weight_decay=wd, seed=args.seed, run_name=run_name,
                                ckpt_dir=args.ckpt_dir, results_dir=args.results_dir))
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    best = max(rows, key=lambda r: r["best_val_acc"])
    print("\n=== Search finished ===")
    for r in sorted(rows, key=lambda r: -r["best_val_acc"]):
        print(f"  lr={r['lr']:<8g} batch={r['batch_size']:<4} dropout={r['dropout']:<4g} "
              f"wd={r['weight_decay']:<6g} -> val acc {r['best_val_acc']:.4f}")
    print(f"\nBest: {best['run_name']} (val acc {best['best_val_acc']:.4f})")
    print(f"Table saved to {out_csv}")


if __name__ == "__main__":
    main()
