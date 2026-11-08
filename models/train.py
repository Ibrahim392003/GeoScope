"""Phase 3 - Train one model and save its best checkpoint + per-epoch history.

Examples (run from the repo root):
    python models/train.py --model resnet18 --epochs 10
    python models/train.py --model vit --epochs 5 --lr 1e-5 --dropout 0.3
Outputs:
    models/checkpoints/<run_name>_best.pth    best weights by validation accuracy (git-ignored)
    results/<run_name>_history.json           settings + loss/accuracy per epoch (commit this)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from architectures import count_parameters, get_model
from data_utils import ROOT, EuroSATDataset, get_transform

DEFAULT_CKPT_DIR = ROOT / "models" / "checkpoints"
DEFAULT_RESULTS_DIR = ROOT / "results"


def set_seed(seed):
    """Same seed -> (almost) the same results every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise only measures."""
    training = optimizer is not None
    model.train(training)                       # train mode enables dropout, eval mode disables it
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):      # no gradients needed for validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model_name, batch_size=32, epochs=5, lr=1e-4, dropout=0.0, weight_decay=0.01,
                seed=42, run_name=None, ckpt_dir=DEFAULT_CKPT_DIR, results_dir=DEFAULT_RESULTS_DIR,
                num_workers=2):
    run_name = run_name or model_name
    ckpt_dir, results_dir = Path(ckpt_dir), Path(results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Run '{run_name}' | device: {device}")

    # Data: Phase 2 split (train used for learning, val used to pick the best epoch)
    train_set = EuroSATDataset("train", transform=get_transform(train=True))
    val_set = EuroSATDataset("val", transform=get_transform(train=False))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # Model, loss, optimizer
    model = get_model(model_name, num_classes=10, pretrained=True, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "run_name": run_name,
        "config": {"model": model_name, "batch_size": batch_size, "epochs": epochs, "lr": lr,
                   "dropout": dropout, "weight_decay": weight_decay, "seed": seed,
                   "optimizer": "AdamW", "loss": "CrossEntropyLoss"},
        "num_parameters": count_parameters(model),
        "epochs": [],
    }
    ckpt_path = ckpt_dir / f"{run_name}_best.pth"
    history_path = results_dir / f"{run_name}_history.json"
    best_val_acc, best_val_loss, best_epoch = 0.0, float("inf"), 0
    start = time.time()

    print(f"-> Training {model_name} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["epochs"].append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                                  "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:               # keep the best epoch only
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"   [+] Saved new best model checkpoint to {ckpt_path}")

        # Saved after every epoch, so a Colab disconnect does not lose the history
        history.update(best_val_acc=best_val_acc, best_val_loss=best_val_loss, best_epoch=best_epoch,
                       train_time_sec=round(time.time() - start, 1))
        history_path.write_text(json.dumps(history, indent=2))

    print(f"-> Done. Best val acc {best_val_acc:.4f} at epoch {best_epoch}. History: {history_path}")
    return {"run_name": run_name, "model": model_name, "lr": lr, "batch_size": batch_size,
            "dropout": dropout, "weight_decay": weight_decay, "epochs": epochs,
            "best_val_acc": best_val_acc, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
            "train_time_sec": history["train_time_sec"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoScope models")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "efficientnet", "vit"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None, help="default: the model name")
    parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CKPT_DIR),
                        help="where to save weights (e.g. a Google Drive folder)")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    train_model(args.model, args.batch_size, args.epochs, args.lr, args.dropout, args.weight_decay,
                args.seed, args.run_name, args.ckpt_dir, args.results_dir)
