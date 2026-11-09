"""Phase 4 - Grad-CAM explainability for the trained models.

Run from the repo root after training (same flags style as evaluate.py):
    python models/gradcam.py --runs resnet18 efficientnet vit_8ep \
        --ckpt_dir /content/drive/MyDrive/GeoScope_ckpt \
        --results_dir /content/drive/MyDrive/GeoScope_results

For every run it writes (to figures/):
    gradcam_correct_<run>.png   most confident CORRECT test image of each class + heatmap
    gradcam_errors_<run>.png    most confident MISTAKES + heatmap (what fooled the model?)
and once for all runs:
    gradcam_compare.png         same test images, one column per model (for the report)

Grad-CAM = take the last convolutional / transformer block, weight its feature maps by how
much they increase the predicted class score (the gradient), and show the result as a heatmap.
Red = regions the model relied on most.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from architectures import get_model
from data_utils import (IMAGENET_MEAN, IMAGENET_STD, ROOT, EuroSATDataset, get_transform,
                        load_class_names)

FIG_DIR = ROOT / "figures"


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def get_target_layer(model, model_name):
    """Last spatial feature layer of each backbone."""
    if model_name == "resnet18":
        return model.layer4[-1]                      # output: (B, 512, 7, 7)
    if model_name == "efficientnet":
        return model.features[-1]                    # output: (B, 1280, 7, 7)
    if model_name == "vit":
        return model.encoder.layers[-1].ln_1         # output: (B, 197, 768) tokens
    raise ValueError(model_name)


class GradCAM:
    def __init__(self, model, model_name):
        self.model, self.model_name = model, model_name
        self.acts, self.grads = None, None
        layer = get_target_layer(model, model_name)
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self._handle.remove()

    def _to_spatial(self, t):
        """ViT gives (B, 1+196, C) tokens -> drop the class token, reshape to (B, C, 14, 14)."""
        if self.model_name == "vit":
            t = t[:, 1:, :]
            b, n, c = t.shape
            s = int(n ** 0.5)
            t = t.reshape(b, s, s, c).permute(0, 3, 1, 2)
        return t

    def __call__(self, x, class_idx=None):
        """x: (1,3,224,224). Returns (heatmap 224x224 in [0,1], predicted class, confidence)."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(1))
        target = pred if class_idx is None else class_idx
        logits[0, target].backward()

        acts, grads = self._to_spatial(self.acts), self._to_spatial(self.grads)
        weights = grads.mean(dim=(2, 3), keepdim=True)            # one weight per channel
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))   # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), pred, float(probs[0, pred])


def denormalize(x):
    """(1,3,H,W) normalised tensor -> (H,W,3) float image in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img, cam, alpha=0.45):
    heat = plt.cm.jet(cam)[..., :3]
    return (1 - alpha) * img + alpha * heat


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_model(run, args, device, num_classes):
    from evaluate import resolve_config            # lazy: only the batch script needs it
    model_name, dropout, _ = resolve_config(run, Path(args.results_dir), args.dropout)
    ckpt = Path(args.ckpt_dir) / f"{run}_best.pth"
    if not ckpt.exists():
        print(f"[skip] {ckpt} not found")
        return None, None
    model = get_model(model_name, num_classes=num_classes, pretrained=False, dropout=dropout)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), model_name


def explain(cam_obj, test_set, idx, device):
    x, _ = test_set[idx]
    x = x.unsqueeze(0).to(device)
    cam, pred, conf = cam_obj(x)
    return denormalize(x), cam, pred, conf


def draw_pairs(items, path, title, cols=4):
    """items: list of (image, cam, caption). Each item = original + heatmap side by side."""
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.2, rows * 2.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (img, cam, caption) in enumerate(items):
        r, c = divmod(k, cols)
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c + 1].imshow(overlay(img, cam))
        axes[r, 2 * c].set_title(caption, fontsize=7, loc="left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


# ----------------------------------------------------------------------------
# App API (Streamlit / Gradio / Flask): no test set, no sklearn, one image in -> heatmap out
# ----------------------------------------------------------------------------
def load_model_for_app(ckpt_path, model_name, dropout=0.0, num_classes=10, device=None):
    """Load one checkpoint once at app start-up (not on every request)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes=num_classes, pretrained=False, dropout=dropout)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval(), device


def explain_image(model, model_name, pil_image, class_names, device):
    """Grad-CAM for one PIL image.

    Returns dict: label, confidence, probs (list), overlay (HxWx3 uint8), heatmap (HxW float 0-1).
    Do NOT call this inside torch.no_grad() - Grad-CAM needs gradients.
    """
    x = get_transform(train=False)(pil_image.convert("RGB")).unsqueeze(0).to(device)
    cam_obj = GradCAM(model, model_name)
    try:
        with torch.enable_grad():
            cam, pred, conf = cam_obj(x)
            probs = torch.softmax(model(x), dim=1)[0].detach().cpu().tolist()
    finally:
        cam_obj.remove()                       # avoid stacking hooks across requests
    img = denormalize(x)
    return {"label": class_names[pred], "confidence": conf, "probs": probs,
            "overlay": (overlay(img, cam) * 255).astype(np.uint8), "heatmap": cam}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Grad-CAM for trained GeoScope models")
    ap.add_argument("--runs", nargs="+", default=["resnet18", "efficientnet", "vit"])
    ap.add_argument("--dropout", type=float, default=0.0, help="only for runs without a history file")
    ap.add_argument("--n_errors", type=int, default=8, help="how many confident mistakes to show")
    ap.add_argument("--ckpt_dir", default=str(ROOT / "models" / "checkpoints"))
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test_set = EuroSATDataset("test", transform=get_transform(train=False))
    loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    compare = {}          # run -> {class_id: (img, cam, pred, conf)}
    ref_idx = None        # one test image per class, chosen from the first model

    for run in args.runs:
        model, model_name = load_model(run, args, device, len(class_names))
        if model is None:
            continue
        print(f"\n=== Grad-CAM: {run} ===")
        from evaluate import predict
        y_true, probs = predict(model, loader, device)
        y_pred, conf = probs.argmax(1), probs.max(1)
        cam_obj = GradCAM(model, model_name)

        # 1) most confident correct image per class
        correct_idx = {}
        for c in range(len(class_names)):
            ok = np.where((y_true == c) & (y_pred == c))[0]
            if len(ok):
                correct_idx[c] = int(ok[conf[ok].argmax()])
        if ref_idx is None:
            ref_idx = dict(correct_idx)

        items = []
        for c, i in correct_idx.items():
            img, cam, pred, p = explain(cam_obj, test_set, i, device)
            items.append((img, cam, f"{class_names[c]} ({p:.0%})"))
        draw_pairs(items, FIG_DIR / f"gradcam_correct_{run}.png", f"Grad-CAM, correct predictions - {run}")

        # 2) most confident mistakes
        wrong = np.where(y_pred != y_true)[0]
        wrong = wrong[np.argsort(-conf[wrong])][:args.n_errors]
        items = []
        for i in wrong:
            img, cam, pred, p = explain(cam_obj, test_set, int(i), device)
            items.append((img, cam, f"true: {class_names[y_true[i]]}\npred: {class_names[pred]} ({p:.0%})"))
        if items:
            draw_pairs(items, FIG_DIR / f"gradcam_errors_{run}.png", f"Grad-CAM, confident mistakes - {run}")

        # 3) heatmaps of the reference images for the cross-model comparison
        compare[run] = {}
        for c, i in ref_idx.items():
            img, cam, pred, p = explain(cam_obj, test_set, i, device)
            compare[run][c] = (img, cam, pred, p)

        del model, cam_obj
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Comparison figure: rows = classes, columns = original + each model
    if compare:
        runs = list(compare)
        classes = sorted(ref_idx)
        fig, axes = plt.subplots(len(classes), len(runs) + 1, figsize=(2.3 * (len(runs) + 1), 2.2 * len(classes)),
                                 squeeze=False)
        for r, c in enumerate(classes):
            axes[r, 0].imshow(compare[runs[0]][c][0])
            axes[r, 0].set_ylabel(class_names[c], fontsize=7)
            for k, run in enumerate(runs, start=1):
                img, cam, pred, p = compare[run][c]
                axes[r, k].imshow(overlay(img, cam))
                axes[r, k].set_title(f"{class_names[pred]} ({p:.0%})", fontsize=6)
            if r == 0:
                axes[0, 0].set_title("original", fontsize=7)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Grad-CAM comparison: " + " | ".join(runs))
        fig.tight_layout()
        fig.savefig(FIG_DIR / "gradcam_compare.png", dpi=200)
        plt.close(fig)
        print("saved", FIG_DIR / "gradcam_compare.png")


if __name__ == "__main__":
    main()