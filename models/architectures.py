import torch
import torch.nn as nn
from torchvision import models


def _head(in_features, num_classes, dropout):
    """Final classification layer.

    dropout = 0  -> plain Linear layer (same weight names as the first checkpoints,
                    so old .pth files still load).
    dropout > 0  -> Dropout followed by Linear.
    """
    linear = nn.Linear(in_features, num_classes)
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), linear)
    return linear


def count_parameters(model):
    """Total number of weights in the model (used in the comparison table)."""
    return sum(p.numel() for p in model.parameters())


def get_model(model_name: str, num_classes: int = 10, pretrained: bool = True, dropout: float = 0.0):
    """
    Factory function to initialize vision backbones for EuroSAT land-use classification.

    Args:
        model_name (str): One of 'resnet18', 'efficientnet', or 'vit'.
        num_classes (int): Number of target land-use classes (default: 10).
        pretrained (bool): Whether to load ImageNet pre-trained weights.
        dropout (float): Dropout probability placed before the final layer (0 = none).
                         Note: torchvision's EfficientNet-B0 already has its own built-in
                         Dropout(0.2); this extra dropout is applied on top of it.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = _head(model.fc.in_features, num_classes, dropout)

    elif model_name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = _head(model.classifier[1].in_features, num_classes, dropout)

    elif model_name == "vit":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = _head(model.heads.head.in_features, num_classes, dropout)

    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'. Choose from ['resnet18', 'efficientnet', 'vit'].")

    return model


if __name__ == "__main__":
    # Quick sanity check
    test_x = torch.randn(2, 3, 224, 224)
    for net in ["resnet18", "efficientnet", "vit"]:
        for p in (0.0, 0.3):
            m = get_model(net, num_classes=10, pretrained=False, dropout=p)
            m.eval()
            out = m(test_x)
            print(f"[{net}] dropout={p} | output shape: {tuple(out.shape)} | parameters: {count_parameters(m):,}")
