import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name: str, num_classes: int = 10, pretrained: bool = True):
    """
    Factory function to initialize vision backbones for EuroSAT land-use classification.
    
    Args:
        model_name (str): One of 'resnet18', 'efficientnet', or 'vit'.
        num_classes (int): Number of target land-use classes (default: 10).
        pretrained (bool): Whether to load ImageNet pre-trained weights.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "vit":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'. Choose from ['resnet18', 'efficientnet', 'vit'].")

    return model

if __name__ == "__main__":
    # Quick sanity check
    test_x = torch.randn(2, 3, 224, 224)
    for net in ["resnet18", "efficientnet", "vit"]:
        m = get_model(net, num_classes=10, pretrained=False)
        out = m(test_x)
        print(f"[{net}] Output shape: {out.shape}")