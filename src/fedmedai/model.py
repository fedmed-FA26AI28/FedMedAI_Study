"""Primary ResNet-18 model, optional BloodCNN and parameter exchange."""

from typing import List
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_device() -> torch.device:
    """Automatically select CUDA GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_resnet18(num_classes: int = 8, pretrained: bool = False) -> nn.Module:
    """Instantiate a ResNet-18 model adapted for medical image classification.
    
    Args:
        num_classes: Number of target classification classes (e.g. 8 for BloodMNIST).
        pretrained: If True, loads ImageNet pretrained weights.
        
    Returns:
        nn.Module: Configured ResNet-18 model.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    # Replace final linear classifier head for target num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class BloodCNN(nn.Module):
    """Custom 3-Block CNN architecture for 28x28 RGB medical image classification."""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = 8, model_name: str = "resnet18", pretrained: bool = False, **kwargs) -> nn.Module:
    """Factory function to get model instance by name. Default is ResNet-18.
    
    Args:
        num_classes: Number of output classes.
        model_name: 'resnet18' or 'custom_cnn'.
        pretrained: Whether to load ImageNet pretrained weights.
        
    Returns:
        nn.Module: Model instance.
    """
    if model_name.lower() in ["custom_cnn", "bloodcnn", "cnn"]:
        if pretrained:
            raise ValueError("No pretrained weights are available for BloodCNN")
        return BloodCNN(num_classes=num_classes)
    if model_name.lower() in ["resnet18", "resnet-18", "resnet"]:
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {model_name}")


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy ndarrays for Flower parameter exchange."""
    return [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Load a list of NumPy ndarrays into model state_dict."""
    current = model.state_dict()
    if len(weights) != len(current):
        raise ValueError("Received an incorrect number of model tensors")
    for (name, tensor), weight in zip(current.items(), weights):
        if tensor.shape != weight.shape:
            raise ValueError(f"Shape mismatch for {name}: {weight.shape} != {tensor.shape}")
    state_dict = {name: torch.as_tensor(weight, dtype=tensor.dtype)
                  for (name, tensor), weight in zip(current.items(), weights)}
    model.load_state_dict(state_dict, strict=True)
