"""The single supported CNN architecture and model-state exchange."""

from typing import List
import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Automatically select CUDA GPU if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def validate_model_config(model_name="cnn", pretrained=False, **options):
    """Reject retired architectures and unsupported options instead of ignoring them."""
    if not isinstance(model_name, str) or model_name.lower() != "cnn":
        raise ValueError("Only model.name='cnn' is supported; historical models require their archived source")
    if pretrained is not False:
        raise ValueError("CNN has no pretrained weights; model.pretrained must be false")
    if options:
        raise ValueError(f"Unsupported CNN options: {', '.join(sorted(options))}")


def get_model(num_classes: int = 8, model_name: str = "cnn", pretrained: bool = False, **options) -> nn.Module:
    """Construct the user-specified CNN; no alternative architectures are available."""
    validate_model_config(model_name, pretrained, **options)
    return CNN(num_classes=num_classes)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy ndarrays for Flower parameter exchange."""
    return [val.detach().cpu().numpy().copy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, weights: List[np.ndarray]) -> None:
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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters, excluding BatchNorm buffers."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
