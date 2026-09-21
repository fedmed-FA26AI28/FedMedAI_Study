"""Client-side local model training routines."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    proximal_mu: float = 0.0,
    optimizer=None,
) -> Tuple[float, float]:
    """Train using cross-entropy, optionally with the FedProx proximal penalty.
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader with local client training samples.
        epochs: Number of local training epochs.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        device: Device to run computation on (CPU or CUDA).
        proximal_mu: Penalty coefficient against a snapshot at this call's start.
        optimizer: Optional persistent optimizer for the centralized baseline.
        
    Returns:
        Tuple[float, float]: (cross_entropy_loss, accuracy) across batches/epochs;
        the loss excludes the proximal penalty for comparison with FedAvg.
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    if epochs < 1 or proximal_mu < 0:
        raise ValueError("epochs must be positive and proximal_mu nonnegative")
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    reference = [p.detach().clone() for p in model.parameters()] if proximal_mu else None
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long).reshape(-1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            objective = loss
            if reference is not None:
                penalty = sum((p - initial).square().sum()
                              for p, initial in zip(model.parameters(), reference))
                objective = loss + (proximal_mu / 2) * penalty
            objective.backward()
            optimizer.step()
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            total_samples += batch_size
            
    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataset")
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    return float(avg_loss), float(accuracy)
