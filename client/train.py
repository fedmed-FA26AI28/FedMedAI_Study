"""Client-side local model training routines."""

from numbers import Integral
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algorithms.coverage import (
    CALIBRATION_METHODS, COVERAGE_METHODS, HEAD_METHODS, head_proximal_penalty,
    head_reference, head_row_weights, linear_head, logit_adjustment, relative_head_row_weights,
    training_class_counts, validate_coverage_options,
)


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    proximal_mu: float = 0.0,
    optimizer=None,
    *,
    algorithm=None,
    head_mu: float = 1.0,
    coverage_kappa: float = 32.0,
    logit_tau: float = 1.0,
    prior_smoothing: float = 1.0,
    diagnostics=None,
    max_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """Train with CE, FedProx or the named experimental classifier objectives.
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader with local client training samples.
        epochs: Number of local training epochs.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        device: Device to run computation on (CPU or CUDA).
        proximal_mu: Penalty coefficient against a snapshot at this call's start.
        optimizer: Optional persistent optimizer for the centralized baseline.
        algorithm: Named client objective; None retains the original FedProx API.
        head_mu: Classifier-only stabilization coefficient for head methods.
        coverage_kappa: Positive smoothing count for legacy coverage weights;
            ignored by calibrated_uniform_head and relative_coverage_calibrated.
        logit_tau: Training-only logit adjustment strength for calibrated methods.
        prior_smoothing: Positive pseudocount for training-only logit calibration.
        diagnostics: Optional dict receiving objective/penalty, head movement,
            optimizer_steps and train_samples_seen across this fit.
        max_steps: Optional positive integer cap across all local epochs. Uses
            the loader's existing order and stops after a completed batch;
            does not repeat data beyond epochs to reach the cap. Class counts
            still come from the complete selected local training dataset.
        
    Returns:
        Tuple[float, float]: (cross_entropy_loss, accuracy) across batches/epochs;
        the loss excludes the proximal penalty for comparison with FedAvg.
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    if epochs < 1 or proximal_mu < 0:
        raise ValueError("epochs must be positive and proximal_mu nonnegative")
    if max_steps is not None and (isinstance(max_steps, bool)
                                  or not isinstance(max_steps, Integral) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer or None")
    if algorithm is not None and algorithm not in {"fedavg", "fedprox"} | COVERAGE_METHODS:
        raise ValueError("Unknown local algorithm")
    validate_coverage_options(algorithm, head_mu, coverage_kappa, logit_tau, prior_smoothing)
    if algorithm in COVERAGE_METHODS and algorithm != "calibrated_fedprox" and proximal_mu:
        raise ValueError("Classifier methods do not use the full-model proximal_mu")
    if algorithm == "calibrated_fedprox" and not math.isfinite(proximal_mu):
        raise ValueError("calibrated_fedprox requires finite nonnegative proximal_mu")
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    reference = [p.detach().clone() for p in model.parameters()] if proximal_mu else None
    head_initial, row_weights, adjustment = None, None, None
    encoder_initial = None
    if algorithm in COVERAGE_METHODS:
        head = linear_head(model)
        counts = training_class_counts(train_loader.dataset, head.out_features)
        if algorithm in HEAD_METHODS or diagnostics is not None:
            head_initial = head_reference(model)
        if diagnostics is not None:
            encoder_initial = {name: parameter.detach().clone()
                               for name, parameter in model.named_parameters()
                               if parameter.requires_grad and not name.startswith("fc.")}
        if algorithm in HEAD_METHODS and head_mu:
            if algorithm == "calibrated_uniform_head":
                weights = [1.0] * head.out_features
            elif algorithm == "relative_coverage_calibrated":
                weights = relative_head_row_weights(counts)
            else:
                weights = head_row_weights(counts, coverage_kappa, uniform=algorithm == "head_prox")
            row_weights = torch.as_tensor(
                weights,
                dtype=head.weight.dtype, device=head.weight.device)
        if algorithm in CALIBRATION_METHODS and logit_tau:
            adjustment = logit_adjustment(counts, prior_smoothing, logit_tau,
                                          device=head.weight.device, dtype=head.weight.dtype)
    
    total_loss = 0.0
    total_objective = 0.0
    total_head_penalty = 0.0
    total_proximal_penalty = 0.0
    correct = 0
    total_samples = 0
    optimizer_steps = 0
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long).reshape(-1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Reporting and prediction always use the same raw logits as inference.
            objective = criterion(outputs + adjustment, labels) if adjustment is not None else loss
            full_penalty = None
            if reference is not None:
                from algorithms.fedprox import proximal_penalty
                full_penalty = proximal_penalty(model, reference, proximal_mu)
                objective = objective + full_penalty
            penalty = None
            if row_weights is not None:
                penalty = head_proximal_penalty(model, head_initial, row_weights, head_mu)
                objective = objective + penalty
            objective.backward()
            optimizer.step()
            optimizer_steps += 1
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            if diagnostics is not None:
                total_objective += objective.item() * batch_size
                total_head_penalty += (penalty.item() if penalty is not None else 0.0) * batch_size
                total_proximal_penalty += (full_penalty.item() if full_penalty is not None else 0.0) * batch_size
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            total_samples += batch_size
            if max_steps is not None and optimizer_steps >= max_steps:
                break
        if max_steps is not None and optimizer_steps >= max_steps:
            break
            
    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataset")
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    if diagnostics is not None:
        diagnostics.update(train_objective_loss=float(total_objective / total_samples),
                           train_head_penalty=float(total_head_penalty / total_samples),
                           train_proximal_penalty=float(total_proximal_penalty / total_samples),
                           optimizer_steps=optimizer_steps, train_samples_seen=total_samples)
        if head_initial is not None:
            head = linear_head(model)
            with torch.no_grad():
                distances = (head.weight - head_initial[0]).square().sum(dim=1)
                if head.bias is not None:
                    distances = distances + (head.bias - head_initial[1]).square()
                norms = distances.sqrt()
                diagnostics.update(head_displacement_norms=norms.cpu().tolist(),
                                   head_displacement_l2=float(norms.norm().item()),
                                   head_displacement_max=float(norms.max().item()))
                encoder_squared = sum((parameter - encoder_initial[name]).square().sum()
                                      for name, parameter in model.named_parameters()
                                      if name in encoder_initial)
                diagnostics["encoder_displacement_l2"] = (
                    float(encoder_squared.sqrt().item()) if encoder_initial else 0.0)
    
    return float(avg_loss), float(accuracy)
