"""Experimental classifier stabilization and train-count logit calibration.

All methods use ordinary sample-weighted FedAvg aggregation. Class frequencies
come only from the selected local training dataset, without loading images or
advancing a DataLoader's random generator.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset


KAPPA_FREE_METHODS = frozenset(("calibrated_uniform_head", "relative_coverage_calibrated"))
HEAD_METHODS = frozenset(("head_prox", "coverage_prox", "coverage_calibrated")) | KAPPA_FREE_METHODS
CALIBRATION_METHODS = frozenset(("logit_calibration", "coverage_calibrated", "calibrated_fedprox")) | KAPPA_FREE_METHODS
COVERAGE_METHODS = HEAD_METHODS | CALIBRATION_METHODS


def validate_coverage_options(name, head_mu=1.0, coverage_kappa=32.0, logit_tau=1.0,
                              prior_smoothing=1.0):
    """Reject invalid active coefficients before datasets or artifacts are created."""
    def finite_number(value, field, positive=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or (value <= 0 if positive else value < 0)):
            domain = "positive" if positive else "nonnegative"
            raise ValueError(f"algorithm.{field} must be finite and {domain}")

    if name in HEAD_METHODS:
        finite_number(head_mu, "head_mu")
    if name in COVERAGE_METHODS and name not in KAPPA_FREE_METHODS:
        finite_number(coverage_kappa, "coverage_kappa", positive=True)
    if name in CALIBRATION_METHODS:
        finite_number(logit_tau, "logit_tau")
        finite_number(prior_smoothing, "prior_smoothing", positive=True)


def _selected_labels(dataset):
    """Resolve nested Subset indices against stored targets without __getitem__."""
    if isinstance(dataset, Subset):
        labels = _selected_labels(dataset.dataset)
        indices = dataset.indices
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("Training Subset indices must be a one-dimensional integer array")
        return labels[indices.astype(np.int64)]
    if isinstance(dataset, TensorDataset):
        if len(dataset.tensors) != 2:
            raise ValueError("Class counts require an image/target TensorDataset")
        labels = dataset.tensors[1]
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("Training class counts require stored dataset.labels or TensorDataset targets")
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels).reshape(-1)
    if labels.size != len(dataset):
        raise ValueError("Training target count does not match the dataset size")
    return labels


def training_class_counts(dataset, num_classes):
    """Count selected local TRAIN targets, including zeros for absent classes."""
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    labels = _selected_labels(dataset)
    if (labels.dtype.kind not in "iuf" or not np.isfinite(labels).all()
            or np.any(labels != np.floor(labels)) or np.any(labels < 0)
            or np.any(labels >= num_classes)):
        raise ValueError("Training targets must be integer class IDs within the classifier range")
    return np.bincount(labels.astype(np.int64), minlength=num_classes)


def linear_head(model):
    """Require the current architecture's final nn.Linear classifier at model.fc."""
    head = getattr(model, "fc", None)
    if not isinstance(head, nn.Linear):
        raise ValueError("Coverage methods require model.fc to be nn.Linear")
    return head


def head_reference(model):
    """Snapshot only the final classifier at the beginning of a local fit."""
    head = linear_head(model)
    return (head.weight.detach().clone(),
            head.bias.detach().clone() if head.bias is not None else None)


def head_row_weights(counts, coverage_kappa=32.0, uniform=False):
    """Return one for head_prox, or kappa/(n_c+kappa) for coverage stabilization."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Class counts must be a finite nonnegative vector")
    if not math.isfinite(float(coverage_kappa)) or coverage_kappa <= 0:
        raise ValueError("coverage_kappa must be finite and positive")
    return np.ones_like(counts) if uniform else coverage_kappa / (counts + coverage_kappa)


def relative_head_row_weights(counts):
    """Return mean-one weights u_c/mean(u), with u_c=1/(1+C*n_c/n).

    Counts must be a nonempty finite nonnegative vector with positive total.
    Scaling before summation avoids overflow for large finite counts and leaves
    the class proportions unchanged. No tunable coverage scale is used.
    """
    values = np.asarray(counts, dtype=np.float64)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or np.any(values < 0) or not np.any(values > 0)):
        raise ValueError("Class counts must be a nonempty finite nonnegative vector with positive total")
    scaled = values / values.max()
    proportions = scaled / scaled.sum()
    unnormalized = 1.0 / (1.0 + values.size * proportions)
    return unnormalized / unnormalized.mean()


def head_proximal_penalty(model, reference, row_weights, head_mu):
    """Apply (mu/2) sum_c a_c ||head_c - initial_head_c||^2, including bias."""
    validate_coverage_options("head_prox", head_mu=head_mu)
    head = linear_head(model)
    initial_weight, initial_bias = reference
    if initial_weight.shape != head.weight.shape:
        raise ValueError("Head reference weight shape does not match the model")
    row_distance = (head.weight - initial_weight).square().sum(dim=1)
    if head.bias is not None:
        if initial_bias is None or initial_bias.shape != head.bias.shape:
            raise ValueError("Head reference bias shape does not match the model")
        row_distance = row_distance + (head.bias - initial_bias).square()
    weights = torch.as_tensor(row_weights, dtype=head.weight.dtype, device=head.weight.device)
    if weights.shape != row_distance.shape:
        raise ValueError("Head row-weight count does not match the classifier")
    return (head_mu / 2) * (weights * row_distance).sum()


def logit_adjustment(counts, prior_smoothing, logit_tau, *, device, dtype):
    """Training-only tau*log(n_c+epsilon); inference uses unchanged raw logits."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Class counts must be a finite nonnegative vector")
    validate_coverage_options("logit_calibration", prior_smoothing=prior_smoothing,
                              logit_tau=logit_tau)
    # Compute in float64 to avoid overflow when adding counts in FP16.
    offsets = logit_tau * np.log(values + prior_smoothing)
    if not np.isfinite(offsets).all():
        raise ValueError("Logit calibration offsets must be finite")
    adjustment = torch.as_tensor(offsets, dtype=dtype, device=device)
    if not torch.isfinite(adjustment).all():
        raise ValueError("Logit calibration offsets exceed the model dtype range")
    return adjustment
