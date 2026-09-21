"""Shared validation/test evaluation for centralized and federated experiments."""

import torch
from torch.utils.data import SequentialSampler, Subset

from fedmedai.monitoring.metrics import classification_metrics


def sample_id(dataset, index):
    """Resolve nested Subset indices to IDs within the original official split."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    return int(index)


def evaluate_classification(model, data_loader, device, num_classes,
                            class_names=None, max_failure_cases=0):
    if max_failure_cases and not isinstance(data_loader.sampler, SequentialSampler):
        raise ValueError("Failure-case sample IDs require a sequential evaluation loader")
    model.to(device)
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss_sum, offset = 0.0, 0
    failures = []
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long).reshape(-1)
            logits = model(images)
            if logits.shape[1] != num_classes:
                raise ValueError("Model output and configured class count differ")
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.argmax(dim=1)
            cm += torch.bincount((labels * num_classes + predictions).cpu(),
                                 minlength=num_classes ** 2).reshape(num_classes, num_classes)
            if len(failures) < max_failure_cases:
                confidence = logits.softmax(dim=1).max(dim=1).values
                for i in (predictions != labels).nonzero(as_tuple=True)[0].tolist():
                    failures.append({"sample_id": sample_id(data_loader.dataset, offset + i),
                                     "true_class": int(labels[i]),
                                     "predicted_class": int(predictions[i]),
                                     "confidence": float(confidence[i])})
                    if len(failures) >= max_failure_cases:
                        break
            offset += len(labels)
    report = classification_metrics(cm.numpy(), class_names)
    report["loss"] = float(loss_sum / report["num_samples"])
    if max_failure_cases:
        report["failure_cases"] = failures
        report["failure_cases_policy"] = "first_errors_in_evaluation_order; bounded"
        report["failure_cases_limit"] = int(max_failure_cases)
    return report


def evaluate_model(model, data_loader, device):
    """Compatibility wrapper for callers that only consume loss and accuracy."""
    model.to(device)
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in data_loader:
            labels = labels.to(device, dtype=torch.long).reshape(-1)
            logits = model(images.to(device))
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
    if not count:
        raise ValueError("Cannot evaluate an empty dataset")
    return loss_sum / count, correct / count

