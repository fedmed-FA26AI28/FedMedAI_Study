"""Reproducible IID / paired Dirichlet partitions; no silent IID fallback."""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.medmnist_code import dataset_info, load_medmnist_split


def _validate(dataset, num_clients, minimum=1):
    if num_clients < 1 or minimum < 1 or len(dataset) < num_clients * minimum:
        raise ValueError("Dataset cannot satisfy num_clients * min_samples_per_client")


def partition_iid(dataset, num_clients=3, seed=42):
    _validate(dataset, num_clients)
    indices = np.random.default_rng(seed).permutation(len(dataset))
    return {cid: part.tolist() for cid, part in enumerate(np.array_split(indices, num_clients))}


def _assign(labels, classes, proportions, rng):
    clients = {cid: [] for cid in range(proportions.shape[1])}
    for row, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        counts = rng.multinomial(len(indices), proportions[row])
        for cid, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
            clients[cid].extend(part.tolist())
    for indices in clients.values():
        rng.shuffle(indices)
    return clients


def partition_dirichlet_non_iid(dataset, num_clients=3, alpha=0.5, seed=42,
                                min_samples_per_client=10, max_attempts=500):
    _validate(dataset, num_clients, min_samples_per_client)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet alpha must be finite and positive")
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.labels).reshape(-1)
    classes = np.unique(labels)
    for _ in range(max_attempts):
        proportions = rng.dirichlet(np.full(num_clients, alpha), size=len(classes))
        clients = _assign(labels, classes, proportions, rng)
        if min(map(len, clients.values())) >= min_samples_per_client:
            return clients
    raise ValueError("Dirichlet allocation failed; adjust minimum, alpha or seed. No IID fallback.")


def _paired_dirichlet(train, val, num_clients, alpha, seed, train_min, val_min):
    _validate(train, num_clients, train_min)
    _validate(val, num_clients, val_min)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet alpha must be finite and positive")
    classes = np.union1d(train.labels, val.labels)
    rng = np.random.default_rng(seed)
    for attempt in range(500):
        proportions = rng.dirichlet(np.full(num_clients, alpha), size=len(classes))
        train_splits = _assign(np.asarray(train.labels), classes, proportions, rng)
        val_splits = _assign(np.asarray(val.labels), classes, proportions, rng)
        if (min(map(len, train_splits.values())) >= train_min
                and min(map(len, val_splits.values())) >= val_min):
            return train_splits, val_splits, attempt + 1
    raise ValueError("Paired Dirichlet allocation failed after 500 attempts; no IID fallback")


def _distribution(labels, indices, num_classes):
    return {str(c): int(n) for c, n in enumerate(
        np.bincount(np.asarray(labels)[indices], minlength=num_classes))}


def create_client_dataloaders(dataset_name="BloodMNIST", partition_type="iid",
                             num_clients=3, alpha=0.3, batch_size=32, image_size=28,
                             seed=42, max_train_samples=None, max_val_samples=None,
                             min_train_samples=10, min_val_samples=1,
                             root=None, download=True, num_workers=0):
    partition_type = partition_type.lower()
    if partition_type not in ("iid", "dirichlet"):
        raise ValueError(f"Unknown partition_type: {partition_type}")
    for cap in (max_train_samples, max_val_samples):
        if cap is not None and cap < 1:
            raise ValueError("Sample caps must be positive or null")
    train = load_medmnist_split(dataset_name, "train", image_size, download, root)
    val = load_medmnist_split(dataset_name, "val", image_size, download, root)
    num_classes = len(dataset_info(dataset_name)["label"])
    attempts = 1
    if partition_type == "iid":
        train_splits = partition_iid(train, num_clients, seed)
        val_splits = partition_iid(val, num_clients, seed + 1)
    else:
        train_splits, val_splits, attempts = _paired_dirichlet(
            train, val, num_clients, alpha, seed, min_train_samples, min_val_samples)
    metadata = {
        "schema_version": 2, "dataset": dataset_name, "partition_type": partition_type,
        "alpha": float(alpha) if partition_type == "dirichlet" else None,
        "random_seed": int(seed), "num_clients": num_clients,
        "client_ids": list(range(num_clients)), "clients": {},
        "source_split_sizes": {"train": len(train), "val": len(val)},
        "sample_id_namespace": "zero_based_index_within_official_split",
        "allocation_policy": ("shared_class_proportions_train_val"
                              if partition_type == "dirichlet" else "uniform_shuffle"),
        "partition_attempts": attempts,
        "sample_caps": {"train_per_client": max_train_samples, "val_per_client": max_val_samples},
    }
    train_loaders, val_loaders = {}, {}
    for cid in range(num_clients):
        train_ids = train_splits[cid][:max_train_samples]
        val_ids = val_splits[cid][:max_val_samples]
        train_loaders[cid] = DataLoader(
            Subset(train, train_ids), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed + cid), drop_last=False, num_workers=num_workers)
        val_loaders[cid] = DataLoader(Subset(val, val_ids), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_dist = _distribution(train.labels, train_ids, num_classes)
        val_dist = _distribution(val.labels, val_ids, num_classes)
        metadata["clients"][cid] = {
            "client_id": cid, "num_train_samples": len(train_ids), "num_val_samples": len(val_ids),
            "allocated_train_samples": len(train_splits[cid]),
            "allocated_val_samples": len(val_splits[cid]),
            "class_distribution": train_dist, "val_class_distribution": val_dist,
            "class_proportions": {c: n / len(train_ids) for c, n in train_dist.items()},
            "sample_ids": {"train_sample_ids": train_ids, "val_sample_ids": val_ids},
        }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata["partition_sha256"] = hashlib.sha256(encoded).hexdigest()
    return train_loaders, val_loaders, metadata


def pooled_loader(loaders, batch_size, shuffle=False, seed=42):
    """Reconstruct the exact union of selected samples, for fair central baselines."""
    first = next(iter(loaders.values())).dataset.dataset
    indices = sorted(i for loader in loaders.values() for i in loader.dataset.indices)
    if len(indices) != len(set(indices)):
        raise ValueError("Client partitions overlap")
    return DataLoader(Subset(first, indices), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))



def visualize_partition(metadata, path):
    """Plot class counts from saved indices metadata; does not reload the dataset."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clients = list(metadata["clients"].values())
    matrix = np.asarray([list(client["class_distribution"].values()) for client in clients])
    figure, axis = plt.subplots()
    bottom = np.zeros(len(clients))
    for label in range(matrix.shape[1]):
        axis.bar(range(len(clients)), matrix[:, label], bottom=bottom, label=str(label))
        bottom += matrix[:, label]
    axis.set(xlabel="Client ID", ylabel="Training samples", title="Partition class distribution")
    axis.legend(title="Class", bbox_to_anchor=(1.02, 1))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
