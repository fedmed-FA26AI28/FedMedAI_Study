"""Locked, deterministic synthetic heterogeneity over official BloodMNIST splits.

These are synthetic appearance stress tests, not hospital domains or a claim of
pure covariate shift. Images supplied to the injectable helper must already use
the project's RGB float32 [-1, 1] preprocessing. Only train labels determine
the label/mixed test assignment probabilities; test labels are used to stratify
that fixed assignment, never to select a transform or tune a coefficient.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.medmnist_code import load_medmnist_split
from datasets.partition import _paired_dirichlet


NUM_CLIENTS = 3
NUM_CLASSES = 8
COHORTS = ("label", "appearance", "mixed", "quantity", "label_quantity")
QUANTITY_PROPORTIONS = (0.1, 0.3, 0.6)
APPEARANCE_POLICY = {
    "version": 1,
    "input": "RGB float32; unit=(normalized+1)/2",
    "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "domains": {
        0: {"name": "identity", "channel_scale": [1.0, 1.0, 1.0],
            "brightness": 1.0, "blur_kernel": None, "blur_sigma": None},
        1: {"name": "warm_darker", "channel_scale": [1.05, 0.90, 0.85],
            "brightness": 0.90, "blur_kernel": None, "blur_sigma": None},
        2: {"name": "cooler_blur", "channel_scale": [0.90, 0.95, 1.05],
            "brightness": 1.0, "blur_kernel": 3, "blur_sigma": 0.8},
    },
    "order": "unit -> channel_scale*brightness -> clamp[0,1] -> optional blur -> normalize",
    "blur": "separable Gaussian 3x3, sigma=0.8, reflection padding, independent RGB channels",
    "selection": "fixed before results, independent of sample labels and all metrics",
    "inference": "same client domain in train, validation and test; no target adaptation",
    "scope": "synthetic appearance perturbations; no real-site or pure-covariate-shift claim",
}


def metadata_digest(value):
    """Canonical SHA-256 for metadata, excluding its own partition_sha256 field.

    Callers pass a dictionary with that field removed; JSON round-tripping
    integer client/domain keys preserves this canonical representation.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)},
                             sort_keys=True).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class CachedTensorDataset(Dataset):
    """Official split order with stored labels; reads cannot mutate cached images."""

    def __init__(self, images, labels):
        self._images = images.contiguous()
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        self.labels.setflags(write=False)
        self.tensor_sha256 = _tensor_digest(self._images)
        self.labels_sha256 = metadata_digest(self.labels.tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self._images[index].clone(), int(self.labels[index])


def _materialize(dataset, num_classes):
    labels = np.asarray(dataset.labels).reshape(-1)
    if (len(labels) != len(dataset) or not len(labels) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or np.any(labels != np.floor(labels))
            or np.any(labels < 0) or np.any(labels >= num_classes)):
        raise ValueError("Dataset labels must be stored integer class IDs in range")
    if isinstance(dataset, CachedTensorDataset):
        return dataset
    first, _ = dataset[0]
    first = torch.as_tensor(first)
    if (first.ndim != 3 or first.shape[0] != 3 or min(first.shape[1:]) < 2
            or not first.is_floating_point()):
        raise ValueError("Expected preprocessed RGB floating tensors shaped (3,H,W)")
    images = torch.empty((len(dataset), *first.shape), dtype=torch.float32)
    for index in range(len(dataset)):
        image, label = dataset[index]
        image = torch.as_tensor(image)
        if image.shape != first.shape or int(np.asarray(label).reshape(-1)[0]) != labels[index]:
            raise ValueError("Image shapes or stored labels disagree with dataset items")
        images[index] = image.detach().cpu().to(torch.float32)
    if (not torch.isfinite(images).all() or images.min() < -1.00001
            or images.max() > 1.00001):
        raise ValueError("Images must use the existing normalized RGB [-1,1] preprocessing")
    return CachedTensorDataset(images, labels)


@lru_cache(maxsize=12)
def _domain_dataset(source, domain):
    if domain == 0:
        return source
    spec = APPEARANCE_POLICY["domains"][domain]
    output = torch.empty_like(source._images)
    scale = torch.tensor(spec["channel_scale"], dtype=torch.float32).view(1, 3, 1, 1)
    kernel = None
    if spec["blur_kernel"] is not None:
        coordinates = torch.arange(-1, 2, dtype=torch.float32)
        one_d = torch.exp(-coordinates.square() / (2 * spec["blur_sigma"] ** 2))
        one_d /= one_d.sum()
        kernel = (one_d[:, None] * one_d[None, :]).expand(3, 1, 3, 3).contiguous()
    # Bound temporary tensors; image/domain realizations are independent of batch size.
    for start in range(0, len(source), 256):
        unit = ((source._images[start:start + 256] + 1) / 2
                * scale * spec["brightness"]).clamp(0, 1)
        if kernel is not None:
            unit = F.conv2d(F.pad(unit, (1, 1, 1, 1), mode="reflect"), kernel, groups=3)
        output[start:start + 256] = unit * 2 - 1
    return CachedTensorDataset(output, source.labels)


def largest_remainder(total, proportions):
    """Allocate every integer item, resolving equal fractional remainders by ID."""
    weights = np.asarray(proportions, dtype=np.float64)
    if (not isinstance(total, Integral) or isinstance(total, bool) or total < 0
            or weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid largest-remainder allocation")
    expected = int(total) * weights / weights.sum()
    counts = np.floor(expected).astype(np.int64)
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:int(total) - int(counts.sum())]] += 1
    return counts


def _allocate(labels, class_proportions, rng):
    result = {cid: [] for cid in range(NUM_CLIENTS)}
    for label, proportions in enumerate(class_proportions):
        ids = rng.permutation(np.flatnonzero(labels == label))
        counts = largest_remainder(len(ids), proportions)
        for cid, part in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
            result[cid].extend(part.tolist())
    for ids in result.values():
        rng.shuffle(ids)
    return result


def _round_transport(matrix, row_totals, column_totals):
    """Maximize residual remainders subject to exact integer row/column margins.

    A tiny deterministic min-cost residual flow avoids a scipy dependency.
    Each cell receives its floor or floor+1; reverse edges permit correcting
    early choices when independent largest-remainder rounding would miss totals.
    """
    base = np.floor(matrix).astype(np.int64)
    rows = np.asarray(row_totals, dtype=np.int64) - base.sum(axis=1)
    cols = np.asarray(column_totals, dtype=np.int64) - base.sum(axis=0)
    if np.any(rows < 0) or np.any(cols < 0) or rows.sum() != cols.sum():
        raise ValueError("IPF residual margins are invalid")
    nr, nc = matrix.shape
    source, sink = nr + nc, nr + nc + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(u, v, capacity, cost):
        forward = [v, len(graph[v]), int(capacity), float(cost)]
        reverse = [u, len(graph[u]), 0, -float(cost)]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for row in range(nr):
        edge(source, row, rows[row], 0)
    cells = {}
    for row in range(nr):
        for col in range(nc):
            cells[row, col] = edge(row, nr + col, 1, -(matrix[row, col] - base[row, col]))
    for col in range(nc):
        edge(nr + col, sink, cols[col], 0)
    for _ in range(int(rows.sum())):
        distance, previous = [float("inf")] * len(graph), [None] * len(graph)
        distance[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for u, adjacency in enumerate(graph):
                if not np.isfinite(distance[u]):
                    continue
                for ei, (v, _, capacity, cost) in enumerate(adjacency):
                    if capacity and distance[u] + cost < distance[v] - 1e-13:
                        distance[v] = distance[u] + cost
                        previous[v] = (u, ei)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            raise ValueError("Controlled integer rounding has no feasible residual flow")
        node = sink
        while node != source:
            u, ei = previous[node]
            item = graph[u][ei]
            item[2] -= 1
            graph[node][item[1]][2] += 1
            node = u
    for (row, col), item in cells.items():
        base[row, col] += 1 - item[2]
    if (not np.array_equal(base.sum(axis=1), row_totals)
            or not np.array_equal(base.sum(axis=0), column_totals)):
        raise AssertionError("Controlled rounding failed its exact margin invariant")
    return base


def quantity_label_counts(class_counts, probabilities, client_totals):
    """IPF of positive Dirichlet odds followed by exact controlled rounding."""
    row_totals = np.asarray(class_counts, dtype=np.int64)
    columns = np.asarray(client_totals, dtype=np.int64)
    weights = np.asarray(probabilities, dtype=np.float64)
    if (weights.shape != (len(row_totals), NUM_CLIENTS) or np.any(row_totals < 0)
            or np.any(columns <= 0) or row_totals.sum() != columns.sum()
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Invalid label/quantity margins or probabilities")
    # Explicit positive floor prevents underflowed Dirichlet zeros blocking IPF.
    matrix = np.maximum(weights, 1e-12) * row_totals[:, None]
    for iteration in range(1, 10001):
        matrix *= columns / matrix.sum(axis=0)
        matrix *= np.divide(row_totals, matrix.sum(axis=1),
                            out=np.zeros(len(row_totals), dtype=float),
                            where=row_totals != 0)[:, None]
        error = float(np.max(np.abs(matrix.sum(axis=0) - columns)))
        if error < 1e-9:
            break
    else:
        raise ValueError("Label/quantity IPF did not converge; no fallback")
    rounded = _round_transport(matrix, row_totals, columns)
    return rounded, {"iterations": iteration, "maximum_margin_error": error,
                     "probability_floor": 1e-12, "fractional_matrix": matrix.tolist(),
                     "rounding": "min_cost_largest_residual_remainders_exact_both_margins"}


def _counts(labels, partitions, num_classes):
    return np.stack([np.bincount(labels[partitions[cid]], minlength=num_classes)
                     for cid in range(NUM_CLIENTS)], axis=1)


def _prepare(cohort, seed, alpha, sources, num_classes):
    rng = np.random.default_rng(seed + 7000)
    partitions, details = {}, {}
    if cohort in ("label", "mixed"):
        train_ids, val_ids, attempts = _paired_dirichlet(
            sources["train"], sources["val"], NUM_CLIENTS, alpha, seed, 10, 1)
        partitions.update(train=train_ids, val=val_ids)
        count_matrix = _counts(sources["train"].labels, train_ids, num_classes)
        totals = count_matrix.sum(axis=1)
        probabilities = np.divide(count_matrix, totals[:, None],
                                  out=np.full(count_matrix.shape, 1 / NUM_CLIENTS),
                                  where=totals[:, None] != 0)
        partitions["test"] = _allocate(sources["test"].labels, probabilities, rng)
        details.update(policy="existing_paired_dirichlet_train_val; train_empirical_class_assignment_test",
                       partition_attempts=attempts,
                       train_class_assignment_probabilities=probabilities.tolist(),
                       absent_train_class_test_policy="uniform across clients; fixed fallback")
    elif cohort == "label_quantity":
        probabilities = rng.dirichlet(np.full(NUM_CLIENTS, alpha), size=num_classes)
        details.update(policy="shared_dirichlet_odds_IPF_with_prescribed_split_client_sizes",
                       requested_client_proportions=list(QUANTITY_PROPORTIONS),
                       dirichlet_odds=probabilities.tolist(), ipf={})
        for split, dataset in sources.items():
            class_counts = np.bincount(dataset.labels, minlength=num_classes)
            totals = largest_remainder(len(dataset), QUANTITY_PROPORTIONS)
            matrix, ipf = quantity_label_counts(class_counts, probabilities, totals)
            # Convert exact counts to probabilities for the shared integer allocator.
            assignment = np.divide(matrix, class_counts[:, None],
                                   out=np.full(matrix.shape, 1 / NUM_CLIENTS),
                                   where=class_counts[:, None] != 0)
            partitions[split] = _allocate_exact(dataset.labels, matrix, rng)
            details["ipf"][split] = dict(ipf, target_client_counts=totals.tolist(),
                                        realized_class_assignment_probabilities=assignment.tolist())
    else:
        proportions = QUANTITY_PROPORTIONS if cohort == "quantity" else (1 / 3,) * 3
        probabilities = np.tile(proportions, (num_classes, 1))
        for split, dataset in sources.items():
            partitions[split] = _allocate(dataset.labels, probabilities, rng)
        details.update(policy="per_class_largest_remainders_fixed_client_proportions",
                       requested_client_proportions=list(proportions))
    domains = {cid: cid if cohort in ("appearance", "mixed") else 0
               for cid in range(NUM_CLIENTS)}
    policy = deepcopy(APPEARANCE_POLICY)
    policy["active_client_domains"] = domains
    metadata = {
        "schema_version": 1, "dataset": "BloodMNIST", "cohort": cohort,
        "random_seed": int(seed), "alpha": float(alpha) if cohort in ("label", "mixed", "label_quantity") else None,
        "num_clients": NUM_CLIENTS, "num_classes": num_classes,
        "client_ids": list(range(NUM_CLIENTS)),
        "source_split_sizes": {split: len(data) for split, data in sources.items()},
        "sample_id_namespace": "official_split:zero_based_original_index",
        "split_overlap_policy": "official disjoint split namespaces; no source item reused across splits",
        "allocation": details, "transform_policy": policy,
        "transform_policy_sha256": metadata_digest(policy), "clients": {},
        "source_tensor_sha256": {split: data.tensor_sha256 for split, data in sources.items()},
        "source_labels_sha256": {split: data.labels_sha256 for split, data in sources.items()},
        "limitations": ["synthetic domains are not hospital provenance or verified label-preserving shifts",
                        "integer allocations only approximate equal class priors or target proportions",
                        "label_quantity IPF changes realized label proportions; axes are not independent",
                        "original official split disjointness is trusted; no patient identities supplied"],
    }
    transformed = {split: {} for split in sources}
    for cid in range(NUM_CLIENTS):
        client = {"client_id": cid, "domain_id": domains[cid], "sample_ids": {},
                  "class_counts": {}, "class_proportions": {}, "assignment_proportions": {},
                  "tensor_realization_sha256": {}}
        for split, source in sources.items():
            ids = partitions[split][cid]
            all_ids = [index for group in partitions[split].values() for index in group]
            if sorted(all_ids) != list(range(len(source))):
                raise AssertionError("Split allocation must cover each original ID exactly once")
            if not ids:
                raise ValueError(f"Empty {split} client {cid}; choose another predeclared seed/design")
            data = _domain_dataset(source, domains[cid])
            transformed[split][cid] = data
            counts = np.bincount(source.labels[ids], minlength=num_classes)
            total_class = np.bincount(source.labels, minlength=num_classes)
            client[f"num_{split}_samples"] = len(ids)
            client["sample_ids"][f"{split}_sample_ids"] = list(ids)
            client["class_counts"][split] = counts.tolist()
            client["class_proportions"][split] = (counts / len(ids)).tolist()
            client["assignment_proportions"][split] = np.divide(
                counts, total_class, out=np.zeros(num_classes, dtype=float), where=total_class != 0).tolist()
            client["tensor_realization_sha256"][split] = _tensor_digest(data._images[ids])
        # Familiar fields for consumers of the existing partition metadata.
        client["class_distribution"] = {str(c): n for c, n in enumerate(client["class_counts"]["train"])}
        client["val_class_distribution"] = {str(c): n for c, n in enumerate(client["class_counts"]["val"])}
        metadata["clients"][cid] = client
    metadata["partition_sha256"] = metadata_digest(metadata)
    return transformed, partitions, metadata


def _allocate_exact(labels, matrix, rng):
    result = {cid: [] for cid in range(NUM_CLIENTS)}
    for label, counts in enumerate(matrix):
        ids = rng.permutation(np.flatnonzero(labels == label))
        for cid, part in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
            result[cid].extend(part.tolist())
    for ids in result.values():
        rng.shuffle(ids)
    return result


def _validate_options(cohort, seed, alpha, batch_size, num_classes):
    if cohort not in COHORTS:
        raise ValueError(f"Unknown heterogeneity cohort: {cohort}")
    if (not isinstance(seed, Integral) or isinstance(seed, bool)
            or seed < 0 or seed > 2 ** 63 - 10000):
        raise ValueError("seed must be an integer in [0, 2**63-10000]")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number))
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("alpha must be finite and positive")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(num_classes, Integral) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")


def _loaders(prepared, batch_size, seed):
    datasets, partitions, metadata = prepared
    groups = []
    for offset, split in enumerate(("train", "val", "test")):
        loaders = {}
        for cid in range(NUM_CLIENTS):
            loaders[cid] = DataLoader(
                Subset(datasets[split][cid], list(partitions[split][cid])),
                batch_size=int(batch_size), shuffle=split == "train", drop_last=False,
                num_workers=0, generator=torch.Generator().manual_seed(int(seed) + 100 * offset + cid))
        groups.append(loaders)
    return groups[0], groups[1], groups[2], deepcopy(metadata)


def build_heterogeneity_from_datasets(cohort, seed, train, val, test, alpha=0.3,
                                     batch_size=32, num_classes=NUM_CLASSES):
    """Inject preprocessed datasets with .labels for offline tests (no disk access).

    The caller supplies disjoint official split objects. All IDs are retained;
    no global NumPy or torch RNG is seeded or consumed by this implementation.
    Injectable datasets themselves must have deterministic __getitem__ methods.
    """
    _validate_options(cohort, seed, alpha, batch_size, num_classes)
    sources = {split: _materialize(data, num_classes)
               for split, data in (("train", train), ("val", val), ("test", test))}
    return _loaders(_prepare(cohort, int(seed), float(alpha), sources, int(num_classes)), batch_size, seed)


@lru_cache(maxsize=1)
def _cached_sources(source_key):
    root = source_key[0]
    sources = {split: _materialize(load_medmnist_split(
        "BloodMNIST", split, image_size=28, download=False, root=root), NUM_CLASSES)
        for split in ("train", "val", "test")}
    expected = {"train": 11959, "val": 1712, "test": 3421}
    if {split: len(data) for split, data in sources.items()} != expected:
        raise ValueError("Research API requires the complete official BloodMNIST splits")
    return sources


@lru_cache(maxsize=32)
def _cached_prepared(cohort, seed, alpha, source_key):
    return _prepare(cohort, seed, alpha, _cached_sources(source_key), NUM_CLASSES)


def clear_heterogeneity_cache():
    """Release the bounded process-local cache; no files or results are removed."""
    _cached_prepared.cache_clear()
    _domain_dataset.cache_clear()
    _cached_sources.cache_clear()


def build_heterogeneity_data(cohort, seed, alpha=0.3, batch_size=32, root=None):
    """Return fresh train/val/test loader dictionaries and auditable metadata.

    Full cached BloodMNIST only, download=False. Tensors/domains/partitions are
    reused across cells, but DataLoaders, index lists, metadata and all private
    shuffle/evaluation generators are fresh on each call. File stat changes
    invalidate source/partition cache keys; realization hashes bind actual data.
    """
    _validate_options(cohort, seed, alpha, batch_size, NUM_CLASSES)
    if root is None:
        from medmnist.dataset import DEFAULT_ROOT
        root = DEFAULT_ROOT
    resolved = Path(root).expanduser().resolve()
    archive = resolved / "bloodmnist.npz"
    if not archive.is_file():
        raise FileNotFoundError(f"Cached BloodMNIST required, downloads disabled: {archive}")
    stat = archive.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    return _loaders(_cached_prepared(cohort, int(seed), float(alpha), key), batch_size, seed)
