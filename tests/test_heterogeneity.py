"""Offline allocation, domain, provenance and RNG checks for new data cohorts."""

import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.coverage import training_class_counts
from client.evaluate import sample_id
from datasets.heterogeneity import (
    APPEARANCE_POLICY, COHORTS, CachedTensorDataset, _domain_dataset, _tensor_digest,
    build_heterogeneity_data, build_heterogeneity_from_datasets,
    clear_heterogeneity_cache, largest_remainder, metadata_digest, quantity_label_counts,
)
from datasets.partition import _paired_dirichlet


class SyntheticRGB(Dataset):
    def __init__(self, class_counts, offset=0):
        self.labels = np.repeat(np.arange(len(class_counts)), class_counts)
        generator = torch.Generator().manual_seed(100 + offset)
        self.images = torch.rand(len(self.labels), 3, 6, 6, generator=generator) * 2 - 1
        self.reads = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.reads += 1
        return self.images[index], int(self.labels[index])


class HeterogeneityTests(unittest.TestCase):
    def setUp(self):
        self.sources = (SyntheticRGB([61, 80, 45, 54]),
                        SyntheticRGB([17, 19, 13, 15], 1),
                        SyntheticRGB([23, 25, 19, 21], 2))

    def tearDown(self):
        clear_heterogeneity_cache()

    def build(self, cohort, seed=2031, **kwargs):
        return build_heterogeneity_from_datasets(
            cohort, seed, *self.sources, num_classes=4, batch_size=11, **kwargs)

    def test_all_cohorts_cover_every_official_id_once_with_stored_labels(self):
        for cohort in COHORTS:
            groups = self.build(cohort)
            metadata = groups[3]
            for split, loaders, original in zip(("train", "val", "test"), groups[:3], self.sources):
                ids = [i for loader in loaders.values() for i in loader.dataset.indices]
                self.assertEqual(sorted(ids), list(range(len(original))), (cohort, split))
                for cid, loader in loaders.items():
                    client = metadata["clients"][cid]
                    expected = np.bincount(original.labels[loader.dataset.indices], minlength=4)
                    self.assertEqual(client["class_counts"][split], expected.tolist())
                    self.assertEqual(client["sample_ids"][f"{split}_sample_ids"], loader.dataset.indices)
                    np.testing.assert_array_equal(training_class_counts(loader.dataset, 4), expected)
                    self.assertEqual(sample_id(loader.dataset, 0), loader.dataset.indices[0])
                    self.assertIsInstance(loader.sampler, RandomSampler if split == "train" else SequentialSampler)
                    self.assertIsNotNone(loader.generator)
            digest_data = {key: value for key, value in metadata.items() if key != "partition_sha256"}
            encoded = json.dumps(digest_data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            self.assertEqual(metadata["partition_sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(metadata_digest(json.loads(json.dumps(digest_data))), metadata["partition_sha256"])

    def test_label_and_mixed_reuse_existing_paired_partition(self):
        expected_train, expected_val, attempts = _paired_dirichlet(
            *self.sources[:2], 3, 0.3, 2031, 10, 1)
        label = self.build("label")
        mixed = self.build("mixed")
        for cid in range(3):
            self.assertEqual(label[0][cid].dataset.indices, expected_train[cid])
            self.assertEqual(label[1][cid].dataset.indices, expected_val[cid])
            for split in range(3):
                self.assertEqual(label[split][cid].dataset.indices, mixed[split][cid].dataset.indices)
        self.assertEqual(label[3]["allocation"]["partition_attempts"], attempts)
        train_counts = np.array([label[3]["clients"][cid]["class_counts"]["train"] for cid in range(3)]).T
        probabilities = train_counts / train_counts.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(label[3]["allocation"]["train_class_assignment_probabilities"], probabilities)
        for cls, n in enumerate(np.bincount(self.sources[2].labels)):
            expected = largest_remainder(int(n), probabilities[cls])
            actual = [label[3]["clients"][cid]["class_counts"]["test"][cls] for cid in range(3)]
            np.testing.assert_array_equal(actual, expected)

    def test_test_labels_cannot_change_train_or_val_partition_or_domains(self):
        before = self.build("mixed")
        self.sources[2].labels = self.sources[2].labels[::-1].copy()
        after = self.build("mixed")
        for split in (0, 1):
            for cid in range(3):
                self.assertEqual(before[split][cid].dataset.indices, after[split][cid].dataset.indices)
                self.assertEqual(before[3]["clients"][cid]["tensor_realization_sha256"]["train"],
                                 after[3]["clients"][cid]["tensor_realization_sha256"]["train"])
        self.assertEqual(before[3]["transform_policy"], after[3]["transform_policy"])

    def test_equal_class_and_quantity_allocations_have_bounded_rounding_error(self):
        for cohort, target in (("appearance", np.ones(3) / 3), ("quantity", np.array([0.1, 0.3, 0.6]))):
            groups = self.build(cohort)
            for split_index, split in enumerate(("train", "val", "test")):
                matrix = np.array([groups[3]["clients"][cid]["class_counts"][split] for cid in range(3)]).T
                expected = np.bincount(self.sources[split_index].labels)[:, None] * target
                self.assertTrue(np.all(np.abs(matrix - expected) < 1))
                if cohort == "appearance":
                    self.assertTrue(np.all(matrix.max(axis=1) - matrix.min(axis=1) <= 1))

    def test_ipf_rounding_preserves_both_margins_for_diverse_seeded_cases(self):
        rng = np.random.default_rng(19)
        for _ in range(100):
            counts = rng.integers(0, 500, size=8)
            odds = rng.dirichlet([0.3] * 3, size=8)
            totals = largest_remainder(int(counts.sum()), [0.1, 0.3, 0.6])
            matrix, detail = quantity_label_counts(counts, odds, totals)
            np.testing.assert_array_equal(matrix.sum(axis=1), counts)
            np.testing.assert_array_equal(matrix.sum(axis=0), totals)
            self.assertTrue(np.all(matrix >= 0))
            self.assertLess(detail["maximum_margin_error"], 1e-9)
            self.assertTrue(np.all(np.abs(matrix - np.array(detail["fractional_matrix"])) <= 1 + 1e-9))
        groups = self.build("label_quantity")
        for split_index, split in enumerate(("train", "val", "test")):
            actual = [len(loader.dataset) for loader in groups[split_index].values()]
            np.testing.assert_array_equal(actual, largest_remainder(len(self.sources[split_index]), [0.1, 0.3, 0.6]))
        self.assertIn("axes are not independent", " ".join(groups[3]["limitations"]))

    def test_zero_count_classes_ipf_and_largest_remainder_ties(self):
        counts = np.array([0, 30, 0, 50])
        odds = np.array([[0, 0, 0], [0.01, 0.3, 0.69], [0, 0, 0], [0.8, 0.1, 0.1]])
        result, _ = quantity_label_counts(counts, odds, [8, 24, 48])
        np.testing.assert_array_equal(result[[0, 2]], np.zeros((2, 3)))
        np.testing.assert_array_equal(largest_remainder(2, [1, 1, 1]), [1, 1, 0])

    def test_domain_policy_has_exact_identity_fixed_colour_and_gaussian_blur(self):
        images = torch.zeros(2, 3, 5, 5)
        images[1, :, 2, 2] = 1
        source = CachedTensorDataset(images, [0, 1])
        self.assertIs(_domain_dataset(source, 0), source)
        warm = _domain_dataset(source, 1)
        expected = torch.tensor([0.945, 0.81, 0.765]) - 1
        torch.testing.assert_close(warm[0][0][:, 0, 0], expected)
        cool = _domain_dataset(source, 2)
        self.assertTrue(torch.all(cool._images >= -1))
        self.assertTrue(torch.all(cool._images <= 1))
        # An impulse is spread only spatially, with no inter-channel mixing.
        difference = cool[1][0] - cool[0][0]
        self.assertTrue(torch.all(difference[:, 1:4, 1:4] > 0))
        self.assertTrue(torch.all(difference[:, 0, :] == 0))
        returned = warm[0][0]
        returned.fill_(77)
        self.assertLessEqual(float(warm[0][0].max()), 1)
        self.assertEqual(APPEARANCE_POLICY["domains"][2]["blur_sigma"], 0.8)

    def test_realization_hashes_bind_transforms_order_and_image_contents(self):
        groups = self.build("mixed")
        for split, loaders in zip(("train", "val", "test"), groups[:3]):
            for cid, loader in loaders.items():
                actual = torch.stack([loader.dataset[i][0] for i in range(len(loader.dataset))])
                self.assertEqual(_tensor_digest(actual), groups[3]["clients"][cid]["tensor_realization_sha256"][split])
        before = groups[3]["partition_sha256"]
        self.sources[0].images[0, 0, 0, 0] *= -1
        after = self.build("mixed")[3]["partition_sha256"]
        self.assertNotEqual(before, after)

    def test_build_and_all_loader_iterations_preserve_global_rng_states(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        for cohort in COHORTS:
            groups = self.build(cohort)
            for loaders in groups[:3]:
                for loader in loaders.values():
                    next(iter(loader))
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_public_api_caches_sources_and_partitions_but_returns_fresh_loaders(self):
        # A full-size cheap fake satisfies public full-split validation without downloads.
        class FullSize(Dataset):
            def __init__(self, size):
                self.labels = np.arange(size) % 8

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                return torch.zeros(3, 2, 2), int(self.labels[index])

        source = {"train": FullSize(11959), "val": FullSize(1712), "test": FullSize(3421)}
        calls = []

        def fake_load(name, split, image_size, download, root):
            calls.append((name, split, image_size, download, root))
            return source[split]

        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "bloodmnist.npz"
            archive.write_bytes(b"synthetic cache-key fixture only")
            with patch("datasets.heterogeneity.load_medmnist_split", side_effect=fake_load):
                first = build_heterogeneity_data("quantity", 2040, batch_size=13, root=temp)
                # Advance first generator and mutate first metadata/indices.
                iterator = iter(first[0][0])
                next(iterator)
                first[3]["clients"][0]["class_counts"]["train"][0] = -1
                first[0][0].dataset.indices.reverse()
                second = build_heterogeneity_data("quantity", 2040, batch_size=19, root=temp)
                third = build_heterogeneity_data("quantity", 2041, root=temp)
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(not call[3] for call in calls))
                self.assertIsNot(first[0][0], second[0][0])
                self.assertIsNot(first[0][0].generator, second[0][0].generator)
                self.assertGreaterEqual(second[3]["clients"][0]["class_counts"]["train"][0], 0)
                self.assertEqual(second[0][0].batch_size, 19)
                self.assertNotEqual(second[3]["partition_sha256"], third[3]["partition_sha256"])
                repeat = build_heterogeneity_data("quantity", 2040, batch_size=19, root=temp)
                self.assertEqual(second[3], repeat[3])
                torch.testing.assert_close(next(iter(second[0][0]))[1], next(iter(repeat[0][0]))[1])
                archive.write_bytes(b"changed archive fingerprint fixture")
                build_heterogeneity_data("quantity", 2040, root=temp)
                self.assertEqual(len(calls), 6)

    def test_invalid_options_images_and_missing_cache_fail_before_training(self):
        for kwargs in ({"cohort": "unknown"}, {"seed": -1}, {"seed": True},
                       {"alpha": 0}, {"alpha": float("nan")}, {"batch_size": 0}):
            options = dict(cohort="label", seed=2031, alpha=0.3, batch_size=11)
            options.update(kwargs)
            with self.assertRaises(ValueError):
                build_heterogeneity_from_datasets(**options, train=self.sources[0],
                                                  val=self.sources[1], test=self.sources[2], num_classes=4)
        self.sources[0].images[0, 0, 0, 0] = 12
        with self.assertRaises(ValueError):
            self.build("quantity")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "downloads disabled"):
                build_heterogeneity_data("label", 2031, root=temp)


if __name__ == "__main__":
    unittest.main()
