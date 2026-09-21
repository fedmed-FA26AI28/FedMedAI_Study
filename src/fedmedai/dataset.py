"""Dataset loading and preprocessing for MedMNIST (BloodMNIST)."""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO


def dataset_info(dataset_name):
    """Validate the supported single-label 2D task and preserve official labels."""
    matches = [info for info in INFO.values() if info["python_class"] == dataset_name]
    if not matches or "3D" in dataset_name:
        raise ValueError(f"Unsupported 2D MedMNIST dataset: {dataset_name}")
    info = matches[0]
    if info["task"] not in ("multi-class", "binary-class"):
        raise ValueError("Only single-label classification is supported")
    return info


def get_transforms(image_size: int = 28) -> transforms.Compose:
    """Image preprocessing transforms for 28x28 RGB medical image classification."""
    transform_list = []
    if image_size != 28:
        transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(transform_list)


class MedMNISTTorchDataset(Dataset):
    """PyTorch Dataset wrapper around MedMNIST dataset.
    
    Ensures targets are 1D LongTensors and images are transformed to RGB 28x28 tensors.
    """

    def __init__(self, medmnist_dataset, transform=None):
        self.dataset = medmnist_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.dataset[idx]
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(np.squeeze(target))
        return img, label

    @property
    def labels(self) -> np.ndarray:
        """Return all labels as a flat 1D numpy array."""
        return np.asarray(self.dataset.labels).reshape(-1)


def load_medmnist_split(
    dataset_name: str = "BloodMNIST",
    split: str = "train",
    image_size: int = 28,
    download: bool = True,
    root: Optional[str] = None,
) -> MedMNISTTorchDataset:
    """Instantiate and wrap a MedMNIST dataset split."""
    dataset_info(dataset_name)
    data_class = getattr(medmnist, dataset_name)
    transform = get_transforms(image_size)
    kwargs = {"root": root} if root is not None else {}
    raw_dataset = data_class(split=split, download=download, **kwargs)
    return MedMNISTTorchDataset(raw_dataset, transform=transform)


def get_centralized_test_loader(
    dataset_name: str = "BloodMNIST",
    batch_size: int = 32,
    image_size: int = 28,
    max_samples: Optional[int] = None,
    seed: int = 42,
    root: Optional[str] = None,
    download: bool = True,
) -> DataLoader:
    """Create DataLoader for the centralized evaluation/test set."""
    test_dataset = load_medmnist_split(dataset_name, split="test", image_size=image_size,
                                     root=root, download=download)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or null")
    if max_samples is not None and max_samples < len(test_dataset):
        indices = np.random.default_rng(seed).permutation(len(test_dataset))[:max_samples].tolist()
        test_dataset = Subset(test_dataset, indices)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
