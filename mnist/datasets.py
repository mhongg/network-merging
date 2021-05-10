from torch.utils.data import ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from config import DATA_DIR
import torch
import numpy as np
import imgaug.augmenters as iaa
import PIL


class DisjointMNIST(MNIST):
    def __init__(self, root, start_idx=0, end_idx=10, **kwargs):
        super(DisjointMNIST, self).__init__(root, **kwargs)
        self.start_idx = start_idx
        self.end_idx = end_idx
        classes = [
            "0 - zero",
            "1 - one",
            "2 - two",
            "3 - three",
            "4 - four",
            "5 - five",
            "6 - six",
            "7 - seven",
            "8 - eight",
            "9 - nine",
        ]
        # Slicing begins here
        self.classes = classes[start_idx:end_idx]
        sliced_labels_idx = [
            i
            for i in range(len(self.targets))
            if self.targets[i] in list(range(self.start_idx, self.end_idx))
        ]
        self.data = self.data[sliced_labels_idx]
        self.targets = self.targets[sliced_labels_idx] - self.start_idx


mnist_first5_train_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=0,
    end_idx=5,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_first5_test_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=0,
    end_idx=5,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_last5_train_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=5,
    end_idx=10,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_last5_test_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=5,
    end_idx=10,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_combined_train_dataset = MNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)


mnist_combined_test_dataset = MNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)


# Easy datasets for testing FPAN purposes
mnist_first5_train_dataset_easy = DisjointMNIST(
    DATA_DIR,
    start_idx=0,
    end_idx=5,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            lambda x: np.array(x),
            iaa.Add(value=-10).augment_image,
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_last5_train_dataset_easy = DisjointMNIST(
    DATA_DIR,
    start_idx=5,
    end_idx=10,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            lambda x: np.array(x),
            iaa.Add(value=10).augment_image,
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_first5_test_dataset_easy = DisjointMNIST(
    DATA_DIR,
    start_idx=0,
    end_idx=5,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            lambda x: np.array(x),
            iaa.Add(value=-10).augment_image,
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

mnist_last5_test_dataset_easy = DisjointMNIST(
    DATA_DIR,
    start_idx=5,
    end_idx=10,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            lambda x: np.array(x),
            iaa.Add(value=10).augment_image,
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)

# Concat the easy datasets
mnist_combined_train_dataset_easy = ConcatDataset(
    [mnist_first5_train_dataset_easy, mnist_last5_train_dataset_easy]
)
mnist_combined_test_dataset_easy = ConcatDataset(
    [mnist_first5_test_dataset_easy, mnist_last5_test_dataset_easy]
)


# Unseen dataset
mnist_first5_train_dataset_unseen, mnist_first5_test_dataset_unseen \
    = random_split(mnist_first5_test_dataset, [4000, len(mnist_first5_test_dataset)-4000], torch.Generator().manual_seed(0))

mnist_last5_train_dataset_unseen, mnist_last5_test_dataset_unseen \
    = random_split(mnist_last5_test_dataset, [4000, len(mnist_last5_test_dataset)-4000], torch.Generator().manual_seed(0))

mnist_combined_train_dataset_unseen, mnist_combined_test_dataset_unseen \
    = random_split(mnist_combined_test_dataset, [8000, len(mnist_combined_test_dataset)-8000], torch.Generator().manual_seed(0))