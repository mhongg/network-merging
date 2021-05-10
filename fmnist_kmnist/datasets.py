from torch.utils.data import ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST, KMNIST
from config import DATA_DIR
import torch
import numpy as np
import imgaug.augmenters as iaa
import PIL

fmnist_classes = [
    "0 - T-shirt/top",
    "1 - Trouser",
    "2 - Pullover",
    "3 - Dress",
    "4 - Coat",
    "5 - Sandal",
    "6 - Shirt",
    "7 - Sneaker",
    "8 - Bag",
    "9 - Ankle boot",
]

kmnist_classes = [
    "0 - お",
    "1 - き",
    "2 - す",
    "3 - つ",
    "4 - な",
    "5 - は",
    "6 - ま",
    "7 - や",
    "8 - れ",
    "9 - を",
]


class ExtendedFMNIST(FashionMNIST):
    """
    FashionMNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedFMNIST, self).__init__(root, **kwargs)


class ExtendedKMNIST(KMNIST):
    """
    KMNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedKMNIST, self).__init__(root, **kwargs)
        extended_class_len = len(extended_classes)
        self.targets = [t + extended_class_len for t in self.targets]


fmnist_train_dataset = FashionMNIST(
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
fmnist_test_dataset = FashionMNIST(
    DATA_DIR,
    train=False,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
)
kmnist_train_dataset = KMNIST(
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
kmnist_test_dataset = KMNIST(
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


# Dataset for training PAN purpose
extended_fmnist_train_dataset = ExtendedFMNIST(
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
    extended_classes=kmnist_classes,
)

extended_kmnist_train_dataset = ExtendedKMNIST(
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
    extended_classes=fmnist_classes,
)


extended_fmnist_test_dataset = ExtendedFMNIST(
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
    extended_classes=kmnist_classes,
)

extended_kmnist_test_dataset = ExtendedKMNIST(
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
    extended_classes=fmnist_classes,
)


# Concat the datasets
fmnist_kmnist_train_dataset = ConcatDataset(
    [extended_fmnist_train_dataset, extended_kmnist_train_dataset]
)
fmnist_kmnist_test_dataset = ConcatDataset(
    [extended_fmnist_test_dataset, extended_kmnist_test_dataset]
)


# Easy datasets for testing FPAN purposes
extended_fmnist_train_dataset_easy = ExtendedFMNIST(
    DATA_DIR,
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
    extended_classes=kmnist_classes,
)

extended_kmnist_train_dataset_easy = ExtendedKMNIST(
    DATA_DIR,
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
    extended_classes=fmnist_classes,
)


extended_fmnist_test_dataset_easy = ExtendedFMNIST(
    DATA_DIR,
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
    extended_classes=kmnist_classes,
)

extended_kmnist_test_dataset_easy = ExtendedKMNIST(
    DATA_DIR,
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
    extended_classes=fmnist_classes,
)

# Concat the easy datasets
fmnist_kmnist_train_dataset_easy = ConcatDataset(
    [extended_fmnist_train_dataset_easy, extended_kmnist_train_dataset_easy]
)
fmnist_kmnist_test_dataset_easy = ConcatDataset(
    [extended_fmnist_test_dataset_easy, extended_kmnist_test_dataset_easy]
)


# Unseen dataset
fmnist_train_dataset_unseen, fmnist_test_dataset_unseen \
    = random_split(fmnist_test_dataset, [4000, len(fmnist_test_dataset)-4000], torch.Generator().manual_seed(0))

kmnist_train_dataset_unseen, kmnist_test_dataset_unseen \
    = random_split(kmnist_test_dataset, [4000, len(kmnist_test_dataset)-4000], torch.Generator().manual_seed(0))

fmnist_kmnist_train_dataset_unseen, fmnist_kmnist_test_dataset_unseen \
    = random_split(fmnist_kmnist_test_dataset, [8000, len(fmnist_kmnist_test_dataset)-8000], torch.Generator().manual_seed(0))