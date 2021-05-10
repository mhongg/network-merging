from torch.utils.data import ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from config import DATA_DIR
import torch
import numpy as np
import imgaug.augmenters as iaa
import PIL

mnist_classes = [
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

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ExtendedMNIST(MNIST):
    """
    MNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedMNIST, self).__init__(root, **kwargs)


class ExtendedCIFAR10(CIFAR10):
    """
    MNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedCIFAR10, self).__init__(root, **kwargs)
        extended_class_len = len(extended_classes)
        self.targets = [t + extended_class_len for t in self.targets]


# Rgb transform
transform_rgb = transforms.Lambda(lambda img: img.convert("RGB"))


mnist_train_dataset = MNIST(
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
mnist_test_dataset = MNIST(
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
cifar10_train_dataset = CIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
cifar10_test_dataset = CIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)


# Dataset for testing the mnist_cifar10 combined models
extended_mnist_test_dataset = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_test_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    extended_classes=mnist_classes,
)


# Dataset for training PAN purpose
extended_mnist_single_channel_train_dataset = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_single_channel_train_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    extended_classes=mnist_classes,
)


extended_mnist_single_channel_test_dataset = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_single_channel_test_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    extended_classes=mnist_classes,
)


extended_mnist_3_channel_train_dataset = ExtendedMNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transform_rgb,
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    extended_classes=cifar10_classes,
)

extended_cifar10_3_channel_train_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    extended_classes=mnist_classes,
)

extended_mnist_3_channel_test_dataset = ExtendedMNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transform_rgb,
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    extended_classes=cifar10_classes,
)

extended_cifar10_3_channel_test_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    extended_classes=mnist_classes,
)


# Concat the datasets
mnist_cifar10_single_channel_train_dataset = ConcatDataset(
    [extended_mnist_single_channel_train_dataset, extended_cifar10_single_channel_train_dataset]
)
mnist_cifar10_single_channel_test_dataset = ConcatDataset(
    [extended_mnist_single_channel_test_dataset, extended_cifar10_single_channel_test_dataset]
)
mnist_cifar10_3_channel_train_dataset = ConcatDataset(
    [extended_mnist_3_channel_train_dataset, extended_cifar10_3_channel_train_dataset]
)
mnist_cifar10_3_channel_test_dataset = ConcatDataset(
    [extended_mnist_3_channel_test_dataset, extended_cifar10_3_channel_test_dataset]
)


# Easy datasets for testing FPAN purposes
extended_mnist_single_channel_train_dataset_easy = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_single_channel_train_dataset_easy = ExtendedCIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Grayscale(),
            lambda x: np.array(x),
            iaa.Add(value=10).augment_image,
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    extended_classes=mnist_classes,
)


extended_mnist_single_channel_test_dataset_easy = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_single_channel_test_dataset_easy = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Grayscale(),
            lambda x: np.array(x),
            iaa.Add(value=10).augment_image,
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    extended_classes=mnist_classes,
)

# Concat the easy datasets
mnist_cifar10_single_channel_train_dataset_easy = ConcatDataset(
    [extended_mnist_single_channel_train_dataset_easy, extended_cifar10_single_channel_train_dataset_easy]
)
mnist_cifar10_single_channel_test_dataset_easy = ConcatDataset(
    [extended_mnist_single_channel_test_dataset_easy, extended_cifar10_single_channel_test_dataset_easy]
)


# Unseen dataset
mnist_single_channel_train_dataset_unseen, mnist_single_channel_test_dataset_unseen \
    = random_split(extended_mnist_single_channel_test_dataset, [4000, len(extended_mnist_single_channel_test_dataset)-4000], torch.Generator().manual_seed(0))

mnist_3_channel_train_dataset_unseen, mnist_3_channel_test_dataset_unseen \
    = random_split(extended_mnist_3_channel_test_dataset, [4000, len(extended_mnist_3_channel_test_dataset)-4000], torch.Generator().manual_seed(0))

cifar10_single_channel_train_dataset_unseen, cifar10_single_channel_test_dataset_unseen \
    = random_split(extended_cifar10_single_channel_test_dataset, [4000, len(extended_cifar10_3_channel_test_dataset)-4000], torch.Generator().manual_seed(0))

cifar10_3_channel_train_dataset_unseen, cifar10_3_channel_test_dataset_unseen \
    = random_split(extended_cifar10_3_channel_test_dataset, [4000, len(extended_cifar10_3_channel_test_dataset)-4000], torch.Generator().manual_seed(0))

mnist_cifar10_single_channel_train_dataset_unseen, mnist_cifar10_single_channel_test_dataset_unseen \
    = random_split(mnist_cifar10_single_channel_test_dataset, [8000, len(mnist_cifar10_single_channel_test_dataset)-8000], torch.Generator().manual_seed(0))

mnist_cifar10_3_channel_train_dataset_unseen, mnist_cifar10_3_channel_test_dataset_unseen \
    = random_split(mnist_cifar10_3_channel_test_dataset, [8000, len(mnist_cifar10_3_channel_test_dataset)-8000], torch.Generator().manual_seed(0))