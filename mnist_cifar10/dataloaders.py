from torch.utils.data import DataLoader
from .datasets import *


def mnist_train_loader(batch_size):
    return DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True,)

def mnist_test_loader(batch_size):
    return DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False,)


def cifar10_train_loader(batch_size):
    return DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True,)

def cifar10_test_loader(batch_size):
    return DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False,)

def cifar10_single_channel_train_loader(batch_size):
    return DataLoader(extended_cifar10_single_channel_train_dataset, batch_size=batch_size, shuffle=True,)


# For training the PAN models
def mnist_cifar10_single_channel_train_loader(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_train_dataset, batch_size=batch_size, shuffle=True,
    )

def mnist_cifar10_single_channel_train_loader_noshuffle(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_train_dataset, batch_size=batch_size, shuffle=False,
    )

def mnist_cifar10_single_channel_test_loader(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_test_dataset, batch_size=batch_size, shuffle=False,
    )


def mnist_cifar10_3_channel_train_loader(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_train_dataset, batch_size=batch_size, shuffle=True,
    )

def mnist_cifar10_3_channel_train_loader_noshuffle(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_train_dataset, batch_size=batch_size, shuffle=False,
    )

def mnist_cifar10_3_channel_test_loader(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_test_dataset, batch_size=batch_size, shuffle=False,
    )


def mnist_cifar10_single_channel_train_loader_easy(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_train_dataset_easy, batch_size=batch_size, shuffle=True,
    )

def mnist_cifar10_single_channel_test_loader_easy(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_test_dataset_easy, batch_size=batch_size, shuffle=False,
    )


def mnist_single_channel_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(mnist_single_channel_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def mnist_3_channel_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(mnist_3_channel_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def mnist_single_channel_test_loader_unseen(batch_size):
    return DataLoader(mnist_single_channel_test_dataset_unseen, batch_size=batch_size, shuffle=False,)


def cifar10_single_channel_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(cifar10_single_channel_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def cifar10_3_channel_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(cifar10_3_channel_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def cifar10_single_channel_test_loader_unseen(batch_size):
    return DataLoader(cifar10_single_channel_test_dataset_unseen, batch_size=batch_size, shuffle=False,)


def mnist_cifar10_single_channel_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_train_dataset_unseen, batch_size=batch_size, shuffle=False,
    )

def mnist_cifar10_3_channel_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_train_dataset_unseen, batch_size=batch_size, shuffle=False,
    )

def mnist_cifar10_single_channel_test_loader_unseen(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_test_dataset_unseen, batch_size=batch_size, shuffle=False,
    )

def mnist_cifar10_3_channel_test_loader_unseen(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_test_dataset_unseen, batch_size=batch_size, shuffle=False,
    )