from torch.utils.data import DataLoader
from .datasets import *


def mnist_first5_train_loader(batch_size):
    return DataLoader(mnist_first5_train_dataset, batch_size=batch_size, shuffle=True,)

def mnist_first5_test_loader(batch_size):
    return DataLoader(mnist_first5_test_dataset, batch_size=batch_size, shuffle=False,)


def mnist_last5_train_loader(batch_size):
    return DataLoader(mnist_last5_train_dataset, batch_size=batch_size, shuffle=True,)

def mnist_last5_test_loader(batch_size):
    return DataLoader(mnist_last5_test_dataset, batch_size=batch_size, shuffle=False,)


def mnist_combined_train_loader(batch_size):
    return DataLoader(mnist_combined_train_dataset, batch_size=batch_size, shuffle=True,)

def mnist_combined_train_loader_noshuffle(batch_size):
    return DataLoader(mnist_combined_train_dataset, batch_size=batch_size, shuffle=False,)

def mnist_combined_test_loader(batch_size):
    return DataLoader(mnist_combined_test_dataset, batch_size=batch_size, shuffle=False,)


def mnist_combined_train_loader_easy(batch_size):
    return DataLoader(mnist_combined_train_dataset_easy, batch_size=batch_size, shuffle=True,)

def mnist_combined_test_loader_easy(batch_size):
    return DataLoader(mnist_combined_test_dataset_easy, batch_size=batch_size, shuffle=False,)


def mnist_first5_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(mnist_first5_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def mnist_first5_test_loader_unseen(batch_size):
    return DataLoader(mnist_first5_test_dataset_unseen, batch_size=batch_size, shuffle=False,)


def mnist_last5_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(mnist_last5_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def mnist_last5_test_loader_unseen(batch_size):
    return DataLoader(mnist_last5_test_dataset_unseen, batch_size=batch_size, shuffle=False,)


def mnist_combined_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(mnist_combined_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def mnist_combined_test_loader_unseen(batch_size):
    return DataLoader(mnist_combined_test_dataset_unseen, batch_size=batch_size, shuffle=False,)