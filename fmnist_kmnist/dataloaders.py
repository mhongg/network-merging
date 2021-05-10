from torch.utils.data import DataLoader
from .datasets import *


def fmnist_train_loader(batch_size):
    return DataLoader(fmnist_train_dataset, batch_size=batch_size, shuffle=True,)

def fmnist_test_loader(batch_size):
    return DataLoader(fmnist_test_dataset, batch_size=batch_size, shuffle=False,)


def kmnist_train_loader(batch_size):
    return DataLoader(kmnist_train_dataset, batch_size=batch_size, shuffle=True,)

def kmnist_test_loader(batch_size):
    return DataLoader(kmnist_test_dataset, batch_size=batch_size, shuffle=False,)


# For training the PAN models
def fmnist_kmnist_train_loader(batch_size):
    return DataLoader(fmnist_kmnist_train_dataset, batch_size=batch_size, shuffle=True,)

def fmnist_kmnist_train_loader_noshuffle(batch_size):
    return DataLoader(fmnist_kmnist_train_dataset, batch_size=batch_size, shuffle=False,)

def fmnist_kmnist_test_loader(batch_size):
    return DataLoader(fmnist_kmnist_test_dataset, batch_size=batch_size, shuffle=False,)


def fmnist_kmnist_train_loader_easy(batch_size):
    return DataLoader(fmnist_kmnist_train_dataset_easy, batch_size=batch_size, shuffle=True,)

def fmnist_kmnist_test_loader_easy(batch_size):
    return DataLoader(fmnist_kmnist_test_dataset_easy, batch_size=batch_size, shuffle=False,)


def fmnist_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(fmnist_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def fmnist_test_loader_unseen(batch_size):
    return DataLoader(fmnist_test_dataset_unseen, batch_size=batch_size, shuffle=False,)


def kmnist_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(kmnist_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def kmnist_test_loader_unseen(batch_size):
    return DataLoader(kmnist_test_dataset_unseen, batch_size=batch_size, shuffle=False,)


def fmnist_kmnist_train_loader_unseen_noshuffle(batch_size):
    return DataLoader(fmnist_kmnist_train_dataset_unseen, batch_size=batch_size, shuffle=False,)

def fmnist_kmnist_test_loader_unseen(batch_size):
    return DataLoader(fmnist_kmnist_test_dataset_unseen, batch_size=batch_size, shuffle=False,)