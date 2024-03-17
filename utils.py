## Imports
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from constants import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if multi-GPU
    torch.backends.cudnn.deterministic = (
        True  # only applies to CUDA convolution operations
    )
    torch.backends.cudnn.benchmark = False
    # usually CuDNN has heuristics as to which algorithm to pick.
    # cudnn.benchmark benchmarks several algorithms and picks the fastest, which is often helpful
    # if your input shapes are fixed and not changing a lot during training. However, this means it
    # may pick a different algorithm even when the deterministic flag is set.
    # As such it is good practice to turn off cudnn.benchmark when turning on cudnn.deterministic


transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors and [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1, 1]
    ]
)


def load_data(name):

    assert name in ["CIFAR-10", "MNIST"], "`name` must be either `CIFAR-10` or `MNIST`"

    if name == "CIFAR-10":
        train_data = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_fn
        )
        val_data = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_fn
        )

    elif name == "MNIST":
        train_data = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_fn
        )
        val_data = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_fn
        )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)

    return train_data, val_data, train_loader, val_loader


## Label dropout util fn to employ in train function, applied in-place
def drop_label(labels, p):
    """
    labels = batch of conditional labels
    p = dropout prob
    """
    mask = np.random.rand(*labels.shape) < p
    labels[mask] = UNCOND_LABEL


def cfg_loss(preds, uncond_preds, noise, train_guidance):
    niket = (1 + train_guidance) * preds - train_guidance * uncond_preds
    # niket = train_guidance * preds + (1-train_guidance) * uncond_preds ###### if using this line, train_guidance would need to be 2, not 1
    loss = F.mse_loss(niket, noise)
    return loss


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
