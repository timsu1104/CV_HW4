import torch
from torch import nn, Tensor

def eval(pred: Tensor, gt: Tensor):
    """
    pred: (B, H, W)
    gt: (B, H, W)
    """
    