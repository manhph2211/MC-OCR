import torch


def accuracy(pred, gt):

    return torch.sum(pred == gt) / len(pred)
