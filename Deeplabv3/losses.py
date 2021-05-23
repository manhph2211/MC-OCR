from torch import nn
import torch
from torch import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.size_average = size_average

    def forward(self, input_tensor, target_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), input_tensor.size(2), -1)
        input_tensor = input_tensor.transpose(1, 2)
        input_tensor = input_tensor.contiguous().view(-1, input_tensor.size(2))

        target_tensor = target_tensor.view(target_tensor.size(0), target_tensor.size(2), -1)
        target_tensor = target_tensor.transpose(1, 2)
        target_tensor = target_tensor.contiguous().view(-1, target_tensor.size(2))

        logpt = F.log_softmax(input)

        return logpt
