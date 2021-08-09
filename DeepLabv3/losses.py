from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha, 1-alpha])
        self.size_average = size_average

    def forward(self, input_tensor, target_tensor):
        if input_tensor.dim() > 2:
            input_tensor = input_tensor.view(input_tensor.size(0), input_tensor.size(1), -1)  # N,C,H,W => N,C,H*W
            input_tensor = input_tensor.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input_tensor = input_tensor.contiguous().view(-1, input_tensor.size(2))   # N,H*W,C => N*H*W,C
        target_tensor = target_tensor.view(-1, 1)

        logpt = F.log_softmax(input_tensor)
        logpt = logpt.gather(1, target_tensor)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input_tensor.data.type():
                self.alpha = self.alpha.type_as(input_tensor.data)
            at = self.alpha.gather(0, target_tensor.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
