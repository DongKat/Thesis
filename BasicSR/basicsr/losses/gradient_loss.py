import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']

@LOSS_REGISTRY.register()
class GradientProfileLoss(nn.Module):
    ''' Gradient Profile Loss.
    Implementation from TextZoom repository: https://github.com/WenjiaWang0312/TextZoom/blob/master/src/loss/gradient_loss.py

    Args:
        pred (Tensor): Prediction images.
        target (Tensor): Target images.

    '''
    def __init__(self, loss_weight=1.0, reduction='mean', criterion_type='l1'):
        super(GradientProfileLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight

        if criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction=reduction)
        elif criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError(f'{criterion_type} criterion has not been supported.')

    def gradient_map(self, x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad

    def forward(self, pred, target):
        map_pred = self.gradient_map(pred)
        map_target = self.gradient_map(target)
        return self.criterion(map_pred, map_target) * self.loss_weight

