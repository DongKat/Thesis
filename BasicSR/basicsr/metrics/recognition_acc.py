import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_accuracy(imgs, gt_labels, model, device):
    """ Calculate recognition accuracy.

    Args:
        img (Tensor): Image tensor with shape (n, c, h, w).
        gt_label (int): Ground truth label for each image.
        model (nn.Module): Torch Recognition model.

    Returns:
        float: Recognition accuracy.
    """

    model.eval()
    model.to(device)
    with torch.no_grad():
        imgs = imgs.to(device)
        gt_labels = gt_labels.to(device)

        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)

        total = gt_labels.size(0)
        correct = (predicted == gt_labels).sum().item()

    return correct / total