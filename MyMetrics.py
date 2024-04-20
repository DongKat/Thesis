# Thu's Resnet ckpt accuracy, maybe PP-OCR Nom acc too

# %%
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
import torch
import torch.functional as F
import cv2

# psnr and ssim is already implemented in basicsr


def calculate_RecognitionAccuracy(recognitionModel: torch.nn.Module, 
                                    img: torch.Tensor, 
                                    label: torch.Tensor, 
                                    crop_border: int = 0):
    """ Calculate recognition accuracy of a recognition model on a batch of images and labels
    
    Args:
        recognitionModel (torch.nn.Module): Recognition model
        img (torch.Tensor): Batch of images
        label (torch.Tensor): Batch of labels
        crop_border (int): Crop border size
    
    Output:
        tensor.float: Recognition accuracy
    """
    
    with torch.no_grad:
        img = img.to(torch.float64)
        label = label.to(torch.float64)
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        # This probably works for most recognition models
        pred = recognitionModel(img)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        # Calculate accuracy
        correct = torch.sum(pred == label)
        total = label.size(0)
        accuracy = correct / total
        
    return accuracy
        


    
