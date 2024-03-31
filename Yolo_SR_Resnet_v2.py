#%%
import cv2
import numpy as np
import os
import pandas as pd
import pickle
import torch
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import pybboxes as pybbx

import torchvision
from torch.nn.functional import softmax
from torchmetrics import Accuracy, Precision, Recall, F1Score

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Images path for yolo. I test on 1902 first
YOLO_WEIGHTS = 'yolov5/weights/yolo_one_label.pt'
TOK1902_RAW = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-raw-images'
TOK1902_RAW_ANNOTATIONS = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-annotation/annotation-mynom'

LVT_RAW = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-raw-images'
LVT_RAW_ANNOTATIONS = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-annotation/annotation-mynom'
BATCH_SIZE = 16




#%%

# Call yolo detect.py
# TODO: This works, in acceptable condition
from yolov5.detect import run as YoloInference

project = 'TempResources/yolov5_runs_ToK1902/detect'
name = 'yolo_SR'
YOLO_RESULTS = os.path.join(project, name)
annotations = YOLO_RESULTS + '/labels' # Yolo will save the labels here

#%%
args = {
    'weights': YOLO_WEIGHTS,
    'source': TOK1902_RAW,
    'imgsz': (640, 640),
    'conf_thres': 0.5,
    'iou_thres': 0.5,   
    'device': '',       # Let YOLO decide
    'save_txt': True,
    'save_crop': True,  # Save cropped prediction boxes, for debugging
    'project': project,
    'name': name,
    'exist_ok': True,
    'hide_labels': True,    # Hide labels from output images
    'hide_conf': True,      # Hide confidence, these two ommited for better visualization
}
YoloInference(**args)


#%%
from NomDataset import NomDataset_Yolo

dataset = NomDataset_Yolo(TOK1902_RAW, annotations, TOK1902_RAW_ANNOTATIONS, unicode_dict_path='NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst-label-set-ucode.pkl')
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

#%%
from ESRGAN import RRDBNet_arch as ESRGAN_arch
esrgan_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'

esrgan_model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
esrgan_model.load_state_dict(torch.load(esrgan_path), strict=True)
esrgan_model.eval()
esrgan_model = esrgan_model.to(DEVICE)

#%%
from torchsummary import summary
from NomResnet import PytorchResNet101
import pickle
data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
with open(data_path_label, 'rb') as f:
    unicode_labels = pickle.load(f)
    unicode_labels = sorted(list(unicode_labels.keys()))
print("Total number of unicode: ", len(unicode_labels))

#%%
resnet_weights = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
resnet_model = PytorchResNet101.load_from_checkpoint(resnet_weights, num_labels=len(unicode_labels))
resnet_model.eval()
resnet_model.freeze()
resnet_model.to(DEVICE)

#%%
# Test the pipeline with a single batch
# batch_img, batch_label, batch_ulabel = next(iter(dataloader))
# with torch.no_grad():
#     batch_img = batch_img.to(DEVICE)
#     esrgan_result = esrgan_model(batch_img)
#     print('ESRGAN result shape: ', esrgan_result.shape)
#     resnet_result = resnet_model(esrgan_result).cpu()
#     pred = softmax(resnet_result)
#     pred = torch.argmax(pred, dim=1)

# results = []
# u_results = []
# for idx, p in enumerate(pred, start=0):
#     p = unicode_labels[p.item()]
#     u_results.append(p)
#     if p != 'UNK':
#         p = "0x" + p
#         p = int(p, 16)
#         p = chr(p)
#     results.append(p)
    
# print("Predicted label unicode: ", u_results)
# print("Predicted label: ", results)

# print("Ground truth label: ", batch_ulabel)
# print("Ground truth label unicode: ", batch_label)
#%%
torch.cuda.empty_cache()


#%%
# Test the pipeline with the whole dataset
precision_table = np.zeros((len(unicode_labels)))
total_correct = 0
total_samples = 0
for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    # batch_img, batch_ulabel, batch_label = batch
    batch_img, batch_ulabel = batch
    with torch.no_grad():
        batch_img = batch_img.to(DEVICE)
        
        esrgan_result = esrgan_model(batch_img)
        # resnet_result = resnet_model(esrgan_result).cpu()

        resnet_result = resnet_model(torch.nn.functional.interpolate(esrgan_result, size=(64, 64), mode='bilinear', align_corners=False)).cpu()
        
        # resnet_result = resnet_model(batch_img).cpu()
        

        
        pred = softmax(resnet_result)
        pred = torch.argmax(pred, dim=1)
        
        precision_table[pred] += 1
        
        pred = [unicode_labels[p.item()] for p in pred]
        
        # Calculate accuracy
        total_samples += len(batch_ulabel)
        for p, gt in zip(pred, batch_ulabel):
            if p == gt:
                total_correct += 1
        
# TODO: Fix dataset label from unicode to unicode_labels index
        

#%%
# print(precision_table)
print("Total correct: ", total_correct)
print("Total samples: ", total_samples)
print("Accuracy: ", total_correct / total_samples)
# Get model precision


#%%
# import pickle
# data_path_label = 'C:/Users/Soppo/Documents/GitHub/Thesis Thu/SOURCE-20230605T100152Z-001/SOURCE/Data/transformed-data/for-AlexNet-and-ResNet101/HWDB1.1-bitmap64-ucode-hannom-v2-labelset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
# with open(data_path_label, 'rb') as f:
#     unicode_labels = pickle.load(f)
# for i, (k, v) in enumerate(unicode_labels.items()):
#     unicode_labels[k] = i
# print(unicode_labels)
# print(unicode_labels['899A'])

#%%
