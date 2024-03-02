## This notebook for Yolo -> SR -> ResNet
## And I'm trying to use Python Interactive Window in VSCode instead of Jupyter
#%%
from NomDataset import NomDatasetV1, NomDatasetV2

# Standard libraries imports
import importlib
import os
import glob
import random
import shutil
from tqdm.autonotebook import tqdm
import pybboxes as pbx


# Data manipulation libraries imports (for image processing)
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Torch libraries imports
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR


# Pytorch Lightning libraries imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Torch utilities libraries imports
import torchmetrics
from torchsummary import summary
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import resnet101
from torchvision import transforms

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
ToK1871_annotation = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-annotation/annotation-mynom/'
ToK1871_image = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images/'

ToK1902_annotation = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-annotation/annotation-mynom/'
Tok1902_image = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-raw-images/'

LVT_annotation = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-annotation/annotation-mynom/'
LVT_image = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-raw-images/'

ToK1871_Dataset = NomDatasetV2(root_annotation=ToK1871_annotation, root_image=ToK1871_image, scale=4, input_dim=(256, 256))
ToK1902_Dataset = NomDatasetV2(root_annotation=ToK1902_annotation, root_image=Tok1902_image, scale=4, input_dim=(256, 256))
LVT_Dataset = NomDatasetV2(root_annotation=LVT_annotation, root_image=LVT_image, scale=4, input_dim=(256, 256))

NomDataloader = DataLoader(ToK1871_Dataset, batch_size=1, shuffle=True, num_workers=0)

#%%
# Test the dataloader
# This cell is meant to demonstrate the output of the NomDataLoader
print("A batch sample of NomDataset:")

for i, batch in enumerate(NomDataloader):
    img_path, img_hr, img_lr, coords_hr, coords_lr, labels_char, labels_unicode_cn, labels_unicode_vn = batch
    break   # To get first batch only
print("Image path: ", img_path)
print("Char labels: ", labels_char)
print("Unicode CN labels: ", labels_unicode_cn)
print("Unicode VN labels: ", labels_unicode_vn)

img_hr = img_hr.squeeze(0).numpy()
img_lr = img_lr.squeeze(0).numpy()


for coord_hr in coords_hr:
    x1, y1, x2, y2 = map(int, coord_hr)
    img_hr = cv2.rectangle(img_hr, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
for coord_lr in coords_lr:
    x1, y1, x2, y2 = map(int, coord_lr)
    img_lr = cv2.rectangle(img_lr, (x1, y1), (x2, y2), (0, 255, 0), 1)    

plt.subplot(1, 2, 1)
plt.imshow(img_hr)
plt.title("HR/Original image")
plt.subplot(1, 2, 2)
plt.imshow(img_lr)
plt.title("x4 bicubic LR image")
plt.show()


#%% Run yolo on raw images
!python yolov5/detect.py --weights yolov5/weights/yolo_one_label.pt --imgsz 640 --conf 0.25 --source NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images --save-txt 

#%% Yolo box crop
yolo_box_path = 'yolov5/runs/detect/exp6/labels/'
img_file_path = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images/'
crop_path = 'TempResources/YoloBoxCrops'

for yolo_box, img_name in zip(os.listdir(yolo_box_path), os.listdir(img_file_path)):
    print(yolo_box, img_name)
    # Check same name
    assert yolo_box.split('.')[0] == img_name.split('.')[0]
    
    img = cv2.imread(os.path.join(img_file_path, img_name))
    # IT'S (H, W, C) in CV2; (C, H, W) in Tensor
    h, w, _ = img.shape
    
    with open(os.path.join(yolo_box_path, yolo_box), 'r') as f:
        for i, line in enumerate(f.readlines()):
            bb_list = line.split()
            _, x_center, y_center, bb_width, bb_height = map(float, bb_list)
            x1, y1, x2, y2 = pbx.convert_bbox((x_center, y_center, bb_width, bb_height), from_type='yolo', to_type='voc', image_size=(w, h))
            
            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(os.path.join(crop_path, f"{img_name.split('.')[0]}_{i}.jpg"), crop)
    
    # plt.imshow(img)

#%% SR on Yolo box crops
from ESRGAN import RRDBNet_arch as ESRGAN_arch

model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
yolo_box_crops = 'TempResources/YoloBoxCrops'
yolo_box_crops_SR = 'TempResources/SR_from_YoloBoxCrops'

model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(DEVICE)

for i, box_image in tqdm(enumerate(os.listdir(yolo_box_crops))):
    img = cv2.imread(os.path.join(yolo_box_crops, box_image))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    
    with torch.no_grad():
        sr_img = model(img).data.squeeze(0).float().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
        sr_img = (sr_img * 255).round().astype(np.uint8)
        
        cv2.imwrite(os.path.join(yolo_box_crops_SR, f"{box_image.split('.')[0]}_SR.jpg"), sr_img) 
        
#%% ResNet on SR images
from NomResnet import PytorchResNet101
data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
with open(data_path_label, 'rb') as f:
    unicode_labels = pickle.load(f)
    unicode_labels = sorted(list(unicode_labels.keys()))
print("Total number of unicode: ", len(unicode_labels))
print(unicode_labels)

weights_path = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
model = PytorchResNet101.load_from_checkpoint(weights_path, num_labels=len(unicode_labels))
model.eval()
model.freeze()
model.to(DEVICE)

yolo_box_crops_SR = 'TempResources/SR_from_YoloBoxCrops'

file_list = os.listdir(yolo_box_crops_SR)
random.shuffle(file_list)

i = 0

for i, image_file in enumerate(file_list):
    image_path = os.path.join(yolo_box_crops_SR, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    pred = model(image)
    pred = softmax(pred)
    pred = torch.argmax(pred, dim=1)
    pred = unicode_labels[pred.item()]
    pred = "0x" + pred
    
    print("Predicted label unicode: ", pred)
    
    pred = int(pred, 16)
    pred = chr(pred)

    print("Predicted label: ", pred)

    plt.imshow(image.squeeze(0).cpu().numpy().transpose(1, 2, 0))
    plt.show()
    i+=1
    if i == 11:
        break