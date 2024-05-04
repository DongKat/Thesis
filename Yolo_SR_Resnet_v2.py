# This python notebook is for testing ResNet checkpoints
# on different type of data.
# E.g: Yolo detection, Raw_crops, SR_crops, etc.

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
import pickle


import torchvision
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchsummary import summary


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

UCODE_DICT_PATH = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl' 

# Images path for yolo. I test on 1902 first
YOLO_WEIGHTS = 'yolov5/weights/yolo_one_label.pt'
TOK1902_RAW = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-raw-images'
TOK1902_RAW_ANNOTATIONS = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-annotation/annotation-mynom'
TOK1902_CROP = 'TempResources/ToK1902/Tok1902_crops'
TOK1902_CROP_ANNOTATIONS = 'TempResources/ToK1902/ToK1902_crops.txt'
TOK1902_REAL_ESRGAN_CROP = 'TempResources/ToK1902/ToK1902_SR_crop/Real_ESRGAN/RealESRGAN_x4plus'
TOK1902_REAL_ESRGAN_NOM_CROP = 'TempResources/ToK1902/ToK1902_SR_crop/Real_ESRGAN/RealESGRANx2plus_RealCE_1K'
TOK1902_REAL_ESRGAN_GP = 'TempResources/ToK1902/ToK1902_SR_crop/Real_ESRGAN/RealESRNETx2plus_ToK1871_GPloss_20epoch'
TOK1902_REAL_ESRGAN_GP_USM = 'TempResources/ToK1902/ToK1902_SR_crop/Real_ESRGAN/RealESRNETx2plus_ToK1871_GPloss_20epoch_USM'

TOK1871_CROP_ANNOTATIONS = 'TempResources/ToK1871/ToK1871_crops.txt'
TOK1871_REAL_ESRGAN_CROP = 'TempResources/ToK1871/SR_ForResNet/Real-ESRGAN_224'
TOK1871_REAL_ESRGAN_RealCE_CROP = 'TempResources/ToK1902/ToK1902_SR_crop/Real_ESRGAN/RealESGRANx2plus_RealCE_1K'
TOK1871_ESRGAN_CROP = 'TempResources/ToK1871/SR_ForResNet/ESRGAN_224'
TOK1871_ESRGAN_NOM_CROP = 'TempResources/ToK1871/SR_ForResNet/ESRGAN_224_Nom'


LVT_RAW = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-raw-images'
LVT_RAW_ANNOTATIONS = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-annotation/annotation-mynom'
LVT_CROP = 'TempResources/LVT/LVT_crops'
LVT_CROP_ANNOTATIONS = 'TempResources/LVT/LVT_crop.txt'
LVT_REAL_ESRGAN_RealCE_CROP = 'TempResources/LVT/LVT_SR_crop/Real_ESRGAN/RealESGRANx2plus_RealCE_1K'
LVT_REAL_ESRGAN_GP = 'TempResources/LVT/LVT_SR_crop/Real_ESRGAN/RealESRNETx2plus_ToK1871_GPloss_20epoch'
LVT_REAL_ESRGAN_GP_USM = 'TempResources/LVT/LVT_SR_crop/Real_ESRGAN/RealESRNETx2plus_ToK1871_GPloss_20epoch_USM'
#%%
# Call yolo detect.py
# TODO: This works, in acceptable condition

project = 'TempResources/yolov5_runs_ToK1902/detect'
name = 'yolo_GT'
YOLO_RESULTS = os.path.join(project, name)
annotations = YOLO_RESULTS + '/labels' # Yolo will save the labels here

# # #%%
# from yolov5.detect import run as YoloInference
# args = {
#     'weights': YOLO_WEIGHTS,
#     'source': TOK1902_RAW,
#     'imgsz': (640, 640),
#     'conf_thres': 0.5,
#     'iou_thres': 0.5,   
#     'device': '',       # Let YOLO decide
#     'save_txt': True,
#     'save_crop': True,  # Save cropped prediction boxes, for debugging
#     'project': project,
#     'name': name,
#     'exist_ok': True,
#     'hide_labels': True,    # Hide labels from output images
#     'hide_conf': True,      # Hide confidence, these two ommited for better visualization
# }
# YoloInference(**args)


#%%
from NomDataset import NomDataset_Yolo
from SR_NomDataset import NomDatasetCrop

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# dataset_yolo = NomDataset_Yolo(TOK1902_RAW, annotations, TOK1902_RAW_ANNOTATIONS, 
#                                image_size=(224, 224),
#                                unicode_dict_path=UCODE_DICT_PATH,
#                                transform=preprocess)


# dataset_crop = NomDatasetCrop(TOK1871_REAL_ESRGAN_CROP, TOK1871_CROP_ANNOTATIONS,
#                                 input_size=(224, 224),
#                                 ucode_dict_path=UCODE_DICT_PATH,
#                                 transforms=preprocess)

dataset_crop = NomDatasetCrop(LVT_REAL_ESRGAN_GP_USM, LVT_CROP_ANNOTATIONS,
                              input_size=(224, 224),
                              ucode_dict_path=UCODE_DICT_PATH,
                              transforms=preprocess)

# dataloader_yolo = DataLoader(dataset=dataset_yolo, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
dataloader_crop = DataLoader(dataset=dataset_crop, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
with open(data_path_label, 'rb') as f:
    unicode_labels = pickle.load(f)
    unicode_labels = sorted(list(unicode_labels.keys()))
print("Total number of unicode: ", len(unicode_labels))

#%%
# Sampling the dataset batch of 16
# batch = next(iter(dataloader_yolo))
# imgs, labels = batch

# plt.figure()
# for idx, i in enumerate(imgs, 1):
#     if idx == 17:
#         break
#     img = i.permute(1, 2, 0).numpy()
#     img = img * 255
#     img = img.clip(0, 255).astype('uint8')
#     plt.subplot(4, 4, idx)
#     plt.imshow(img)
# plt.show()    
# print("Labels:", end=' ')
# for j in labels:
#     if j == 'UNK':
#         print(j, end=', ')
#     else:
#         print(chr(int(j, 16)), end=', ')
        
batch = next(iter(dataloader_crop))
imgs, labels = batch
plt.figure()
for idx, i in enumerate(imgs, 1):
    if idx == 17:
        break
    img = i.permute(1, 2, 0).numpy()
    img = img * 255
    img = img.clip(0, 255).astype('uint8')
    plt.subplot(4, 4, idx)
    plt.imshow(img)
plt.show()

labels = labels.tolist()
print("Labels:", [unicode_labels[i] for i in labels])
print("Labels:", end=' ')
for idx, i in enumerate(labels):
    if unicode_labels[i] == 'UNK':
        print("UNK", end=', ')
    else:
        print(chr(int(unicode_labels[i], 16)), end=', ')

#%%

# # randint = np.random.randint(0, len(dataset_yolo))
# randint = 3
# img, label = dataset_yolo[randint]
# img = img.permute(1, 2, 0).numpy()
# # Just the mean is enough to grasp the value range of image
# print("Image mean: ", img.mean())
# plt.imshow(img)
# plt.show()
# if label == 'UNK':
#     print("Label: ", label)
# else:
#     print("Label: ", chr(int(label, 16)))


randint = np.random.randint(0, len(dataset_crop))
# randint = 4
img, label = dataset_crop[randint]
img = img.permute(1, 2, 0).numpy()
print("Image mean: ", img.mean())
plt.imshow(img)
plt.show()
print("Label: ", unicode_labels[label])
if unicode_labels[label] == 'UNK':
    print("Label: ", unicode_labels[label])
else:
    print("Label: ", chr(int(unicode_labels[label], 16)))


# #%%
# from ESRGAN import RRDBNet_arch as ESRGAN_arch
# esrgan_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'

# esrgan_model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
# esrgan_model.load_state_dict(torch.load(esrgan_path), strict=True)
# esrgan_model.eval()
# esrgan_model = esrgan_model.to(DEVICE)


#%%
from NomResnet import PytorchResNet101
resnet_weights = 'Backup/PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039_old.ckpt'
resnet_model = PytorchResNet101.load_from_checkpoint(resnet_weights, num_labels=len(unicode_labels))
resnet_model.eval()
resnet_model.freeze()
resnet_model.to(DEVICE)

#%%
torch.cuda.empty_cache()


#%%
# Test the Resnet with the test dataset NomDatatsetYolo
# dataloader = dataloader_yolo

# losses = []
# for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#     # batch_img, batch_ulabel, batch_label = batch
#     batch_img, batch_label = batch
#     batch_label = list(batch_label)
#     for i in range(len(batch_label)):
#         try:
#             batch_label[i] = unicode_labels.index(batch_label[i])
#         except:
#             batch_label[i] = unicode_labels.index('UNK')
#     batch_label = torch.tensor(batch_label, dtype=torch.long)
#     with torch.no_grad():
#         batch_img = batch_img.to(DEVICE)   
#         batch_label = batch_label.to(DEVICE)     
        
#         resnet_result = resnet_model(batch_img)
        
#         loss = resnet_model.criterion(resnet_result, batch_label)
#         losses.append(loss.item())
        
#         pred = softmax(resnet_result, dim=1)
#         pred = torch.argmax(pred, dim=1)
                
#         acc = resnet_model.metrics(pred, batch_label)
    
        
    
# losses = np.array(losses) / len(dataloader)
# print("Average loss: ", losses.mean())
# print("Average accuracy: ", resnet_model.metrics.compute())

#%%
# Test the Resnet with the test dataset NomDatasetCrop
dataloader = dataloader_crop

losses = []
for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    # batch_img, batch_ulabel, batch_label = batch
    batch_img, batch_label = batch
    with torch.no_grad():
        batch_img = batch_img.to(DEVICE)   
        batch_label = batch_label.to(DEVICE)     
        
        resnet_result = resnet_model(batch_img)
        
        loss = resnet_model.criterion(resnet_result, batch_label)
        losses.append(loss.item())
        
        pred = softmax(resnet_result, dim=1)
        pred = torch.argmax(pred, dim=1)
                
        resnet_model.metrics(pred, batch_label)
        
losses = np.array(losses) / len(dataloader)
print("Average loss: ", losses.mean())
print("Average accuracy: ", resnet_model.metrics.compute())