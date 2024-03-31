#%%
import torch
import torchvision
from torch import Tensor
import os
import cv2

yolo_model = torch.hub.load(repo_or_dir='./yolov5', model='custom', path='yolov5\weights\yolo_one_label.pt', source='local')

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes

#%%
# Define the batch size
batch_size = 32

image_file_path = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images'
image_list = os.listdir(image_file_path)

batch_list = []

for i in range(batch_size):
    img = cv2.imread(os.path.join(image_file_path, image_list[i]))
    batch_list.append(img)
    
#%%
print(len(batch_list))

#%%
result = yolo_model(batch_list, size=640)
print(type(result))
#%%
print(len(result.xyxy))
print(result.xyxy[0].shape)
print(result.xyxy[1].shape)
print(result.xyxy[2].shape)
#%%
print(result.pandas().xyxy[0].round())
