#%%
import cv2
import numpy as np
import os

# For reading excel files, require openpyxl
import pandas as pd

import pickle
import torch
import torchvision
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import pybboxes as pybbx
from torchmetrics import Accuracy, Precision, Recall, F1Score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Images path for yolo. I test on 1902 first
YOLO_WEIGHTS = 'yolov5\weights\yolo_one_label.pt'
TOK1902_RAW = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-raw-images'
TOK1902_RAW_ANNOTATIONS = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1902/1902-annotation/annotation-mynom'

BATCH_SIZE = 32

# Commented out because it is not used

# #%%
# # Run yolo model on image
# img = 'test.jpg'
# img = cv2.imread(img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # Preprocess the image for yolo, i.e. resize to a multiple of stride s=32 by padding
# tmp_img = letterbox(img)[0]
# h0, w0, _ = tmp_img.shape
# # cv2 to torch tensor
# tmp_img = torch.from_numpy(tmp_img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

# #%%
# with torch.no_grad():
#     yolo_result = yolo_model(tmp_img)
#     yolo_result = non_max_suppression(yolo_result, conf_thres=0.5, iou_thres=0.5)
# print(yolo_result[0])

# # %%
# # Scale bounding boxes to original image size
# h1, w1, _ = img.shape
# print("Original image shape: ", h1, w1)
# print("Yolo Letterbox shape: ", h0, w0)
# count = 0
# for row in yolo_result[0]:
#     # Draw bounding boxes
#     box = scale_boxes([h0, w0], row[:4], [h1, w1]).round()
#     x1, y1, x2, y2 = map(int, box)
#     print(x1, y1, x2, y2)
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#     count += 1
# print("Number of objects detected: ", count)
# cv2.imwrite('./test_yolo.jpg', img)

# #%%
# # Evaluate the yolo model
# torch.cuda.empty_cache()



# #%%
# # Run SR with ESRGAN on image
# from ESRGAN import RRDBNet_arch as ESRGAN_arch

# model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'

# sr_model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
# sr_model.load_state_dict(torch.load(model_path), strict=True)
# sr_model.eval()
# sr_model = sr_model.to(DEVICE)

# #%% 
# print(yolo_result[0])

# # %%
# # TODO: Run SR on multiple regions of interest and pass to ResNet
# h1, w1, _ = img.shape
# print("Original image shape: ", h1, w1)
# print("Yolo Letterbox shape: ", h0, w0)
# for row in yolo_result[0]:
#     # Draw bounding boxes
#     x1, y1, x2, y2 = map(int, row[:4])
#     print(x1, y1, x2, y2)
#     # Crop the region of interest
#     roi = img[int(y1):int(y2), int(x1):int(x2)]
#     roi = torch.from_numpy(roi).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)
#     # # Resize the roi to 4x
#     with torch.no_grad():
#         sr_result = sr_model(roi).data.squeeze(0).float().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
#         sr_result = (sr_result * 255).round().astype(np.uint8)
    
#     # Save the sr result
#     cv2.imwrite(f'./test_yolo_sr_{count}.jpg', sr_result)
    
# # %%
# # Load the ResNet model
# from NomResnet import PytorchResNet101
# data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
# with open(data_path_label, 'rb') as f:
#     unicode_labels = pickle.load(f)
#     unicode_labels = sorted(list(unicode_labels.keys()))
# print("Total number of unicode: ", len(unicode_labels))
# print(unicode_labels)

# weights_path = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
# resnet_model = PytorchResNet101.load_from_checkpoint(weights_path, num_labels=len(unicode_labels))
# resnet_model.eval()
# resnet_model.freeze()
# resnet_model.to(DEVICE)

# # %%
# with torch.no_grad():
#     tmp_img = torch.from_numpy(sr_result).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
#     resnet_result = resnet_model(tmp_img)
#     pred = softmax(resnet_result)
#     pred = torch.argmax(pred, dim=1)
#     pred = unicode_labels[pred.item()]
#     pred = "0x" + pred
    
#     print("Predicted label unicode: ", pred)
    
#     pred = int(pred, 16)
#     pred = chr(pred)

#     print("Predicted label: ", pred)
    
    
    
    
    
    
    

# # %%
# from ESRGAN import RRDBNet_arch as ESRGAN_arch
# from NomResnet import PytorchResNet101

# yolo_weights = 'yolov5\weights\yolo_one_label.pt'
# yolo_model = torch.hub.load(repo_or_dir='./yolov5', model='custom', path=yolo_weights, source='local')
# yolo_model.eval()

# esrgan_weights = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
# esrgan_model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
# esrgan_model.load_state_dict(torch.load(esrgan_weights), strict=True)
# esrgan_model.eval()

# data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
# with open(data_path_label, 'rb') as f:
#     unicode_labels = pickle.load(f)
#     unicode_labels = sorted(list(unicode_labels.keys()))
# print("Total number of unicode: ", len(unicode_labels))
# print(unicode_labels)

# resnet_weights = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
# resnet_model = PytorchResNet101.load_from_checkpoint(resnet_weights, num_labels=len(unicode_labels))
# resnet_model.eval()

# #%%
# test_image_file_path = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images'
# test_annotation_file_path = 'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-annotation/annotation-mynom'



# class NomDataset(torch.utils.data.Dataset):
#     """
#         Dataloader for loading images for Yolo - SR - ResNet pipeline
#     """

#     def __init__(self, image_file_path, annotation_file_path, transform=None) -> None:
#         self.image_file_path = image_file_path
#         self.annotation_file_path = annotation_file_path
#         self.image_files = []
#         self.annotation_files = []
#         self.transform = transform
#         self.load_files_list()
        
#     def load_files_list(self) -> None:
#         for file in os.listdir(self.image_file_path):
#             if file.endswith('.jpg'):
#                 self.image_files.append(file)
#         for file in os.listdir(self.annotation_file_path):
#             if file.endswith('.xlsx'):
#                 self.annotation_files.append(file)
#         assert len(self.image_files) == len(self.annotation_files), "Number of image files and annotation files do not match"
    
#     def __getitem__(self, index: int):
#         assert index <= len(self), "Index out of range"
        
#         image = cv2.imread(os.path.join(self.image_file_path, self.image_files[index]))
#         annotations_file = pd.read_excel(os.path.join(self.annotation_file_path, self.annotation_files[index]))
        
#         annotations = {"boxes": [], "labels": []}
#         for _, row in annotations_file.iterrows():
#             x1, y1, x2, y2 = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
#             unicode_label = row['UNICODE']
#             chinese_char_label = row['CHAR']
            
#             annotations['boxes'].append((x1, y1, x2, y2))
#             annotations['labels'].append(unicode_label)
        
#         return image, annotations
    
#     def __len__(self) -> int:
#         return len(self.image_files)
    
# nomDataset = NomDataset(test_image_file_path, test_annotation_file_path)
# nomDataloader = torch.utils.data.DataLoader(dataset=nomDataset, batch_size=8, num_workers=0, shuffle=False)

# from yolov5.utils.augmentations import letterbox
# from yolov5.utils.general import non_max_suppression, scale_boxes

# # TODO: Passing parameters to the pipeline
# class Pipeline:
#     def __init__(self, yolo_model, esrgan_model, resnet_model) -> None:
#         self.yolo_model = yolo_model
#         self.esrgan_model = esrgan_model
#         self.resnet_model = resnet_model
    
#     # Run by batch
#     def run(self, batch: list):
#         yolo_predictions = yolo_model(batch)
#         print(type(yolo_predictions))

        
# pipeline = Pipeline(yolo_model, esrgan_model, resnet_model)

#%%

# Call yolo detect.py
# TODO: This works, in acceptable condition
from yolov5.detect import run as YoloInference

project = 'runs/detect'
name = 'yolo_SR'
YOLO_RESULTS = os.path.join(project, name)
annotations = YOLO_RESULTS + '/labels' # Yolo will save the labels here

# args = {
#     'weights': YOLO_WEIGHTS,
#     'source': TOK1902_RAW,
#     'imgsz': (640, 640),
#     'conf_thres': 0.5,
#     'iou_thres': 0.5,   
#     'device': '',       # Let YOLO decide
#     'save_txt': True,
#     'save_crop': True,  # Save cropped prediction boxes, for debugging
#     'project': 'yolov5_runs/detect',
#     'name': 'yolo_SR',
#     'exist_ok': True,
#     'hide_labels': True,    # Hide labels from output images
#     'hide_conf': True,      # Hide confidence, these two ommited for better visualization
# }
# YoloInference(**args)


#%%

def find_best_IOU(ref_box, boxes) -> float | tuple | int:
    def calculate_IOU(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        x5, y5 = max(x1, x3), max(y1, y3)
        x6, y6 = min(x2, x4), min(y2, y4)
        intersection = max(0, x6 - x5) * max(0, y6 - y5)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        return intersection / union
    
    best_iou = 0
    best_box = None
    best_index = -1
    for index, box in enumerate(boxes, 0):
        iou = calculate_IOU(ref_box, box)
        if iou > best_iou:
            best_iou = iou
            best_box = box
            best_index = index
    return best_iou, best_box, best_index

#%%
# Load yolo results with data loader
class NomDataset_Yolo(torch.utils.data.Dataset):
    def __init__(self, image_file_path, annotation_file_path, label_file_path, transform=None) -> None:
        self.image_file_path = image_file_path
        self.annotation_file_path = annotation_file_path
        self.label_file_path = label_file_path
        
        self.image_files = []
        self.annotation_files = []
        self.label_files = []
        
        self.transform = transform
        self.load_files_list()
        
        # 
        self.crop_dict = {'crops': [], 'original_images_name': [], 'labels': [], 'unicode_labels': []}
        self.load_crops()
    
    def load_files_list(self) -> None:
        for file in os.listdir(self.image_file_path):
            if file.endswith('.jpg'):
                self.image_files.append(file)
        for file in os.listdir(self.annotation_file_path):
            if file.endswith('.txt'):
                self.annotation_files.append(file)
        assert len(self.image_files) == len(self.annotation_files), "Number of image files and annotation files do not match"
        
        for file in os.listdir(self.label_file_path):
            if file.endswith('.xlsx'):
                self.label_files.append(file)
        assert len(self.image_files) == len(self.label_files), "Number of image files and label files do not match"

    def load_crops(self) -> None:
        # For reading yolo txt files
        total_n = len(self.image_files)
        for image_file, txt_file, excel_file in tqdm(zip(self.image_files, self.annotation_files, self.label_files), total=total_n):
            image = cv2.cvtColor(cv2.imread(os.path.join(self.image_file_path, image_file)), cv2.COLOR_BGR2RGB)    # Grayscale, so I can stack 3 channels later
            h, w, _ = image.shape
            df = pd.read_excel(os.path.join(self.label_file_path, excel_file))
            # tmp_img = image.copy()

            label_dict = {'boxes': [], 'labels': [], 'unicode_labels': []}
            for _, row in df.iterrows():
                x1, y1, x2, y2 = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
                label = row['UNICODE']
                ucode_label = row['CHAR']
                label_dict['boxes'].append((x1, y1, x2, y2))
                label_dict['labels'].append(label)
                label_dict['unicode_labels'].append(ucode_label)
                # tmp_img = cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
            
            
                
            with open(os.path.join(self.annotation_file_path, txt_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, x, y, b_w, b_h = map(float, line.split(' '))
                    bbox = pybbx.YoloBoundingBox(x, y, b_w, b_h, image_size=(w, h)).to_voc(return_values=True)
                    x1, y1, x2, y2 = bbox
                    
                    # Find the best IOU to label the cropped image
                    iou, box, idx = find_best_IOU(bbox, label_dict['boxes'])
                    
                    self.crop_dict['original_images_name'].append(image_file)
                    self.crop_dict['labels'].append(label_dict['labels'][idx])
                    self.crop_dict['unicode_labels'].append(label_dict['unicode_labels'][idx])
                        

                    crop_img = image[int(y1):int(y2), int(x1):int(x2)]
                    self.crop_dict['crops'].append(crop_img)
                    
                    # print("Label: ", label_dict['labels'][idx], "Unicode: ", label_dict['unicode_labels'][idx])
                    # plt.imshow(crop_img, cmap='gray')
                    # plt.show()
                    
                    # tmp_img = cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.imwrite(f'./bbox_yolo_and_gt_{image_file}', tmp_img)            
            
            # print("Image: ", image_file)
            # print(len(label_dict['boxes']), len(self.crop_dict['crops']))
            # print(len(label_dict['labels']), len(self.crop_dict['labels']))
    
        # For reading raw annotation excel files
        # for image_file, excel_file in zip(self.image_files, self.annotation_files):
        #     image = cv2.cvtColor(cv2.imread(os.path.join(self.image_file_path, image_file)), cv2.COLOR_BGR2RGB)
        #     df = pd.read_excel(os.path.join(self.annotation_file_path, excel_file))
            
        #     for _, row in df.iterrows():
        #         x_tl, y_tl, x_br, y_br = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
        #         label = row['UNICODE']
        #         ucode_label = row['CHAR']
                
        #         crop_img = image[int(y_tl):int(y_br), int(x_tl):int(x_br)]
                
        #         self.crop_dict['crops'].append(crop_img)
        #         self.crop_dict['labels'].append(label)
        #         self.crop_dict['unicode_labels'].append(ucode_label)   
    
            
    def __len__(self) -> int:
        return len(self.crop_dict['crops'])
        
    def __getitem__(self, index: int) -> torch.Tensor | str:
        assert index <= len(self), "Index out of range"
                
        image = self.crop_dict['crops'][index]
        label = self.crop_dict['labels'][index]
        ucode_label = self.crop_dict['unicode_labels'][index]

        
        if self.transform:
            image = self.transform(image)
        else:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            image = image *  1.0 / 255
            
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            image = (image - mean) / std
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label, ucode_label


dataset = NomDataset_Yolo(TOK1902_RAW, annotations, TOK1902_RAW_ANNOTATIONS)
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
print(unicode_labels)

#%%
resnet_weights = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
resnet_model = PytorchResNet101.load_from_checkpoint(resnet_weights, num_labels=len(unicode_labels))
resnet_model.eval()
resnet_model.freeze()
resnet_model.to(DEVICE)

#%%
# Test the pipeline with a single batch
batch_img, batch_label, batch_ulabel = next(iter(dataloader))
with torch.no_grad():
    batch_img = batch_img.to(DEVICE)
    esrgan_result = esrgan_model(batch_img)
    print('ESRGAN result shape: ', esrgan_result.shape)
    resnet_result = resnet_model(esrgan_result).cpu()
    pred = softmax(resnet_result)
    pred = torch.argmax(pred, dim=1)

results = []
u_results = []
for idx, p in enumerate(pred, start=0):
    p = unicode_labels[p.item()]
    u_results.append(p)
    if p != 'UNK':
        p = "0x" + p
        p = int(p, 16)
        p = chr(p)
    results.append(p)
    
print("Predicted label unicode: ", u_results)
print("Predicted label: ", results)

print("Ground truth label: ", batch_ulabel)
print("Ground truth label unicode: ", batch_label)


#%%

# Test the pipeline with the whole dataset
precision_table = np.zeros((len(unicode_labels)))
total_correct = 0
total_samples = 0
for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    batch_img, batch_ulabel, batch_label = batch
    with torch.no_grad():
        batch_img = batch_img.to(DEVICE)
        
        esrgan_result = esrgan_model(batch_img)
        resnet_result = resnet_model(esrgan_result).cpu()
        
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
print(precision_table)
print("Total correct: ", total_correct)
print("Total samples: ", total_samples)
print("Accuracy: ", total_correct / total_samples)

