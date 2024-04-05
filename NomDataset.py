#%%
import cv2
import os
import pandas as pd
import pybboxes as pybbx
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import pytorch_lightning as pl
from matplotlib import pyplot as plt

class NomDatasetV1(Dataset):
    """
        This dataset is used for cropping the images and saving the cropped images, with labels
        
        root_annotation: the path to the folder containing the annotation files
        root_image: the path to the folder containing the images
        scale: the scale of the images
        
        The dataset will return:
            - img_hr: the high resolution cropped image
            - img_lr: the low resolution cropped image
            - label_char: the label of the image in char
            - label_unicode_cn: the label of the image in unicode_cn
            - label_unicode_vn: the label of the image in unicode_vn
    """
    def __init__(self, root_annotation: str, root_image: str, scale: int):
        super().__init__()
        
        if root_annotation is None:
            raise ValueError("root_annotation is None")
        if root_image is None:
            raise ValueError("root_image is None")
        if scale is None:
            raise ValueError("scale is None")
        
        self.scale = scale
        self.total_images = 0
        self.image_paths =          []
        self.labels_unicode_cn =    []
        self.labels_char =          []
        self.labels_unicode_vn =    []
        
        for annotation_excel in tqdm(os.listdir(root_annotation)):
            
            # This dictionary will store the information for each annotation file, which annotates a page
            page_dict = {
                "crop_coords": [],
                "unicode_cn": [],
                "char": [],
                "unicode_vn": []
            }
            if annotation_excel.endswith(".xlsx") or annotation_excel.endswith(".xls"):
                # print("Reading file: " + os.path.join(root_annotation, annotation_excel))
                df = pd.read_excel(os.path.join(
                    root_annotation, annotation_excel))
                
                for index, row in df.iterrows():
                    # Extract the image name and the label
                    page_dict["crop_coords"].append(
                        (row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']))
                    page_dict["unicode_cn"].append(row['UNICODE'])
                    page_dict["char"].append(row['CHAR'])
                    page_dict["unicode_vn"].append(row['UNICODE_QN'])
            
            # Cropped the image
            # The image name is the same as the annotation file name
            image_name = annotation_excel.split(".")[0] + ".jpg"
            image_path = os.path.join(root_image, image_name)
            image = cv2.imread(image_path)
            for i in range(len(page_dict["crop_coords"])):
                crop_coords = page_dict["crop_coords"][i]
                cropped_image = image[crop_coords[1]:crop_coords[3],
                                      crop_coords[0]:crop_coords[2]]
                
                # Save the cropped image
                cropped_image_name = f"{image_name.split('.')[0]}_cropped_{i:03d}.jpg"
                cropped_directory = "NomDataset/datasets/cropped_images/"
                cropped_image_path = os.path.join(cropped_directory, cropped_image_name)
                # print(cropped_image_path)
                
                cv2.imwrite(cropped_image_path, cropped_image)
                
                # Save the cropped image path
                self.image_paths.append(cropped_image_path)
                # Save the label
                self.labels_char.append(page_dict["char"][i])
                self.labels_unicode_cn.append(page_dict["unicode_cn"][i])
                self.labels_unicode_vn.append(page_dict["unicode_vn"][i])
                
                self.total_images += 1
                
                
    def __len__(self):
        return self.total_images
    
    def resize_to_divisible_by_scale(self, img, scale=4):
        """
        Resize the image to be divisible by scale
        """
        h, w, _ = img.shape
        h = h - h % scale
        w = w - w % scale
        img = img[:h, :w, :]
        return img
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        image_path = self.image_paths[index]
        print(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        img_hr = self.resize_to_divisible_by_scale(img, scale=self.scale)
        img_lr = cv2.resize(img_hr, (img_hr.shape[1] // self.scale, img_hr.shape[0] // self.scale), interpolation=cv2.INTER_CUBIC)
        label_char = self.labels_char[index]
        label_unicode_cn = self.labels_unicode_cn[index]
        label_unicode_vn = self.labels_unicode_vn[index]


        return img_hr, img_lr, label_char, label_unicode_cn, label_unicode_vn
    
class NomDatasetV2:
    """
    This dataset is used for getting the images and labels from the raw images, then stored the coords for future use in SR image

    root_annotation: the path to the folder containing the annotation files
    root_image: the path to the folder containing the images
    scale: the scale of the images
    
    The dataset will return:
        - img_hr: the high resolution image (divisible by scale)
        - img_lr: the low resolution image
        - crop_coords: the coords of the cropped image
        - label_char: list of char label
        - label_unicode_cn: list of unicode_cn label
        - label_unicode_vn: list of unicode_vn label
    
    """
    
    
    def __init__(self, root_annotation: str, root_image: str, scale: int, input_dim:tuple):
        super().__init__()
        
        if root_annotation is None:
            raise ValueError("root_annotation is None")
        if root_image is None:
            raise ValueError("root_image is None")
        if scale is None:
            raise ValueError("scale is None")
        
        self.input_dim = input_dim
        self.scale = scale
        self.total_images = len(os.listdir(root_image))
        self.image_paths =          []
        self.image_coords =         []
        self.labels_char =          []
        self.labels_unicode_cn =    []
        self.labels_unicode_vn =    []
                
        
        for path in os.listdir(root_image):
            self.image_paths.append(os.path.join(root_image, path))        
        
        for annotation_excel in tqdm(os.listdir(root_annotation)):
            df = pd.read_excel(os.path.join(
                    root_annotation, annotation_excel))
            
            tmp_coords = df.loc[:, ['LEFT', 'TOP', 'RIGHT', 'BOTTOM']].values.tolist()
            tmp_labels_char = df.loc[:, ['CHAR']].values.tolist()
            tmp_labels_unicode_cn = df.loc[:, ['UNICODE']].values.tolist()
            tmp_labels_unicode_vn = df.loc[:, ['UNICODE_QN']].values.tolist()
            
            self.image_coords.append(tmp_coords)
            self.labels_char.append(tmp_labels_char)
            self.labels_unicode_cn.append(tmp_labels_unicode_cn)
            self.labels_unicode_vn.append(tmp_labels_unicode_vn)
            
    def __len__(self):
        return self.total_images
    
    def resize_to_divisible_by_scale(self, img, scale=4):
        """
        Resize the image to be divisible by scale (Reuse the one from NomDatasetV1)
        """
        
        h, w, _ = img.shape
        h = h - h % scale
        w = w - w % scale
        
        # Assume image shape needs to be trimmed down to be divisible by scale
        dif_h = img.shape[0] - h
        dif_w = img.shape[1] - w
        
        img = img[:h, :w, :]
        
        return img
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        image_path = self.image_paths[index]
        # print(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_hr = self.resize_to_divisible_by_scale(img, scale=self.scale)
        img_lr = cv2.resize(img_hr, (img_hr.shape[1] // self.scale, img_hr.shape[0] // self.scale), interpolation=cv2.INTER_CUBIC)
        
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                std=[0.229, 0.224, 0.225]),
        # ])
        
        crop_coords_hr = self.image_coords[index]
        crop_coords_lr = []
        for i in range(len(crop_coords_hr)):
            crop_coords_lr.append([x // self.scale for x in crop_coords_hr[i]])
        
        label_char = self.labels_char[index]
        label_unicode_cn = self.labels_unicode_cn[index]
        label_unicode_vn = self.labels_unicode_vn[index]


        return image_path, img_hr, img_lr, crop_coords_hr, crop_coords_lr, label_char, label_unicode_cn, label_unicode_vn

class NomDataset_Yolo(torch.utils.data.Dataset):
    def __init__(self, image_file_path, annotation_file_path, label_file_path, unicode_dict_path, image_size, transform=None) -> None:
        self.image_file_path = image_file_path
        self.annotation_file_path = annotation_file_path
        self.label_file_path = label_file_path
        self.unicode_dict_path = unicode_dict_path
        self.image_size = image_size    # Target crop image size 
        
        self.image_files = []
        self.annotation_files = []
        self.label_files = []
        
        self.transform = transform
        self.load_files_list()
        
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
        
        
        # # Label dictionary
        # with open(self.unicode_dict_path, 'rb') as f:
        #     unicode_labels = pickle.load(f)
        # for i, (k, v) in enumerate(unicode_labels.items()):
        #     unicode_labels[k] = i
        
        # For reading yolo txt files
        total_n = len(self.image_files)
        for image_file, txt_file, excel_file in tqdm(zip(self.image_files, self.annotation_files, self.label_files), total=total_n, disable=True):
            image = cv2.cvtColor(cv2.imread(os.path.join(self.image_file_path, image_file)), cv2.COLOR_BGR2RGB)    # Grayscale, so I can stack 3 channels later
            h, w, _ = image.shape
            df = pd.read_excel(os.path.join(self.label_file_path, excel_file))

            label_dict = {'boxes': [], 'labels': [], 'unicode_labels': []}
            for _, row in df.iterrows():
                x1, y1, x2, y2 = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
                label = row['UNICODE']
                ucode_label = row['CHAR']
                label_dict['boxes'].append((x1, y1, x2, y2))
                label_dict['labels'].append(label)
                label_dict['unicode_labels'].append(ucode_label)

            with open(os.path.join(self.annotation_file_path, txt_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, x, y, b_w, b_h = map(float, line.split(' '))
                    bbox = pybbx.YoloBoundingBox(x, y, b_w, b_h, image_size=(w, h)).to_voc(return_values=True)
                    x1, y1, x2, y2 = bbox
                    
                    # Find the best IOU to label the cropped image
                    iou, box, idx = find_best_IOU(bbox, label_dict['boxes'])
                    
                    self.crop_dict['original_images_name'].append(image_file)
                    crop_img = image[int(y1):int(y2), int(x1):int(x2)]
                    self.crop_dict['crops'].append(crop_img)       
                    self.crop_dict['unicode_labels'].append(label_dict['unicode_labels'][idx])
                    
                    self.crop_dict['labels'].append(label_dict['labels'][idx])
                    # try:
                    #     self.crop_dict['labels'].append(unicode_labels[label_dict['labels'][idx]])
                    # except:
                    #     print("Error at: ", image_file, label_dict['labels'][idx])
                    #     plt.imshow(crop_img)
                    #     print(label_dict['labels'][idx])
            
                    
                    
        
    
    
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
            # Resize the image to 56x56
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            image = image *  1.0 / 255
            
            # TODO: This is the mean and std of ImageNet dataset, need to change to the mean and std of the dataset
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # mean = [0.799, 0.818, 0.829]
            # std = [0.183, 0.179, 0.179]

            image = (image - mean) / std
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # return image, label, ucode_label
        return image, label

# #%%
# with open('NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst-label-set-ucode.pkl', 'rb') as f:
#     unicode_labels = pickle.load(f)
# for i, (k, v) in enumerate(unicode_labels.items()):
#     unicode_labels[k] = i
#     print(i, k)
# print(unicode_labels)
# print(len(unicode_labels))

# #%%
# dataset = NomDataset_Yolo('NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images',
#                           'TempResources/yolov5_runs_ToK1871/detect/yolo_SR/labels',
#                           'NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-annotation/annotation-mynom',
#                           'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst-label-set-ucode.pkl')
# dataset[0]
