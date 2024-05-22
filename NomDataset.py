#%%
import cv2
import os
import pandas as pd
import pybboxes as pybbx
import pickle
import torch

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms

import pytorch_lightning as pl


#%%
class YoloCropDataset(Dataset):
    def __init__(self, image_file_path : str, annotation_file_path : str, label_file_path : str, unicode_dict_path : str, image_size : int | int, transform : transforms.Compose):
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
        assert len(self.image_files) == len(self.label_files), f"Number of image files and label files do not match. {len(self.image_files)} != {len(self.label_files)}"

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
        
        
        # Label dictionary
        with open(self.unicode_dict_path, 'rb') as f:
            unicode_labels = pickle.load(f)
        for i, (k, v) in enumerate(unicode_labels.items()):
            unicode_labels[k] = i
        
        # For reading yolo txt files
        total_n = len(self.image_files)
        for image_file, txt_file, excel_file in zip(self.image_files, self.annotation_files, self.label_files):
            image = cv2.cvtColor(cv2.imread(os.path.join(self.image_file_path, image_file)), cv2.COLOR_BGR2RGB)    # Grayscale, so I can stack 3 channels later
            h, w, _ = image.shape
            df = pd.read_excel(os.path.join(self.label_file_path, excel_file))

            label_dict = {'boxes': [], 'labels': []}
            for _, row in df.iterrows():
                x1, y1, x2, y2 = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
                label = row['UNICODE']
                label_dict['boxes'].append((x1, y1, x2, y2))
                label_dict['labels'].append(label)

            with open(os.path.join(self.annotation_file_path, txt_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    _, x, y, b_w, b_h = map(float, line.split(' '))
                    bbox = pybbx.YoloBoundingBox(x, y, b_w, b_h, image_size=(w, h)).to_voc(return_values=True)
                    x1, y1, x2, y2 = bbox
                    
                    # Find the best IOU to label the cropped image
                    iou, box, idx = find_best_IOU(bbox, label_dict['boxes'])
                    
                    crop_img = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    self.crop_dict['crops'].append(crop_img)
                    self.crop_dict['labels'].append(unicode_labels[label_dict['labels'][idx]])

        assert len(self.crop_dict['crops']) == len(self.crop_dict['labels']), "Number of crops and labels do not match"

    def __len__(self) -> int:
        return len(self.crop_dict['crops'])
        
    def __getitem__(self, index: int) -> torch.Tensor | torch.Tensor:
        assert index <= len(self), "Index out of range"
                
        image = self.crop_dict['crops'][index]
        label = self.crop_dict['labels'][index]

        
        if self.transform:
            image = self.transform(image)
        else:
            # Resize the image to 224x224
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
            image = image *  1.0 / 255
            
            # TODO: This is the mean and std of ImageNet dataset, need to change to the mean and std of the dataset
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # mean = [0.799, 0.818, 0.829]
            # std = [0.183, 0.179, 0.179]

            image = (image - mean) / std
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

class YoloCropDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs : dict, input_size : int | int, batch_size : int, num_workers : int, transforms=None):
        super().__init__()
        self.data_dir = data_dirs
        self.input_size = input_size
        self.transforms = transforms

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = YoloCropDataset(self.data_dir['train'][0], self.data_dir['train'][1], self.data_dir['train'][2], self.data_dir['train'][3], self.data_dir['train'][4], self.input_size, self.transforms)
            self.val_dataset = YoloCropDataset(self.data_dir['val'][0], self.data_dir['val'][1], self.data_dir['val'][2], self.data_dir['val'][3], self.data_dir['val'][4], self.input_size, self.transforms)
        elif stage == 'test':
            self.test_dataset = YoloCropDataset(self.data_dir['test'][0], self.data_dir['test'][1], self.data_dir['test'][2], self.data_dir['test'][3], self.data_dir['test'][4], self.input_size, self.transforms)
        elif stage is None:
            pass
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class ImageCropDataset(Dataset):
    """ Image Crop Dataset Loader, used for loading Crop images and labels of crop image

    Args:
        crop_path (str): Path to the directory containing the crop images.
        label_path (str): Path to the file containing the labels of the crop images.
        input_size (tuple(int, int)): Image size to return.
        ucode_dict_path (str): Path to the file containing the unicode dictionary. For translate unicode to dictionary index
        transforms (Callable): Transforms to apply to the crop images.

    """
    def __init__(self, crop_path : str, label_path : str, input_size : int | int, ucode_dict : dict, transforms):
        self.crop_path = crop_path
        self.label_path = label_path
        self.ucode_dict = ucode_dict
        self.transforms = transforms

        self.input_size = input_size
        self.num_labels = 0

        self.crop_list = []
        self.labels_list = []

        def read_crop_and_label(crop_path, label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.split(', ')
                    self.labels_list.append(line[1].strip())
                    crop = line[0].strip()
                    # Check path exists
                    if not os.path.exists(os.path.join(crop_path, crop)):
                        raise FileNotFoundError(f'Crop image {os.path.join(crop_path, crop)} not found')
                    else:
                        self.crop_list.append(crop)

        read_crop_and_label(crop_path, label_path)

        assert self.crop_list is not None, 'No crop images found'
        assert self.labels_list is not None, 'No labels found'
        assert self.ucode_dict is not None, 'No unicode dictionary found'
        assert len(self.crop_list) == len(self.labels_list), 'Number of crops and labels do not match'

        # Display statistics of dataset
        print(f'Number of crops: {len(self.crop_list)}')
        print(f'Number of labels: {self.num_labels}')
        print(f'Crop images shape: {self.input_size}')
        print(f'Number of unique labels: {len(self.ucode_dict)}')

    def __len__(self):
        return len(self.crop_list)

    def __getitem__(self, idx):
        assert idx < len(self), 'Index out of range'
        img_path = os.path.join(self.crop_path, self.crop_list[idx])
        x_crop_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = x_crop_img.shape
        if (h, w) != self.input_size:
            x_crop_img = cv2.resize(x_crop_img, self.input_size, cv2.INTER_LANCZOS4)

        if transforms is not None:
            x_crop_img = self.transforms(x_crop_img).float()
        else:
            x_crop_img = torch.tensor(x_crop_img).float()

        y_label = self.labels_list[idx]
        try:
            y_label = self.ucode_dict[y_label]
        except KeyError:
            # TODO: Handle unknown labels, cuz current dict does not have all Sino-Nom ucode
            y_label = self.ucode_dict['UNK']
        y_label = torch.tensor(y_label, dtype=torch.long)

        return x_crop_img, y_label

class ImageCropDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs : dict, ucode_dict_path : str, input_size : int | int, batch_size : int, num_workers : int, transforms=None):
        super().__init__()
        self.data_dir = data_dirs
        self.ucode_dict_path = ucode_dict_path
        
        self.input_size = input_size
        self.transforms = transforms

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        def read_ucode_dict(ucode_dict_path):
            with open(ucode_dict_path, 'rb') as f:
                ucode_dict = pickle.load(f)
            for i, (k, v) in enumerate(ucode_dict.items()):
                ucode_dict[k] = i
            return ucode_dict
        self.ucode_dict = read_ucode_dict(ucode_dict_path)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = ImageCropDataset(self.data_dir['train'][0], self.data_dir['train'][1], self.input_size, self.ucode_dict, self.transforms)
            self.val_dataset = ImageCropDataset(self.data_dir['val'][0], self.data_dir['val'][1], self.input_size, self.ucode_dict, self.transforms)
        elif stage == 'test':
            self.test_dataset = ImageCropDataset(self.data_dir['test'][0], self.data_dir['test'][1], self.input_size, self.ucode_dict, self.transforms)
        elif stage is None:
            pass
        else:
            raise ValueError(f"Stage {stage} not recognized")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)