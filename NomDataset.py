#%%
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import pytorch_lightning as pl

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

    
class NomDatasetV3(pl.LightningDataModule):
    def __init__(self, root_annotation: str, root_image: str, scale: int, input_dim:tuple):
        super().__init__()
        
        self.root_annotation = root_annotation
        self.root_image = root_image
        self.scale = scale
        self.input_dim = input_dim
        
    def prepare_data():
        
    
    def setup(self, stage=None):


# #%%
# if __name__ == "__main__":
    # root_annotation = "NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-annotation/annotation-mynom"
    # root_image = "NomDataset/datasets/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images"
    # scale = 2
    # dataset = NomDatasetV1(root_annotation, root_image, scale)
        
    # #%%
    # from matplotlib import pyplot as plt
    # img_hr, img_lr, label_char, label_unicode_cn, label_unicode_vn = dataset[0]
    # print("Label char: " + label_char)
    # print("Label CN: " + label_unicode_cn)
    # print("Label VN: " + label_unicode_vn)

    # plt.subplot(1,2,1)
    # plt.imshow(img_hr)
    # plt.subplot(1,2,2)  
    # plt.imshow(img_lr)

    # dataset = NomDatasetV2(root_annotation, root_image, scale)

    # img_path, img_hr, img_lr, crop_coords_hr, crop_coords_lr, label_char, label_unicode_cn, label_unicode_vn = dataset[0]

    # print("Label char: " + str(label_char))
    # print("Label CN: " + str(label_unicode_cn))
    # print("Label VN: " + str(label_unicode_vn))

    # for i in range(len(crop_coords_hr)):
    #     cv2.rectangle(img_hr, (crop_coords_hr[i][0], crop_coords_hr[i][1]), (crop_coords_hr[i][2], crop_coords_hr[i][3]), (0, 255, 0), 2)
        
    # for i in range(len(crop_coords_lr)):
    #     cv2.rectangle(img_lr, (crop_coords_lr[i][0], crop_coords_lr[i][1]), (crop_coords_lr[i][2], crop_coords_lr[i][3]), (0, 255, 0), 2)

    # plt.subplot(1,2,1)
    # plt.imshow(img_hr)
    # plt.subplot(1,2,2)  
    # plt.imshow(img_lr)