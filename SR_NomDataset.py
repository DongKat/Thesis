#%%
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import cv2
import os
from matplotlib import pyplot as plt
import pickle
import torch

#%%
class NomDatasetCrop(Dataset):
    def __init__(self, crop_path : str, label_path : str, input_size : int | int, ucode_dict_path : str, transforms):
        self.crop_path = crop_path
        self.label_path = label_path
        self.ucode_dict_path = ucode_dict_path
        self.transforms = transforms
        
        self.input_size = input_size
        self.num_labels = 0
        
        self.crop_list = []
        self.labels_list = []
        self.ucode_dict = {}
                
        def read_crop_and_label(crop_path, label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.split(', ')  
                    self.labels_list.append(line[1].strip())
                    crop = line[0].strip()
                    # Check path exists
                    if not os.path.exists(os.path.join(crop_path, crop)):
                        raise FileNotFoundError(f'Crop image {crop} not found')
                    else:
                        self.crop_list.append(crop)
                        
        def read_ucode_dict(ucode_dict_path):
            with open(ucode_dict_path, 'rb') as f:
                labels = pickle.load(f)
                labels = sorted(list(labels.keys()))
            self.num_labels = len(labels)
            for i, label in enumerate(labels):
                self.ucode_dict[label] = i      # Dictionary to convert unicode to index int        
        
        read_crop_and_label(crop_path, label_path)
        read_ucode_dict(ucode_dict_path)
        assert self.crop_list is not None, 'No crop images found'
        assert self.labels_list is not None, 'No labels found'
        assert self.ucode_dict is not None, 'No unicode dictionary found'
    
        # print('Finished reading labels')
        
        
    def __len__(self):
        assert len(self.crop_list) == len(self.labels_list), 'Number of crops and labels do not match'
        return len(self.crop_list)
    
    def __getitem__(self, idx):
        assert idx < len(self), 'Index out of range'
        img_path = os.path.join(self.crop_path, self.crop_list[idx])
        x_crop_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = x_crop_img.shape
        if (h, w) != self.input_size:
            x_crop_img = cv2.resize(x_crop_img, self.input_size, cv2.INTER_LANCZOS4)
            
        x_crop_img = self.transforms(x_crop_img).float()
        
        y_label = self.labels_list[idx]
        try:
            y_label = self.ucode_dict[y_label]
        except KeyError:
            # TODO: Handle unknown labels, cuz current dict does not have all Sino-Nom ucode 
            y_label = self.ucode_dict['UNK']
        y_label = torch.tensor(y_label, dtype=torch.long) 

        return x_crop_img, y_label

class NomDataModule(LightningDataModule):
    def __init__(self, data_dirs : dict, batch_size : int, input_size : int | int, num_workers : int):
        super().__init__()
        self.data_dir = data_dirs
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = NomDatasetCrop(self.data_dir['train'][0], self.data_dir['train'][1], self.input_size, self.data_dir['ucode_dict'])
            self.val_dataset = NomDatasetCrop(self.data_dir['val'][0], self.data_dir['val'][1], self.input_size, self.data_dir['ucode_dict'])
        elif stage == 'test':
            self.test_dataset = NomDatasetCrop(self.data_dir['test'][0], self.data_dir['test'][1], self.input_size, self.data_dir['ucode_dict'])
        else:
            raise ValueError(f"Stage {stage} not recognized")
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
#%%

# opt_dict = {'train': ('TempResources/ToK1871/SR_ForResNet/Real-ESRGAN_224', 'TempResources/ToK1871/ToK1871_crops.txt'),
#              'val': ('TempResources/ToK1902/ToK1902_crops', 'TempResources/ToK1902/ToK1902_crops.txt'),
#              'test': ('TempResources/LVT/LVT_crops', 'TempResources/LVT/LVT_crop.txt'),
#              'ucode_dict': 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
# }
# # datamodule = NomDataModule(opt_dict, 8, (224, 224), 0)

# # #%%
# # datamodule.setup('fit')
# # dataloader = datamodule.train_dataloader()
# # temp = next(iter(dataloader))
# # temp_img = temp[0][2].permute(1, 2, 0).numpy()
# # temp_img = temp_img * 255
# # plt.imshow(temp_img.astype('uint8'))
# # plt.show()
# # print(temp[0].shape)
# # print(temp[1])
# dataset = NomDatasetCrop(opt_dict['test'][0], opt_dict['test'][1], 224, opt_dict['ucode_dict'])
# print(len(dataset))
# tmp = dataset[0][1]
# print(tmp)
# tmp_img = dataset[0][0].permute(1, 2, 0).numpy()
# tmp_img = tmp_img * 255
# tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
# plt.imshow(tmp_img.astype('uint8'))
# plt.show()
# #%%
# img = temp[0][5].permute(1, 2, 0).numpy()
# img = img * 255
# plt.imshow(img.astype('uint8'))
# plt.show()
# print(temp[1][5])
# print(chr(int('0x' + temp[1][5], 16)))
# # %%

# # %%
