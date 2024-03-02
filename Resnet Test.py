# Test ResNet model on few Yolo crops of original images from ToK1871?
#%%
from NomDataset import NomDatasetV1, NomDatasetV2

# Standard libraries imports
import importlib
import os
import glob
import random
import shutil
from tqdm import tqdm
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

#%%
data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
with open(data_path_label, 'rb') as f:
    unicode_labels = pickle.load(f)
    unicode_labels = sorted(list(unicode_labels.keys()))
print("Total number of unicode: ", len(unicode_labels))
print(unicode_labels)

#%%
class PytorchResNet101(pl.LightningModule):
    def __init__(self, num_labels):
        super(PytorchResNet101, self).__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels

        # get ResNet architecture
        backbone = resnet101(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.num_labels)

        self.criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.metrics_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        classification = self.classifier(representations)

        return classification


    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.00001, weight_decay=1e-4, momentum=0.9)
        # optimizer = Adam(self.parameters(), lr=0.001, eps=1)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.0005)
        # return dict(
        #     optimizer=optimizer,
        #     lr_scheduler=dict(
        #         scheduler=scheduler,
        #         interval='step'
        #     )
        # )
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # accuracy
        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        acc = self.metrics_accuracy(y_hat_argmax, y)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append(loss)

        return {'loss': loss, 'train_accuracy': acc}


    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()

        self.log('train_acc_epoch', epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.clear()


    # def convert_to_one_hot(self, y):
    #     vector = np.zeros((y.shape[0], self.num_labels))
    #     vector = torch.eye(self.num_labels)[y]
    #     return torch.tensor(vector, dtype=torch.int)

    # VALIDATION
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # accuracy
        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        acc = self.metrics_accuracy(y_hat_argmax, y)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.append(loss)

        return {'val_loss': loss, 'val_accuracy': acc}

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()

        self.log('val_acc_epoch', epoch_average, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()


    # Saver
    def save_metrics(self, folder, filename, content):
        with open(folder  + filename + '.txt', 'w', encoding='utf-8') as the_txt_file:
            the_txt_file.write(str(content))
        with open(folder + filename + '.pickle', 'wb') as f:
            pickle.dump(content, f)


    # TEST
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)

        # accuracy
        acc = self.metrics_accuracy(y_hat_argmax, y)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # loss
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append(loss, acc)

        # return {'test_loss': loss, 'test_accuracy': acc, 'ap': ap, 'auroc': auroc, 'confusion_matrix':confmat, 'f1': f1}
        return {'test_loss': loss, 'test_accuracy': acc, 'y': y, 'y_hat':y_hat}

    def test_step_end(self, batch_parts):
        return batch_parts

    def on_test_epoch_end(self, test_step_outputs):
        acc = 0
        loss = 0
        count = 0
        y = None
        y_hat = None
        for test_step_out in test_step_outputs:
            if count == 0:
                y = test_step_out['y']
                y_hat = test_step_out['y_hat']
            else:
                y = torch.cat([y, test_step_out['y']], 0)
                y_hat = torch.cat([y_hat, test_step_out['y_hat']], 0)
            loss += test_step_out['test_loss']
            acc += test_step_out['test_accuracy']
            count += 1

        acc = acc / count
        self.log('test_acc_epoch', acc, on_epoch=True, prog_bar=True, logger=True)
        self.save_metrics('metrics/', 'test_acc_epoch', acc)

        loss = loss / count
        self.log('test_loss_epoch', loss, on_epoch=True, prog_bar=True, logger=True)
        self.save_metrics('metrics/', 'test_loss_epoch', loss)

        y_hat_softmax = softmax(y_hat)
        y_hat_argmax = torch.argmax(y_hat_softmax, dim=1)
        self.save_metrics('ys/', 'y', y)
        self.save_metrics('ys/', 'y_hat', y_hat)
        self.save_metrics('ys/', 'y_hat_softmax', y_hat_softmax)
        self.save_metrics('ys/', 'y_hat_argmax', y_hat_argmax)


#%%
weights_path = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
model = PytorchResNet101.load_from_checkpoint(weights_path, num_labels=len(unicode_labels))
model.eval()
model.freeze()
model.to(DEVICE)

#%%
yolo_crop_path = "TempResources/YoloBoxCrops"

file_list = os.listdir(yolo_crop_path)
random.shuffle(file_list)

i = 0

for i, image_file in enumerate(file_list):
    image_path = os.path.join(yolo_crop_path, image_file)
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
    pred = int(pred, 16)
    pred = chr(pred)

    print("Predicted label: ", pred)

    plt.imshow(image.squeeze(0).cpu().numpy().transpose(1, 2, 0))
    plt.show()
    i+=1
    if i == 11:
        break
    
#%% Welp it's 10/10 acc, the model is working fine. ResNet test concluded
print(chr(0x20C4B))