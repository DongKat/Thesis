#%%
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "TempResources/YoloBoxCrops/tale-of-kieu-1871-page001_13.jpg"
image = cv2.imread(file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
image = image / 255.0
image = np.transpose(image, (2, 0, 1))
image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

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