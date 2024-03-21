#%%
import cv2
import numpy as np
import pickle
import torch
import torchvision
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_yolo_weights = 'yolov5\weights\yolo_one_label.pt'
yolo_model = torch.hub.load(repo_or_dir='./yolov5', model='custom', path=custom_yolo_weights, source='local')
yolo_model.eval()
yolo_model = yolo_model.to(DEVICE)

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes

#%%
# Run yolo model on image
img = 'test.jpg'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Preprocess the image for yolo, i.e. resize to a multiple of stride s=32 by padding
tmp_img = letterbox(img)[0]
h0, w0, _ = tmp_img.shape
# cv2 to torch tensor
tmp_img = torch.from_numpy(tmp_img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

#%%
with torch.no_grad():
    yolo_result = yolo_model(tmp_img)
    yolo_result = non_max_suppression(yolo_result, conf_thres=0.5, iou_thres=0.5)
print(yolo_result[0])

# %%
# Scale bounding boxes to original image size
h1, w1, _ = img.shape
print("Original image shape: ", h1, w1)
print("Yolo Letterbox shape: ", h0, w0)
count = 0
for row in yolo_result[0]:
    # Draw bounding boxes
    box = scale_boxes([h0, w0], row[:4], [h1, w1]).round()
    x1, y1, x2, y2 = map(int, box)
    print(x1, y1, x2, y2)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    count += 1
print("Number of objects detected: ", count)
cv2.imwrite('./test_yolo.jpg', img)

#%%
# Evaluate the yolo model
torch.cuda.empty_cache()

#%%
# Run SR with ESRGAN on image
from ESRGAN import RRDBNet_arch as ESRGAN_arch

model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'

sr_model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
sr_model.load_state_dict(torch.load(model_path), strict=True)
sr_model.eval()
sr_model = sr_model.to(DEVICE)

#%% 
print(yolo_result[0])

# %%
# TODO: Run SR on multiple regions of interest and pass to ResNet
h1, w1, _ = img.shape
print("Original image shape: ", h1, w1)
print("Yolo Letterbox shape: ", h0, w0)
for row in yolo_result[0]:
    # Draw bounding boxes
    x1, y1, x2, y2 = map(int, row[:4])
    print(x1, y1, x2, y2)
    # Crop the region of interest
    roi = img[int(y1):int(y2), int(x1):int(x2)]
    roi = torch.from_numpy(roi).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)
    # # Resize the roi to 4x
    with torch.no_grad():
        sr_result = sr_model(roi).data.squeeze(0).float().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
        sr_result = (sr_result * 255).round().astype(np.uint8)
    
    # Save the sr result
    cv2.imwrite(f'./test_yolo_sr_{count}.jpg', sr_result)
    
# %%
# Load the ResNet model
from NomResnet import PytorchResNet101
data_path_label = 'NomDataset/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl'
with open(data_path_label, 'rb') as f:
    unicode_labels = pickle.load(f)
    unicode_labels = sorted(list(unicode_labels.keys()))
print("Total number of unicode: ", len(unicode_labels))
print(unicode_labels)

weights_path = 'PytorchResNet101Pretrained-data-v2-epoch=14-val_loss_epoch=1.42927-train_acc_epoch=0.99997-val_acc_epoch=0.79039.ckpt'
resnet_model = PytorchResNet101.load_from_checkpoint(weights_path, num_labels=len(unicode_labels))
resnet_model.eval()
resnet_model.freeze()
resnet_model.to(DEVICE)

# %%
with torch.no_grad():
    tmp_img = torch.from_numpy(sr_result).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    resnet_result = resnet_model(tmp_img)
    pred = softmax(resnet_result)
    pred = torch.argmax(pred, dim=1)
    pred = unicode_labels[pred.item()]
    pred = "0x" + pred
    
    print("Predicted label unicode: ", pred)
    
    pred = int(pred, 16)
    pred = chr(pred)

    print("Predicted label: ", pred)
# %%
