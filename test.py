from ESRGAN import RRDBNet_arch as ESRGAN_arch
import os
import cv2
import numpy as np
import torch
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
raw_path = 'NomDataset/datasets/mono-domain-datasets/luc-van-tien/lvt-raw-images'
sr_result_path = 'TempResources/SR_from_HR_LVT'

model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(DEVICE)

for i, box_image in tqdm(enumerate(os.listdir(raw_path))):
    if os.path.exists(os.path.join(sr_result_path, f"{box_image.split('.')[0]}_SR.jpg")):
        continue
    img = cv2.imread(os.path.join(raw_path, box_image))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    
    with torch.no_grad():
        sr_img = model(img).data.squeeze(0).float().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
        sr_img = (sr_img * 255).round().astype(np.uint8)
        
    # check if file exists
    cv2.imwrite(os.path.join(sr_result_path, f"{box_image.split('.')[0]}_SR.jpg"), sr_img) 
