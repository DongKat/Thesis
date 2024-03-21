#%%
import cv2
import torch
import torchvision
from matplotlib import pyplot as plt

custom_yolo_weights = 'yolov5\weights\yolo_one_label.pt'
yolo_model = torch.hub.load(repo_or_dir='./yolov5', model='custom', path=custom_yolo_weights, source='local')
yolo_model.eval()

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes

#%%
# Run yolo model on image
img = 'test.jpg'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Preprocess the image for yolo, i.e. resize to a multiple of stride s=32 by padding
img = letterbox(img)[0]
h0, w0, _ = img.shape
# cv2 to torch tensor
img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
img.to(device='cuda')


#%%
result = yolo_model(img)
result = non_max_suppression(result, conf_thres=0.5, iou_thres=0.5)


# %%
# Scale bounding boxes to original image size
img = cv2.imread('test.jpg')
h1, w1, _ = img.shape
# print("Original image shape: ", h1, w1)
# print("Yolo Letterbox shape: ", h0, w0)
count = 0
for row in result[0]:
    # Draw bounding boxes
    box = scale_boxes([h0, w0], row[:4], [h1, w1]).round()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    count += 1
print("Number of objects detected: ", count)
cv2.imwrite('./test_yolo.jpg', img)

#%%
# Evaluate the yolo model

#%%
# Run SR with ESRGAN on image
from ESRGAN import RRDBNet_arch as ESRGAN_arch

model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
yolo_box_crops = 'TempResources/YoloBoxCrops'
yolo_box_crops_SR = 'TempResources/SR_from_YoloBoxCrops'

model = ESRGAN_arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(DEVICE)

for i, box_image in tqdm(enumerate(os.listdir(yolo_box_crops))):
    img = cv2.imread(os.path.join(yolo_box_crops, box_image))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    
    with torch.no_grad():
        sr_img = model(img).data.squeeze(0).float().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
        sr_img = (sr_img * 255).round().astype(np.uint8)
        
        cv2.imwrite(os.path.join(yolo_box_crops_SR, f"{box_image.split('.')[0]}_SR.jpg"), sr_img) 
