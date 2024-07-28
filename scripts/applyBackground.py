import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil
# %%
def add_background(background_img : np.ndarray, overlay_img : np.ndarray):
    background = background_img
    overlay = overlay_img

    background = cv2.resize(background, (overlay.shape[1], overlay.shape[0]), interpolation = cv2.INTER_CUBIC)
    # cv2.imwrite("/content/resized_background.jpg", background)

    alpha = 0.5
    beta = 1 - alpha

    added_image = cv2.addWeighted(background, alpha, overlay, beta, 0.0)

    return added_image


background_dir = 'Image Texture Backgrounds'
backgrounds = os.listdir(background_dir)
backgrounds = [os.path.join(background_dir, i) for i in backgrounds]

img_dir = 'TempResources/ToK1871/ToK1871_mixedSR_crops'
output_dir = 'TempResources/ToK1871/ToK1871_mixedSR_crops_wBackground'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for img in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img)
    image = cv2.imread(img_path)
    background = cv2.imread(np.random.choice(backgrounds))
    image = add_background(background, image)

    img = img.split('.')[0] + '.jpg'
    cv2.imwrite(os.path.join(output_dir, img), image)
# %%
mixed_output_dir = 'TempResources/ToK1871/ToK1871_mixedSR_crops_wMixedbackground'

if not os.path.exists(mixed_output_dir):
    os.makedirs(mixed_output_dir)

# Copy the images to the mixed_output_dir
for img in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img)
    shutil.copy(img_path, os.path.join(mixed_output_dir, img))

for img in tqdm(os.listdir(output_dir)):
    img_path = os.path.join(output_dir, img)
    shutil.copy(img_path, os.path.join(mixed_output_dir, img))
