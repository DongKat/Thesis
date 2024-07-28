from multiprocessing import Pool
from tqdm import tqdm
import shutil
import os

hr_folders = ['./train/52mm_sub_512', './train/52mm_sub_512']
lr_folders = ['./train/26mm_sub_512', './train/13mm_sub_512']

if not os.path.exists('processed'):
    os.makedirs('processed')

if not os.path.exists('processed/HR'):
    os.makedirs('processed/HR')
    
if not os.path.exists('processed/LR'):
    os.makedirs('processed/LR')
    
    
pbar = tqdm(total=len(hr_folders), unit='file', desc='Moving')
index = 0
for folder in hr_folders:
    for file in os.listdir(folder):
        shutil.copy(f'{folder}/{file}', f'processed/HR/{index:09d}.jpg')
        pbar.update(1)
        index += 1
pbar.close()

pbar = tqdm(total=len(lr_folders), unit='file', desc='Moving')
index = 0
for folder in lr_folders:
    for file in os.listdir(folder):
        shutil.copy(f'{folder}/{file}', f'processed/LR/{index:09d}.jpg')
        pbar.update(1)
        index += 1
pbar.close()
    
