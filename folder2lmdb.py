
#%%
import lmdb
import os
import h5py
import pickle
import cv2
from tqdm import tqdm
from time import time
import numpy as np

def buf2img(buf) -> np.ndarray:
    imageBuf = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def folder2lmdb(folder_root, lmdb_path, image_size=256):
    map_size = len(os.listdir(folder_root)) * 3 * image_size * image_size * 8 # Should be enough?
    
    def checkImageIsValid(imageBin):
        if imageBin is None:
            return False
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True
    
    def writeCache(env, cache:dict):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                k = k.encode('utf-8')
                v = v.encode('utf-8')
                txn.put(k, v)
                
    env = lmdb.open(lmdb_path, map_size=map_size)
    counter = 1
    
    nSamples = len(os.listdir(folder_root))
            

# Store images in a folder into an LMDB, map size 
def folder2lmdb_text(folder_root, label_file, lmdb_path, image_size=256):
    map_size = len(os.listdir(folder_root)) * 3 * image_size * image_size * 8 # Should be enough?
    # map_size = 100
    
    def checkImageIsValid(imageBin):
        if imageBin is None:
            return False
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True
    
    def writeCache(env, cache:dict):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                k = k.encode('utf-8')
                v = v.encode('utf-8') if isinstance(v, str) else v
                
                txn.put(k, v)
                
    env = lmdb.open(lmdb_path, map_size=map_size)
    counter = 1
    
    nSamples = len(os.listdir(folder_root))
    #number of lines in file

    
    with open(label_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == nSamples, f'Number of lines in label file does not match number of images in folder ({len(lines)} != {nSamples})'
        cache = dict()
        for line in lines:
            img_name, label = line.strip().split(', ')
            
            if not os.path.exists(os.path.join(folder_root, img_name)):
                print('%s does not exist' % img_name)
                raise FileNotFoundError
            
            img = cv2.imread(os.path.join(folder_root, img_name))
            imgBin = cv2.imencode('.jpg', img)[1]
            if not checkImageIsValid(imgBin):
                print('%s is not a valid image' % img_name)
                raise ValueError
            
            imageKey = f'image-{counter:09d}'
            labelKey = f'label-{counter:09d}'
            cache[imageKey] = imgBin.tobytes()
            cache[labelKey] = label.encode('utf-8')

            if counter % 1000 == 0:
                writeCache(env, cache)
                del cache
                cache = dict()
                print(f'Written {counter} / {nSamples}')
            counter += 1
            
        nSamples = counter - 1
        cache['num-samples'] = str(nSamples)
        writeCache(env, cache)
        print(f'Created dataset with {nSamples} samples')
    env.close()
    
def pairedFolder2lmdb(folders: tuple, label_file, lmdb_path):
    map_size = 1 * 1024 * 1024 * 1024  # 1GB in bytes
    lq_folder, gt_folder = folders
    
    def checkImageIsValid(imageBin):
        if imageBin is None:
            return False
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True
    
    def writeCache(env, cache:dict):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                k = k.encode('utf-8')
                v = v.encode('utf-8') if isinstance(v, str) else v
                txn.put(k, v)

    env = lmdb.open(lmdb_path, map_size=map_size)
    counter = 1
    
    assert len(os.listdir(lq_folder)) == len(os.listdir(gt_folder)), 'Number of images in folders do not match'
    nSamples = len(os.listdir(gt_folder))
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == nSamples, f'Number of lines in label file does not match number of images in folder ({len(lines)} != {nSamples})'
        cache = dict()
        for line in lines:
            img_name, label = line.strip().split(', ')
            
            if not os.path.exists(os.path.join(folders[0], img_name)):
                print('%s does not exist' % img_name)
                raise FileNotFoundError(f'{img_name} does not exist in {folders[0]}')
            
            img = cv2.imread(os.path.join(folders[0], img_name))
            imgBin = cv2.imencode('.jpg', img)[1]
            if not checkImageIsValid(imgBin):
                print('%s is not a valid image' % img_name)
                raise ValueError
            imageKey = f'image_gt_{counter:09d}'
            cache[imageKey] = imgBin.tobytes()
            
            if not os.path.exists(os.path.join(folders[1], img_name)):
                print('%s does not exist' % img_name)
                raise FileNotFoundError(f'{img_name} does not exist in {folders[1]}')
            
            img = cv2.imread(os.path.join(folders[1], img_name))
            imgBin = cv2.imencode('.jpg', img)[1]
            if not checkImageIsValid(imgBin):
                print('%s is not a valid image' % img_name)
                raise ValueError
            imageKey = f'image_lq_{counter:09d}'
            cache[imageKey] = imgBin.tobytes()


            labelKey = f'label_{counter:09d}'
            cache[labelKey] = label.encode('utf-8')
            
            if counter % 1000 == 0:
                writeCache(env, cache)
                del cache
                cache = dict()
                print(f'Written {counter} / {nSamples}')
            counter += 1
            
        nSamples = counter - 1
        cache['dataset_size'] = str(nSamples)
        writeCache(env, cache)
        print(f'Created dataset with {nSamples} samples')
    env.close()
        
            
#%%
if __name__ == '__main__':
    folder_root = ('TempResources/ToK1902/temp/gt', 'TempResources/ToK1902/temp/lqx2')
    label_file = 'temp.txt'
    lmdb_path = 'TempResources/ToK1902/temp/lmdb'
    folder2lmdb(folder_root, lmdb_path)
    # folder2lmdb(folder_root, label_file, lmdb_path)
    # pairedFolder2lmdb(folder_root, label_file, lmdb_path)
    print('Done')
    
    idx = 11
    
    # Test read lmdb file
    start_time = time()
    env = lmdb.open(lmdb_path)
    with env.begin(write=False) as txn:
        label_key = f'label_{idx:09d}'.encode()
        word = txn.get(label_key)
        
        img = txn.get(f'image_gt_{idx:09d}'.encode())
        img = buf2img(img)
        print(img.shape)
        
        img = txn.get(f'image_lq_{idx:09d}'.encode())
        img = buf2img(img)
        print(img.shape)
        
        print(chr(int(word, 16)))
        print('Total samples:', txn.get('dataset_size'.encode()).decode())
        
        
    env.close()
    print('Read from lmdb:', (time() - start_time) * 1000)
    
    
    # # Compare read time from disk and lmdb
    # start_time = time()
    # with open(label_file, 'r') as f:
    #     for line in f:
    #         img_name, label = line.strip().split(', ')
    #         img = cv2.imread(os.path.join(folder_root, img_name))
    # # print('Read from disk:', img.shape, chr(int(label, 16)))
    # print('Read from disk:', (time() - start_time) * 1000)
    
    
