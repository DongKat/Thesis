#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import sys
import os

# with open('./valid_list.txt', 'r') as f:
#     nameList = list()
#     for line in f:
#         nameList.append(line.strip('.JPG\n'))
#     nameList = sorted(nameList)
# for name in nameList[39:]:
#     imgName = f'{name}.JPG'
#     imgFolder = f'52mm'
#     annoFilename = f'res_{name}.txt'
#     annoFolderName = 'det_annos'
    
#     try:
#         dirPath = f'./{annoFolderName}/{annoFilename}'
#         with open(dirPath, 'r') as f:
#             anno = f.readlines()
#             anno = [x.strip() for x in anno]
#         print(len(anno))

#     except:
#         print(f'File {dirPath} not found')
    
#     try:
#         dirPath = f'./{imgFolder}/{imgName}'
#         img = cv2.imread(dirPath)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         if img is None:
#             raise
#         print(img.shape)
#     except:
#         print(f'File {dirPath} not found')

#     # Draw a binary mask, 1 for the object, 0 for the background
#     mask = np.zeros((img.shape[0], img.shape[1]))
    
#     for dBox in anno:
#         x_tl, y_tl, _, _, x_br, y_br, _, _, _ = dBox.split(',')
#         x_tl, y_tl, x_br, y_br = int(x_tl), int(y_tl), int(x_br), int(y_br)
#         mask[y_tl:y_br, x_tl:x_br] = 255
    
#     def checkMask(rectBox, mask):
#         # Check box intersect with mask
#         x_tl, y_tl, x_br, y_br = rectBox
#         if np.sum(mask[y_tl:y_br, x_tl:x_br]) > 0:
#             return True
#         else:
#             return False
def createTextMask(image, anno):
    # Create a binary for detection boxes
    mask = np.zeros((image.shape[0], image.shape[1]))
    
    for dBox in anno:
        x_tl, y_tl, _, _, x_br, y_br, _, _, _ = dBox
        x_tl, y_tl, x_br, y_br = int(x_tl), int(y_tl), int(x_br), int(y_br)
        mask[y_tl:y_br, x_tl:x_br] = 255
        
    mask = mask.astype(np.uint8)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1) 
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def checkInMask(mask, rectBox):
    x_tl, y_tl, x_br, y_br = rectBox
    if np.sum(mask[y_tl:y_br, x_tl:x_br]) > 0:
        return True 
    else:
        return False
    
def getRectList(anno_file):
    with open(anno_file, 'r') as f:
        anno = f.readlines()
        anno = [x.strip().split(',') for x in anno]
    return anno


def worker(path, anno_file, opt):
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    
    thresh_border = 0
    

    img_name, extension = os.path.splitext(os.path.basename(path))
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2.imwrite('test.jpg', img)
    anno = getRectList(anno_file)
    textMask = createTextMask(img, anno)
    
    h, w = img.shape[0:2]
    
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)
        
    
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            # img = cv2.rectangle(img, (y, x), (y + crop_size, x + crop_size), (0, 255, 0), 2)
            # if checkInMask(textMask, (y, x, y + crop_size, x + crop_size)):
            cv2.imwrite(
                os.path.join(opt['output'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
            #     pass
            # else:
            #     continue
    # cv2.imwrite(
    #     os.path.join(opt['output'], f'{img_name}_s{index:03d}{extension}'), img,
    #     [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    # cv2.imwrite('mask.png', textMask)
    # cv2.imwrite('test.jpg', img)
    process_info = f'Processing {img_name} ...'
    return process_info



def main(args):
    opt = dict()
    opt['input'] = args.input_folder
    opt['detectionAnno'] = args.detectionAnno_folder
    opt['output'] = args.output_folder
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size
    opt['n_thread'] = args.n_threads
    opt['compression_level'] = args.compression_level
    
    img_list = list(os.listdir(opt['input']))
    img_list = [os.path.join(opt['input'], v) for v in img_list]
        
    anno_list = list(os.listdir(opt['detectionAnno']))
    anno_list = [os.path.join(opt['detectionAnno'], v) for v in anno_list]
    
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    
    assert os.path.exists(opt['input']), f'Input folder not found: {opt["input"]}'
    assert os.path.exists(opt['detectionAnno']), f'Detection annotation folder not found: {opt["detectionAnno"]}'
    
    if not os.path.exists(opt['output']):
        os.makedirs(opt['output'])
    
    for path in img_list:
        img_name = os.path.basename(path).split('.')[0]
        anno = os.path.join(opt['detectionAnno'], f'res_{img_name}.txt')
        

        pool.apply_async(worker, args=(path, anno, opt), 
                         callback=lambda arg: pbar.update(1),
        )
        
    pool.close()
    pool.join()
    pool.close()
    
    
    print('\nAll processes done.')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='./train/26mm')
    parser.add_argument('--detectionAnno_folder', type=str, default='./train/det_annos')
    parser.add_argument('--output_folder', type=str, default='./train/26mm_sub_512')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--step', type=int, default=512)
    parser.add_argument('--thresh_size', type=int, default=0)
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--compression_level', type=int, default=3)
    
    args = parser.parse_args()
    main(args)



    