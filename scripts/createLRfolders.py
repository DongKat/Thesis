folder_image = ['NomDataset/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images', 'NomDataset/mono-domain-datasets/tale-of-kieu/1902/1902-raw-images', 'NomDataset/mono-domain-datasets/luc-van-tien/lvt-raw-images']
folder_annotation = ['NomDataset/mono-domain-datasets/tale-of-kieu/1871/1871-annotation/annotation-mynom', 'NomDataset/mono-domain-datasets/tale-of-kieu/1902/1902-annotation/annotation-mynom', 'NomDataset/mono-domain-datasets/luc-van-tien/lvt-annotation/annotation-mynom']

from tqdm import tqdm
import os
import pandas as pd
import cv2
import pickle

def createLRFolder(image_dir, anno_dir, out_dir, scale):
    image_paths = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]
    anno_paths = [os.path.join(anno_dir, os.path.splitext(image)[0] + '.xlsx') for image in os.listdir(image_dir)]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pbar = tqdm(total=len(image_paths), desc='Processing images')
    with open(out_dir + '/label_boxes.txt', 'w') as f:
        img_boxes_dict = {}
        for image_path, anno_path in zip(image_paths, anno_paths):
            img = cv2.imread(image_path)
            anno_df = pd.read_excel(anno_path)

            h, w, _ = img.shape
            mod4_folder = os.path.join(out_dir, 'GTmod4')
            mod4_image = os.path.join(mod4_folder, os.path.basename(image_path))
            mod4_anno = os.path.join(mod4_folder, os.path.basename(anno_path))
            if not os.path.exists(mod4_folder):
                os.makedirs(mod4_folder)
            # create image with mod4 scale
            h_mod4, w_mod4 = h - h % scale, w - w % scale
            img_mod4 = cv2.resize(img, (w_mod4, h_mod4), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(mod4_image, img_mod4)

            f.write(f'{os.path.basename(image_path)}: ')
            img_labelBoxes = []
            for _, row in anno_df.iterrows():
                x1, y1, x2, y2 = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
                x1, y1, x2, y2 = x1 - x1 % scale, y1 - y1 % scale, x2 - x2 % scale, y2 - y2 % scale
                label = row['UNICODE']
                img_labelBoxes.append(((x1, y1, x2, y2), label))
                f.write(f'({x1},{y1},{x2},{y2},{label}),')
            img_boxes_dict[os.path.basename(image_path)] = img_labelBoxes

            f.write('\n')





            lrx2_folder = os.path.join(out_dir, 'LRx2')
            lrx2_image = os.path.join(lrx2_folder, os.path.basename(image_path))
            lrx2_anno = os.path.join(lrx2_folder, os.path.basename(anno_path))
            if not os.path.exists(lrx2_folder):
                os.makedirs(lrx2_folder)
            # create image with 1/2 scale
            h_lrx2, w_lrx2 = h // 2, w // 2
            img_lrx2 = cv2.resize(img, (w_lrx2, h_lrx2), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(lrx2_image, img_lrx2)

            lrx4_folder = os.path.join(out_dir, 'LRx4')
            lrx4_image = os.path.join(lrx4_folder, os.path.basename(image_path))
            lrx4_anno = os.path.join(lrx4_folder, os.path.basename(anno_path))
            if not os.path.exists(lrx4_folder):
                os.makedirs(lrx4_folder)
            # create image with 1/4 scale
            h_lrx4, w_lrx4 = h // 4, w // 4
            img_lrx4 = cv2.resize(img, (w_lrx4, h_lrx4), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(lrx4_image, img_lrx4)


            pbar.update(1)


        with open(os.path.join(out_dir, 'label_boxes.pkl'), 'wb') as pkl:
            pickle.dump(img_boxes_dict, pkl)
        
        pbar.close()
        print(f'Processed {image_dir} and saved to {out_dir}, with scale {scale}')

# createLRFolder(folder_image[0], folder_annotation[0], 'TempResources/ToK1871/ToK1871_GTmod4', 4)
# createLRFolder(folder_image[1], folder_annotation[1], 'TempResources/ToK1902/ToK1902_GTmod4', 4)
# createLRFolder(folder_image[2], folder_annotation[2], 'TempResources/LVT/LVT_GTmod4', 4)




            


        
        

            

