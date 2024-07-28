import os
import cv2
from tqdm.auto import tqdm


anno_dir = './val/annos'
img_dirs = ['./val/52mm', './val/26mm', './val/13mm']
output_dirs = ['./cropbox/52mm', './cropbox/26mm', './cropbox/13mm']
counter = 0

with open('./cropbox/label.txt', 'w') as label_file:
    for img_dir, output_dir in zip(img_dirs, output_dirs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for anno_file in tqdm(os.listdir(anno_dir)):
            anno_path = os.path.join(anno_dir, anno_file)
            img_path = os.path.join(img_dir, anno_file.replace('txt', 'JPG'))
            with open(anno_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(',')
                    x_tl, y_tl, _, _, x_br, y_br, _, _, label = line
                    x_tl, y_tl, x_br, y_br = int(x_tl), int(y_tl), int(x_br), int(y_br)
                    img = cv2.imread(img_path)
                    crop_img = img[y_tl:y_br, x_tl:x_br]

                    output_path = os.path.join(output_dir, f'{counter:09d}.png')
                    try:
                        cv2.imwrite(output_path, crop_img)
                        label_file.write(f'{output_path}, {label}\n')
                    except:
                        print('\nError: ', img_path)

                    counter += 1

                    # break
            # break
        # break



