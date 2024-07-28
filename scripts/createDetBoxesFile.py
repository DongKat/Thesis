# Get annotation boxes from the dataset and save them in a file

import os
import pandas as pd
from tqdm import tqdm

tok1871_raw_images = 'NomDataset/mono-domain-datasets/tale-of-kieu/1871/1871-raw-images'
tok1871_annotations = 'NomDataset/mono-domain-datasets/tale-of-kieu/1871/1871-annotation'

tok1902_raw_images = 'NomDataset/mono-domain-datasets/tale-of-kieu/1902/1902-raw-images'
tok1902_annotations = 'NomDataset/mono-domain-datasets/tale-of-kieu/1902/1902-annotation'

lvt_raw_images = 'NomDataset/mono-domain-datasets/luc-van-tien/lvt-raw-images'
lvt_annotations = 'NomDataset/mono-domain-datasets/luc-van-tien/lvt-annotation/annotation-mynom'

for image_file in tqdm(os.listdir(tok1871_raw_images), desc='Processing 1871 images'):
    image_name = os.path.splitext(image_file)[0]
    annotation_file = os.path.join(tok1871_annotations, f'{image_name}.xlsx')
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f'Annotation file {annotation_file} not found')
    annotation_df = pd.read_excel(annotation_file)

    label_dict = {'boxes': [], 'labels': []}
    for _, row in annotation_df.iterrows():
        x1, y1, x2, y2 = row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM']
        label = row['UNICODE']
        label_dict['boxes'].append((x1, y1, x2, y2))
        label_dict['labels'].append(label)
    


