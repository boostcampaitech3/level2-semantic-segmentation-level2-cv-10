import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import os
import os.path as osp
import numpy as np

cat2label = {
    'Background': 0,
    'General trash': 1,
    'Paper': 2,
    'Paper pack': 3,
    'Metal': 4,
    'Glass': 5,
    'Plastic': 6,
    'Styrofoam': 7,
    'Plastic bag': 8,
    'Battery': 9,
    'Clothing': 10
}


class COCOSegDataset(Dataset):
    def __init__(self, data_dir, json_path, mode='train', transform=None):
        '''
        Parameters:
            data_dir (string): data directory path
            json_path (string): .json file path
            mode (string): (train, val, test)
            transform (A.transform): transform
        '''
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.coco = COCO(json_path)

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        # [1] Get img_info
        img_id = self.coco.getImgIds(idx)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = osp.join(self.data_dir, img_info['file_name'])
        img_size = (img_info['height'], img_info['width'])

        # [2] Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # .astype(np.float32) np.uint8 to np.float32
        # img /= 255.0 # Albumentations support both uint8(0~255) or float32(0.0~1.0)

        if self.mode in ('train', 'val'):
            # [3] Load Annotations
            ann_ids = self.coco.getAnnIds(img_id)
            ann_infos = self.coco.loadAnns(ann_ids)

            # [4] Load Categories
            cat_ids = self.coco.getCatIds()
            cat_infos = self.coco.loadCats(cat_ids)

            # [5] Initialize  Mask
            mask = np.zeros(img_size) # (512, 512) initialized to 0

            # [6] Sort annotations in descending order of area
            ann_infos = sorted(ann_infos, key = lambda ann_info: ann_info['area'], reverse=True)

            # [7] Iterate through annotations and fill mask
            for ann_info in ann_infos:
                binary_mask = self.coco.annToMask(ann_info)

                ann_cat_id = ann_info['category_id']
                ann_cat_name = self.get_classname(ann_cat_id, cat_infos)
                pixel_value = cat2label[ann_cat_name]
                mask[binary_mask == 1] = pixel_value

            mask = mask.astype(np.uint8)

            # [8] transform
            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

            return img, mask, img_info

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed['image']

            return img, img_info


    def get_classname(self, cat_id, cats):
        return [x['name'] for x in cats if x['id'] == cat_id][0]
