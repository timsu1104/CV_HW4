# camera-ready

# https://github.com/fregu856/deeplabv3/blob/master/datasets.py

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import os, json

from cityscapesscripts.helpers.csHelpers import getCoreImageFileName

cityscapes_data_path = "./data"
assert os.path.exists(cityscapes_data_path)

class CityscapesDataset(Dataset):
    def __init__(self, cityscapes_data_path, split: str, augment: bool):
        super().__init__()
        
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 1024
        self.new_img_w = 2048
        
        self.augment = augment
        
        regions = json.load(open('./datasets/directories.json', 'r'))

        self.file_names = []
        
        region_split = regions[split]
            
        self.img_dir = os.path.join(cityscapes_data_path, f"leftImg8bit/{split}/")
        self.label_dir = os.path.join(cityscapes_data_path, f"gtFine/{split}/")
        for region in region_split:
            data_path = os.path.join(self.img_dir, region)
            label_root = os.path.join(self.label_dir, region)
            
            for file_name in os.listdir(data_path):
                img_id = getCoreImageFileName(file_name)
                img_path = os.path.join(data_path, file_name)
                label_path = os.path.join(label_root, img_id + "_gtFine_labelTrainIds.png")

                fn = {
                    "img_id": img_id,
                    "img_path": img_path,
                    "label_path": label_path
                }
                self.file_names.append(fn)
    
    def load_and_preprocess(self, img_path):
        img = cv2.imread(img_path, -1) # (1024, 2048, 3)
        img = cv2.resize(img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST) # (512, 1024, 3)
        return img

    def __getitem__(self, index):
        f_name = self.file_names[index]

        img = self.load_and_preprocess(f_name["img_path"])
        label = self.load_and_preprocess(f_name["label_path"])
        
        ########## Data augmentation ############

        if self.augment:
            # random horizontal flip
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                label = cv2.flip(label, 1)
                
            # random vertical flip
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 0)
                label = cv2.flip(label, 0)
            
            # random scale
            scale_h = np.random.uniform(low=0.8, high=1.2)
            scale_w = np.random.uniform(low=0.8, high=1.2)
            new_h = int(scale_h*self.new_img_h)
            new_w = int(scale_w*self.new_img_w)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST) 
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # random crop
            x_left = np.random.randint(low=0, high=(new_w - 256))
            y_left = np.random.randint(low=0, high=(new_h - 256))
            img = img[y_left: y_left + 256, x_left: x_left + 256] # (256, 256, 3)
            label = label[y_left: y_left + 256, x_left: x_left + 256] # (256, 256)
        
        # normalization
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # (256, 256, 3)
        img = np.transpose(img, (2, 0, 1)) # (3, 256, 256)

        # ToTensor
        img = torch.from_numpy(img).float() # (3, 256, 256)
        label = torch.from_numpy(label).long() # (256, 256)

        datum = (img, label)
        
        return datum

    def __len__(self):
        return len(self.file_names)

def build_dataset(args):
    train_dataset = CityscapesDataset(args.data_path, 'train', True)
    val_dataset = CityscapesDataset(args.data_path, 'train', False)
    test_dataset = CityscapesDataset(args.data_path, 'val', False)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    build_dataset()