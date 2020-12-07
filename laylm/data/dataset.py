import os
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import dataset
from sklearn.model_selection import train_test_split

from laylm.data import utils 
from ..ops import boxes_ops

import pandas as pd
import numpy as np
import json
from tqdm import tqdm, trange



class IDCardDataset(dataset.Dataset):
    def __init__(self, root, tokenizer, labels=None, mode='train', 
                 test_size=0.2, max_seq_length=512, 
                 rand_seq=True, rand_seq_prob=0.5):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.labels = labels
        self.mode = mode
        self.test_size = test_size
        self.max_seq_length = max_seq_length
        self.rand_seq = rand_seq
        self.rand_seq_prob = rand_seq_prob
        
        self._build_files()
        
    def _build_files(self):
        names = self._get_names("*_json.json")
        self.train_names, self.test_names = self._split_dataset(names)
        
        if self.mode=="train":
            self.names = self.train_names
        else:
            self.names = self.test_names
        
    def _split_dataset(self, data):
        train, test = train_test_split(data, 
                                       test_size=self.test_size, 
                                       random_state=1261)
        return train, test
    
    def _get_names(self, path_pattern):
        names = []
        files = self._glob_filter(path_pattern)
#         files_iter = tqdm(files, desc="Read All JSON Files")
        for file in files:
            names.append(file.name.split("_")[0])
        return names
    
    def _glob_filter(self, pattern):
        return sorted(list(self.root.glob(pattern)))
    
    def _get_data(self, idx):
        name = self.names[idx]
        
        json_file = f'{name}_json.json'
        json_path = self.root.joinpath(json_file)
#         print('json_path exist: ',json_path.exists())
        
        img_file = f'{name}_image.jpg'
        img_path = self.root.joinpath(img_file)
        
        mask_file = f'{name}_mask.jpg'
        mask_path = self.root.joinpath(mask_file)
        
        data = {
            'anno': str(json_path),
            'image': str(img_path),
            'mask': str(mask_path),
        }
        
        return data
        
        
    def _load_anno(self, path):
        path = str(path)
        with open(path) as f:
            data_dict = json.load(f)
        return data_dict
    
    def _load_image(self, path):
        path = str(path)
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        return img
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        record = self._get_data(idx)
        anno = self._load_anno(record['anno'])
        img = self._load_image(record['image'])
        mask = self._load_image(record['mask'])
        
        anno_objects = utils.format_annotation_objects(
            anno, self.tokenizer, self.max_seq_length,
            rand_seq=self.rand_seq, rand_seq_prob=self.rand_seq_prob
        )
        
        input_ids, input_masks, segment_ids, label_ids, boxes = anno_objects
#         print(input_ids)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_masks = torch.tensor(input_masks, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        boxes = torch.tensor(boxes, dtype=torch.long)
        data = (
            input_ids, input_masks, 
            segment_ids, label_ids, boxes,
            img, mask
        )
        
        return data
    

class IDCardAnnoDataset(object):
    def __init__(self, root, tokenizer, max_seq_length=512, 
                 rand_seq=True, rand_seq_prob=0.5):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.rand_seq = rand_seq
        self.rand_seq_prob = rand_seq_prob
        
        self.anno = self._glob_filter("*_json.json")
    
    def _glob_filter(self, pattern):
        return sorted(list(self.root.glob(pattern)))
    
    def _load_anno(self, path):
        path = str(path)
        with open(path) as f:
            data_dict = json.load(f)
        return data_dict
    
    def _format_anno(self, anno):
        objects = anno['objects']
        data = []
        for obj in objects:
            pts = obj['points']
            pts = np.array(pts).astype(np.int)
            pts = boxes_ops.order_points(np.array(pts).astype(np.int))
            pts = list(boxes_ops.to_xyminmax(pts))
            
            dct = {
                'text': obj['text'],
                'bbox': pts,
                'label': obj['label']
            }
            data.append(dct)
            
        return data
    
    def __getitem__(self, idx):
        path = str(self.anno[idx])
        anno = self._load_anno(path)
        objects = self._format_anno(anno)
        
        return objects
        
    
    
    