
import os
from pathlib import Path

import torch
from torch.utils.data import dataset


class IDCardDataset(dataset.Dataset):
    def __init__(self, root, tokenizer, labels=None, mode='train'):
        self.root = root
        self.tokenizer = tokenizer
        self.labels = labels
        self.mode = mode
        
        
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    
    