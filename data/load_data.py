from torch.utils.data import *
from torchvision import transforms
import numpy as np
import torch
import random
import json
import cv2
from PIL import Image
import os

from .data_augment import train_transforms, test_transforms

from typing import Optional, Callable

CHARS = [
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 lpr_max_len: int = 7,
                 transform: Optional[Callable] = None,
                 max_samples: int = -1):
        if not os.path.exists(root):
            raise ValueError(f"Data path is not exist")
        self.root = root
        self.train = train
        self.lpr_max_len = lpr_max_len
        self.max_samples = max_samples
        if transform:
            self.transform = transform
        elif train:
            self.transform = train_transforms
        else:
            self.transform = test_transforms
        self.n_samples = 0
        self.track_path = []
        for sc in os.listdir(self.root):
            for type in os.listdir(os.path.join(self.root, sc)):
                track_root = os.path.join(self.root, sc, type)
                self.n_samples += len(os.listdir(track_root))
                self.track_path.extend([os.path.join(track_root, track_name) for track_name in os.listdir(track_root)])
        self._valid_dataset()
        self.chars = CHARS
    
    def __len__(self):
        return min(self.n_samples, self.max_samples) if self.max_samples > 0 else self.n_samples
    
    def __getitem__(self, idx: int):
        if idx < 0:
            raise ValueError(f"Index must be non-negative integer")
        track_dir = self.track_path[idx]
        img_path = os.path.join(track_dir, "hr-001.png" if "hr-001.png" in os.listdir(track_dir) else "hr-001.jpg")
        img = self.transform(Image.open(img_path))
        if not self.train:
            return img
        target = self._read_annotation(track_dir)
        encoded_target = [CHARS_DICT.get(c, len(CHARS) - 1) for c in target]
        return img, encoded_target, len(target)
    
    def _read_annotation(self, track_dir: str) -> str:
        if not os.path.exists(track_dir):
            raise ValueError(f"{track_dir} does not exists")
        try:
            with open(os.path.join(track_dir, "annotations.json"), 'r', encoding="utf-8") as f:
                anno = json.load(f)
            target = anno.get('plate_text', "")
            return target
        except Exception as e:
            raise e
    
    def _valid_dataset(self):
        for i in range(len(self)):
            try:
                self[i]
            except Exception as e:
                raise e