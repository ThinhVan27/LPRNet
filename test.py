# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: ThinhVan27 .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataset
from data.data_augment import *
from model.LPRNet import build_lprnet
from model.small_LPRNet import build_small_lprnet
from utils import *

import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import torch
import time
import os
import wandb

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def test():
    args = get_parser(train=False)
    
    print(f"[INFO] Start testing...")
    #----------Build LPRNet model-----------
    lprnet = build_small_lprnet(lpr_max_len=args.lpr_max_len, train=False, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    lprnet.to(device)
    print("[INFO] Successful to build network!")
    
    #----------Load pretrained model--------
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, weights_only=True, map_location=torch.device(device)))
        print("[INFO] Load pretrained model successful!")
    else:
        lprnet.apply(weights_init_optimal)
        print("[INFO] Initial net weights successful!")
        
    #----------Create test dataset------------
    test_dataset = LPRDataset(root=args.test_img_dirs,
                              train=True,
                              lpr_max_len=args.lpr_max_len,
                              transform=test_transforms)
    print("[INFO] Load data successful!")
    
    # Test
    test_acc, test_loss = decode(lprnet, test_dataset, args, valid=False)
    show(lprnet, test_dataset, args)


if __name__ == "__main__":
    test()
