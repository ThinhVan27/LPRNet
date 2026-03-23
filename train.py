# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: thinhvan27.
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataset
from model.LPRNet import build_lprnet
from utils import *

from sklearn.model_selection import train_test_split
from torchinfo import summary
from tqdm import tqdm
from torch.utils.data import Subset
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

def train():
    args = get_parser()
    
    #---------Initialize wandb run------------
    wandb.init(
        entity=args.entity,
        project=args.project,
        name=f"lprnet-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "test_batch_size": args.test_batch_size,
            "max_epoch": args.max_epoch,
            "dropout_rate": args.dropout_rate,
            "lpr_max_len": args.lpr_max_len,
            "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu",
            'mode': f"{args.mode} search",
            'topk': args.topk if args.mode == "beam" else 1
        }
    )
    
    #----------Build LPRNet model-----------
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, train=True, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    lprnet.to(device)
    print("[INFO] Successful to build network!")
    
    #----------Log model architecture-------
    wandb.log({"model_architecture": str(summary(lprnet, input_size=(1, 3, 40, 100), verbose=0))})

    #----------Load pretrained model--------
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, weights_only=True, map_location=torch.device(device)))
        print("[INFO] Load pretrained model successful!")
    else:
        lprnet.apply(weights_init_optimal)
        print("[INFO] Initial net weights successful!")

    #----------Define optimizer-------------
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)

    #----------Define LR Scheduler----------
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=args.gamma)
    
    #----------Create train/test dataset------------
    ds = LPRDataset(args.train_img_dirs)
    if args.test:
        test_dataset = LPRDataset(args.test_img_dirs)
    train_idx, valid_idx = train_test_split(np.arange(len(ds)), test_size=0.2, random_state=42, shuffle=True)
    train_dataset = Subset(ds, train_idx)
    valid_dataset = Subset(ds, valid_idx)
    print("[INFO] Load data successful!")
    
    #----------Create DataLoader---------------
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=args.num_workers)

    #-----------Loss function--------------
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    #----------Initialize hyperparameters---------
    T_length = args.T_length # Prediction length
    n_epochs = args.max_epoch
    epoch_loss = 0
    min_loss = float('inf')
    best = None
    
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    for epoch in tqdm(range(n_epochs), leave=False):
        epoch_loss = 0
        for (images, labels, lengths) in train_loader:
            start_time = time.time()
            # Prepare input
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
            images = images.to(device).requires_grad_(False)
            labels = labels.to(device).requires_grad_(False)
            # Forward
            logits = lprnet(images).permute(2, 0, 1) # (T, N, C) shape for CTCLoss
            log_probs = logits.log_softmax(2).requires_grad_()
            # Backward
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if loss.item() == np.inf:
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            end_time = time.time()
        print(f'[INFO] Train loss: {epoch_loss}')
        # Validation
        val_acc, val_loss = decode(lprnet, valid_dataset, args)
        # Tracking
        log_dict = {
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
            'batch_time': end_time - start_time,
        }
        wandb.log(log_dict)
        if (epoch + 1) % args.save_interval == 0:
            save_path = args.save_folder + f"epoch_{epoch+1}.pth"
            torch.save(lprnet.state_dict(), save_path)
            wandb.save(save_path)
        
        if val_loss < min_loss:
            min_loss = val_loss
            best = lprnet.state_dict()
        scheduler.step()
    
    # Test
    if args.test:
        test_acc, test_loss = decode(lprnet, test_dataset, args)
        wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
    last_path = args.save_folder + "last.pth"
    torch.save(lprnet.state_dict(), last_path)
    wandb.save(last_path)
    best_path = args.save_folder + "best.pth"
    torch.save(best, best_path)
    wandb.save(best_path)
               
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()
