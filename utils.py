from model.LPRNet import build_lprnet
from data.load_data import *

from typing import Literal
import argparse
import torch
from torch import nn
import time
from matplotlib import pyplot as plt

def weights_init_optimal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.01)
        
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int8)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = [T_length] * len(lengths)
    target_lengths = lengths

    return tuple(input_lengths), tuple(target_lengths)

def decode(model, dataset, args: argparse.Namespace, mode: Literal['beam', 'greedy'] = 'greedy',valid=True):
    Tp = 0 # Correct prediction
    Tn_1 = 0 # Wrong length
    Tn_2 = 0 # True length but not correct
    t1 = time.time()

    loss = 0
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    loader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    for (images, labels, lengths) in loader:
        # load train data
        start = 0
        targets = []
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el[:7].numpy() for el in targets])

        images = images.to(device)
        model = model.to(device)

        # Forward
        logits = model(images)
        # Calc loss
        input_lengths, target_lengths = sparse_tuple_for_ctc(args.T_length, lengths)
        log_probs = logits.permute(2, 0, 1).log_softmax(2)
        loss += ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths).item()
        logits = logits.cpu().detach().numpy()
        if mode == 'greedy':
            pred_labels = greedy_search(logits)
        else:
            pred_labels = beam_search(logits, topk=args.topk)
        for i, label in enumerate(pred_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i], dtype=np.int8) == np.asarray(label, dtype=np.int8)).all():
                Tp += 1
            else:
                Tn_2 += 1

    acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2 + 1e-6)
    t2 = time.time()
    phase = "Test" if not valid else "Validation"
    print("[INFO] {} accuracy: {} [{}:{}:{}:{}] | {} loss: {}".format(phase, acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2), phase, loss))
    print("[INFO] {} speed: {}s 1/{}]".format(phase, (t2 - t1) / len(dataset), len(dataset)))

    return acc, loss

def greedy_search(logits):
    pred_labels = list()
    for i in range(logits.shape[0]):
        pred = logits[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
    
    return pred_labels

def beam_search(logits, topk=3):
    pred_labels = list()
    for i in range(logits.shape[0]):
        pred = logits[i, :, :]
        pred_space = list()
        for j in range(pred.shape[1]):
            ids = np.argpartition(pred[:, j], -topk)[-topk:]
            pred_space.append([idx for idx in sorted(list(ids), key=lambda x: pred[x, j])])
        dp = np.zeros(shape=(topk, pred.shape[1]))
        dp[:, 0] = np.asarray([pred[idx, j] for idx in pred_space[0]])
        for t in range(1, len(pred_space)):
            pass
            
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
    
    return pred_labels

        