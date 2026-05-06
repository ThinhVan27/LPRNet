from model.LPRNet import build_lprnet
from data.load_data import *

from typing import Literal
import argparse
import torch
from torch import nn
import time
import yaml
from matplotlib import pyplot as plt
import random

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def get_parser(train=True):
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--config', default="config/" + f"{'train' if train else 'test'}_config.yaml", help='path to configuration file')
    args = parser.parse_args()
    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

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

def decode(model, dataset, args: argparse.Namespace, dequan: False):
    Tp = 0 # Correct prediction
    Tn_1 = 0 # Wrong length
    Tn_2 = 0 # True length but not correct

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
        if dequan:
            logits = logits.dequantize()
        # Calc loss
        input_lengths, target_lengths = sparse_tuple_for_ctc(args.T_length, lengths)
        log_probs = logits.permute(2, 0, 1).log_softmax(2)
        loss += ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths).item()
        log_probs = log_probs.permute(1, 2, 0).cpu().detach().numpy()
        if args.mode == 'beam':
            pred_labels = beam_search(log_probs, topk=args.topk)
            
        else:
            pred_labels = greedy_search(log_probs)
        for i, label in enumerate(pred_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i], dtype=np.int8) == np.asarray(label, dtype=np.int8)).all():
                Tp += 1
            else:
                Tn_2 += 1

    acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2 + 1e-6)

    return acc, loss

def predict(model, dataset, args: argparse.Namespace, dequan=False):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    loader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    all_predicts, all_targets = [], [] 
    for (images, _, _) in loader:
        # load train data
        images = images.to(device)
        model = model.to(device)
        # Forward
        logits = model(images)
        if dequan:
            logits = logits.dequantize()
        # Calc loss
        probs = logits.softmax(1).squeeze()
        probs.cpu().detach().numpy()
        if args.mode == 'beam':
            pred_labels = beam_search(probs, topk=args.topk)
        else:
            pred_labels = greedy_search(probs)
        all_predicts.extend(pred_labels)
    return all_predicts

def eval(predicts, targets):
    Tp, Tn_1, Tn_2 = 0, 0, 0
    for i, label in enumerate(predicts):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i], dtype=np.int8) == np.asarray(label, dtype=np.int8)).all():
                Tp += 1
            else:
                Tn_2 += 1
    acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2 + 1e-6)
    phase = "Validation"
    print("[INFO] {} accuracy: {} [{}:{}:{}:{}]".format(phase, acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    return acc
    
def reduce_seq(pred_label):
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
    return no_repeat_blank_label
            
def greedy_search(log_probs):
    pred_labels = list()
    for i in range(log_probs.shape[0]):
        pred = log_probs[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        
        pred_labels.append(reduce_seq(pred_label))
    
    return pred_labels

def beam_search(log_probs, topk=3):
    pred_labels = list()
    T_len = log_probs.shape[2]
    for i in range(log_probs.shape[0]):
        pred = log_probs[i, :, :]
        pred_space = list()
        for j in range(T_len):
            ids = np.argsort(pred[:, j])[-topk:]
            pred_space.append([idx for idx in sorted(list(ids))])
        # Find the sequence with maximum probabilities
        dp = np.zeros((topk, T_len), dtype=np.float32)
        trace = np.zeros_like(dp, dtype=np.int8)
        dp[:, 0] = np.asarray([pred[idx, 0] for idx in pred_space[0]])
        for t in range(1, T_len):
            max_probs = np.asarray([dp[pre_k,t-1] for pre_k in range(topk)])
            for k in range(topk):
                dp[k, t] = np.max(max_probs) + pred[pred_space[t][k], t]
                trace[k, t] = np.argmax(max_probs)
        pred_label = list()
        last_idx = np.argmax(dp[:, -1])
        pred_label.append(pred_space[T_len-1][last_idx])
        t = T_len - 1
        while t > 0:
            last_idx = trace[last_idx, t]
            pred_label.append(pred_space[t-1][last_idx])
            t = t - 1
        pred_labels.append(reduce_seq(pred_label[::-1]))
    return pred_labels

def decode_label(lst):
    return ''.join([CHARS[i] for i in lst])

def show(model, dataset, args):
    encode_path = lambda path: f"{path[19]}/{path[21]}/{path[-11:]}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = len(dataset.track_path)
    ids = []
    for i in range(4):
        ids += random.sample(range(i*1000, i*1000+1000), 2)
    imgs, labels = [], []
    for idx in ids:
        for k in range(5):
            img, label, _ = dataset[idx*5+k]
            imgs.append(img)
            labels.append(label)
    imgs = torch.stack(imgs, dim=0).to(device)
    model = model.to(device)
    logits = model(imgs)
    log_probs = logits.permute(2, 0, 1).log_softmax(2).permute(1, 2, 0).cpu().detach().numpy()
    pred_labels = beam_search(log_probs) if args.mode == "beam" else greedy_search(log_probs)
    
    fig, axes = plt.subplots(len(ids), 5, figsize=(7, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).cpu().detach().numpy())
        ax.set_title(f"{encode_path(dataset.track_path[ids[i//5]])}\n Label: {decode_label(labels[i])}\n Predict: {decode_label(pred_labels[i])}", fontsize=8)
        ax.axis(False)
    plt.tight_layout()
    plt.savefig('predict_samples.png')
