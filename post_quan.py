#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml
import torch.ao.quantization
from torchinfo import summary

from data.data_augment import test_transforms
from data.load_data import CHARS, LPRDataset
from model.LPRNet import build_lprnet
from model.small_LPRNet import build_small_lprnet, build_quan_small_lprnet

from utils import predict, eval, decode, quan_predict


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_test_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return SimpleNamespace(**cfg)


def build_model_and_load(weights_path: str, args: SimpleNamespace, device: torch.device) -> nn.Module:
    model = build_quan_small_lprnet(
        lpr_max_len=args.lpr_max_len,
        train=False,
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_test_dataset(args: SimpleNamespace) -> LPRDataset:
    return LPRDataset(
        root=args.test_img_dirs,
        train=False, # Chú ý: Ở đây test dataset nên để train=False
        lpr_max_len=args.lpr_max_len,
        transform=test_transforms,
        max_samples=-1
    )


def eval_model(model: nn.Module, dataset: LPRDataset, args: SimpleNamespace, title: str, dequan: bool=False) -> dict:
    print(f"\n[INFO] ===== {title} =====")
    acc, loss = decode(model, dataset, args, dequan=dequan)
    return {"acc": acc, "loss": loss}


def main():
    set_seed(42)
    args = load_test_config(config_path="config/test_config.yaml")
    
    # 1. ÉP KIỂU CHẠY TRÊN CPU CHO QUÁ TRÌNH LƯỢNG TỬ HÓA
    device = torch.device("cpu")
    
    # Thiết lập backend qnnpack (Khuyến nghị cho phần cứng nhúng/FPGA/ARM)
    # torch.backends.quantized.engine = 'x86' 

    model = build_model_and_load(weights_path=args.pretrained_model,
                                 args=args,
                                 device=device)
    ds = get_test_dataset(args)
    
    # Đảm bảo model ở chế độ eval trước khi fuse và quantize
    model.eval()

    # 2. THỰC HIỆN FUSE MODULES (GỘP LỚP)
    # Cập nhật đường dẫn fuse theo kiến trúc QuantizedSmallLPRNet mới
    modules_to_fuse = [
        # Gộp Conv2d và BatchNorm ngoài backbone
        ['backbone.0', 'backbone.1'],
        ['backbone.13', 'backbone.14'],
        
        # Gộp Conv2d và BatchNorm bên TRONG các small_basic_block
        ['backbone.4.block.6', 'backbone.4.block.7'],
        ['backbone.7.block.6', 'backbone.7.block.7'],
        ['backbone.9.block.6', 'backbone.9.block.7'],
    ]
    torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

    # 3. CÀI ĐẶT QCONFIG CỤ THỂ
    model.qconfig = torch.ao.quantization.get_default_qconfig()
    
    # Prepare: Chèn các hàm Observer để thu thập dải phân phối dữ liệu
    model = torch.ao.quantization.prepare(model, inplace=True)
    
    # Calibration: Chạy suy luận qua toàn bộ tập Test để Observer tính scale/zero_point chính xác nhất
    print(eval_model(model, ds, args, "Calibrating Small LPRNet...", quan=True))
    
    # Convert: Lượng tử hóa mô hình sang INT8 dựa trên thông số Calibration
    quantized_model = torch.ao.quantization.convert(model, inplace=True)
    
    # # Lưu trọng số
    os.makedirs("weights/small_LPRNet", exist_ok=True)
    torch.save(quantized_model.state_dict(), "weights/small_LPRNet/quan.bin")
    
    quantized_model.load_state_dict(torch.load("weights/small_LPRNet/quan.bin", map_location=device, weights_only=True))
    
    # Đánh giá mô hình đã lượng tử hóa
    print(eval_model(quantized_model, ds, args, "Quantized small LPRNet", dequan=True))

if __name__ == "__main__":
    main()