from torchinfo import summary
from model.LPRNet import build_lprnet
from data.load_data import *
import torch

from matplotlib import pyplot as plt

model = build_lprnet()
print(summary(model, input_size=(1, 3, 30, 60), col_names=['trainable', 'input_size', 'output_size']))

# ds = LPRDataset(root='data\\train')
# for i, (img, label, length) in enumerate(ds):
#     imgs = []
#     if length != 7:
#         print(f"[INFO] Track {i} | Label: {label} | Length: {length}")
#         plt.axis(False)
#         plt.title(f'LP Text: {''.join([CHARS[j] for j in label])}')
#         plt.imshow(img.permute(1, 2, 0))
    
# plt.show()
        