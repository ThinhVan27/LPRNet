import torch.nn as nn
import torch

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, train, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = train
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0, 64x28x58
            nn.BatchNorm2d(num_features=64), # 64x28x58
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)), # 64x26x56
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 *** # 128x26x56
            nn.BatchNorm2d(num_features=128), # 128x26x56
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)), # 64x24x27
            small_basic_block(ch_in=64, ch_out=256),   # 8# 256x24x27
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 *** 256x24x27
            nn.BatchNorm2d(num_features=256),   # 12 
            nn.ReLU(), # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 1)),  # 14 64x22x25
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 4), stride=1),  # 16 256x19x22
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(16, 1), stride=1), # 20 37x4x22
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        pool_lst = [
            nn.AvgPool2d(kernel_size=(6, 15), stride=(6, 2)),
            nn.AvgPool2d(kernel_size=(6, 13), stride=(6, 2)),
            nn.AvgPool2d(kernel_size=(5, 6), stride=(5, 1))
        ]
        for i, f in enumerate(keep_features):
            if i < 3:
                f = pool_lst[i](f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

def build_lprnet(lpr_max_len=7, train=False, class_num=37, dropout_rate=0.5):

    Net = LPRNet(lpr_max_len, train, class_num, dropout_rate)

    if train == True:
        return Net.train()
    else:
        return Net.eval()
