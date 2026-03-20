from torchinfo import summary
import torch.nn as nn
import torch

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(),
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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0, 64x38x98
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)), # 64x36x96
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 *** 128x36x96
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)), # 64x34x47
            small_basic_block(ch_in=64, ch_out=256),   # 8 256x34x47
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),  # 14
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 *** 256x34x47
            nn.BatchNorm2d(num_features=256),   # 12
            nn.LeakyReLU(), # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14 64x32x23
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16 256x32x20
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=(2,1)), # 20 # 37x10x20
            nn.BatchNorm2d(num_features=class_num),
            nn.LeakyReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )
        
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.full(size=(c, ), fill_value=20.0)) for c in [64, 128, 256, 37]
        ])

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        #-----Feature map shapes-----
        # 2: 64x38x98
        # 6: 128x36x96
        # 13: 256x34x47
        # 22: 37x10x20
        for i, f in enumerate(keep_features):
            if i in [0,1]:
                f = nn.AvgPool2d(kernel_size=(9, 19), stride=(3, 4))(f)
            if i == 2:
                f = nn.AvgPool2d(kernel_size=(7, 9), stride=(3, 2))(f)
            f_pow = torch.pow(f, 2)
            f_norm = torch.sqrt(torch.sum(f_pow, dim=1, keepdim=True)) + 1e-10
            f = torch.mul(torch.div(f, f_norm), self.gammas[i].view(1,-1,1,1))
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

if __name__ == "__main__":
    model = build_lprnet()
    summary(model=model,
            input_size=(1, 3, 40, 100),
            col_names=['input_size', 'output_size', 'trainable'])
