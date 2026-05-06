from torchinfo import summary
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
            nn.BatchNorm2d(num_features=ch_out),
        )

    def forward(self, x):
        return self.block(x)


class SmallLPRNet(nn.Module):
    """A lightweight LPRNet variant with a single forward flow.

    This version removes the multi-scale global_context aggregation branch
    and predicts logits directly from the final feature map.
    """

    def __init__(self, lpr_max_len, train, class_num, dropout_rate):
        super(SmallLPRNet, self).__init__()
        self.phase = train
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            small_basic_block(ch_in=64, ch_out=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=(1, 1)),

            small_basic_block(ch_in=128, ch_out=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)),

            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(9, 1), stride=(1, 1)),
        )

    def forward(self, x):
        x = self.backbone(x)
        logits = torch.squeeze(x)
        return logits


def build_small_lprnet(lpr_max_len=7, train=False, class_num=37, dropout_rate=0.5):
    net = SmallLPRNet(lpr_max_len, train, class_num, dropout_rate)
    if train:
        return net.train()
    return net.eval()


class QuantizedSmallLPRNet(nn.Module):
    """Quantization small LPRNet model
    """

    def __init__(self, lpr_max_len, train, class_num, dropout_rate):
        super(QuantizedSmallLPRNet, self).__init__()
        self.phase = train
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.quan = torch.quantization.QuantStub()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            small_basic_block(ch_in=64, ch_out=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=(1, 1)),

            small_basic_block(ch_in=128, ch_out=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)),

            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(9, 1), stride=(1, 1)),
        )
        self.de_quan = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quan(x)
        x = self.backbone(x)
        logits = torch.squeeze(x)
        x = self.de_quan(x)
        return logits
    
def build_quan_small_lprnet(lpr_max_len=7, train=False, class_num=37, dropout_rate=0.5):
    net = QuantizedSmallLPRNet(lpr_max_len, train, class_num, dropout_rate)
    if train:
        return net.train()
    return net.eval()


if __name__ == "__main__":
    model = build_small_lprnet()
    summary(
        model=model,
        input_size=(1, 3, 40, 100),
        col_names=["input_size", "output_size", "trainable"],
    )
