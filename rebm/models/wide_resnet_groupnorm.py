# Modified from https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/wide_resnet.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.gn1(x))
        else:
            out = self.relu1(self.gn1(x))
        out = self.relu2(self.gn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetWithGroupNorm(nn.Module):
    """Based on code from https://github.com/yaodongyu/TRADES"""

    def __init__(
        self,
        depth=28,
        num_classes=10,
        widen_factor=10,
        sub_block1=False,
        dropRate=0.0,
        bias_last=True,
        one_class=False,
    ):
        super(WideResNetWithGroupNorm, self).__init__()
        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(
                n, nChannels[0], nChannels[1], block, 1, dropRate
            )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        # global average pooling and classifier
        self.gn1 = nn.GroupNorm(
            num_groups=16, num_channels=nChannels[3]
        )  # just added to try to aviod nans
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]
        self.one_class = one_class

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, labels=None):
        # print("self device", self.device)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.gn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)

        # one class training
        if self.one_class:
            return out[:, 0]

        # logits for specific labels
        if labels is not None:
            print(f"{out.shape=}")
            print(f"{labels.shape=}")
            out = out.gather(1, labels.unsqueeze(1)).squeeze(1)

        # self.check_for_nans()
        return out

    def check_for_nans(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaNs found in parameter: {name}")
