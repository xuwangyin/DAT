import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# A helper module that applies LayerNorm to a 4D tensor.
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        # We normalize over the channel dimension only.
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x has shape (N, C, H, W); we permute it to (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        # Permute back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        # Replace BatchNorm2d with LayerNorm2d
        self.ln1 = LayerNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.ln2 = LayerNorm2d(out_planes)
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
            x = self.relu1(self.ln1(x))
            out = self.conv1(x)
        else:
            out = self.relu1(self.ln1(x))
            out = self.conv1(out)
        out = self.relu2(self.ln2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return shortcut + out


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
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetNoBatchnorm(nn.Module):
    """Based on code from https://github.com/RobustBench/robustbench"""

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
        super(WideResNetNoBatchnorm, self).__init__()
        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
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
        self.ln1 = LayerNorm2d(nChannels[3], eps=1e-5)  # replaced bn1 with ln1
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]
        self.one_class = one_class

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_val = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n_val))
            elif isinstance(m, (nn.BatchNorm2d, LayerNorm2d)):
                # Initialize affine parameters for normalization layers.
                if hasattr(m, "layer_norm"):
                    nn.init.constant_(m.layer_norm.weight, 1)
                    nn.init.constant_(m.layer_norm.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, labels=None):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.ln1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)

        if self.one_class:
            return out[:, 0]

        if labels is not None:
            out = out.gather(1, labels.unsqueeze(1)).squeeze(1)
        return out

    def check_for_nans(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaNs found in parameter: {name}")
