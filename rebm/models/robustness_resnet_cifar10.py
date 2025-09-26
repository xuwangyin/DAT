'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l - 1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, normalize_input: bool = True, use_batchnorm: bool = False):
        super(ResNet, self).__init__()

        self.augment_dim = 0
        self.num_classes = num_classes
        self.normalize_input = normalize_input
        self.use_batchnorm = use_batchnorm

        first_stride = 1

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=first_stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=first_stride)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, y=None, augment_labels=None):
        if self.normalize_input:
            # mean = torch.as_tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype,
            #                     device=x.device)
            # std = torch.as_tensor([0.2023, 0.1994, 0.2010], dtype=x.dtype,
            #                     device=x.device)
            # https://github.com/M4xim4l/InNOutRobustness/blob/d81d1d26e5ebc9193009e3d92bd67b5e01d6cfd6/utils/model_normalization.py#L28
            mean = torch.as_tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618], dtype=x.dtype,
                                device=x.device)
            std = torch.as_tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628], dtype=x.dtype,
                                device=x.device)            
            x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
            
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=False)
        out = self.pool(out)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        # if with_latent:
            # return final, pre_out
    
        if y is None:
            return final

        return final[torch.arange(final.size(0)), y]
    
    def train(self, mode=True):
        """
        Override the train method to optionally keep BatchNorm layers in evaluation mode.
        
        Args:
            mode (bool): Whether to set training mode for the model
        """
        super(ResNet, self).train(mode)
        if mode and not self.use_batchnorm:
            # Set all BatchNorm layers to eval mode even when the model is in training mode
            for submodule in self.modules():
                if 'batchnorm' in submodule.__class__.__name__.lower():
                    submodule.train(False)
        return self


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wd=1.5, **kwargs)


def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wd=.75, **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet50Wide(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], wm=1.5, **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
