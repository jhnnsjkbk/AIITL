# the code in this file is intended to provide the ResNet model architecture
# Code is based on https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/networks/wideresnet.py

# general libraries
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import torch
from typing import Tuple, Sequence, Union
from torch import Tensor

_bn_momentum = 0.1


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1) -> nn.Conv2d:
    """
    Creation of Convolutional layer. Please refer to `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_ for technical details.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class WideBasic(nn.Module):
    """
    Provides the WideBasic Block architecture for the WideResNet model as suggested in
     `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_.
    """

    def __init__(self: object,
                 in_planes: int,
                 planes: int,
                 dropout_rate: float,
                 stride: int = 1) -> None:
        """
        Initialization of WideBasic Block of WideResNet architecture. Please refer to `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_ for technical details.
        """

        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self: object,
                x: Sequence[Tensor]) -> Sequence[Tensor]:
        """
        Forward function to compute output logits based on input image.

        :param x: Sequence of tensors of input image
        :return: Sequence of tensors of output logits
        """
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """
    Provides the WideResNet architecture as suggested in
     `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_.
    """

    def __init__(self: object,
                 depth: int,
                 widen_factor: int,
                 dropout_rate: float,
                 num_classes: int) -> None:
        """
        Initialization of WideResNet architecture. Please refer to `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_ for technical details.
        """

        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)
        self.linear = nn.Linear(nStages[3], num_classes)


    def _wide_layer(self: object,
                    block: WideBasic,
                    planes: int,
                    num_blocks: int,
                    dropout_rate: float,
                    stride: int) -> Sequence:
        """
        Creation of layers. Please refer to `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_ for technical details.
        """

        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)


    def forward(self: object,
                x: Sequence[Tensor]) -> Sequence[Tensor]:
        """
        Forward function to compute output logits based on input image.

        :param x: Sequence of tensors of input image
        :return: Sequence of tensors of output logits
        """

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    # maha functions

    def feature_list(self: object,
                     x: Sequence[Tensor]) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        """
        Computes all intermediate outputs and adds them to a list.

        :param self: Object instance.
        :param x: Input image.
        :return: Ouput of network, Sequence of all intermediate forward outputs
        """

        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out_list.append(out)
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out), out_list


    def intermediate_forward(self: object,
                             x: Sequence[Tensor],
                             layer_index: int) -> Sequence[Tensor]:
        """
        Computes the intermediate output of the forward function up to the specified layer.
        This function needs to be added to an Object class.

        :param self: Object instance.
        :param x: Input image.
        :param layer_index: Number of layer
        :return: Output (logits)
        """

        out = self.conv1(x)
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
        return out


    def feature_list_sizes(self: object) -> Sequence:
        """
        Computes sizes of features based on random input.
        Length of this list provides number of features of network.

        :param self: Object instance
        :return: Sequence of sizes of features
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return [out.size(1) for out in self.feature_list(torch.rand(2,3,32,32).to(device))[1]]
