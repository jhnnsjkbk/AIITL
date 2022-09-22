# the code in this file is intended to provide the ResNet model architecture
# Code is based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence,Union
from torch import Tensor


class BasicBlock(nn.Module):
    """
    Provides the Basic Block architecture for the ResNet model as suggested in
     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_.
    """

    expansion = 1

    def __init__(self: object,
                 in_planes: int,
                 planes: int,
                 stride: int = 1) -> None:
        """
        Initialization of Basic Block of ResNet architecture. Please refer to `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_ for technical details.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self: object,
                x: Sequence[Tensor]) -> Sequence[Tensor]:
        """
        Forward function to compute output logits based on input image.

        :param x: Sequence of tensors of input image
        :return: Sequence of tensors of output logits
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Provides the Bottleneck architecture for the ResNet model as suggested in
     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_.
    """

    expansion = 4

    def __init__(self: object,
                 in_planes: int,
                 planes: int,
                 stride: int = 1) -> None:
        """
        Initialization of Bottleneck of ResNet architecture. Please refer to `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_ for technical details.
        """


        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self: object,
                x: Sequence[Tensor]) -> Sequence[Tensor]:
        """
        Forward function to compute output logits based on input image.

        :param x: Sequence of tensors of input image
        :return: Sequence of tensors of output logits
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Provides the ResNet architecture as suggested in
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_.
    """

    def __init__(self: object,
                 block: Union[BasicBlock, Bottleneck],
                 num_blocks: int,
                 num_classes: int = 10) -> None:
        """
        Initialization of ResNet architecture. Please refer to `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_ for technical details.
        """

        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self: object,
                    block: Union[BasicBlock, Bottleneck],
                    planes: int,
                    num_blocks: int,
                    stride: int) -> Sequence:
        """
        Creation of layers. Please refer to `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_ for technical details.
        """

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self: object,
                x: Sequence[Tensor]) -> Sequence[Tensor]:
        """
        Forward function to compute output logits based on input image.

        :param x: Sequence of tensors of input image
        :return: Sequence of tensors of output logits
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    # maha functions

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
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        if layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        if layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        if layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out


    def feature_list(self: object,
                     x: Sequence[Tensor]) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        """
        Computes all intermediate outputs and adds them to a list.

        :param self: Object instance.
        :param x: Input image.
        :return: Ouput of network, Sequence of all intermediate forward outputs
        """

        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out), out_list


    def feature_list_sizes(self: object) -> Sequence:
        """
        Computes sizes of features based on random input.
        Length of this list provides number of features of network.

        :param self: Object instance
        :return: Sequence of sizes of features
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return [out.size(1) for out in self.feature_list(torch.rand(2,3,32,32).to(device))[1]]


def ResNet18() -> ResNet:
    """
    :return: ResNet 18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:
    """
    :return: ResNet 34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:
    """
    :return: ResNet 50 model
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101() -> ResNet:
    """
    :return: ResNet 101 model
    """
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152() -> ResNet:
    """
    :return: ResNet 152 model
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])


