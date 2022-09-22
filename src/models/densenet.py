# the code in this file is intended to provide the DenseNet model architecture

# general libraries
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from typing import Tuple, Sequence
from torch import Tensor


def set_parameter_requires_grad(network: object,
                                feature_extracting: bool) -> None:
    """
    Switches off requires_grad for the parameters of the network if featute_extracting == True.

    :param network: Network object
    :param feature_extracting: Activate feature extracting (switch-off requires_grad)
    """
    if feature_extracting:
        for param in network.parameters():
            param.requires_grad = False



def initialize_model(model_name: str,
                     num_classes: int,
                     feature_extract: bool,
                     use_pretrained: bool = True) -> Tuple[object, int]:
    """
    Downloads and provides the model architecture of the specified model.

    :param model_name: Name of model, only implemented for 'densenet'
    :param num_classes: Number of classes to predict
    :param feature_extract: Used to call set_parameter_requires_grad() and adjust final layer structure
    :param use_pretrained: Include pre-trained weights
    :return: Network model object, input size
    """

    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "densenet":
        """ Densenet-121
        """
        model_ft = models.densenet121(pretrained=use_pretrained, progress=True, drop_rate=0.2)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if feature_extract:
            model_ft.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        raise NotImplementedError

    return model_ft, input_size


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


    out = self.features.pool0(self.features.relu0((self.features.norm0(self.features.conv0(x)))))
    if layer_index == 1:
        out = self.features.transition1(self.features.denseblock1(out))
    if layer_index == 2:
        out = self.features.transition1(self.features.denseblock1(out))
        out = self.features.transition2(self.features.denseblock2(out))
    if layer_index == 3:
        out = self.features.transition1(self.features.denseblock1(out))
        out = self.features.transition2(self.features.denseblock2(out))
        out = self.features.transition3(self.features.denseblock3(out))
    if layer_index == 4:
        out = self.features.transition1(self.features.denseblock1(out))
        out = self.features.transition2(self.features.denseblock2(out))
        out = self.features.transition3(self.features.denseblock3(out))
        out = F.relu(self.features.norm5(self.features.denseblock4(out)), inplace=True)
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
    out = self.features.pool0(self.features.relu0((self.features.norm0(self.features.conv0(x)))))
    out_list.append(out)
    out = self.features.transition1(self.features.denseblock1(out))
    out_list.append(out)
    out = self.features.transition2(self.features.denseblock2(out))
    out_list.append(out)
    out = self.features.transition3(self.features.denseblock3(out))
    out_list.append(out)
    out = F.relu(self.features.norm5(self.features.denseblock4(out)), inplace=True)
    out_list.append(out)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)

    return self.classifier(out), out_list


def feature_list_sizes(self: object) -> Sequence:
    """
    Computes sizes of features based on random input.
    Length of this list provides number of features of network.

    :param self: Object instance
    :return: Sequence of sizes of features
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [out.size(1) for out in self.feature_list(torch.rand(2,3,224,224).to(device))[1]]
