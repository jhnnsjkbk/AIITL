# this code is intended to provide all functions needed for the ODIN allocation mechanism
# major parts of the code are based on https://github.com/facebookresearch/odin

# general libraries
import torch
import torch.nn as nn
import numpy as np
import src.config
import src.config
import src.data.make_data as md
from torch.utils.data import DataLoader
import src.utils.utils as utils
import time
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from typing import Tuple, Any, Dict, Callable, Sequence
import pandas as pd

# get config
config = src.config.cfg

def get_odin_score(network: object,
                   testloader: DataLoader,
                   temper: int,
                   eps: float) -> Sequence[float]:
    """
    Compute ODIN scores for the testloader samples.

    :param network: Network object
    :param testloader: DataLoader of samples considered
    :param temper: Temperature scaling parameter
    :param eps: Input perturbation parameters
    :return: Odin scores
    """

    # initialize
    criterion = nn.CrossEntropyLoss()
    network.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scores = []

    print('ODIN scores are computed.')

    for idx, (data, target) in enumerate(testloader):
        data = data.to(device)
        data.requires_grad = True
        outputs = network(data)
        _, labels = torch.max(outputs.data, 1)

        # temp scaling
        scaled_outputs = outputs / temper

        # get gradient
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(data.data, -eps, gradient)
        tempOutputs = network(Variable(tempInputs))
        tempOutputs = tempOutputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = tempOutputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        scores.append(np.max(nnOutputs, axis=1)[0])

    print('Done.')

    return scores


def tune_odin_params(inNetwork: object,
                     tpr_target: float) -> Tuple[float, float, float, float]:
    """
    Tunes ODIN hyperparameters to find best eps for respective delta (95% TPR). Temperature set to 1000 for all experiments.

    :param inNetwork: Network object, whose data is to be tuned
    :return: best_eps, resp_delta, resp_score (FNR), resp_tpr
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get data
    if config.args.pipe_case == 'benchmark':
        # get data based on in-distribution dataset key
        inloader, outloader = md.get_odin_tune_data(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                                index(inNetwork.trainloader.dataset.name)])
    elif config.args.pipe_case == 'sat':
        # get data based on in-distribution dataset key
        inloader, outloader = md.get_odin_tune_data(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                                   index(inNetwork.trainloader.dataset.name)])
    y_true = np.concatenate((np.ones(len(inloader), dtype="int32"), np.zeros(len(outloader), dtype="int32")))   # in distribution set to 1 --> targeted for

    # initialize grids
    eps = torch.linspace(0, 0.004, 21).tolist()
    if config.args.pipe_case == 'benchmark':
        if inNetwork.trainloader.dataset.name == 'CIFAR100':
            delta = [i/1000000 for i in range(10000, 20000)]
        else:
            delta = [i/1000000 for i in range(100000, 200000)]
    elif config.args.pipe_case == 'sat':
        if inNetwork.trainloader.dataset.name == 'FMOW_public':
            delta = [i/1000000 for i in range(100000)]
        elif inNetwork.trainloader.dataset.name == 'FMOW_private':
            delta = [i/1000000 for i in range(100000)]
        elif inNetwork.trainloader.dataset.name == 'FMOW' or inNetwork.trainloader.dataset.name == 'AID' or inNetwork.trainloader.dataset.name == 'RESISC':
            delta = [i/1000000 for i in range(100000)]
        else:
            delta = [i/1000000 for i in range(100000, 200000)]
    scores = np.ones((len(eps), 4))

    # gridsearch for eps
    for idx, i in enumerate(eps):
        print('eps ' + str(idx+1) + '..')
        start = time.time()

        in_odin_score = get_odin_score(inNetwork, inloader, config.args.odin_temper, i)
        out_odin_score = get_odin_score(inNetwork, outloader, config.args.odin_temper, i)
        combined_odin_scores = np.concatenate((in_odin_score, out_odin_score))

        for d_idx, d in enumerate(delta):
            # simulate prediction as would have resulted for a given delta
            y_pred = []
            for j in range(len(y_true)):
                if combined_odin_scores[j] > d:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            # get the delta that results in a TPR of 95% --> end d gridsearch
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp/(tp+fn)
            if tpr <= tpr_target:
                fpr = fp/(fp+tn)
                scores[idx, 0] = i
                scores[idx, 1] = d
                scores[idx, 2] = fpr
                scores[idx, 3] = tpr
                break

        now = time.time()
        dur = now-start
        print('Duration for iteration ' + str(idx+1) + ': ' + utils.format_time(dur))


    # report best eps and respective delta that minimizes FPR
    minIndex = np.argmin(scores, axis=0)[2]

    best_eps = scores[minIndex, 0]
    resp_delta = scores[minIndex, 1]
    resp_score = scores[minIndex, 2]
    resp_tpr = scores[minIndex, 3]

    return best_eps, resp_delta, resp_score, resp_tpr


def tune_all_odin(networks: Sequence[object]) -> None:
    """
    Function to tune ODIN for all the networks considered.

    :param networks: Sequence of networks considered
    """

    # tuning needs to be activated
    if config.args.pipe_tune_thresholds:
        odin_results = pd.DataFrame(columns=[model.trainloader.dataset.name for model in networks], index=['best_eps', 'resp_delta', 'resp_score', 'resp_tpr'])
        for model in networks:
            best_eps, resp_delta, resp_score, resp_tpr = tune_odin_params(model, 0.95)
            odin_result = [best_eps, resp_delta, resp_score, resp_tpr]
            odin_results[model.trainloader.dataset.name] = odin_result
            print('ODIN results:')
            print(odin_results[[model.trainloader.dataset.name]])

            # set parameters to respective config arguments
            if not config.args.dyn_pipe:
                if config.args.pipe_case == 'benchmark':

                    if config.args.pipe_tune_dataset == 'iSUN':
                        if model.trainloader.dataset.name == 'CIFAR10':
                            config.args.c10_wideresnet_odin_delta_tpr95_iSUN = resp_delta
                            config.args.c10_wideresnet_odin_eps_tpr95_iSUN = best_eps
                        elif model.trainloader.dataset.name == 'SVHN':
                            config.args.svhn_wideresnet_odin_delta_tpr95_iSUN = resp_delta
                            config.args.svhn_wideresnet_odin_eps_tpr95_iSUN = best_eps
                        elif model.trainloader.dataset.name == 'MNIST':
                            config.args.mnist_wideresnet_odin_delta_tpr95_iSUN = resp_delta
                            config.args.mnist_wideresnet_odin_eps_tpr95_iSUN = best_eps
                        elif model.trainloader.dataset.name == 'FMNIST':
                            config.args.fmnist_wideresnet_odin_delta_tpr95_iSUN = resp_delta
                            config.args.fmnist_wideresnet_odin_eps_tpr95_iSUN = best_eps
                    elif config.args.pipe_tune_dataset == 'pipe':
                        if model.trainloader.dataset.name == 'CIFAR10':
                            config.args.c10_wideresnet_odin_delta_tpr95_pipe = resp_delta
                            config.args.c10_wideresnet_odin_eps_tpr95_pipe = best_eps
                        elif model.trainloader.dataset.name == 'SVHN':
                            config.args.svhn_wideresnet_odin_delta_tpr95_pipe = resp_delta
                            config.args.svhn_wideresnet_odin_eps_tpr95_pipe = best_eps
                        elif model.trainloader.dataset.name == 'MNIST':
                            config.args.mnist_wideresnet_odin_delta_tpr95_pipe = resp_delta
                            config.args.mnist_wideresnet_odin_eps_tpr95_pipe = best_eps
                        elif model.trainloader.dataset.name == 'FMNIST':
                            config.args.fmnist_wideresnet_odin_delta_tpr95_pipe = resp_delta
                            config.args.fmnist_wideresnet_odin_eps_tpr95_pipe = best_eps

                elif config.args.pipe_case == 'sat':
                    if model.trainloader.dataset.name == 'Euro_SAT_countryside':
                        config.args.eurosat_odin_delta_tpr95 = resp_delta
                        config.args.eurosat_odin_eps_tpr95 = best_eps
                    elif model.trainloader.dataset.name == 'FMOW':
                        config.args.fmow_odin_delta_tpr95 = resp_delta
                        config.args.fmow_odin_eps_tpr95 = best_eps
                    elif model.trainloader.dataset.name == 'AID':
                        config.args.aid_odin_delta_tpr95 = resp_delta
                        config.args.aid_odin_eps_tpr95 = best_eps
                    elif model.trainloader.dataset.name == 'RESISC':
                        config.args.resisc_odin_delta_tpr95 = resp_delta
                        config.args.resisc_odin_eps_tpr95 = best_eps

        print('ODIN results:')
        print(odin_results)

    else:
        print('ODIN tuning already conducted. Respective parameters are provided in config.py')


def tune_dyn_odin_params(inNetwork: object,
                         inloader: DataLoader,
                         outloader: DataLoader,
                         tpr_target: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Function to tune ODIN for the network considered in the dynamic system.

    :param inNetwork: Network object
    :param inloader: DataLoader of in-distribution data
    :param outloader: Dataloader of out-of-distribtion data
    :param tpr_target: Desired TPR target
    :return: best_eps, resp_delta, resp_score (FNR), resp_tpr
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    y_true = np.concatenate((np.ones(len(inloader), dtype="int32"), np.zeros(len(outloader), dtype="int32")))   # in distribution set to 1 --> targeted for

    # initialize grids
    if config.args.dyn_MVP_test:
        eps = [0.001]
    else:
        eps = torch.linspace(0, 0.004, 21).tolist()
    if config.args.pipe_case == 'benchmark':
        delta = [i/1000000 for i in range(100000, 200000)]
    elif config.args.pipe_case == 'sat':
        delta = [i/1000000 for i in range(1000000)]

    scores = np.ones((len(eps), 4))

    for idx, i in enumerate(eps):
        print('eps ' + str(idx+1) + '..')
        start = time.time()

        in_odin_score = get_odin_score(inNetwork, inloader, config.args.odin_temper, i)
        out_odin_score = get_odin_score(inNetwork, outloader, config.args.odin_temper, i)
        combined_odin_scores = np.concatenate((in_odin_score, out_odin_score))

        for d_idx, d in enumerate(delta):
            # simulate prediction as would have resulted for a given delta
            y_pred = []
            for j in range(len(y_true)):
                if combined_odin_scores[j] > d:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            # get the delta that results in a TPR of 95% --> end d gridsearch
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp/(tp+fn)
            if tpr <= tpr_target:
                fpr = fp/(fp+tn)
                scores[idx, 0] = i
                scores[idx, 1] = d
                scores[idx, 2] = fpr
                scores[idx, 3] = tpr
                break

        now = time.time()
        dur = now-start
        print('Duration for iteration ' + str(idx+1) + ': ' + utils.format_time(dur))


    # report best eps and respective delta that minimizes FPR
    minIndex = np.argmin(scores, axis=0)[2]

    best_eps = scores[minIndex, 0]
    resp_delta = scores[minIndex, 1]
    resp_score = scores[minIndex, 2]
    resp_tpr = scores[minIndex, 3]

    # assign odin scores
    if config.args.dyn_multi_tune_dataset == 'pipe':
        if hasattr(inNetwork, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                config.args.c10_wideresnet_odin_delta_tpr95_pipe = resp_delta
                config.args.c10_wideresnet_odin_eps_tpr95_pipe = best_eps
            else:
                config.args.eurosat_odin_delta_tpr95 = resp_delta
                config.args.eurosat_odin_eps_tpr95 = best_eps
        else:
            if inNetwork.name == 'exp_AI_1_dyn':
                config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe = resp_delta
                config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe = best_eps
            elif inNetwork.name == 'exp_AI_2_dyn':
                config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe = resp_delta
                config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe = best_eps
            elif inNetwork.name == 'exp_AI_3_dyn':
                config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe = resp_delta
                config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe = best_eps
    else:
        if hasattr(inNetwork, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                config.args.c10_wideresnet_odin_delta_tpr95_iSUN = resp_delta
                config.args.c10_wideresnet_odin_eps_tpr95_iSUN = best_eps
            else:
                config.args.eurosat_odin_delta_UCM = resp_delta
                config.args.eurosat_odin_eps_UCM = best_eps
        else:
            if inNetwork.name == 'exp_AI_1_dyn':
                config.args.exp_AI_1_wideresnet_odin_delta_tpr95_ext = resp_delta
                config.args.exp_AI_1_wideresnet_odin_eps_tpr95_ext = best_eps
            elif inNetwork.name == 'exp_AI_2_dyn':
                config.args.exp_AI_2_wideresnet_odin_delta_tpr95_ext = resp_delta
                config.args.exp_AI_2_wideresnet_odin_eps_tpr95_ext = best_eps
            elif inNetwork.name == 'exp_AI_3_dyn':
                config.args.exp_AI_3_wideresnet_odin_delta_tpr95_ext = resp_delta
                config.args.exp_AI_3_wideresnet_odin_eps_tpr95_ext = best_eps

    print('ODIN results:')
    print('best eps: ' + str(best_eps))
    print('best delta: ' + str(resp_delta))
    print('TPR: ' + str(resp_tpr))
    print('FPR: ' + str(resp_score))

    return best_eps, resp_delta, resp_score, resp_tpr
