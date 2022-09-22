# this code is intended to provide all functions needed to tune the Softmax thresholds

# general libraries
import src.data.make_data as md
import src.utils.pipe as pipe
import src.config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
import numpy as np
from typing import Tuple, Sequence, Any, Dict, Callable
import src.utils.utils as utils

# get config
config = src.config.cfg

def tune_gen_AI_threshold(main_network: object,
                          unc_score: str,
                          experts: Sequence[object] = []) -> float:
    """
    Tunes the threshold for the Softmax allocation mechanism (A1).
    By design, tuning is conducted on validation data of 1000 known and 600 unknown samples

    :param main_network: Network object of general ML model
    :param unc_score: Uncertainty score, only implemented for 'conf' (Softmax)
    :param experts: sequence of artificial experts, needed to compute pipeline
    :return: threshold
    """

    # initialize
    outloader_type = config.args.pipe_tune_dataset
    selec_mech_cache = config.args.selec_mech
    comb_mech_cache = config.args.comb_mech
    unc_score_cache = config.args.unc_score

    # get data
    if outloader_type == 'pipe':
        pipeloader, pipe_true_labels, inds = md.gen_pipe_data(1000, [0.2, 0.2, 0.2], split='val')

    # in case iSUN is used instead of validation data
    elif outloader_type == 'iSUN':
        _, inloader, _ = md.get_data('0')
        inset = Subset(inloader.dataset, [i for i in range(5000) if i % 5 == 0])
        intruelabels = []
        for i in range(len(inset)):
            intruelabels.append(inloader.dataset.classes[inset[i][1]])

        transform_isun = transforms.Compose([
            transforms.Resize(len(main_network.trainloader.dataset[0][0][0])),
            transforms.ToTensor(),
            transforms.Normalize([0.4861, 0.4633, 0.4275], [0.2331, 0.2343, 0.2441])
        ])
        outset = ImageFolder(config.args.pipe_root + 'data/raw/iSUN/', transform=transform_isun)
        outset = Subset(outset, [i for i in range(600)])
        outtruelabels = ['iSUN'] * len(outset)

        pipeloader = DataLoader(ConcatDataset([inset, outset]), batch_size=1, shuffle=False, num_workers=2)
        pipe_true_labels = intruelabels + outtruelabels


    perf = 0

    # tune Softmax
    if unc_score == 'conf':
        config.args.selec_mech = 'unc_thresh'
        config.args.comb_mech = 'min_uncertain'
        config.args.unc_score = 'conf'
        print(config.args.unc_score + ' thresh is tuned for case ' + config.args.pipe_case + ' on ' + outloader_type + ' (dimension: ' + config.args.domain + ')')
        if config.args.pipe_case == 'benchmark':
            thresholds = [0.950, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995,
                          0.996, 0.997, 0.998, 0.999]
        elif config.args.pipe_case == 'sat':
            #thresholds = list(i/100 for i in range(70, 89))
            thresholds = [0.91, 0.92, 0.93, 0.935, 0.94, 0.945, 0.950, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99]

        # run val_pipeline several times
        for idx, i in enumerate(thresholds):
            # enforce reproducibility
            main_seed = config.args.main_seed
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)
            print('Threshold ' + str(idx+1) + ' of ' + str(len(thresholds)))
            # select items
            if config.args.pipe_case == 'benchmark':
                if outloader_type == 'pipe':
                    config.args.conf_thresh_benchmark_wideresnet_pipe_wFMNIST = i
                elif outloader_type == 'iSUN':
                    config.args.conf_thresh_benchmark_wideresnet_iSUN_wFMNIST = i
            elif config.args.pipe_case == 'sat':
                config.args.conf_thresh_sat = i
            # run pipeline
            predloader, pred_indices, forwardloader, forward_indices = pipe.select_instances(main_network, pipeloader)
            # do prediction and get step1 accuracy
            pred_step1 = pipe.pipe_pred(main_network, predloader)
            pipe_true_labels_step1 = list(np.array(pipe_true_labels)[pred_indices])
            # call experts and get final accuracy
            pipe_true_labels_step2 = list(np.array(pipe_true_labels)[forward_indices])
            pred_step2, _ = pipe.call_experts(experts, forwardloader, true_labels_call_experts=pipe_true_labels_step2)
            pred_pipe = pred_step1 + pred_step2
            pipe_true_labels_tot = pipe_true_labels_step1 + pipe_true_labels_step2
            perf_try = pipe.pipe_eval(pred_pipe, pipe_true_labels_tot)[0]
            print('try: ' + str(perf_try))

            if perf_try > perf:
                perf = perf_try
                best_threshold = i


    config.args.selec_mech = selec_mech_cache
    config.args.comb_mech = comb_mech_cache
    config.args.unc_score = unc_score_cache

    return best_threshold


def tune_all_gen_AI_thresholds(gen_AI: object,
                               experts: Sequence[object] = []) -> None:
    """
    Tunes all implemented thresholds (currently, only Softmax implemented)

    :param gen_AI: Network objects
    :param experts: sequence of artificial experts, needed to compute pipeline
    """

    outloader_type = config.args.pipe_tune_dataset
    size_cache = config.args.pipe_size
    config.args.pipe_size = 1000

    # tuning needs to be activated
    if config.args.pipe_tune_thresholds:
        # tuning
        best_th_conf_thresh = tune_gen_AI_threshold(main_network=gen_AI, experts=experts, unc_score='conf')
        if config.args.pipe_case == 'benchmark':

            # assign thresholds
            if outloader_type == 'pipe':
                config.args.conf_thresh_benchmark_wideresnet_pipe_wFMNIST = best_th_conf_thresh
                print('conf_thresh_benchmark_wideresnet_pipe_wFMNIST: ' + str(best_th_conf_thresh))
            elif outloader_type == 'iSUN':
                config.args.conf_thresh_benchmark_wideresnet_iSUN_wFMNIST = best_th_conf_thresh
                print('conf_thresh_benchmark_wideresnet_iSUN_wFMNIST: ' + str(best_th_conf_thresh))
        elif config.args.pipe_case == 'sat':
            config.args.conf_thresh_sat = best_th_conf_thresh
            print('conf_thresh_sat: ' + str(best_th_conf_thresh))

    else:
        print('Threshold tuning already conducted. Respective thresholds are provided in config.py')


    config.args.pipe_size = size_cache


def tune_all_gen_AI_thresholds_dyn(network_list: Sequence[object],
                                   valloader: DataLoader,
                                   val_true_labels: Sequence[str],
                                   acc_target: float) -> None:
    """
    Tunes the Softmax threshold for all models in network_list.
    Only applicable to single-domain AIITL-system.

    :param network_list: Sequence of networks to be tuned
    :param valloader: DataLoader of validation data
    :param val_true_labels: True labels to evaluate quality
    :param acc_target: desired accuracy targeted for when deploying the Softmax allocation mechanism
    """

    # initialize
    selec_mech_cache = config.args.selec_mech
    comb_mech_cache = config.args.comb_mech
    unc_score_cache = config.args.unc_score
    assert len(network_list) == len(acc_target)
    perf = 0
    config.args.selec_mech = 'unc_thresh'
    config.args.comb_mech = 'min_uncertain'
    config.args.unc_score = 'conf'

    thresholds = [0.950, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9992, 0.9994, 0.9996, 0.9998]

    # run val_pipeline for all models
    for idx_network, network in enumerate(network_list):

        if not hasattr(network, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                config.args.conf_thresh_singledom_cache = config.args.conf_thresh_singledom
            elif config.args.pipe_case == 'sat':
                config.args.conf_thresh_singledom_sat_cache = config.args.conf_thresh_singledom_sat

        # run val_pipeline for all thresholds
        for idx, i in enumerate(thresholds):
            # enforce reproducibility
            main_seed = config.args.main_seed
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)
            print('Threshold ' + str(idx+1) + ' of ' + str(len(thresholds)))
            # select items
            if config.args.pipe_case == 'benchmark':
                config.args.conf_thresh_singledom = i
            else:
                config.args.conf_thresh_singledom_sat = i

            # run pipeline, always with HITL (no expert, just for simplicity)
            pipeloader = valloader
            pipe_true_labels_tot = val_true_labels

            predloader, pred_indices, forwardloader, forward_indices = pipe.select_instances(network, pipeloader)

            # classify instances and get gen AI accuracy
            pred_step1 = pipe.pipe_pred(network, predloader)
            pipe_true_labels_step1 = list(np.array(pipe_true_labels_tot)[pred_indices])

            pred_step2 = list(np.array(pipe_true_labels_tot)[forward_indices])

            pred_pipe_wH = pred_step1 + pred_step2
            pipe_true_labels_wH = pipe_true_labels_step1 + pred_step2

            # get final accuracy, matrix and human effort
            perf_try = pipe.pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[0]
            print('try: ' + str(perf_try))

            if perf_try >= acc_target[idx_network]:
                best_threshold = i
                break

        # assign tuned parameters
        if hasattr(network, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                config.args.conf_thresh_singledom = best_threshold
                print('conf_thresh_singledom: ' + str(best_threshold))
            elif config.args.pipe_case == 'sat':
                config.args.conf_thresh_singledom_sat = best_threshold
                print('conf_thresh_singledom_sat: ' + str(best_threshold))
        else:
            config.args.conf_thresh_singledom_exp = best_threshold
            print('conf_thresh_singledom_exp: ' + str(best_threshold))
            if config.args.pipe_case == 'benchmark':
                config.args.conf_thresh_singledom = config.args.conf_thresh_singledom_cache
            elif config.args.pipe_case == 'sat':
                config.args.conf_thresh_singledom_sat = config.args.conf_thresh_singledom_sat_cache

    config.args.selec_mech = selec_mech_cache
    config.args.comb_mech = comb_mech_cache
    config.args.unc_score = unc_score_cache
