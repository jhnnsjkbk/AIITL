
# general libraries
import copy
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from typing import Tuple, Any, Dict, Callable, Sequence
import src.config
import src.data.make_data as md
import src.utils.utils as utils
import src.utils.model_building as mb
import src.utils.odin as odin
import src.utils.mahalanobis as maha
import torch.backends.cudnn as cudnn
import pickle
from sklearn.metrics import classification_report
import src.visualization.reports as rs
import collections
import time
import os
import mlflow
import src.utils.thresholds as th

# get config
config = src.config.cfg

def get_Softmax_prob(network: object,
                     testloader: DataLoader) -> Sequence[Sequence[float]]:
    """
    Computes Softmax probabilities for test data as classification uncertainty measure.

    :param network: Network object for Softmax function
    :param testloader: DataLoader of data for Softmax probabilities
    :return: Sequence of Softmax probabilities. Per test sample, a list of probabilities for all classes is computed
    """

    # prepare network
    network.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize uncertainty measures
    softmax = nn.Softmax(dim=config.args.softmax_dim)
    softmax_prop = []

    print('Softmax probabilities are retrieved.')

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            softmax_cache = softmax(output)
            softmax_prop.append(softmax_cache.tolist()[0])

    print("Done.")

    return softmax_prop

def get_unc_score(prob: Sequence[Sequence[float]],
                  unc_score: str) -> Sequence[float]:
    """
    Transforms the uncertainty output into a specific score to be used by the allocation mechanisms.

    :param prob: Sequence of probabilities per sample and class
    :param unc_score: str, 'conf' for Softmax
    :return: Sequence of score per sample
    """

    if unc_score == 'conf':

        # get max of prob
        score = np.max(np.array(prob), axis=1).tolist()

    return score


def select_instances(network: object,
                     testloader: DataLoader) -> Tuple[DataLoader, Sequence, DataLoader, Sequence]:
    """
    Implementation of ALLOCATION MECHANISM 1.
    Splits the provided DataLoader into the known and unknown samples based on allocation mechanism specified in config.py

    :param network: Network object
    :param testloader: DataLoader of data samples to be allocated
    :return: predloader, pred_indices, forwardloader, forward_indices
    """

    # initialize
    forward_indices = []


    # SOFTMAX
    if config.args.selec_mech == 'unc_thresh':

        if config.args.unc_score == 'conf':

            # get threshold
            if config.args.domain == 'multi':
                if config.args.pipe_case == 'benchmark':
                    if config.args.pipe_tune_dataset == 'pipe':
                        thresh = config.args.conf_thresh_benchmark_wideresnet_pipe_wFMNIST
                    elif config.args.pipe_tune_dataset == 'iSUN':
                        thresh = config.args.conf_thresh_benchmark_wideresnet_iSUN
                elif config.args.pipe_case == 'sat':
                    thresh = config.args.conf_thresh_sat
            elif config.args.domain == 'single':
                if config.args.pipe_case == 'benchmark':
                    thresh = config.args.conf_thresh_singledom
                elif config.args.pipe_case == 'sat':
                    thresh = config.args.conf_thresh_singledom_sat

        print('threshold: ' + str(thresh))

        # get uncertainties for provided testdata
        if config.args.unc_mech == 'S':
            prob = get_Softmax_prob(network, testloader)

        # get uncertainty scores
        unc_score = get_unc_score(prob, config.args.unc_score)

        # reject samples
        for batch_idx, (data, target) in enumerate(testloader):
            if unc_score[batch_idx] < thresh:
                forward_indices.append(batch_idx)


    # BOTTOM X
    if config.args.selec_mech == 'unc_bottom_x':

        # get bottom_x value
        bottom_x = config.args.bottom_x

        # get uncertainties for provided testdata
        if config.args.unc_mech == 'S':
            prob = get_Softmax_prob(network, testloader)

        # get uncertainty scores
        unc_score = get_unc_score(prob, config.args.unc_score)

        # select bottom x samples to reject
        if len(unc_score) == bottom_x:
            forward_indices = list(range(0, len(unc_score)))
        else:
            forward_indices = np.argpartition(unc_score, bottom_x)
            forward_indices = forward_indices[:bottom_x]
            forward_indices = np.sort(forward_indices).tolist()


    # ODIN
    if config.args.selec_mech == 'odin' or config.args.selec_mech == 'odin_ext':

        # adjust tune dataset based on type of ODIN
        if config.args.selec_mech == 'odin_ext':
            tune_dataset_cache = config.args.pipe_tune_dataset
            if config.args.pipe_case == 'benchmark':
                config.args.pipe_tune_dataset = 'iSUN'
            else:
                config.args.pipe_tune_dataset = 'UCM'

        delta, eps = utils.get_delta_eps(network, 'odin')

        print('')
        print('delta: ' + str(delta))
        print('eps: ' + str(eps))
        print('')

        if config.args.selec_mech == 'odin_ext':
            config.args.pipe_tune_dataset = tune_dataset_cache

        # get scores
        odin_scores = odin.get_odin_score(network, testloader, config.args.odin_temper, eps)

        # reject samples
        for idx, score in enumerate(odin_scores):
            if score <= delta:
                forward_indices.append(idx)


    # MAHA
    if config.args.selec_mech == 'maha':

        # load network's maha regressors
        lr = utils.get_maha_models(network)

        # get maha params
        delta, eps = utils.get_delta_eps(network, 'maha')

        print('delta: ' + str(delta))
        print('eps: ' + str(eps))
        print('')

        num_layers = len(network.feature_list_sizes())

        # initialize
        if config.args.dyn_pipe:
            if config.args.pipe_case == 'benchmark':
                num_classes = 10
            elif config.args.pipe_case == 'sat':
                if hasattr(network, 'trainloader'):
                    num_classes = len(np.unique(network.trainloader.dataset.classes))
                else:
                    if network.name == 'exp_AI_1_dyn':
                        num_classes = 35
                    elif network.name == 'exp_AI_2_dyn':
                        num_classes = 24
                    elif network.name == 'exp_AI_3_dyn':
                        num_classes = 12
        else:
            num_classes = len(np.unique(network.trainloader.dataset.classes))

        # compute layer-wise mahascores for testloader
        maha_scores_tot = np.zeros((len(testloader), num_layers))

        # reduce batch size temporarily
        batch_size_cache = utils.adjust_train_batch_size(network, 'reduce')

        # get trainloader
        if hasattr(network, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                working_batch_size = 64
            elif config.args.pipe_case == 'sat':
                working_batch_size = 32
            if config.args.pipe_case == 'benchmark':
                if config.args.dyn_pipe and network.trainloader.dataset.name != 'CIFAR10':
                    trainloader = DataLoader(network.trainloader.dataset, num_workers=2, shuffle=True, batch_size=working_batch_size)
                else:
                    trainloader, _, _ = md.get_data(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                                index(network.trainloader.dataset.name)])
            elif config.args.pipe_case == 'sat':
                if config.args.dyn_pipe and network.trainloader.dataset.name != 'Euro_SAT_countryside':
                    trainloader = DataLoader(network.trainloader.dataset, num_workers=2, shuffle=True, batch_size=working_batch_size)
                else:
                    trainloader, _, _ = md.get_data(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                                index(network.trainloader.dataset.name)])
        else:
            print('dummy gen AI trainloaders for dyn pipe experts.')
            # trainloader only needed as argument for maha_mean_cov, however, mean and cov are just loaded and not computed again
            trainloader, _, _ = md.get_data('0')

        # get mean, precision
        sample_class_mean, precision = maha.maha_mean_cov(network, num_classes, network.feature_list_sizes(), trainloader)

        # switch back batch size
        utils.adjust_train_batch_size(network, 'switch_back', batch_size_cache)

        # compute layer-wise maha scores
        for layer in range(num_layers):
            maha_score = maha.get_maha_layer_score(network, testloader, num_classes, sample_class_mean, precision, layer, eps)
            maha_scores_tot[:, layer] = maha_score

        # adjust scores by scaler - if applicable
        if hasattr(network, 'trainloader'):
            if network.trainloader.dataset.name == 'MNIST':
                scaler_path = config.args.pipe_root + 'models/benchmark/MAHA/MNIST_scaler_' + 'wideresnet' + '_' + config.args.pipe_tune_dataset + '_wFMNIST' + '.pickle'
                filehandler = open(scaler_path, 'rb')
                scaler = pickle.load(filehandler)
                maha_scores_tot = scaler.transform(maha_scores_tot)
            if network.trainloader.dataset.name == 'FMNIST':
                scaler_path = config.args.pipe_root + 'models/benchmark/MAHA/FMNIST_scaler_' + 'wideresnet' + '_' + config.args.pipe_tune_dataset + '.pickle'
                filehandler = open(scaler_path, 'rb')
                scaler = pickle.load(filehandler)
                maha_scores_tot = scaler.transform(maha_scores_tot)
            if config.args.pipe_case == 'sat':
                if config.args.dyn_pipe:
                    scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_DYN_scaler' + config.args.ablation_study + '.pickle'
                else:
                    if network.trainloader.dataset.name != 'Euro_SAT_countryside':
                        scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_scaler_' + config.args.pipe_tune_dataset + '.pickle'
                    else:
                        scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_scaler_' + config.args.pipe_tune_dataset + '.pickle'

                filehandler = open(scaler_path, 'rb')
                scaler = pickle.load(filehandler)
                maha_scores_tot = scaler.transform(maha_scores_tot)
        else:
            if network.name == 'exp_AI_2_dyn' and config.args.pipe_case == 'benchmark':
                scaler_path = config.args.pipe_root + 'models/benchmark/MAHA/exp_AI_2_scaler_' + config.args.ablation_study + 'wideresnet' + '_' + 'pipe' + '_wFMNIST' + '.pickle'
                filehandler = open(scaler_path, 'rb')
                scaler = pickle.load(filehandler)
                maha_scores_tot = scaler.transform(maha_scores_tot)
            if config.args.pipe_case == 'sat':
                scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.name + '_DYN_scaler' + config.args.ablation_study + '.pickle'
                filehandler = open(scaler_path, 'rb')
                scaler = pickle.load(filehandler)
                maha_scores_tot = scaler.transform(maha_scores_tot)

        if 'scaler_path' in locals() or 'scaler_path' in globals():
            print('')
            print('file path of scaler: ' + scaler_path)
            print('')

        # get maha score
        for layer in range(num_layers):
            maha_scores_tot[:, layer] = maha_scores_tot[:, layer] * lr.coef_[0][layer]
        maha_scores = list(np.sum(maha_scores_tot, axis=1))

        # reject samples
        forward_indices = []
        for idx, score in enumerate(maha_scores):
            if score <= delta:
                forward_indices.append(idx)


    # GATING
    if config.args.selec_mech == 'gating':
        print('Instances are selected based on gating model.')
        # load expert gating model
        if config.args.dyn_pipe:
            model_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/GATING/selec_mech_dyn_pipe_gating{}.pickle'.format(config.args.ablation_study)
        else:
            if config.args.pipe_case == 'benchmark':
                model_path = config.args.pipe_root + 'models/benchmark/GATING/C10vRestWFMNISTGatingModel.pickle'
            elif config.args.pipe_case == 'sat':
                model_path = config.args.pipe_root + 'models/sat/GATING/EUROSATvRestV2GatingModel.pickle'
        filehandler = open(model_path, 'rb')
        network = utils.Unpickler(filehandler).load()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network = network.module.to(device)
        if device == 'cuda':
            network = torch.nn.DataParallel(network)
            cudnn.benchmark = True

        # define instance ownership based on model prediction
        network.eval()
        instance_owner = np.zeros((len(testloader), 1))

        # change data preprocessing for gating model
        if config.args.domain == 'multi' and not config.args.dyn_pipe:
            test_set = testloader.dataset
            test_set_copy = copy.deepcopy(test_set)
            _ = utils.set_transform(test_set_copy, 'test')
            testloader_gating = DataLoader(test_set_copy, num_workers=2, shuffle=False, batch_size=1)
        else:
            testloader_gating = copy.deepcopy(testloader)

        # no need for gradients, predict allocations
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(testloader_gating):
                data, target = data.to(device), target.to(device)
                output = network(data)
                _, predicted = torch.max(output.data, 1)
                instance_owner[batch_idx, 0] = predicted.item()

        # check if all instances were handled
        not_all_included = 2 in instance_owner
        if not_all_included:
            raise ArithmeticError

        # reject samples
        forward_indices = list(np.where(instance_owner == 0)[0])



    # prepare output

    # rejected data
    forwardloader = DataLoader(Subset(testloader.dataset, forward_indices), batch_size=1, shuffle=False,
                               num_workers=2)

    # indices of assigned/kept data
    pred_indices = []
    for i in range(0, len(testloader)):
        if i not in forward_indices:
            pred_indices.append(i)
    if config.args.selec_mech == 'odin' or config.args.selec_mech == 'odin_ext' or config.args.selec_mech == 'maha':
        pred_indices_cache = {}
        pred_indices_cache['ind'] = pred_indices
        if config.args.selec_mech == 'odin' or config.args.selec_mech == 'odin_ext':
            pred_indices_cache[config.args.selec_mech + '_scores'] = np.array(odin_scores)[pred_indices]
        else:
            pred_indices_cache[config.args.selec_mech + '_scores'] = np.array(maha_scores)[pred_indices]
        pred_indices = pred_indices_cache
        pred_indices['delta'] = delta

    # assigned/kept data
    if config.args.selec_mech == 'odin' or config.args.selec_mech == 'odin_ext' or config.args.selec_mech == 'maha':
        predloader = DataLoader(Subset(testloader.dataset, pred_indices['ind']), batch_size=1, shuffle=False,
                            num_workers=2)
    else:
        predloader = DataLoader(Subset(testloader.dataset, pred_indices), batch_size=1, shuffle=False,
                                num_workers=2)



    return predloader, pred_indices, forwardloader, forward_indices


def pipe_pred(network: object,
              testloader: DataLoader) -> Sequence[str]:
    """
    Predicts data within the Ai-in-the-loop system using the unique class label names (strings).

    :param network: Network object
    :param testloader: DataLoader of data to be predicted
    :return: sequence of predictions as true class labels
    """

    # initialize values
    network.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get class names and create respective output
    if config.args.dyn_pipe:

        # general AI
        if hasattr(network, 'trainloader'):
            if network.trainloader.dataset.name == 'CIFAR10':
                result_classes = network.trainloader.dataset.classes
            elif network.trainloader.dataset.name == 'Euro_SAT_countryside':
                result_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
        # expert AIs
        else:
            if network.name == 'dyn_single':
                if config.args.pipe_case == 'benchmark':
                    all_classes = ['airplane-CIFAR10', 'automobile-CIFAR10', 'bird-CIFAR10', 'cat-CIFAR10', 'deer-CIFAR10', 'dog-CIFAR10', 'frog-CIFAR10', 'horse-CIFAR10', 'ship-CIFAR10', 'truck-CIFAR10']
                    no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat
                    result_classes = all_classes[no_strong_classes:]
                elif config.args.pipe_case == 'sat':
                    all_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
                    no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat
                    result_classes = all_classes[no_strong_classes:]
            if network.name == 'exp_AI_1_dyn':
                if config.args.pipe_case == 'benchmark':
                    result_classes = [str(s) + '-SVHN' for s in range(10)]
                else:
                    result_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

            elif network.name == 'exp_AI_2_dyn':
                if config.args.pipe_case == 'benchmark':
                    result_classes = [str(s) + '-MNIST' for s in range(10)]
                else:
                    result_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']

            elif network.name == 'exp_AI_3_dyn':
                if config.args.pipe_case == 'benchmark':
                    result_classes = ['T-shirt/top-FMNIST', 'Trouser-FMNIST', 'Pullover-FMNIST', 'Dress-FMNIST', 'Coat-FMNIST', 'Sandal-FMNIST',
                                            'Shirt-FMNIST', 'Sneaker-FMNIST', 'Bag-FMNIST', 'Ankle boot-FMNIST']
                else:
                    result_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']

        log_interval_fix = config.args.dyn_log_interval

    else:
        result_classes = network.trainloader.dataset.classes
        _, log_interval_fix = utils.get_epochs_loginterval(network.trainloader)

    if len(testloader) > 1:
        print('Predictions are computed.')
    pred = []

    # no need for gradients, compute predictions
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = network(data)
            _, predicted = torch.max(output.data, 1)
            pred.append(result_classes[predicted.item()])
            if len(testloader) > 1:
                if batch_idx % log_interval_fix == 0:
                    utils.progress_bar(batch_idx, len(testloader))

    return pred


def call_experts(networks: Sequence[object],
                 testloader: DataLoader,
                 true_labels_call_experts: Sequence[str] = None,
                 forward_ind: Sequence[int] = None) -> Tuple[Sequence[str], Dict]:
    """
    Implementation of ALLOCATION MECHANISM 2.
    Allocates the samples of the provided DataLoader to one of the networks or the human expert.

    :param networks: Sequence of artificial expert objects
    :param testloader: DataLoader of data samples to be allocated
    :param true_labels_call_experts: true labels needed to simulate human intervention
    :param forward_ind=None: forward indices are needed to get total human intervention index
    :return: pred, human_effort
    """

    # intialize
    tot_pipe_size = config.args.pipe_size
    for noise in config.args.pipe_noise:
        tot_pipe_size = tot_pipe_size + int(noise*config.args.pipe_size)

    if len(testloader) == 0:
        pred = []
        # default format of human effort
        human_effort = {
            '#_inst': 0,
            'human_ind_step2': [],
            'no_human_ind_step2': [],
            'human_ind_total': [],
            'no_human_ind_total': []
        }
        return pred, human_effort


    # SOFTMAX
    if config.args.comb_mech == 'min_uncertain':
        print('Experts are combined based on minimum uncertainty. Uncertainty score used: ' + config.args.unc_score)

        # compute uncertainties for all classifiers
        unc = {}
        for net in networks:
            if config.args.unc_mech == 'S':
                prob = get_Softmax_prob(net, testloader)
            else:
                raise NotImplementedError

            unc[net.trainloader.dataset.name] = prob
        keys = list(unc.keys())

        # get uncertainty scores per net and instance
        score = []
        pred = []
        for key in keys:
            if unc[key] == []:
                return pred
            else:
                score.append(get_unc_score(unc[key], config.args.unc_score))
        score = np.array(score)

        # get the network with maximum probability (minimum uncertainty) per instance
        net_argmax = list(np.argmax(score, axis=0))
        # get the class/label index of the chosen network
        class_argmax = []
        for i in range(0, len(net_argmax)):
            class_argmax.append(np.argmax(unc[keys[net_argmax[i]]][i]))
        # get the correct class label
        for i in range(0, len(net_argmax)):
            result_classes = networks[net_argmax[i]].trainloader.dataset.classes
            pred.append(result_classes[class_argmax[i]])

        human_effort = {
            '#_inst': 0,
            'human_ind_step2': [],
            'no_human_ind_step2': [idx for idx, i in enumerate(pred)],
            'human_ind_total': [],
            'no_human_ind_total': [i for i in range(tot_pipe_size)]
        }


    # ODIN
    if config.args.comb_mech == 'odin' or config.args.comb_mech == 'odin_ext':
        print('Experts are combined based on ODIN.')

        selection = collections.defaultdict(dict)

        # for each expert: call allocation mechanism 1 to define claims
        for net in networks:
            selec_mech_cache = config.args.selec_mech
            config.args.selec_mech = config.args.comb_mech
            if hasattr(net, 'trainloader'):
                name_for_selection = net.trainloader.dataset.name
            else:
                name_for_selection = net.name
            predloader, pred_indices, forwardloader, forward_indices = select_instances(net, testloader)
            config.args.selec_mech = selec_mech_cache
            selection[name_for_selection]['predloader'] = predloader
            selection[name_for_selection]['pred_indices'] = pred_indices['ind']
            if config.args.comb_mech == 'odin_ext':
                selection[name_for_selection]['pred_odin_scores'] = pred_indices['odin_ext_scores']
            else:
                selection[name_for_selection]['pred_odin_scores'] = pred_indices['odin_scores']
            selection[name_for_selection]['delta'] = pred_indices['delta']
            selection[name_for_selection]['forwardloader'] = forwardloader
            selection[name_for_selection]['forward_indices'] = forward_indices

        # for each sample, find and collect one-one relation with allocated agent (1: svhn, 2: mnist, 3:c100, 99: human)
        instance_owner = np.zeros((len(testloader), 1), dtype='int')

        # compare all 3 nets:
        # unique claims are to be predicted individually,
        # call human for duplicates/triplets in predloaders
        # call human for triplets in forwardloaders (samples not yet handled before)
        for net in networks:
            if hasattr(net, 'trainloader'):
                name_for_selection = net.trainloader.dataset.name
            else:
                name_for_selection = net.name

            #uniques in pred
            uniques = set(selection[name_for_selection]['pred_indices'])
            for other_net in networks:
                if net != other_net:
                    if hasattr(other_net, 'trainloader'):
                        name_for_selection_on = other_net.trainloader.dataset.name
                    else:
                        name_for_selection_on = other_net.name
                    uniques = uniques - set(selection[name_for_selection_on]['pred_indices'])
            uniques = list(uniques)
            for i in uniques:
                if name_for_selection == 'SVHN' or name_for_selection == 'FMOW_utilities' or name_for_selection == 'exp_AI_1_dyn' or name_for_selection == 'FMOW':
                    instance_owner[i, 0] = 1
                if name_for_selection == 'MNIST' or name_for_selection == 'FMOW_public' or name_for_selection == 'exp_AI_2_dyn' or name_for_selection == 'AID':
                    instance_owner[i, 0] = 2
                if name_for_selection == 'CIFAR100' or name_for_selection == 'FMNIST' or name_for_selection == 'FMOW_private' or name_for_selection == 'exp_AI_3_dyn' or name_for_selection == 'RESISC':
                    instance_owner[i, 0] = 3

            #dupl/tripl in pred
            for idx, j in enumerate(selection[name_for_selection]['pred_indices']):
                if j not in uniques:
                    instance_owner[j, 0] = 99

            #tripl in forward
            tripl = set(selection[name_for_selection]['forward_indices'])
            for other_net in networks:
                if net != other_net:
                    if hasattr(other_net, 'trainloader'):
                        name_for_selection_on = other_net.trainloader.dataset.name
                    else:
                        name_for_selection_on = other_net.name
                    tripl = tripl & set(selection[name_for_selection_on]['forward_indices'])
            tripl = list(tripl)

            for i in tripl:
                instance_owner[i, 0] = 99

        if len(networks) == 0:
            instance_owner[:, 0] = 99

        # check if all instances were handled
        not_all_included = 0 in instance_owner
        if not_all_included:
            raise ArithmeticError

        # predict subsets and concatenate based on instance number as key to keep everything in old order
        pred = np.empty((len(testloader)), dtype=object)
        for net in networks:
            if hasattr(net, 'trainloader'):
                name_for_selection = net.trainloader.dataset.name
            else:
                name_for_selection = net.name
            if config.args.pipe_case == 'benchmark':
                if config.args.dyn_pipe:
                    if name_for_selection == 'exp_AI_1_dyn':
                        net_number = 1
                    elif name_for_selection == 'exp_AI_2_dyn':
                        net_number = 2
                    elif name_for_selection == 'exp_AI_3_dyn':
                        net_number = 3
                else:
                    net_number = int(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                 index(net.trainloader.dataset.name)])
            elif config.args.pipe_case == 'sat':
                if config.args.dyn_pipe:
                    if name_for_selection == 'exp_AI_1_dyn':
                        net_number = 1
                    elif name_for_selection == 'exp_AI_2_dyn':
                        net_number = 2
                    elif name_for_selection == 'exp_AI_3_dyn':
                        net_number = 3
                else:
                    net_number = int(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                 index(net.trainloader.dataset.name)])
            idx, _ = np.where(instance_owner == net_number)
            pred_ind_unique = list(idx)
            subset_unique = Subset(testloader.dataset, pred_ind_unique)
            predloader_unique = DataLoader(subset_unique, batch_size=1, shuffle=False, num_workers=2)
            pred[pred_ind_unique] = pipe_pred(net, predloader_unique)

        # human "prediction"
        true_labels = np.array(true_labels_call_experts)
        human_idx = np.where(instance_owner == 99)[0]
        pred[list(human_idx)] = true_labels[list(human_idx)]
        pred = list(pred)

        # check whether all instances are handled
        not_all_included = None in pred
        if not_all_included:
            raise ArithmeticError

        human_effort = {
            '#_inst': len(list(np.where(instance_owner==99)[0])),
            'human_ind_step2': list(np.where(instance_owner==99)[0]),
            'no_human_ind_step2': [idx for idx, i in enumerate(pred) if idx not in list(np.where(instance_owner==99)[0])],
            'human_ind_total': list(np.array(forward_ind)[list(np.where(instance_owner==99)[0])]),
            'no_human_ind_total': [i for i in range(tot_pipe_size) if i not in list(np.array(forward_ind)[list(np.where(instance_owner==99)[0])])]
        }

    # MAHA
    if config.args.comb_mech == 'maha':
        print('Experts are combined based on Mahalanobis OOD detection.')
        # ownership for each net
        selection = collections.defaultdict(dict)
        selec_mech_cache = config.args.selec_mech
        config.args.selec_mech = 'maha'

        # for each expert: call allocation mechanism 1 to define claims
        for net in networks:
            if hasattr(net, 'trainloader'):
                name_for_selection = net.trainloader.dataset.name
            else:
                name_for_selection = net.name
            predloader, pred_indices, forwardloader, forward_indices = select_instances(net, testloader)
            selection[name_for_selection]['predloader'] = predloader
            selection[name_for_selection]['pred_indices'] = pred_indices['ind']

            selection[name_for_selection]['pred_maha_scores'] = pred_indices['maha_scores']
            selection[name_for_selection]['delta'] = pred_indices['delta']
            selection[name_for_selection]['forwardloader'] = forwardloader
            selection[name_for_selection]['forward_indices'] = forward_indices
        config.args.selec_mech = selec_mech_cache

        # for each sample, find and collect one-one relation with allocated agent (1: svhn, 2: mnist, 3:c100, 99: human)
        instance_owner = np.zeros((len(testloader), 1), dtype='int')

        # compare all 3 nets:
        # unique claims are to be predicted individually,
        # call human for duplicates/triplets in predloaders
        # call human for triplets in forwardloaders (samples not yet handled before)
        for net in networks:
            if hasattr(net, 'trainloader'):
                name_for_selection = net.trainloader.dataset.name
            else:
                name_for_selection = net.name

            #uniques in pred
            uniques = set(selection[name_for_selection]['pred_indices'])
            for other_net in networks:
                if net != other_net:
                    if hasattr(other_net, 'trainloader'):
                        name_for_selection_on = other_net.trainloader.dataset.name
                    else:
                        name_for_selection_on = other_net.name
                    uniques = uniques - set(selection[name_for_selection_on]['pred_indices'])
            uniques = list(uniques)
            for i in uniques:
                if name_for_selection == 'SVHN' or name_for_selection == 'FMOW_utilities' or name_for_selection == 'exp_AI_1_dyn' or name_for_selection == 'FMOW':
                    instance_owner[i, 0] = 1
                if name_for_selection == 'MNIST' or name_for_selection == 'FMOW_public' or name_for_selection == 'exp_AI_2_dyn' or name_for_selection == 'AID':
                    instance_owner[i, 0] = 2
                if name_for_selection == 'CIFAR100' or name_for_selection == 'FMNIST' or name_for_selection == 'FMOW_private' or name_for_selection == 'exp_AI_3_dyn' or name_for_selection == 'RESISC':
                    instance_owner[i, 0] = 3

            #dupl/tripl in pred
            for idx, j in enumerate(selection[name_for_selection]['pred_indices']):
                if j not in uniques:
                    instance_owner[j, 0] = 99

            #tripl in forward
            tripl = set(selection[name_for_selection]['forward_indices'])
            for other_net in networks:
                if net != other_net:
                    if hasattr(other_net, 'trainloader'):
                        name_for_selection_on = other_net.trainloader.dataset.name
                    else:
                        name_for_selection_on = other_net.name
                    tripl = tripl & set(selection[name_for_selection_on]['forward_indices'])
            tripl = list(tripl)

            for i in tripl:
                instance_owner[i, 0] = 99

        if len(networks) == 0:
            instance_owner[:, 0] = 99

        # check if all instances were handled
        not_all_included = 0 in instance_owner
        if not_all_included:
            raise ArithmeticError

        # predict subsets and concatenate based on instance number as key to keep everything in old order
        pred = np.empty((len(testloader)), dtype=object)
        for net in networks:
            if hasattr(net, 'trainloader'):
                name_for_selection = net.trainloader.dataset.name
            else:
                name_for_selection = net.name
            if config.args.pipe_case == 'benchmark':
                if config.args.dyn_pipe:
                    if name_for_selection == 'exp_AI_1_dyn':
                        net_number = 1
                    elif name_for_selection == 'exp_AI_2_dyn':
                        net_number = 2
                    elif name_for_selection == 'exp_AI_3_dyn':
                        net_number = 3
                else:
                    net_number = int(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                 index(net.trainloader.dataset.name)])
            elif config.args.pipe_case == 'sat':
                if config.args.dyn_pipe:
                    if name_for_selection == 'exp_AI_1_dyn':
                        net_number = 1
                    elif name_for_selection == 'exp_AI_2_dyn':
                        net_number = 2
                    elif name_for_selection == 'exp_AI_3_dyn':
                        net_number = 3
                else:
                    net_number = int(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                 index(net.trainloader.dataset.name)])
            idx, _ = np.where(instance_owner == net_number)
            pred_ind_unique = list(idx)
            subset_unique = Subset(testloader.dataset, pred_ind_unique)
            predloader_unique = DataLoader(subset_unique, batch_size=1, shuffle=False, num_workers=2)
            pred[pred_ind_unique] = pipe_pred(net, predloader_unique)

        # human "prediction"
        true_labels = np.array(true_labels_call_experts)
        human_idx, _ = np.where(instance_owner == 99)
        pred[list(human_idx)] = true_labels[list(human_idx)]
        pred = list(pred)

        # check whether all instances are handled
        not_all_included = None in pred
        if not_all_included:
            raise ArithmeticError

        human_effort = {
            '#_inst': len(list(np.where(instance_owner==99)[0])),
            'human_ind_step2': list(np.where(instance_owner==99)[0]),
            'no_human_ind_step2': [idx for idx, i in enumerate(pred) if idx not in list(np.where(instance_owner==99)[0])],
            'human_ind_total': list(np.array(forward_ind)[list(np.where(instance_owner==99)[0])]),
            'no_human_ind_total': [i for i in range(tot_pipe_size) if i not in list(np.array(forward_ind)[list(np.where(instance_owner==99)[0])])]
        }


    # GATING
    if config.args.comb_mech == 'gating':
        print('Experts are combined/called based on gating model.')
        # load expert gating model
        if config.args.dyn_pipe:
            model_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/GATING/dyn_pipe_gating{}.pickle'.format(config.args.ablation_study)
        else:
            if config.args.pipe_case == 'benchmark':
                model_path = config.args.pipe_root + 'models/benchmark/GATING/SVHNvMNISTvFMNISTGatingModel.pickle'
            elif config.args.pipe_case == 'sat':
                model_path = config.args.pipe_root + 'models/sat/GATING/EurosatvFMOWvAIDvRESISCModel.pickle'

        # intialize allocation
        instance_owner = np.zeros((len(testloader), 1))
        if len(networks) > 0:
            filehandler = open(model_path, 'rb')
            network = utils.Unpickler(filehandler).load()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            network = network.module.to(device)
            if device == 'cuda':
                network = torch.nn.DataParallel(network)
                cudnn.benchmark = True

            # define instance ownership based on model prediction (similar to above)
            network.eval()

            # change data preprocessing for gating model
            if config.args.domain == 'multi' and not config.args.dyn_pipe:
                test_set = testloader.dataset
                test_set_copy = copy.deepcopy(test_set)
                _ = utils.set_transform(test_set_copy, 'test')
                testloader_gating = DataLoader(test_set_copy, num_workers=2, shuffle=False, batch_size=1)
            else:
                testloader_gating = copy.deepcopy(testloader)

            # no need for gradients, predict allocations
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(testloader_gating):
                    data, target = data.to(device), target.to(device)
                    output = network(data)
                    _, predicted = torch.max(output.data, 1)

                    instance_owner[batch_idx, 0] = predicted.item()

        # for net in networks: predict own instances
        pred = np.empty((len(testloader)), dtype=object)
        for net in networks:
            if config.args.dyn_pipe:
                net_number = int(net.name[7])
            else:
                if config.args.pipe_case == 'benchmark':
                    net_number = int(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                     index(net.trainloader.dataset.name)])
                elif config.args.pipe_case == 'sat':
                    net_number = int(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                     index(net.trainloader.dataset.name)])
            idx, _ = np.where(instance_owner == net_number)
            pred_ind_unique = list(idx)
            subset_unique = Subset(testloader.dataset, pred_ind_unique)
            predloader_unique = DataLoader(subset_unique, batch_size=1, shuffle=False, num_workers=2)
            pred[pred_ind_unique] = pipe_pred(net, predloader_unique)


        # human predicts for not trusted experts and gen AI instances
        # get not covered ids:
        not_covered_ids = [0,1,2,3]
        covered_ids = []
        for net in networks:
            if config.args.dyn_pipe:
                id = int(net.name[7])
            else:
                if net.trainloader.dataset.name == 'SVHN' or net.trainloader.dataset.name == 'FMOW_utilities' or net.trainloader.dataset.name == 'FMOW':
                    id = 1
                elif net.trainloader.dataset.name == 'MNIST' or net.trainloader.dataset.name == 'FMOW_public' or net.trainloader.dataset.name == 'AID':
                    id = 2
                elif net.trainloader.dataset.name == 'FMNIST' or net.trainloader.dataset.name == 'FMOW_private' or net.trainloader.dataset.name == 'RESISC':
                    id = 3
            covered_ids.append(id)
        not_covered_ids = list(set(not_covered_ids) - set(covered_ids))

        for i in range(np.shape(instance_owner)[0]):
            if instance_owner[i, 0] in not_covered_ids:
                instance_owner[i, 0] = 99

        # human pred
        true_labels = np.array(true_labels_call_experts)
        human_idx, _ = np.where(instance_owner == 99)
        pred[list(human_idx)] = true_labels[list(human_idx)]
        pred = list(pred)

        # check whether all instances are handled
        not_all_included = None in pred
        if not_all_included:
            raise ArithmeticError

        human_effort = {
            '#_inst': len(list(np.where(instance_owner==99)[0])),
            'human_ind_step2': list(np.where(instance_owner==99)[0]),
            'no_human_ind_step2': [idx for idx, i in enumerate(pred) if idx not in list(np.where(instance_owner==99)[0])],
            'human_ind_total': list(np.array(forward_ind)[list(np.where(instance_owner==99)[0])]),
            'no_human_ind_total': [i for i in range(tot_pipe_size) if i not in list(np.array(forward_ind)[list(np.where(instance_owner==99)[0])])]
        }

    return pred, human_effort



def pipe_eval(pred: Sequence[str],
              true: Sequence[str]) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Computes several (weighted) metrics for predicted vs. true classes.

    :param pred: sequence of predictions with correct class label names to account for different datasets
    :param true: sequence of true label names
    :return: accuracy, precision, recall, f1_score, avg_auroc, avg_auprc, support, classification_rep
    """

    # average metrics
    if pred.__class__.__name__ == 'list':
        if len(pred) != len(true):
            return print('Warning: length of lists are not equal!')
        if pred == []:
            accuracy = 0
            precision = 0
            recall = 0
            f1_score = 0
            avg_auroc = 0
            avg_auprc = 0
            support = 0
            classification_rep = None
            true_inds = []
        else:
            classification_rep = classification_report(true, pred, output_dict=True, zero_division=0)
            accuracy = classification_rep['accuracy']
            precision = classification_rep['weighted avg']['precision']
            recall = classification_rep['weighted avg']['recall']
            f1_score = classification_rep['weighted avg']['f1-score']
            #avg_auroc = roc_auc_score(true, pred, average='weighted')
            avg_auroc = 1
            avg_auprc = 1
            #avg_auprc = average_precision_score(true, pred, average='weighted')
            support = classification_rep['weighted avg']['support']
            classification_rep = classification_report(true, pred, output_dict=False, zero_division=0)
            true_inds = [i for i, x in enumerate(true) if true[i] == pred[i]]

        return accuracy, precision, recall, f1_score, avg_auroc, avg_auprc, support, classification_rep, true_inds

    elif pred.__class__.__name__ == 'ndarray':
        if pred.size == 0:
            accuracy = 0
            precision = 0
            recall = 0
            f1_score = 0
            avg_auroc = 0
            avg_auprc = 0
            support = 0
            classification_rep = None
            true_inds = []
        else:
            classification_rep = classification_report(true, pred, output_dict=True, zero_division=0)
            accuracy = classification_rep['accuracy']
            precision = classification_rep['weighted avg']['precision']
            recall = classification_rep['weighted avg']['recall']
            f1_score = classification_rep['weighted avg']['f1-score']
            #avg_auroc = roc_auc_score(true, pred, average='weighted')
            avg_auroc = 1
            avg_auprc = 1
            #avg_auprc = average_precision_score(true, pred, average='weighted')
            support = classification_rep['weighted avg']['support']
            classification_rep = classification_report(true, pred, output_dict=False, zero_division=0)
            true_inds = [i for i, x in enumerate(true) if true[i] == pred[i]]

        return accuracy, precision, recall, f1_score, avg_auroc, avg_auprc, support, classification_rep, true_inds

    else:
        raise NotImplementedError


def run_pipeline(main_net: object,
                 experts: Sequence[object] = [],
                 metric: int = 0,
                 split: str = 'test') -> Tuple[float, float, float, float, float, pd.DataFrame, Dict, float, float, Dict, float,]:
    """
    Central implementation for the STATIC AIITL-system of both single- and multi-domain data.

    :param main_net: Network object of general ML model
    :param experts: sequence of artificial expert objects
    :param metric: 0: accuracy, ...
    :return: step1_perf (accuracy of general model on assigned samples, step2_perf_wH (accuracy of artificial experts after allocation mechanism 2 incl. human prediction),
    step2_perf_woH (accuracy of artificial experts after allocation mechanism 2 without human prediction), pipe_perf_wH (system accuracy with human), pipe_perf_woH (system accuracy without human),
    matrix (allocation matrix), human_effort (number of samples predicted by human),
    HITL_base_acc (accuracy for HITL-system baseline),
    HITL_acc_same_effort_result (system accuracy when same human effort as for AIITL-system is used for HITL-system baseline,
    exp_claimed_acc (accuracy of artificial experts on allocated data),
    global_model_HITL (accuracy of global model together with human effort of AIITl-system)
    """

    if config.args.domain == 'multi':

        # generate pipeline data (final batch of dynamic system)
        if config.args.pipe_size == 0 and config.args.pipe_noise == [0, 0, 0]:
            if config.args.dyn_batch_design == 'independent':
                raise NotImplementedError
            elif config.args.dyn_batch_design == 'dependent':
                if config.args.dyn_MVP_test:
                    no_batches = 1000
                else:
                    no_batches = config.args.dyn_no_batches
                if config.args.pipe_case == 'benchmark':
                    config.args.pipe_size = int(5000/(no_batches+1))
                    config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [78257, 65000, 65000]]
                elif config.args.pipe_case == 'sat':
                    config.args.pipe_size = int(1900/(no_batches+1))
                    config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [48565, 7164, 7560]]

        pipeloader, pipe_true_labels_tot, inds = md.gen_pipe_data(config.args.pipe_size, config.args.pipe_noise)

        # select instances (allocation mechanism 1)
        predloader, pred_indices, forwardloader, forward_indices = select_instances(main_net, pipeloader)


        # classify instances that are not selected to forward to experts
        pred_step1 = pipe_pred(main_net, predloader)
        if config.args.selec_mech == 'odin' or config.args.selec_mech == 'odin_ext' or config.args.selec_mech == 'maha':
            pipe_true_labels_step1 = list(np.array(pipe_true_labels_tot)[pred_indices['ind']])
        else:
            pipe_true_labels_step1 = list(np.array(pipe_true_labels_tot)[pred_indices])
        step1_perf = pipe_eval(pred_step1, pipe_true_labels_step1)[metric]
        pipe_true_labels_step2 = list(np.array(pipe_true_labels_tot)[forward_indices])

        # allocation mechanism 2
        pred_step2, human_effort = call_experts(experts, forwardloader, pipe_true_labels_step2, forward_indices)
        step2_perf_wH = pipe_eval(pred_step2, pipe_true_labels_step2)[metric]
        step2_perf_woH = pipe_eval(np.array(pred_step2)[human_effort['no_human_ind_step2']],
                                   np.array(pipe_true_labels_step2)[human_effort['no_human_ind_step2']])[metric]
        pred_pipe_wH = pred_step1 + pred_step2
        pred_pipe_woH = pred_step1 + list(np.array(pred_step2)[human_effort['no_human_ind_step2']])
        pipe_true_labels_wH = pipe_true_labels_step1 + pipe_true_labels_step2
        pipe_true_labels_woH = pipe_true_labels_step1 + list(np.array(pipe_true_labels_step2)[human_effort['no_human_ind_step2']])

        # get expert claimed acc
        exp_claimed_acc = {}
        for i in range(len(experts)):
            if i == 0:
                label = '-SVHN'
                if config.args.pipe_case == 'sat':
                    exp_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
            elif i == 1:
                label = '-MNIST'
                if config.args.pipe_case == 'sat':
                    exp_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
            else:
                label = '-FMNIST'
                if config.args.pipe_case == 'sat':
                    exp_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
            if config.args.pipe_case == 'benchmark':
                ind_of_expAI_pred = [True if i.endswith(label) else False for i in pred_pipe_woH]
            elif config.args.pipe_case == 'sat':
                ind_of_expAI_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
            expAI_pred = list(np.array(pred_pipe_woH)[ind_of_expAI_pred])
            expAI_true = list(np.array(pipe_true_labels_woH)[ind_of_expAI_pred])
            exp_claimed_acc['exp_AI_' + str(i+1) + '_claimed_acc'] = pipe_eval(expAI_pred, expAI_true)[metric]

        # get final accuracy
        pipe_perf_wH = pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[metric]
        pipe_perf_woH = pipe_eval(pred_pipe_woH, pipe_true_labels_woH)[metric]

        true_inds = pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[8]

        network_list = [main_net] + experts
        matrix = rs.allocation_matrix(network_list, pipe_true_labels_woH, pred_pipe_woH)

        # HITL base accuracy (general ML model predicts assigned samples and human predicts rest)
        HITL_base_pred = pred_step1 + pipe_true_labels_step2
        HITL_base_true = pipe_true_labels_wH
        HITL_base_acc = pipe_eval(HITL_base_pred, HITL_base_true)[0]

        #HITL_acc_same_effort = 0 (human helps with same effort as in AIITL-system)
        HITL_acc_same_effort_result = {}
        if config.args.pipe_HITL_same_effort:
            classes = main_net.trainloader.dataset.classes
            print('HITL same effort baseline is computed..')
            start = time.time()
            # Same human effort as in pipeline (assumption: optimally allocated only to OOD), rest general AI
            ind_of_human = []
            not_ind_of_human = []
            for i in range(len(pipe_true_labels_step2)):
                if human_effort['#_inst'] == 0:
                    break
                if not pipe_true_labels_step2[i] in classes:
                    ind_of_human.append(i)
                    if len(ind_of_human) > human_effort['#_inst']:
                        break
            for i in range(len(pipe_true_labels_step2)):
                if i not in ind_of_human:
                    not_ind_of_human.append(i)

            assert (len(ind_of_human) + len(not_ind_of_human)) == len(pipe_true_labels_step2)

            pred_human = list(np.array(pipe_true_labels_step2)[ind_of_human])

            gen_AI_loader = DataLoader(Subset(forwardloader.dataset, not_ind_of_human), batch_size=1, shuffle=False, num_workers=2)
            pred_gen_AI = pipe_pred(main_net, gen_AI_loader)

            HITL_acc_same_effort_pred = pred_step1 + pred_gen_AI + pred_human
            HITL_acc_same_effort_true = pipe_true_labels_step1 + list(np.array(pipe_true_labels_step2)[not_ind_of_human]) + pred_human
            HITL_acc_same_effort = pipe_eval(HITL_acc_same_effort_pred, HITL_acc_same_effort_true)[0]

            end = time.time()
            dur = start - end

            HITL_acc_same_effort_result['acc'] = HITL_acc_same_effort
            HITL_acc_same_effort_result['dur'] = dur

        # baseline of global model referring samples to the human expert with same effort as in AIITL-system
        global_model_HITL = {}
        if config.args.global_model_HITL_same_effort:

            print('Global model + HITL (same effort) baseline is computed..')
            start = time.time()

            bottom_x_cache = config.args.bottom_x
            config.args.bottom_x = human_effort['#_inst']
            selec_mech_cache = config.args.selec_mech
            config.args.selec_mech = 'unc_bottom_x'

            # single domain select instances
            predloader, pred_indices, forwardloader, forward_indices = select_instances(main_net, pipeloader)

            # predict predloader with global model
            global_pred = mb.pred_global(main_net, experts[0], experts[1], experts[2], predloader)
            global_true_label = list(np.array(pipe_true_labels_tot)[pred_indices])

            # rest is true label (human prediction)
            human_pred_true = list(np.array(pipe_true_labels_tot)[forward_indices])

            # get accuracy
            global_model_HITL['acc'] = pipe_eval(global_pred + human_pred_true, global_true_label + human_pred_true)[metric]

            end = time.time()
            dur = start - end

            global_model_HITL['dur'] = dur
            config.args.bottom_x = bottom_x_cache
            config.args.selec_mech = selec_mech_cache



        return step1_perf, step2_perf_wH, step2_perf_woH, pipe_perf_wH, pipe_perf_woH, matrix, human_effort, HITL_base_acc, HITL_acc_same_effort_result, exp_claimed_acc, global_model_HITL, true_inds, predloader, forwardloader



    elif config.args.domain == 'single':

        # generate data (first batch of dynamic system)
        batches = md.gen_train_batches('single')
        pipeloader, pipe_true_labels_tot = batches[0]  # static pipeline is run on very first dynamic batch

        _ = utils.set_transform(pipeloader.dataset, 'test')

        # get uncertainty and refer all uncertain instances to human
        assert config.args.selec_mech == 'unc_thresh'
        predloader, pred_indices, forwardloader, forward_indices = select_instances(main_net, pipeloader)

        # classify instances and get gen AI accuracy
        pred_step1 = pipe_pred(main_net, predloader)
        pipe_true_labels_step1 = list(np.array(pipe_true_labels_tot)[pred_indices])
        step1_perf = pipe_eval(pred_step1, pipe_true_labels_step1)[metric]

        pred_step2 = list(np.array(pipe_true_labels_tot)[forward_indices])

        pred_pipe_wH = pred_step1 + pred_step2
        pipe_true_labels_wH = pipe_true_labels_step1 + pred_step2

        # get final accuracy, matrix and human effort
        pipe_perf_wH = pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[metric]
        network_list = [main_net]
        matrix = rs.allocation_matrix(network_list, pipe_true_labels_step1, pred_step1, pipe_true_labels_tot)
        human_effort = len(forward_indices) / len(pipeloader.dataset)

        return step1_perf, pipe_perf_wH, matrix, human_effort


def run_dyn_pipeline(gen_AI: object, 
                     metric: int = 0, 
                     run_name: str = '', 
                     exp_name: str = '') -> Dict:
    """
    Central implementation of the DYNAMIC AIITL-system of both single- and multi-domain data.
        
    :param gen_AI: general ML model object
    :param metric: 0: accuracy, ... 
    :param run_name: name of mlflow run
    :param exp_name: name of mlflow experiment to log metrics
    :return: Dictionary of result metrics/statistics
    """

    # check that MAHA is never tuned on external data
    assert not (config.args.dyn_multi_tune_dataset == 'iSUN' and config.args.dyn_multi_selec_mech == 'maha')
    assert not (config.args.dyn_multi_tune_dataset == 'iSUN' and config.args.dyn_multi_comb_mech == 'maha')
    assert not (config.args.dyn_multi_tune_dataset == 'UCM' and config.args.dyn_multi_comb_mech == 'maha')
    assert not (config.args.dyn_multi_tune_dataset == 'UCM' and config.args.dyn_multi_comb_mech == 'maha')

    num_experts = config.args.dyn_num_experts

    # delete existing maha cov/mean files, if pipeline not resumed (so, if started from the very beginning)
    if config.args.dyn_resume == '':
        if config.args.dyn_multi_selec_mech == 'maha' or config.args.dyn_multi_comb_mech == 'maha':
            for id in range(num_experts):
                mean_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/MAHA/' + 'exp_AI_' + str(id+1) + '_dyn' + config.args.ablation_study + (('_' + 'wideresnet') if config.args.pipe_case == 'benchmark' else '') + '_mean.pickle'
                cov_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/MAHA/' + 'exp_AI_' + str(id+1) + '_dyn' + config.args.ablation_study + (('_' + 'wideresnet') if config.args.pipe_case == 'benchmark' else '') + '_cov.pickle'
                try:
                    os.remove(mean_path)
                    os.remove(cov_path)
                except OSError:
                    pass

    # initialization
    trust_thresh = config.args.dyn_trust_thresh
    config.args.dyn_pipe = True
    gen_AI_exp_name = 'gen_AI_exp_trustLevel' + str(trust_thresh)

    print('Dynamic pipeline started for ' + config.args.domain + ' domain.')

    if config.args.domain == 'multi':
        config.args.dyn_checkpoint = 'models/' + config.args.pipe_case + '/DYN_PIPE/MULTI'
    elif config.args.domain == 'single':
        config.args.dyn_checkpoint = 'models/' + config.args.pipe_case + '/DYN_PIPE/SINGLE'


    # get dynamic data batches
    train_batches = md.gen_train_batches(config.args.domain)

    # intialize datasets iteratively increased throughout system
    gen_AI_train_data = Subset(gen_AI.trainloader.dataset, [])
    gen_AI_test_data = Subset(gen_AI.trainloader.dataset, [])
    multi_exp_dataset = {}
    if config.args.domain == 'multi':
        for id in range(num_experts):
            data, _, _ = md.get_data(str(id+1))
            multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data'] = Subset(data.dataset, [])
            multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data'] = Subset(data.dataset, [])
            multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data'] = Subset(data.dataset, [])
    exp_data = {}

    # initiliaze baselines
    gen_AI_only = []
    gen_AI_improvement = []
    perf_exp_allocation = []
    perf_exp_allocation_afterselec = []
    gen_AI_HITL = []
    gen_AI_HITL_afterselec = []

    # initialize regular metrics
    step1_perf = []
    pipe_perf = []
    human_effort = []
    batch_forward_indices = []
    exp_claimed_acc = {}
    trust_scores = {}
    exp_test_acc = {}
    exp_test_acc_strong = {}
    exp_coverage = {}
    trust_batchid_exp = {}
    start_batch = 0
    trained_experts = {}
    maha_params = {}
    odin_params = {}
    dataset_sizes = {}
    maha_params['gen_AI_maha_eps'] = 0.0
    maha_params['gen_AI_maha_delta'] = 0.0
    odin_params['gen_AI_odin_eps'] = 0.0
    odin_params['gen_AI_odin_delta'] = 0.0
    gating_score = 0
    gating_trust_batchid = 9999

    # trust score early stopping
    max_delta = config.args.dyn_trust_max_delta
    patience = config.args.dyn_trust_patience

    # initialize further metrics
    if config.args.domain == 'multi':
        if config.args.pipe_case == 'benchmark':
            config.args.pipe_tune_dataset = 'iSUN'
        else:
            config.args.pipe_tune_dataset = 'UCM'
        config.args.selec_mech = 'odin'
        config.args.comb_mech = config.args.dyn_multi_comb_mech

        for id in range(num_experts):
            exp_data['exp_AI_' + str(id+1) + '_score'] = 0
            trust_scores['exp_AI_' + str(id+1) + '_trustscores'] = []
            exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'] = []
            exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'] = []
            exp_coverage['exp_AI_' + str(id+1) + '_coverage'] = []
            trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] = 9999
            maha_params['exp_AI_' + str(id+1) + '_maha_eps'] = 0.0
            maha_params['exp_AI_' + str(id+1) + '_maha_delta'] = 0.0
            odin_params['exp_AI_' + str(id+1) + '_odin_eps'] = 0.0
            odin_params['exp_AI_' + str(id+1) + '_odin_delta'] = 0.0
            dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'] = []
            dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'] = []
            dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'] = []

        # load information if resume is active
        if config.args.dyn_resume != '':
            # Load checkpoint.
            print('')
            print('==> Resuming from checkpoint..')
            if os.path.isfile(config.args.pipe_root + config.args.dyn_resume):
                resume_file = config.args.pipe_root + config.args.dyn_resume
            else:
                resume_file = os.path.join(config.args.pipe_root + config.args.dyn_resume, 'checkpoint_dyn_multi.pth.tar')
            assert os.path.isfile(resume_file), 'Error: no checkpoint directory found!'
            checkpoint_save = os.path.dirname(resume_file)
            checkpoint_info = torch.load(resume_file)
            start_batch = checkpoint_info['train_batch']
            step1_perf = checkpoint_info['step1_perf']
            pipe_perf = checkpoint_info['pipe_perf']
            human_effort = checkpoint_info['human_effort']
            batch_forward_indices = checkpoint_info['batch_forward_indices']
            trust_scores = checkpoint_info['trust_scores']
            exp_claimed_acc = checkpoint_info['exp_claimed_acc']
            exp_test_acc = checkpoint_info['exp_test_acc']
            exp_data = checkpoint_info['exp_data']
            exp_coverage = checkpoint_info['exp_coverage']
            maha_params = checkpoint_info['maha_params']
            odin_params = checkpoint_info['odin_params']
            trust_batchid_exp = checkpoint_info['trust_batchid_exp']
            gating_score = checkpoint_info['gating_score']
            gating_trust_batchid = checkpoint_info['gating_trust_batchid']
            for id in range(num_experts):
                if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] > start_batch:
                    exp_data['exp_AI_' + str(id+1) + '_score'] = exp_data['exp_AI_' + str(id+1) + '_score']
                else:
                    exp_data['exp_AI_' + str(id+1) + '_score'] = 1.0
                    model_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/DYN_PIPE/' + config.args.domain.upper() + '/' + 'exp_AI_' + str(id+1) + '.pickle'
                    #filehandler = open(model_path, 'rb')
                    trained_experts['exp_AI_' + str(id+1)] = utils.load_gpu_pickle(model_path)
                    #trained_experts['exp_AI_' + str(id+1)] = pickle.load(filehandler)
                    
                    if config.args.dyn_multi_comb_mech == 'maha':
                        # assign maha params of trusted experts
                        if id == 0:
                            print('exp AI 1 maha params are assigned.')
                            config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe = maha_params['exp_AI_' + str(id+1) + '_maha_delta']
                            config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe = maha_params['exp_AI_' + str(id+1) + '_maha_eps']
                            print('gen AI maha params are assigned.')
                            if config.args.pipe_case == 'benchmark':
                                config.args.c10_wideresnet_maha_delta_pipe_wFMNIST = maha_params['gen_AI_maha_delta']
                                config.args.c10_wideresnet_maha_eps_pipe_wFMNIST = maha_params['gen_AI_maha_eps']
                            else:
                                config.args.eurosat_maha_delta = maha_params['gen_AI_maha_delta']
                                config.args.eurosat_maha_eps = maha_params['gen_AI_maha_eps']
                            config.args.selec_mech = config.args.dyn_multi_selec_mech
                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                        elif id == 1:
                            print('exp AI 2 maha params are assigned.')
                            config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe = maha_params['exp_AI_' + str(id+1) + '_maha_delta']
                            config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe = maha_params['exp_AI_' + str(id+1) + '_maha_eps']
                            print('gen AI maha params are assigned.')
                            if config.args.pipe_case == 'benchmark':
                                config.args.c10_wideresnet_maha_delta_pipe_wFMNIST = maha_params['gen_AI_maha_delta']
                                config.args.c10_wideresnet_maha_eps_pipe_wFMNIST = maha_params['gen_AI_maha_eps']
                            else:
                                config.args.eurosat_maha_delta = maha_params['gen_AI_maha_delta']
                                config.args.eurosat_maha_eps = maha_params['gen_AI_maha_eps']
                            config.args.selec_mech = config.args.dyn_multi_selec_mech
                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                        else:
                            print('exp AI 3 maha params are assigned.')
                            config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe = maha_params['exp_AI_' + str(id+1) + '_maha_delta']
                            config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe = maha_params['exp_AI_' + str(id+1) + '_maha_eps']
                            print('gen AI maha params are assigned.')
                            if config.args.pipe_case == 'benchmark':
                                config.args.c10_wideresnet_maha_delta_pipe_wFMNIST = maha_params['gen_AI_maha_delta']
                                config.args.c10_wideresnet_maha_eps_pipe_wFMNIST = maha_params['gen_AI_maha_eps']
                            else:
                                config.args.eurosat_maha_delta = maha_params['gen_AI_maha_delta']
                                config.args.eurosat_maha_eps = maha_params['gen_AI_maha_eps']
                            config.args.selec_mech = config.args.dyn_multi_selec_mech
                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset

                    elif config.args.dyn_multi_comb_mech == 'odin':
                        # assign odin params of trusted experts
                        if id == 0:
                            print('exp AI 1 odin params are assigned.')
                            config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe = odin_params['exp_AI_' + str(id+1) + '_odin_delta']
                            config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe = odin_params['exp_AI_' + str(id+1) + '_odin_eps']
                            print('gen AI odin params are assigned.')
                            if config.args.dyn_multi_tune_dataset == 'pipe':
                                if config.args.pipe_case == 'benchmark':
                                    config.args.c10_wideresnet_odin_delta_tpr95_pipe = odin_params['gen_AI_odin_delta']
                                    config.args.c10_wideresnet_odin_eps_tpr95_pipe = odin_params['gen_AI_odin_eps']
                                else:
                                    config.args.eurosat_odin_delta_tpr95 = odin_params['gen_AI_odin_delta']
                                    config.args.eurosat_odin_eps_tpr95 = odin_params['gen_AI_odin_eps']
                            elif config.args.dyn_multi_tune_dataset == 'iSUN':
                                config.args.c10_wideresnet_odin_delta_tpr95_iSUN = odin_params['gen_AI_odin_delta']
                                config.args.c10_wideresnet_odin_eps_tpr95_iSUN = odin_params['gen_AI_odin_eps']
                            elif config.args.dyn_multi_tune_dataset == 'UCM':
                                config.args.eurosat_odin_delta_UCM = odin_params['gen_AI_odin_delta']
                                config.args.eurosat_odin_eps_UCM = odin_params['gen_AI_odin_eps']
                            config.args.selec_mech = config.args.dyn_multi_selec_mech
                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                        elif id == 1:
                            print('exp AI 2 odin params are assigned.')
                            config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe = odin_params['exp_AI_' + str(id+1) + '_odin_delta']
                            config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe = odin_params['exp_AI_' + str(id+1) + '_odin_eps']
                            print('gen AI odin params are assigned.')
                            if config.args.dyn_multi_tune_dataset == 'pipe':
                                if config.args.pipe_case == 'benchmark':
                                    config.args.c10_wideresnet_odin_delta_tpr95_pipe = odin_params['gen_AI_odin_delta']
                                    config.args.c10_wideresnet_odin_eps_tpr95_pipe = odin_params['gen_AI_odin_eps']
                                else:
                                    config.args.eurosat_odin_delta_tpr95 = odin_params['gen_AI_odin_delta']
                                    config.args.eurosat_odin_eps_tpr95 = odin_params['gen_AI_odin_eps']
                            elif config.args.dyn_multi_tune_dataset == 'iSUN':
                                config.args.c10_wideresnet_odin_delta_tpr95_iSUN = odin_params['gen_AI_odin_delta']
                                config.args.c10_wideresnet_odin_eps_tpr95_iSUN = odin_params['gen_AI_odin_eps']
                            elif config.args.dyn_multi_tune_dataset == 'UCM':
                                config.args.eurosat_odin_delta_UCM = odin_params['gen_AI_odin_delta']
                                config.args.eurosat_odin_eps_UCM = odin_params['gen_AI_odin_eps']
                            config.args.selec_mech = config.args.dyn_multi_selec_mech
                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                        else:
                            print('exp AI 3 odin params are assigned.')
                            config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe = odin_params['exp_AI_' + str(id+1) + '_odin_delta']
                            config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe = odin_params['exp_AI_' + str(id+1) + '_odin_eps']
                            print('gen AI odin params are assigned.')
                            if config.args.dyn_multi_tune_dataset == 'pipe':
                                if config.args.pipe_case == 'benchmark':
                                    config.args.c10_wideresnet_odin_delta_tpr95_pipe = odin_params['gen_AI_odin_delta']
                                    config.args.c10_wideresnet_odin_eps_tpr95_pipe = odin_params['gen_AI_odin_eps']
                                else:
                                    config.args.eurosat_odin_delta_tpr95 = odin_params['gen_AI_odin_delta']
                                    config.args.eurosat_odin_eps_tpr95 = odin_params['gen_AI_odin_eps']
                            elif config.args.dyn_multi_tune_dataset == 'iSUN':
                                config.args.c10_wideresnet_odin_delta_tpr95_iSUN = odin_params['gen_AI_odin_delta']
                                config.args.c10_wideresnet_odin_eps_tpr95_iSUN = odin_params['gen_AI_odin_eps']
                            elif config.args.dyn_multi_tune_dataset == 'UCM':
                                config.args.eurosat_odin_delta_UCM = odin_params['gen_AI_odin_delta']
                                config.args.eurosat_odin_eps_UCM = odin_params['gen_AI_odin_eps']
                            config.args.selec_mech = config.args.dyn_multi_selec_mech
                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset

        config.args.dyn_resume = ''

    # initialize further metrics
    elif config.args.domain == 'single':
        score = 0.0
        trust_batchid_exp['single'] = 9999
        trust_scores['single'] = []
        exp_claimed_acc['single'] = []
        exp_test_acc['single'] = []
        exp_test_acc_strong['single'] = []
        exp_coverage['single'] = []
        dataset_sizes['single_trainset_size'] = []
        dataset_sizes['single_testset_size'] = []

        # load information if resume is acitve
        if config.args.dyn_resume != '':
            # Load checkpoint.
            print('')
            print('==> Resuming from checkpoint..')
            if os.path.isfile(config.args.pipe_root + config.args.dyn_resume):
                resume_file = config.args.pipe_root + config.args.dyn_resume
            else:
                resume_file = os.path.join(config.args.pipe_root + config.args.dyn_resume, 'checkpoint_dyn_single.pth.tar')
            assert os.path.isfile(resume_file), 'Error: no checkpoint directory found!'
            checkpoint_save = os.path.dirname(resume_file)
            checkpoint_info = torch.load(resume_file)
            start_batch = checkpoint_info['train_batch']
            step1_perf = checkpoint_info['step1_perf']
            pipe_perf = checkpoint_info['pipe_perf']
            human_effort = checkpoint_info['human_effort']
            batch_forward_indices = checkpoint_info['batch_forward_ind']
            trust_scores['single'] = checkpoint_info['trust_scores']
            exp_claimed_acc['single'] = checkpoint_info['exp_claimed_acc']
            exp_test_acc['single'] = checkpoint_info['exp_test_acc']
            exp_test_acc_strong['single'] = checkpoint_info['exp_test_acc_strong']
            gen_AI_std_acc = checkpoint_info['genAI_test_acc']
            gen_AI_strong_acc = checkpoint_info['genAI_test_acc']
            config.args.conf_thresh_singledom = checkpoint_info['gen_AI_bench_thresh']
            config.args.conf_thresh_singledom_sat = checkpoint_info['gen_AI_sat_thresh']
            config.args.conf_thresh_singledom_exp = checkpoint_info['gen_AI_exp_thresh']
            exp_coverage['single'] = checkpoint_info['exp_coverage']
            config.args.dyn_single_train_epochs = checkpoint_info['dyn_single_train_epoch']
            gen_AI_only = checkpoint_info['gen_AI_only']
            gen_AI_improvement = checkpoint_info['gen_AI_only']
            perf_exp_allocation = checkpoint_info['gen_AI_only']
            perf_exp_allocation_afterselec = checkpoint_info['gen_AI_only']
            gen_AI_HITL = checkpoint_info['gen_AI_only']
            gen_AI_HITL_afterselec = checkpoint_info['gen_AI_only']
            if checkpoint_info['trust_expert']:
                score = 1.0
                model_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/DYN_PIPE/' + config.args.domain.upper() + '/' + gen_AI_exp_name + '.pickle'
                filehandler = open(model_path, 'rb')
                network = pickle.load(filehandler)

        config.args.dyn_resume = ''


    # run DYNAMIC, MULTI-DOMAIN AIITL-SYSTEM
    if config.args.domain == 'multi':

        # loop over batches
        for idx, (train_batch, batch_true_labels_tot, batch_settypes) in enumerate(train_batches):

            # enforce reproducibility
            main_seed = config.args.main_seed
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)


            # if resume: Data sets need to be rebuilt, as they are never saved for computational reasons
            if idx < start_batch:

                print('RESUMING FROM CHECKPOINT REQUIRES ITERATIVE REBUILD OF DATASETS')

                if config.args.dyn_multi_transforms == 'adapted':
                    # go to eval mode
                    _ = utils.set_transform(train_batch.dataset, 'test')

                if idx < len(batch_forward_indices):

                    # allocation mechanism 1
                    forward_dataset = Subset(train_batch.dataset, batch_forward_indices[idx])
                    batch_true_labels_step2 = list(np.array(batch_true_labels_tot)[batch_forward_indices[idx]])
                    forwardloader_datamgmt = DataLoader(forward_dataset, num_workers=2, batch_size=1, shuffle=False)

                    # collect all previously trusted experts
                    prev_trusted_exp = []
                    for id in range(num_experts):
                        if exp_data['exp_AI_' + str(id+1) + '_score'] >= trust_thresh:
                            if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] < (idx+1):
                                prev_trusted_exp.append(trained_experts['exp_AI_' + str(id+1)])

                    # if rule is to start using experts first when all experts are trusted, reset prev_trusted_exp
                    if config.args.dyn_multi_tune_pot == 'when_all_trusted':
                        if len(prev_trusted_exp) < config.args.dyn_num_experts:
                            prev_trusted_exp = []

                    # if gating model not reliable, yet ..
                    if gating_trust_batchid < (idx+1):
                        gating_score = 0
                    else:
                        gating_score = 1

                    # if gating model is not to trusted yet, reset previously experts
                    if config.args.dyn_multi_comb_mech == 'gating':
                        if gating_score < config.args.gating_trust_thresh:
                            prev_trusted_exp = []

                    print('Trusted experts:')
                    print([exp.name for exp in prev_trusted_exp])

                    # allocation mechanism 2
                    pred_step2, batch_human_effort = call_experts(prev_trusted_exp, forwardloader_datamgmt, batch_true_labels_step2, batch_forward_indices[idx])

                    # remaining instances are needed to be allocated to create expert AIs iteratively
                    # only not yet trusted experts will be affected
                    print('Expert allocation of not predicted instances..')
                    if num_experts == 3:
                        print('.. based on HITL.')

                        # classes for sat dataset allocation
                        exp1_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                        exp2_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                        exp3_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']

                        # allocation based on human expert
                        exp_allocation = np.zeros((len(batch_human_effort['human_ind_step2']), 1))
                        for pos, i in enumerate(list(np.array(batch_true_labels_step2)[batch_human_effort['human_ind_step2']])):
                            if i.endswith('-SVHN') or i in exp1_classes:
                                exp_allocation[pos, 0] = 1
                            if i.endswith('-MNIST') or i in exp2_classes:
                                exp_allocation[pos, 0] = 2
                            if i.endswith('-FMNIST') or i.endswith('-CIFAR100') or i in exp3_classes:
                                exp_allocation[pos, 0] = 3

                    #### DATASET REBUILD ####
                    for id in range(num_experts):
                        if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] >= (idx+1):
                            # based on allocated instances, do iterative train, val, test split
                            # split size is designed to create the original val data size after all 10 batches
                            expert_data_ind = list(np.where(exp_allocation == (id+1))[0])

                            expert_dataset = Subset(Subset(forward_dataset, batch_human_effort['human_ind_step2']),
                                                    expert_data_ind)
                            if config.args.dyn_multi_split == 'random':
                                if config.args.pipe_case == 'benchmark':
                                    split = [int(0.773076923*len(expert_dataset)), int(0.076923077*len(expert_dataset)), int(0.15*len(expert_dataset))]
                                else:
                                    raise NotImplementedError
                                if split[0] + split[1] + split[2] != len(expert_dataset):
                                    split[0] = split[0] + (len(expert_dataset) - (split[0] + split[1] + split[2]))
                                exp_batch_trainset, exp_batch_valset, exp_batch_testset = random_split(expert_dataset, split, torch.Generator().manual_seed(config.args.main_seed))
                            elif config.args.dyn_multi_split == 'train_val':
                                forwardset_settypes = np.array(batch_settypes)[np.array(batch_forward_indices[idx])[batch_human_effort['human_ind_step2']]]
                                expert_settypes = forwardset_settypes[expert_data_ind]
                                expert_dataset_traintest_ind = [i for i in range(len(expert_dataset)) if expert_settypes[i] == 0]
                                expert_dataset_val_ind = [i for i in range(len(expert_dataset)) if expert_settypes[i] == 1]
                                assert (len(expert_dataset_traintest_ind) + len(expert_dataset_val_ind)) == len(expert_dataset)
                                exp_batch_valset = Subset(expert_dataset, expert_dataset_val_ind)
                                exp_traintest_set = Subset(expert_dataset, expert_dataset_traintest_ind)
                                split = [int(0.8*len(exp_traintest_set)), int(0.2*len(exp_traintest_set))]
                                if split[0] + split[1] != len(exp_traintest_set):
                                    split[0] = split[0] + (len(exp_traintest_set) - (split[0] + split[1]))
                                exp_batch_trainset, exp_batch_testset = random_split(exp_traintest_set, split, torch.Generator().manual_seed(config.args.main_seed))

                            multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data'] = ConcatDataset([multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data'], exp_batch_trainset])
                            multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data'] = ConcatDataset([multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data'], exp_batch_valset])
                            multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data'] = ConcatDataset([multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data'], exp_batch_testset])

                            dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'].append(len(multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data']))
                            dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'].append(len(multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data']))
                            dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'].append(len(multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data']))

                    for id in range(num_experts):
                        print('Expert AI ' + str(id+1) + ' (last) available dataset sizes: ')
                        print('trainset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'][-1]))
                        print('valset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'][-1]))
                        print('testset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'][-1]))

                    assert len(forward_dataset) == (len(list(np.array(batch_true_labels_step2)[batch_human_effort['no_human_ind_step2']])) + len(list(np.where(exp_allocation == 0)[0])) + len(list(np.where(exp_allocation == 1)[0])) + len(list(np.where(exp_allocation == 2)[0])) + len(list(np.where(exp_allocation == 3)[0])))



            # DYNAMIC, MULTI-DOMAIN AIITL-SYSTEM FROM SCRATCH
            else:

                print('Batch no. ' + str(idx+1) + ':')

                if config.args.dyn_multi_transforms == 'adapted':
                    # go to eval mode
                    _ = utils.set_transform(train_batch.dataset, 'test')

                # check correct parameter assignment
                if config.args.selec_mech == 'maha':
                    if config.args.pipe_case == 'benchmark':
                        print('config.args.c10_wideresnet_maha_delta_pipe_wFMNIST: ' + str(config.args.c10_wideresnet_maha_delta_pipe_wFMNIST))
                        print('config.args.c10_wideresnet_maha_eps_pipe_wFMNIST: ' + str(config.args.c10_wideresnet_maha_eps_pipe_wFMNIST))
                    else:
                        print('config.args.eurosat_maha_delta: ' + str(config.args.eurosat_maha_delta))
                        print('config.args.eurosat_maha_eps: ' + str(config.args.eurosat_maha_eps))
                    print('')
                    print('config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe: ' + str(config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe))
                    print('config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe: ' + str(config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe))
                    print('')
                    print('config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe: ' + str(config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe))
                    print('config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe: ' + str(config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe))
                    print('')
                    print('config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe: ' + str(config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe))
                    print('config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe: ' + str(config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe))
                elif config.args.selec_mech == 'odin':
                    if config.args.pipe_tune_dataset == 'pipe':
                        if config.args.pipe_case == 'benchmark':
                            print('config.args.c10_wideresnet_odin_delta_tpr95_pipe: ' + str(config.args.c10_wideresnet_odin_delta_tpr95_pipe))
                            print('config.args.c10_wideresnet_odin_eps_tpr95_pipe: ' + str(config.args.c10_wideresnet_odin_eps_tpr95_pipe))
                        else:
                            print('config.args.eurosat_odin_delta_tpr95: ' + str(config.args.eurosat_odin_delta_tpr95))
                            print('config.args.eurosat_odin_eps_tpr95: ' + str(config.args.eurosat_odin_eps_tpr95))
                        print('')
                        print('config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe: ' + str(config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe))
                        print('config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe: ' + str(config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe))
                        print('')
                        print('config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe: ' + str(config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe))
                        print('config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe: ' + str(config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe))
                        print('')
                        print('config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe: ' + str(config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe))
                        print('config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe: ' + str(config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe))
                    else:
                        if config.args.pipe_tune_dataset == 'UCM':
                            print('config.args.eurosat_odin_delta_UCM: ' + str(config.args.eurosat_odin_delta_UCM))
                            print('config.args.eurosat_odin_eps_UCM: ' + str(config.args.eurosat_odin_eps_UCM))
                        else:
                            print('config.args.c10_wideresnet_odin_delta_tpr95_iSUN: ' + str(config.args.c10_wideresnet_odin_delta_tpr95_iSUN))
                            print('config.args.c10_wideresnet_odin_eps_tpr95_iSUN: ' + str(config.args.c10_wideresnet_odin_eps_tpr95_iSUN))
                        print('')
                        print('config.args.exp_AI_1_wideresnet_odin_delta_tpr95_ext: ' + str(config.args.exp_AI_1_wideresnet_odin_delta_tpr95_ext))
                        print('config.args.exp_AI_1_wideresnet_odin_eps_tpr95_ext: ' + str(config.args.exp_AI_1_wideresnet_odin_eps_tpr95_ext))
                        print('')
                        print('config.args.exp_AI_2_wideresnet_odin_delta_tpr95_ext: ' + str(config.args.exp_AI_2_wideresnet_odin_delta_tpr95_ext))
                        print('config.args.exp_AI_2_wideresnet_odin_eps_tpr95_ext: ' + str(config.args.exp_AI_2_wideresnet_odin_eps_tpr95_ext))
                        print('')
                        print('config.args.exp_AI_3_wideresnet_odin_delta_tpr95_ext: ' + str(config.args.exp_AI_3_wideresnet_odin_delta_tpr95_ext))
                        print('config.args.exp_AI_3_wideresnet_odin_eps_tpr95_ext: ' + str(config.args.exp_AI_3_wideresnet_odin_eps_tpr95_ext))


                # for batch: instance selection, default ODIN on iSUN, as it performed best in pipeline
                predloader, pred_indices, forwardloader, forward_indices = select_instances(gen_AI, train_batch)
                batch_forward_indices.append(forward_indices)
                # classify instances that are not selected to forward to experts
                pred_step1 = pipe_pred(gen_AI, predloader)
                if config.args.selec_mech == 'odin' or config.args.selec_mech == 'maha':
                    batch_true_labels_step1 = list(np.array(batch_true_labels_tot)[pred_indices['ind']])
                else:
                    batch_true_labels_step1 = list(np.array(batch_true_labels_tot)[pred_indices])
                step1_perf.append(pipe_eval(pred_step1, batch_true_labels_step1)[metric])
                batch_true_labels_step2 = list(np.array(batch_true_labels_tot)[forward_indices])

                #### INFERENCE ####
                #### EXPERT ALLOCATION ####
                # previously trusted experts are used to allocate instances and predict
                # collect experts trusted already in previous batches
                prev_trusted_exp = []
                for id in range(num_experts):
                    if exp_data['exp_AI_' + str(id+1) + '_score'] >= trust_thresh:
                        if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] < (idx+1):
                            prev_trusted_exp.append(trained_experts['exp_AI_' + str(id+1)])

                # if rule is to start using experts first when all experts are trusted, reset prev_trusted_exp
                if config.args.dyn_multi_tune_pot == 'when_all_trusted':
                    if len(prev_trusted_exp) < config.args.dyn_num_experts:
                        prev_trusted_exp = []

                # if gating model not reliable, yet, reset previously trusted experts
                if config.args.dyn_multi_comb_mech == 'gating':
                    if gating_score < config.args.gating_trust_thresh:
                        prev_trusted_exp = []

                print('Trusted experts for allocation accordingly to point of time and gating model rule:')
                print([exp.name for exp in prev_trusted_exp])

                # allocation mechanism 2
                pred_step2, batch_human_effort = call_experts(prev_trusted_exp, forwardloader, batch_true_labels_step2, forward_indices)
                human_effort.append(batch_human_effort['#_inst']/len(batch_true_labels_tot))
                pred_pipe_wH = pred_step1 + pred_step2
                pipe_true_labels_wH = batch_true_labels_step1 + batch_true_labels_step2
                pred_pipe_woH = pred_step1 + list(np.array(pred_step2)[batch_human_effort['no_human_ind_step2']])
                pipe_true_labels_woH = batch_true_labels_step1 + list(np.array(batch_true_labels_step2)[batch_human_effort['no_human_ind_step2']])
                # get final accuracy
                pipe_perf.append(pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[metric])

                # get expert accuracies
                tot_ind_of_exp_pred = 0

                # expert accuracies on allocated data
                for id in range(num_experts):
                    if id == 0:
                        if config.args.pipe_case == 'benchmark':
                            ind_of_exp_pred = [True if i.endswith('-SVHN') else False for i in pred_pipe_woH]
                        elif config.args.pipe_case == 'sat':
                            exp_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                            ind_of_exp_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
                        tot_ind_of_exp_pred = tot_ind_of_exp_pred + sum(ind_of_exp_pred)
                    elif id == 1:
                        if config.args.pipe_case == 'benchmark':
                            ind_of_exp_pred = [True if i.endswith('-MNIST') else False for i in pred_pipe_woH]
                        elif config.args.pipe_case == 'sat':
                            exp_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                            ind_of_exp_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
                        tot_ind_of_exp_pred = tot_ind_of_exp_pred + sum(ind_of_exp_pred)
                    else:
                        if config.args.pipe_case == 'benchmark':
                            ind_of_exp_pred = [True if i.endswith('-FMNIST') else False for i in pred_pipe_woH]
                        elif config.args.pipe_case == 'sat':
                            exp_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                            ind_of_exp_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
                        tot_ind_of_exp_pred = tot_ind_of_exp_pred + sum(ind_of_exp_pred)

                    exp_pred = list(np.array(pred_pipe_woH)[ind_of_exp_pred])
                    exp_true = list(np.array(pipe_true_labels_woH)[ind_of_exp_pred])
                    exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'].append(pipe_eval(exp_pred, exp_true)[metric])
                    exp_coverage['exp_AI_' + str(id+1) + '_coverage'].append((len(exp_pred)/len(batch_true_labels_tot)))


                # remaining instances are needed to be allocated to create expert AIs iteratively
                # only not yet trusted experts will be affected
                print('Expert allocation of not predicted instances..')
                if num_experts == 3:
                    print('.. based on HITL.')

                    # classes for sat dataset allocation
                    exp1_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                    exp2_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                    exp3_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']

                    # allocation based on human expert
                    exp_allocation = np.zeros((len(batch_human_effort['human_ind_step2']), 1))
                    for pos, i in enumerate(list(np.array(batch_true_labels_step2)[batch_human_effort['human_ind_step2']])):
                        if i.endswith('-SVHN') or i in exp1_classes:
                            exp_allocation[pos, 0] = 1
                        if i.endswith('-MNIST') or i in exp2_classes:
                            exp_allocation[pos, 0] = 2
                        if i.endswith('-FMNIST') or i.endswith('-CIFAR100') or i in exp3_classes:
                            exp_allocation[pos, 0] = 3


                if len(prev_trusted_exp) == 0:
                    assert len(train_batch.dataset) == (len(predloader) + len(list(np.where(exp_allocation == 0)[0])) + len(list(np.where(exp_allocation == 1)[0])) + len(list(np.where(exp_allocation == 2)[0])) + len(list(np.where(exp_allocation == 3)[0])))
                else:
                    assert len(train_batch.dataset) == (len(predloader) + len(list(np.where(exp_allocation == 0)[0])) + len(list(np.where(exp_allocation == 1)[0])) + len(list(np.where(exp_allocation == 2)[0])) + len(list(np.where(exp_allocation == 3)[0])) + tot_ind_of_exp_pred)


                #### BASELINES ####
                print('baselines are computed. May take a while..')
                # gen AI for all
                print('')
                print('gen AI for all baseline:')
                gen_AI_pred = pipe_pred(gen_AI, train_batch)
                gen_AI_only.append(pipe_eval(gen_AI_pred, batch_true_labels_tot)[0])

                # gen AI + HITL according to selec mech
                print('')
                print('gen AI + HITL according to selec mech baseline:')
                gen_AI_HITL_selec_pred = pred_step1 + batch_true_labels_step2
                gen_AI_HITL_afterselec.append(pipe_eval(gen_AI_HITL_selec_pred, batch_true_labels_step1+batch_true_labels_step2)[0])

                # gen AI + HITL according to optimal allocation
                print('')
                print('gen AI + HITL according to optimal allocation baseline:')
                if config.args.pipe_case == 'benchmark':
                    gen_AI_pred = pipe_pred(gen_AI, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label.endswith('CIFAR10')]),
                                                               num_workers=2, shuffle=False, batch_size=1))
                    gen_AI_trues = list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label.endswith('CIFAR10')]])
                    HITL_trues = list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if not label.endswith('CIFAR10')]])
                    gen_AI_HITL.append(pipe_eval(gen_AI_pred+HITL_trues, gen_AI_trues+HITL_trues)[0])
                else:
                    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
                    gen_AI_pred = pipe_pred(gen_AI, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label in classes]),
                                                               num_workers=2, shuffle=False, batch_size=1))
                    gen_AI_trues = list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in classes]])
                    HITL_trues = list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in classes]])
                    gen_AI_HITL.append(pipe_eval(gen_AI_pred+HITL_trues, gen_AI_trues+HITL_trues)[0])


                # gen AI + perfect exp allocation according selec mech
                print('')
                print('gen AI + perfect exp allocation (incl. HITL) according to selec mech baseline:')
                human_pred_true = batch_true_labels_step2
                pred_tot = pred_step1
                true_tot = batch_true_labels_step1
                for exp in prev_trusted_exp:
                    if config.args.pipe_case == 'benchmark':
                        if exp.name[7] == '1':
                            title = '-SVHN'
                        elif exp.name[7] == '2':
                            title = '-MNIST'
                        elif exp.name[7] == '3':
                            title = '-FMNIST'
                        pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(forwardloader.dataset, [i for i, label in enumerate(batch_true_labels_step2) if label.endswith(title)]),
                                                                               num_workers=2, shuffle=False, batch_size=1))
                        true_tot = true_tot + list(np.array(batch_true_labels_step2)[[i for i, label in enumerate(batch_true_labels_step2) if label.endswith(title)]])
                        # remaining samples for human:
                        human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label.endswith(title)]))
                    else:
                        if exp.name[7] == '1':
                            title_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                        elif exp.name[7] == '2':
                            title_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                        elif exp.name[7] == '3':
                            title_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                        pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(forwardloader.dataset, [i for i, label in enumerate(batch_true_labels_step2) if label in title_classes]),
                                                                        num_workers=2, shuffle=False, batch_size=1))
                        true_tot = true_tot + list(np.array(batch_true_labels_step2)[[i for i, label in enumerate(batch_true_labels_step2) if label in title_classes]])
                        # remaining samples for human:
                        human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in title_classes]))

                perf_exp_allocation_afterselec.append(pipe_eval(pred_tot+human_pred_true, true_tot+human_pred_true)[0])

                # gen AI + perfect exp allocation to optimal allocation
                print('')
                print('gen AI + perfect exp allocation (incl. HITL) according to optimal allocation baseline:')
                pred_tot = []
                true_tot = []
                human_pred_true = batch_true_labels_tot
                exps = prev_trusted_exp + [gen_AI]
                for exp in exps:
                    if config.args.pipe_case == 'benchmark':
                        if hasattr(exp, 'trainloader'):
                            title = '-CIFAR10'
                        else:
                            if exp.name[7] == '1':
                                title = '-SVHN'
                            elif exp.name[7] == '2':
                                title = '-MNIST'
                            elif exp.name[7] == '3':
                                title = '-FMNIST'
                        pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label.endswith(title)]),
                                                                        num_workers=2, shuffle=False, batch_size=1))
                        true_tot = true_tot + list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label.endswith(title)]])
                        # remaining samples for human:
                        human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label.endswith(title)]))
                    else:
                        if hasattr(exp, 'trainloader'):
                            title_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
                        else:
                            if exp.name[7] == '1':
                                title_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                            elif exp.name[7] == '2':
                                title_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                            elif exp.name[7] == '3':
                                title_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                        pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label in title_classes]),
                                                                        num_workers=2, shuffle=False, batch_size=1))
                        true_tot = true_tot + list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in title_classes]])
                        # remaining samples for human:
                        human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in title_classes]))

                perf_exp_allocation.append(pipe_eval(pred_tot+human_pred_true, true_tot+human_pred_true)[0])


                #### INCREMENTAL EXPERT CREATION ####
                expert_set_length_tot = 0
                for id in range(num_experts):
                    if exp_data['exp_AI_' + str(id+1) + '_score'] < trust_thresh:
                        print('Expert AI ' + str(id+1) + ' is created.')

                        ## 1. STEP: create and extend datasets

                        # based on allocated instances, do iterative train, val, test split
                        expert_data_ind = list(np.where(exp_allocation == (id+1))[0])
                        expert_dataset = Subset(Subset(forwardloader.dataset, batch_human_effort['human_ind_step2']),
                                                expert_data_ind)
                        expert_set_length_tot = expert_set_length_tot + len(expert_dataset)

                        # split size is designed to create the original val data size after all batches
                        if config.args.dyn_multi_split == 'random':
                            if config.args.pipe_case == 'benchmark':
                                split = [int(0.773076923*len(expert_dataset)), int(0.076923077*len(expert_dataset)), int(0.15*len(expert_dataset))]
                            else:
                                raise NotImplementedError
                            if split[0] + split[1] + split[2] != len(expert_dataset):
                                split[0] = split[0] + (len(expert_dataset) - (split[0] + split[1] + split[2]))
                            exp_batch_trainset, exp_batch_valset, exp_batch_testset = random_split(expert_dataset, split, torch.Generator().manual_seed(config.args.main_seed))

                        # split according to original datasets
                        elif config.args.dyn_multi_split == 'train_val':
                            forwardset_settypes = np.array(batch_settypes)[np.array(batch_forward_indices[idx])[batch_human_effort['human_ind_step2']]]
                            expert_settypes = forwardset_settypes[expert_data_ind]
                            expert_dataset_traintest_ind = [i for i in range(len(expert_dataset)) if expert_settypes[i] == 0]
                            expert_dataset_val_ind = [i for i in range(len(expert_dataset)) if expert_settypes[i] == 1]
                            assert (len(expert_dataset_traintest_ind) + len(expert_dataset_val_ind)) == len(expert_dataset)
                            exp_batch_valset = Subset(expert_dataset, expert_dataset_val_ind)
                            exp_traintest_set = Subset(expert_dataset, expert_dataset_traintest_ind)

                            split = [int(0.8*len(exp_traintest_set)), int(0.2*len(exp_traintest_set))]
                            if split[0] + split[1] != len(exp_traintest_set):
                                split[0] = split[0] + (len(exp_traintest_set) - (split[0] + split[1]))
                            exp_batch_trainset, exp_batch_testset = random_split(exp_traintest_set, split, torch.Generator().manual_seed(config.args.main_seed))


                        # concat data
                        multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data'] = ConcatDataset([multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data'], exp_batch_trainset])
                        multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data'] = ConcatDataset([multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data'], exp_batch_valset])
                        multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data'] = ConcatDataset([multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data'], exp_batch_testset])

                        dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'].append(len(multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data']))
                        dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'].append(len(multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data']))
                        dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'].append(len(multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data']))


                        ## 2. STEP: train expert

                        # go to train mode for train data (rest is already test)
                        if config.args.dyn_multi_transforms == 'adapted':
                            train_data_copy = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data'])
                            _ = utils.set_transform(train_data_copy, 'train')
                        else:
                            train_data_copy = multi_exp_dataset['exp_AI_' + str(id+1) + '_train_data']

                        batch_size = config.args.dyn_multi_train_batch if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_train_batch_sat

                        # get train dataloaders
                        trainloader = DataLoader(train_data_copy, num_workers=2, shuffle=True, batch_size=batch_size)
                        valloader = DataLoader(multi_exp_dataset['exp_AI_' + str(id+1) + '_val_data'], num_workers=2, shuffle=False, batch_size=1)
                        testloader = DataLoader(multi_exp_dataset['exp_AI_' + str(id+1) + '_test_data'], num_workers=2, shuffle=False, batch_size=1)
                        trainloader.dataset.name = 'exp_AI_' + str(id+1) + '_dyn'
                        valloader.dataset.name = 'exp_AI_' + str(id+1) + '_dyn'
                        testloader.dataset.name = 'exp_AI_' + str(id+1) + '_dyn'
                        if id == 0:
                            if config.args.pipe_case == 'benchmark':
                                classes = [str(s) + '-SVHN' for s in range(10)]
                            else:
                                classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                        elif id == 1:
                            if config.args.pipe_case == 'benchmark':
                                classes = [str(s) + '-MNIST' for s in range(10)]
                            else:
                                classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                        else:
                            if config.args.pipe_case == 'benchmark':
                                classes = ['T-shirt/top-FMNIST', 'Trouser-FMNIST', 'Pullover-FMNIST', 'Dress-FMNIST', 'Coat-FMNIST', 'Sandal-FMNIST',
                                       'Shirt-FMNIST', 'Sneaker-FMNIST', 'Bag-FMNIST', 'Ankle boot-FMNIST']
                            else:
                                classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                        trainloader.dataset.classes = classes
                        valloader.dataset.classes = classes
                        testloader.dataset.classes = classes

                        """# comment the following, if script was interrupted during retraining on train + test data
                        if (config.args.dyn_multi_resume_expAI1 and id == 0) or (config.args.dyn_multi_resume_expAI2 and id == 1) or (config.args.dyn_multi_resume_expAI3 and id == 2):
                            config.args.resume = 'models/' + config.args.pipe_case + '/DYN_PIPE/' + config.args.domain.upper() + '/' + 'exp_AI_' + str(id+1) + '_checkpoint.pth.tar'"""


                        # TRAINING
                        trained_experts['exp_AI_' + str(id+1)], train_acc, _ = mb.dyn_expert_creation(trainloader, valloader=valloader, name='exp_AI_' + str(id+1))

                        config.args.resume = ''
                        if id == 0:
                            config.args.dyn_multi_resume_expAI1 = False
                        elif id == 1:
                            config.args.dyn_multi_resume_expAI2 = False
                        elif id == 2:
                            config.args.dyn_multi_resume_expAI3 = False
                        test_acc = mb.test(trained_experts['exp_AI_' + str(id+1)], testloader)
                        #print('Train acc: ' + str(train_acc))
                        print('Test acc: ' + str(test_acc))

                        ## 3. STEP: check if expert can be trusted
                        if test_acc < trust_thresh:
                            print('Expert AI ' + str(id+1) + ' is not trusted, yet. Further training needed. All instances are handled by HITL.')
                            trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] = 9999
                            exp_data['exp_AI_' + str(id+1) + '_score'] = test_acc
                            trust_scores['exp_AI_' + str(id+1) + '_trustscores'].append(test_acc)
                        else:
                            print('Expert ' + str(id+1) + ' is trusted now! Future batches will consult expert first.')
                            trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] = (idx+1)
                            exp_data['exp_AI_' + str(id+1) + '_score'] = test_acc
                            trust_scores['exp_AI_' + str(id+1) + '_trustscores'].append(test_acc)

                        # check for early stopping (starting after patience iterations and only if score is > 25% (safety measure, as SVHN sometimes needs longer to kick-off from the start accuracy of ~20%))
                        if trust_scores['exp_AI_' + str(id+1) + '_trustscores'][-1] >= 0.5 and idx >= patience and trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] != (idx+1) and config.args.ablation_study != 'no_trust_early_stop':
                            base_score = trust_scores['exp_AI_' + str(id+1) + '_trustscores'][int(-1*patience-1)]
                            deltas = []
                            for i in range((patience)):
                                deltas.append(abs(trust_scores['exp_AI_' + str(id+1) + '_trustscores'][-1*(i+1)] - base_score))
                            if sum([True if d <= max_delta else False for d in deltas]) == patience:
                                # early stopping!
                                print('Training is stopped early based on non-increasing trust_score.')
                                exp_data['exp_AI_' + str(id+1) + '_score'] = 1
                                trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] = (idx+1)


                # check whether all human effort was handled (or less, due to potential gen AI instances forwarded to expert)
                assert expert_set_length_tot <= batch_human_effort['#_inst']

                # matrix
                matrix = rs.allocation_matrix([gen_AI] + [trained_experts['exp_AI_' + str(id+1)] for id in range(num_experts)], pipe_true_labels_woH, pred_pipe_woH, tot_true_labels_for_dyn_pipe=batch_true_labels_tot)

                if not config.args.dyn_MVP_test:
                    # get standard test accuracy for all trained experts
                    for id in range(num_experts):
                        print('Get standard test accuracy for all trained experts.')
                        _, _, std_testloader = md.get_data(str(id+1))
                        std_testloader_copy = copy.deepcopy(std_testloader)
                        _ = utils.set_transform(std_testloader_copy.dataset, 'test')

                        exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'].append(mb.test(trained_experts['exp_AI_' + str(id+1)], std_testloader_copy))



                ## 4. STEP: HYPERPARAMETER TUNING FOR ALL TRUSTED EXPERTS (ONLY ONCE PER EXPERT)  ####

                # collect all experts that are newly trusted in this iteration
                if config.args.dyn_multi_tune_dataset == 'pipe':
                    trusted_exp = [gen_AI]
                else:
                    trusted_exp = []
                for id in range(num_experts):
                    if exp_data['exp_AI_' + str(id+1) + '_score'] >= trust_thresh:
                        if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] == (idx+1):
                            trusted_exp.append(trained_experts['exp_AI_' + str(id+1)])

                # add previously trusted experts, only if newly trusted experts are added in this iteration
                if len(trusted_exp) > (1 if config.args.dyn_multi_tune_dataset == 'pipe' else 0):
                    for id in range(num_experts):
                        # only add if not tuned on external dataset (iSUN / UCM)
                        if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] < (idx+1) and config.args.dyn_multi_tune_dataset == 'pipe':
                            trusted_exp.append(trained_experts['exp_AI_' + str(id+1)])

                # check whether tuning appliable based on tuning point of time
                if config.args.dyn_multi_comb_mech == 'odin' or config.args.dyn_multi_comb_mech == 'maha':

                    # tuning applicable either when new experts are trusted or all experts are trusted
                    if (config.args.dyn_multi_tune_pot == 'when_trusted' and (len(trusted_exp) > (1 if config.args.dyn_multi_tune_dataset == 'pipe' else 0))) or \
                            (config.args.dyn_multi_tune_pot == 'when_all_trusted' and (len(trusted_exp) == ((config.args.dyn_num_experts+1) if config.args.dyn_multi_tune_dataset == 'pipe' else config.args.dyn_num_experts))):
                        print(config.args.dyn_multi_comb_mech + ' hyperparameter tuning is conducted for trusted experts accordingly to point of time rule. Previously trusted experts are included, as well:')
                        print([exp.name if hasattr(exp, 'name') else exp.trainloader.dataset.name for exp in trusted_exp])

                        # gen AI and previously trusted experts are always retuned using old validation data

                        # tuning process
                        for id, exp in enumerate(trusted_exp):
                            if hasattr(exp, 'trainloader'):

                                # gen AI is only tuned, if pipe (not iSUN or UCM) dataset is used (which is always the case for maha)
                                if config.args.dyn_multi_selec_mech == 'maha' or (config.args.dyn_multi_selec_mech == 'odin' and config.args.dyn_multi_tune_dataset == 'pipe'):

                                    # 1.: get datasets

                                    print('Gen AI is tuned.')

                                    # inloader is the standard val data inloader
                                    _, inloader, _ = md.get_data('0')

                                    #inloader = DataLoader(Subset(inloader.dataset, [i for i in range(100)]), batch_size=1, shuffle=False)

                                    # outloader is either pipe data or iSUN/UCM
                                    if config.args.dyn_multi_tune_dataset == 'pipe':
                                        outloaderset = Subset(inloader.dataset, [])
                                        outdatasets = [1, 2, 3]
                                        print('outdatasets:')
                                        print(outdatasets)

                                        # collect sizes of outloader sets
                                        out_sizes = []
                                        for outdataset in outdatasets:
                                            out_sizes.append(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))

                                        for idx_i, outdataset in enumerate(outdatasets):

                                            # get other indices:
                                            other_ind = [i for i in range(num_experts) if i != idx_i]

                                            # outloader needs to be limited to 5k samples for computational reasons
                                            # make sure each dataset is represented to full length up until 1667 samples (1/3 of 5k)
                                            if len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']) <= 1667:
                                                outloaderset = ConcatDataset([outloaderset, multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']])
                                            else:
                                                assert num_experts == 3  # otherwise this structure does not work
                                                # if all out sets are > 1667 --> all at most 1667
                                                if out_sizes[other_ind[0]] > 1667 and out_sizes[other_ind[1]] > 1667:
                                                    outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(1667, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])
                                                # if only this set > 1667 --> fill up until 5k or length of dataset
                                                elif out_sizes[other_ind[0]] <= 1667 and out_sizes[other_ind[1]] <= 1667:
                                                    rem_size = 5000 - out_sizes[other_ind[0]] - out_sizes[other_ind[1]]
                                                    outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(rem_size, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])
                                                # if one other set is > 1667 --> 50/50 share of remaining space
                                                elif (out_sizes[other_ind[0]] <= 1667 and out_sizes[other_ind[1]] > 1667) or (out_sizes[other_ind[0]] > 1667 and out_sizes[other_ind[1]] <= 1667):
                                                    # get id of the one set < 1667:
                                                    small_id = other_ind[0] if out_sizes[other_ind[0]] <= 1667 else other_ind[1]
                                                    # check out total remaining space
                                                    rem_space_for_both_big_sets = int(5000 - out_sizes[small_id])
                                                    # distribute remaining space 50/50
                                                    space_for_this_set = int(rem_space_for_both_big_sets / 2)
                                                    outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(space_for_this_set, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])


                                        outloader = DataLoader(outloaderset, batch_size=1, num_workers=2, shuffle=True)
                                    else:
                                        print('iSUN/UCM is loadad as outloader.')
                                        outloader = md.get_data('5')


                                    """if len(inloader) >= len(outloader):
                                        inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), len(outloader.dataset))), batch_size=1, shuffle=False)
                                    else:
                                        outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), len(inloader.dataset))), batch_size=1, shuffle=True)"""

                                    #inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), 2)), batch_size=1, shuffle=False)
                                    #outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), 2)), batch_size=1, shuffle=True)


                            else:
                                # 1.: get datasets

                                print('Exp AI ' + str(exp.name[7]) + ' is tuned.')

                                # inloader is the standard val data inloader
                                inloader = DataLoader(multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_val_data'], num_workers=2, shuffle=False, batch_size=1)

                                # outloader is either pipe data or iSUN
                                if config.args.dyn_multi_tune_dataset == 'pipe':

                                    _, outloader1, _ = md.get_data('0')
                                    # gen AI set with same length as outloaders (approx.)
                                    outdatasets = [i for i in range(1, 4) if i != int(exp.name[7])]
                                    print('outdatasets:')
                                    print(outdatasets)

                                    out_sizes = []
                                    for outdataset in outdatasets:
                                        out_sizes.append(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))

                                    # determine length of gen AI
                                    if out_sizes[0] <= 1667 and out_sizes[1] <= 1667:
                                        # if the other sets are both < 1667 --> fill up remaining space with gen AI
                                        rem_space = 5000 - out_sizes[0] - out_sizes[1]
                                        outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(rem_space, len(outloader1.dataset))))
                                    elif out_sizes[0] > 1667 and out_sizes[1] > 1667:
                                        # if both other sets are > 1667 --> limit all sets to 1667
                                        outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(1667, len(outloader1.dataset))))
                                    elif (out_sizes[0] > 1667 and out_sizes[1] <= 1667) or (out_sizes[0] <= 1667 and out_sizes[1] > 1667):
                                        # if one other set is > 1667 --> 50/50 space
                                        # get id of the one set < 1667:
                                        small_id = 0 if out_sizes[0] <= 1667 else 1
                                        # check out total remaining space
                                        rem_space_for_both_big_sets = int(5000 - out_sizes[small_id])
                                        # distribute remaining space 50/50
                                        space_for_this_set = int(rem_space_for_both_big_sets / 2)
                                        outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(space_for_this_set, len(outloader1.dataset))))

                                    for outdataset in outdatasets:
                                        if len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']) <= 1667:
                                            outloaderset = ConcatDataset([outloaderset, multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']])
                                        elif out_sizes[0] > 1667 and out_sizes[1] > 1667:
                                            outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(1667, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])
                                        else:
                                            outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(space_for_this_set, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])


                                    outloader = DataLoader(outloaderset, batch_size=1, num_workers=2, shuffle=True)
                                else:
                                    print('iSUN/UCM is loadad as outloader.')
                                    outloader = md.get_data('5')

                                """# make sure inloader has the same length as outloader
                                if len(inloader) >= len(outloader):
                                    inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), len(outloader.dataset))), batch_size=1, shuffle=False)
                                else:
                                    outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), len(inloader.dataset))), batch_size=1, shuffle=True)"""

                                #inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), 2)), batch_size=1, shuffle=False)
                                #outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), 2)), batch_size=1, shuffle=True)

                            # 2.: actual tuning
                            if config.args.dyn_MVP_test:
                                inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), min(50, len(inloader.dataset)))), batch_size=1, shuffle=False)
                                outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), min(50, len(outloader.dataset)))), batch_size=1, shuffle=True)

                            if config.args.dyn_multi_comb_mech == 'maha':
                                if not hasattr(exp, 'trainloader'):
                                    # go to train mode for train data (rest is already test)
                                    if config.args.dyn_multi_transforms == 'adapted':
                                        train_data_copy = copy.deepcopy(ConcatDataset([multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_train_data'], multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_test_data']]))
                                        _ = utils.set_transform(train_data_copy, 'train')
                                    else:
                                        train_data_copy = ConcatDataset([multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_train_data'], multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_test_data']])

                                    trainloader = DataLoader(train_data_copy, num_workers=2, shuffle=True, batch_size=64 if config.args.pipe_case == 'benchmark' else 32)

                                    best_delta, best_eps, _, _ = maha.tune_dyn_maha_params(exp, inloader, outloader, trainloader=trainloader)
                                else:
                                    best_delta, best_eps, _, _ = maha.tune_dyn_maha_params(exp, inloader, outloader)

                                # show dataset structure
                                print('INLOADER:')
                                print(utils.underlying_datasets(inloader.dataset))
                                print('OUTLOADER:')
                                print(utils.underlying_datasets(outloader.dataset))
                                print('')

                                # save parameters
                                if id == 0:
                                    maha_params['gen_AI_maha_eps'] = best_eps
                                    maha_params['gen_AI_maha_delta'] = best_delta
                                else:
                                    maha_params['exp_AI_' + exp.name[7] + '_maha_eps'] = best_eps
                                    maha_params['exp_AI_' + exp.name[7] + '_maha_delta'] = best_delta

                            elif config.args.dyn_multi_comb_mech == 'odin':

                                best_delta, best_eps, _, _ = odin.tune_dyn_odin_params(exp, inloader, outloader, 0.95)

                                if config.args.dyn_multi_tune_dataset != 'iSUN' and config.args.dyn_multi_tune_dataset != 'UCM':
                                    # show dataset structure
                                    print('INLOADER:')
                                    print(utils.underlying_datasets(inloader.dataset))
                                    print('OUTLOADER:')
                                    print(utils.underlying_datasets(outloader.dataset))
                                    print('')

                                if id == 0:
                                    if config.args.dyn_multi_tune_dataset == 'pipe':
                                        odin_params['gen_AI_maha_eps'] = best_eps
                                        odin_params['gen_AI_maha_delta'] = best_delta
                                else:
                                    odin_params['exp_AI_' + exp.name[7] + '_maha_eps'] = best_eps
                                    odin_params['exp_AI_' + exp.name[7] + '_maha_delta'] = best_delta


                        config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                        config.args.selec_mech = config.args.dyn_multi_selec_mech

                # gating mode training
                elif config.args.dyn_multi_comb_mech == 'gating':

                    # if at least one expert is trusted
                    counter = 0
                    for id in range(num_experts):
                        if trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'] <= (idx+1):
                            counter = counter + 1

                    if counter > (0 if config.args.dyn_multi_tune_pot == 'when_trusted' else (config.args.dyn_num_experts-1)) and gating_score < config.args.gating_trust_thresh:
                        print('gating model is trained.')

                        # create train and valloaders with respective targets
                        gen_AI_trainloader, gen_AI_valloader, _ = md.get_data('0')
                        gating_trainset = copy.deepcopy(gen_AI_trainloader.dataset)
                        gating_valset = copy.deepcopy(gen_AI_valloader.dataset)
                        exp1_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_train_data'])
                        exp1_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_val_data'])
                        exp2_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_train_data'])
                        exp2_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_val_data'])
                        exp3_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_train_data'])
                        exp3_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_val_data'])

                        # ALLO Mechanism 2
                        if config.args.pipe_case == 'benchmark':
                            _ = utils.set_target(gating_trainset, 0, 'CIFAR10')
                            _ = utils.set_target(gating_valset, 0, 'CIFAR10')
                            _ = utils.set_target(exp1_trainset, 1, 'SVHN')
                            _ = utils.set_target(exp1_valset, 1, 'SVHN')
                            _ = utils.set_target(exp2_trainset, 2, 'MNIST')
                            _ = utils.set_target(exp2_valset, 2, 'MNIST')
                            _ = utils.set_target(exp3_trainset, 3, 'FashionMNIST')
                            _ = utils.set_target(exp3_valset, 3, 'FashionMNIST')
                        else:
                            _ = utils.set_target(gating_trainset, 0, 'Euro_SAT_countryside')
                            _ = utils.set_target(gating_valset, 0, 'Euro_SAT_countryside')
                            _ = utils.set_target(exp1_trainset, 1, 'FMOW')
                            _ = utils.set_target(exp1_valset, 1, 'FMOW')
                            _ = utils.set_target(exp2_trainset, 2, 'AID')
                            _ = utils.set_target(exp2_valset, 2, 'AID')
                            _ = utils.set_target(exp3_trainset, 3, 'RESISC')
                            _ = utils.set_target(exp3_valset, 3, 'RESISC')

                        train_length = min(len(gating_trainset), len(exp1_trainset), len(exp2_trainset), len(exp3_trainset))
                        val_length = min(len(gating_valset), len(exp1_valset), len(exp2_valset), len(exp3_valset))

                        gating_trainset = Subset(gating_trainset, random.sample(range(len(gating_trainset)), train_length))
                        gating_valset = Subset(gating_valset, random.sample(range(len(gating_valset)), val_length))
                        exp1_trainset = Subset(exp1_trainset, random.sample(range(len(exp1_trainset)), train_length))
                        exp1_valset = Subset(exp1_valset, random.sample(range(len(exp1_valset)), val_length))
                        exp2_trainset = Subset(exp2_trainset, random.sample(range(len(exp2_trainset)), train_length))
                        exp2_valset = Subset(exp2_valset, random.sample(range(len(exp2_valset)), val_length))
                        exp3_trainset = Subset(exp3_trainset, random.sample(range(len(exp3_trainset)), train_length))
                        exp3_valset = Subset(exp3_valset, random.sample(range(len(exp3_valset)), val_length))

                        gating_tot_trainset = ConcatDataset([gating_trainset, exp1_trainset, exp2_trainset, exp3_trainset])
                        gating_tot_valset = ConcatDataset([gating_valset, exp1_valset, exp2_valset, exp3_valset])

                        # go to train mode for train data (rest is already test)
                        if config.args.dyn_multi_transforms == 'adapted':
                            train_data_copy = copy.deepcopy(gating_tot_trainset)
                            _ = utils.set_transform(train_data_copy, 'train')
                        else:
                            train_data_copy = gating_tot_trainset

                        gating_train_batch = config.args.gating_train_batch if config.args.pipe_case == 'benchmark' else config.args.sat_gating_train_batch

                        gating_trainloader = DataLoader(train_data_copy, num_workers=2, batch_size=gating_train_batch, shuffle=True)
                        gating_valloader = DataLoader(gating_tot_valset, num_workers=2, batch_size=1, shuffle=True)

                        targets = []
                        for data, target in gating_trainloader:
                            #print(target)
                            for i in range(len(target)):
                                targets.append(target[i].item())
                        target_set_train = list(set(targets))
                        targets = []
                        for data, target in gating_valloader:
                            #print(target)
                            for i in range(len(target)):
                                targets.append(target[i].item())
                        target_set_val = list(set(targets))

                        assert target_set_train == [0, 1, 2, 3]
                        assert target_set_val == [0, 1, 2, 3]

                        gating_trainloader.dataset.name = 'gating'
                        gating_valloader.dataset.name = 'gating'
                        print('Len of gating train data: ' + str(len(gating_tot_trainset)))
                        print('Len of gating val data: ' + str(len(gating_tot_valset)))

                        # train model
                        _, gating_val_acc = mb.train_dyn_gating_model(gating_trainloader, gating_valloader)
                        print('gating val acc: ' + str(gating_val_acc))

                        if gating_val_acc >= config.args.gating_trust_thresh:
                            print('Gating model is trusted now.')
                            gating_score = gating_val_acc
                            gating_trust_batchid = (idx+1)

                            config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                            config.args.selec_mech = config.args.dyn_multi_selec_mech

                        else:
                            print('Gating model is not trusted, yet.')

                        # SELEC MECH GATING (allocation mechanism 1)
                        if config.args.dyn_multi_selec_mech == 'gating':
                            print('gating model for selec mech is trained.')
                            print('the selec mech gating model is not checked for a minimum score, as its performance is at least as good as the previous gating model by design (only 2 classes, same data)')

                            # create train and valloaders with respective targets
                            gen_AI_trainloader, gen_AI_valloader, _ = md.get_data('0')
                            gating_trainset_selec = copy.deepcopy(gen_AI_trainloader.dataset)
                            gating_valset_selec = copy.deepcopy(gen_AI_valloader.dataset)
                            exp1_trainset_selec = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_train_data'])
                            exp1_valset_selec = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_val_data'])
                            exp2_trainset_selec = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_train_data'])
                            exp2_valset_selec = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_val_data'])
                            exp3_trainset_selec = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_train_data'])
                            exp3_valset_selec = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_val_data'])

                            if config.args.pipe_case == 'benchmark':
                                _ = utils.set_target(gating_trainset_selec, 1, 'CIFAR10')
                                _ = utils.set_target(gating_valset_selec, 1, 'CIFAR10')
                                _ = utils.set_target(exp1_trainset_selec, 0, 'SVHN')
                                _ = utils.set_target(exp1_valset_selec, 0, 'SVHN')
                                _ = utils.set_target(exp2_trainset_selec, 0, 'MNIST')
                                _ = utils.set_target(exp2_valset_selec, 0, 'MNIST')
                                _ = utils.set_target(exp3_trainset_selec, 0, 'FashionMNIST')
                                _ = utils.set_target(exp3_valset_selec, 0, 'FashionMNIST')
                            else:
                                _ = utils.set_target(gating_trainset_selec, 1, 'Euro_SAT_countryside')
                                _ = utils.set_target(gating_valset_selec, 1, 'Euro_SAT_countryside')
                                _ = utils.set_target(exp1_trainset_selec, 0, 'FMOW')
                                _ = utils.set_target(exp1_valset_selec, 0, 'FMOW')
                                _ = utils.set_target(exp2_trainset_selec, 0, 'AID')
                                _ = utils.set_target(exp2_valset_selec, 0, 'AID')
                                _ = utils.set_target(exp3_trainset_selec, 0, 'RESISC')
                                _ = utils.set_target(exp3_valset_selec, 0, 'RESISC')


                            train_length = min(len(gating_trainset_selec), len(exp1_trainset_selec), len(exp2_trainset_selec), len(exp3_trainset_selec))
                            val_length = min(len(gating_valset_selec), len(exp1_valset_selec), len(exp2_valset_selec), len(exp3_valset_selec))

                            gating_trainset_selec = Subset(gating_trainset_selec, random.sample(range(len(gating_trainset_selec)), train_length))
                            gating_valset_selec = Subset(gating_valset_selec, random.sample(range(len(gating_valset_selec)), val_length))
                            exp1_trainset_selec = Subset(exp1_trainset_selec, random.sample(range(len(exp1_trainset_selec)), train_length))
                            exp1_valset_selec = Subset(exp1_valset_selec, random.sample(range(len(exp1_valset_selec)), val_length))
                            exp2_trainset_selec = Subset(exp2_trainset_selec, random.sample(range(len(exp2_trainset_selec)), train_length))
                            exp2_valset_selec = Subset(exp2_valset_selec, random.sample(range(len(exp2_valset_selec)), val_length))
                            exp3_trainset_selec = Subset(exp3_trainset_selec, random.sample(range(len(exp3_trainset_selec)), train_length))
                            exp3_valset_selec = Subset(exp3_valset_selec, random.sample(range(len(exp3_valset_selec)), val_length))

                            gating_tot_trainset_selec = ConcatDataset([gating_trainset_selec, exp1_trainset_selec, exp2_trainset_selec, exp3_trainset_selec])
                            gating_tot_valset_selec = ConcatDataset([gating_valset_selec, exp1_valset_selec, exp2_valset_selec, exp3_valset_selec])

                            # go to train mode for train data (rest is already test)
                            if config.args.dyn_multi_transforms == 'adapted':
                                train_data_copy_selec = copy.deepcopy(gating_tot_trainset_selec)
                                _ = utils.set_transform(train_data_copy_selec, 'train')
                            else:
                                train_data_copy_selec = gating_tot_trainset_selec

                            gating_trainloader_selec = DataLoader(train_data_copy_selec, num_workers=2, batch_size=gating_train_batch, shuffle=True)
                            gating_valloader_selec = DataLoader(gating_tot_valset_selec, num_workers=2, batch_size=1, shuffle=True)

                            targets = []
                            for data, target in gating_trainloader_selec:
                                #print(target)
                                for i in range(len(target)):
                                    targets.append(target[i].item())
                            target_set_train = list(set(targets))
                            targets = []
                            for data, target in gating_valloader_selec:
                                #print(target)
                                for i in range(len(target)):
                                    targets.append(target[i].item())
                            target_set_val = list(set(targets))

                            assert target_set_train == [0, 1]
                            assert target_set_val == [0, 1]

                            gating_trainloader_selec.dataset.name = 'gating'
                            gating_valloader_selec.dataset.name = 'gating'
                            print('Len of gating train data: ' + str(len(gating_tot_trainset_selec)))
                            print('Len of gating val data: ' + str(len(gating_tot_valset_selec)))

                            # train model
                            _, gating_val_acc_selec = mb.train_dyn_gating_model(gating_trainloader_selec, gating_valloader_selec, type='selec')
                            print('gating selec mech val acc: ' + str(gating_val_acc_selec))



                    elif counter > 0 and gating_score >= config.args.gating_trust_thresh:
                        print('gating model(s) is/are already trained and trusted.')



                elif config.args.dyn_multi_comb_mech == 'unc_thresh':
                    raise NotImplementedError

                # print and track results of batch
                print('Gen AI acc. on claimed instances: ' + str(step1_perf[idx]))
                print('Pipe acc: ' + str(pipe_perf[idx]))
                print('Human effort: ' + str(human_effort[idx]))
                print('allocation matrix:')
                print(matrix)
                for id in range(num_experts):
                    print('Expert AI ' + str(id+1) + ' (last) available dataset sizes: ')
                    print('trainset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'][-1]))
                    print('valset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'][-1]))
                    print('testset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'][-1]))
                    print('Expert AI ' + str(id+1) + ' acc on claimed instances: ')
                    print(exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'][idx])
                    if not config.args.dyn_MVP_test:
                        print('Expert AI ' + str(id+1) + ' acc on standard test set: ')
                        print(exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'][idx])
                    print('Expert AI ' + str(id+1) + ' coverage: ')
                    print(exp_coverage['exp_AI_' + str(id+1) + '_coverage'][idx])
                print('')

                # track results if part of 'run all pipelines'
                # mlflow
                if run_name != '':
                    mlflow.set_experiment(experiment_name=exp_name)
                    run_name_now = str('BATCH_' + str(idx+1) + '_' + run_name)
                    mlflow.start_run(run_name=run_name_now)

                    mlflow.log_metric('gen_AI_claimed_acc', step1_perf[-1])
                    mlflow.log_metric('pipe_performance', pipe_perf[-1])
                    mlflow.log_metric('human_effort', human_effort[-1])
                    mlflow.log_metric('gen_AI_only', gen_AI_only[-1])
                    #mlflow.log_metric('gen_AI_improvement', gen_AI_improvement[-1])
                    mlflow.log_metric('perf_exp_allocation', perf_exp_allocation[-1])
                    mlflow.log_metric('perf_exp_allocation_afterselec', perf_exp_allocation_afterselec[-1])
                    mlflow.log_metric('gen_AI_HITL', gen_AI_HITL[-1])
                    mlflow.log_metric('gen_AI_HITL_afterselec', gen_AI_HITL_afterselec[-1])

                    mlflow.log_params({'pipe_case': config.args.pipe_case})

                    # log odin/maha params
                    if config.args.pipe_case == 'benchmark':
                        odin_maha_params = {
                            'gen_AI_odin_delta_pipe': config.args.c10_wideresnet_odin_delta_tpr95_pipe,
                            'gen_AI_odin_eps_pipe': config.args.c10_wideresnet_odin_eps_tpr95_pipe,
                            'gen_AI_odin_delta_ext': config.args.c10_wideresnet_odin_delta_tpr95_iSUN,
                            'gen_AI_odin_eps_ext': config.args.c10_wideresnet_odin_eps_tpr95_iSUN,
                            'gen_AI_maha_delta_pipe': config.args.c10_wideresnet_maha_delta_pipe_wFMNIST,
                            'gen_AI_maha_eps_pipe': config.args.c10_wideresnet_maha_eps_pipe_wFMNIST,
                        }
                    else:
                        odin_maha_params = {
                            'gen_AI_odin_delta_pipe': config.args.eurosat_odin_delta_tpr95,
                            'gen_AI_odin_eps_pipe': config.args.eurosat_odin_eps_tpr95,
                            'gen_AI_odin_delta_ext': config.args.eurosat_odin_delta_UCM,
                            'gen_AI_odin_eps_ext': config.args.eurosat_odin_eps_UCM,
                            'gen_AI_maha_delta_pipe': config.args.eurosat_maha_delta,
                            'gen_AI_maha_eps_pipe': config.args.eurosat_maha_eps,
                        }
                    
                    odin_maha_params['exp_AI_1_odin_delta_pipe'] = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe
                    odin_maha_params['exp_AI_1_odin_eps_pipe'] = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe
                    odin_maha_params['exp_AI_1_odin_delta_ext'] = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_ext
                    odin_maha_params['exp_AI_1_odin_eps_ext'] = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_ext
                    odin_maha_params['exp_AI_1_maha_delta_pipe'] = config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe
                    odin_maha_params['exp_AI_1_maha_eps_pipe'] = config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe
                    
                    odin_maha_params['exp_AI_2_odin_delta_pipe'] = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe
                    odin_maha_params['exp_AI_2_odin_eps_pipe'] = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe
                    odin_maha_params['exp_AI_2_odin_delta_ext'] = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_ext
                    odin_maha_params['exp_AI_2_odin_eps_ext'] = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_ext
                    odin_maha_params['exp_AI_2_maha_delta_pipe'] = config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe
                    odin_maha_params['exp_AI_2_maha_eps_pipe'] = config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe
                    
                    odin_maha_params['exp_AI_3_odin_delta_pipe'] = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe
                    odin_maha_params['exp_AI_3_odin_eps_pipe'] = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe
                    odin_maha_params['exp_AI_3_odin_delta_ext'] = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_ext
                    odin_maha_params['exp_AI_3_odin_eps_ext'] = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_ext
                    odin_maha_params['exp_AI_3_maha_delta_pipe'] = config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe
                    odin_maha_params['exp_AI_3_maha_eps_pipe'] = config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe

                    mlflow.log_params(odin_maha_params)

                    for id in range(num_experts):
                        mlflow.log_metric('trust_scores_exp_AI_' + str(id+1), trust_scores['exp_AI_' + str(id+1) + '_trustscores'][-1])
                        mlflow.log_metric('exp_claimed_acc_exp_AI_' + str(id+1), exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'][-1])
                        if not config.args.dyn_MVP_test:
                            mlflow.log_metric('exp_test_acc_exp_AI_' + str(id+1), exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'][-1])
                        mlflow.log_metric('exp_coverage_exp_AI_' + str(id+1), exp_coverage['exp_AI_' + str(id+1) + '_coverage'][-1])
                        mlflow.log_metric('trust_batchid_exp_AI_' + str(id+1), trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'])
                        mlflow.log_metric('trainset_sizes_exp_AI_' + str(id+1), dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'][-1])
                        mlflow.log_metric('valset_sizes_exp_AI_' + str(id+1), dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'][-1])
                        mlflow.log_metric('testset_sizes_exp_AI_' + str(id+1), dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'][-1])

                    # log allocation matrix
                    for i in range(matrix.shape[0] - 1):
                        for j in range(matrix.shape[1] - 1):
                            mlflow.log_metric(str(i) + '/' + str(j), matrix.iloc[i, j])

                    mlflow.end_run()

                # save checkpoint
                if config.args.resume == '':
                    checkpoint_save = config.args.pipe_root + config.args.dyn_checkpoint
                state = {}
                state['train_batch'] = idx + 1
                state['step1_perf'] = step1_perf
                state['pipe_perf'] = pipe_perf
                state['human_effort'] = human_effort
                state['trust_scores'] = trust_scores
                state['exp_claimed_acc'] = exp_claimed_acc
                state['exp_test_acc'] = exp_test_acc
                state['exp_data'] = exp_data
                state['exp_coverage'] = exp_coverage
                state['trust_batchid_exp'] = trust_batchid_exp
                state['batch_forward_indices'] = batch_forward_indices
                state['maha_params'] = maha_params
                state['odin_params'] = odin_params
                state['gating_score'] = gating_score
                state['gating_trust_batchid'] = gating_trust_batchid
                state['gen_AI_only'] = gen_AI_only
                state['gen_AI_improvement'] = gen_AI_improvement
                state['perf_exp_allocation'] = perf_exp_allocation
                state['perf_exp_allocation'] = perf_exp_allocation_afterselec
                state['gen_AI_HITL'] = gen_AI_HITL
                state['gen_AI_HITL_afterselec'] = gen_AI_HITL_afterselec
                utils.save_checkpoint(state, checkpoint=checkpoint_save, filename='checkpoint_dyn_multi.pth.tar')


        # FINAL BATCH THAT IS COMPARED WITH STATIC AIITL-SYSTEM
        if config.args.dyn_multi_finalbatch:

            print('Final test batch..')

            # enforce reproducibility
            main_seed = config.args.main_seed
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)

            # all not yet trusted experts are going to be retrained on complete train + test data
            not_yet_trusted_exp = []
            for i in range(num_experts):
                if exp_data['exp_AI_' + str(i+1) + '_score'] < trust_thresh:
                    not_yet_trusted_exp.append('exp_AI_' + str(i+1) + '_dyn')
            print('Training on train + test data for all not yet trusted experts (in contrast to other experts, to boost their performance a bit more, as they did not manage to achieve good performance, yet):')
            print([exp for exp in not_yet_trusted_exp])
            for exp in not_yet_trusted_exp:
                print('Exp AI ' + exp[7] + ' is trained on complete data.')
                trainloader = DataLoader(multi_exp_dataset['exp_AI_' + exp[7] + '_train_data'], num_workers=2, shuffle=True, batch_size=batch_size)
                valloader = DataLoader(multi_exp_dataset['exp_AI_' + exp[7] + '_val_data'], num_workers=2, shuffle=False, batch_size=1)
                testloader = DataLoader(multi_exp_dataset['exp_AI_' + exp[7] + '_test_data'], num_workers=2, shuffle=False, batch_size=1)
                trainset_total = ConcatDataset([trainloader.dataset, testloader.dataset])
                # go to train mode for train data (rest is already test)
                if config.args.dyn_multi_transforms == 'adapted':
                    train_data_copy = copy.deepcopy(trainset_total)
                    _ = utils.set_transform(train_data_copy, 'train')
                else:
                    train_data_copy = trainset_total
                trainloader_total = DataLoader(train_data_copy, num_workers=2, shuffle=True, batch_size=batch_size)
                trainloader_total.dataset.name = 'exp_AI_' + exp[7] + '_dyn'
                if exp[7] == '1':
                    if config.args.pipe_case == 'benchmark':
                        classes = [str(s) + '-SVHN' for s in range(10)]
                    else:
                        classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                elif exp[7] == '2':
                    if config.args.pipe_case == 'benchmark':
                        classes = [str(s) + '-MNIST' for s in range(10)]
                    else:
                        classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                else:
                    if config.args.pipe_case == 'benchmark':
                        classes = ['T-shirt/top-FMNIST', 'Trouser-FMNIST', 'Pullover-FMNIST', 'Dress-FMNIST', 'Coat-FMNIST', 'Sandal-FMNIST',
                                   'Shirt-FMNIST', 'Sneaker-FMNIST', 'Bag-FMNIST', 'Ankle boot-FMNIST']
                    else:
                        classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                trainloader_total.dataset.classes = classes

                """# uncomment the following, if script was interrupted during retraining on train + test data
                # CAUTION: ADJUST id indexing!
                #if (config.args.dyn_multi_resume_expAI1 and id == 0) or (config.args.dyn_multi_resume_expAI2 and id == 1) or (config.args.dyn_multi_resume_expAI3 and id == 2):
                #    config.args.resume = 'models/' + config.args.pipe_case + '/DYN_PIPE/' + config.args.domain.upper() + '/' + 'exp_AI_' + str(id+1) + '_checkpoint.pth.tar'"""

                trained_experts['exp_AI_' + exp[7]] = mb.dyn_expert_creation(trainloader_total, valloader=valloader, name='exp_AI_' + exp[7])[0]

                """config.args.resume = ''
                if id == 0:
                    config.args.dyn_multi_resume_expAI1 = False
                elif id == 1:
                    config.args.dyn_multi_resume_expAI2 = False
                elif id == 2:
                    config.args.dyn_multi_resume_expAI3 = False"""

            if not config.args.dyn_MVP_test:
                # get standard test accuracy for all trained experts
                for id in range(num_experts):
                    print('Get standard test accuracy for all trained experts.')
                    _, _, std_testloader = md.get_data(str(id+1))
                    std_testloader_copy = copy.deepcopy(std_testloader)
                    _ = utils.set_transform(std_testloader_copy.dataset, 'test')

                    exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'].append(mb.test(trained_experts['exp_AI_' + str(id+1)], std_testloader_copy))

            # final tuning of all networks (only if not all experts have already been trusted before final batch)
            if config.args.dyn_multi_tune_dataset == 'pipe':
                all_exp = [gen_AI]
                for i in range(num_experts):
                    all_exp.append(trained_experts['exp_AI_' + str(i+1)])
            else:
                all_exp = []
                for i in range(num_experts):
                    # only append newly trusted experts as iSUN/UCM tuning does not change for old models
                    if trust_batchid_exp['exp_AI_' + str(i+1) + '_trust_batchid'] > (idx+1):
                        all_exp.append(trained_experts['exp_AI_' + str(i+1)])


            # gen AI and previously trusted experts are always retuned using old validation data
            print(config.args.dyn_multi_comb_mech + ' hyperparameter tuning is conducted for all experts - if there are not yet trusted experts present - as previous trusted experts are usually tuned again then. (gen AI only if not tuned on iSUN/UCM, as that already exists).')
            # gen AI and previously trusted experts are always retuned using old validation data
            # (not updated anymore after trust -> no further human effort through retuning), when new trusted experts are added
            if config.args.dyn_multi_comb_mech == 'odin' or config.args.dyn_multi_comb_mech == 'maha':

                if len(not_yet_trusted_exp) > 0:
                    print([exp.name if hasattr(exp, 'name') else exp.trainloader.dataset.name for exp in all_exp])

                    # tuning process
                    for id, exp in enumerate(all_exp):

                        # 1.: get data
                        if hasattr(exp, 'trainloader'):

                            # gen AI is only tuned, if pipe dataset is used (which is always the case for maha)
                            if config.args.dyn_multi_selec_mech == 'maha' or (config.args.dyn_multi_selec_mech == 'odin' and config.args.dyn_multi_tune_dataset == 'pipe'):
                                print('Gen AI is tuned.')

                                # inloader is the standard val data inloader
                                _, inloader, _ = md.get_data('0')
                                #inloader = DataLoader(Subset(inloader.dataset, [i for i in range(100)]), batch_size=1, shuffle=False)

                                # outloader is either pipe data or iSUN
                                if config.args.dyn_multi_tune_dataset == 'pipe':
                                    outloaderset = Subset(inloader.dataset, [])
                                    outdatasets = [1, 2, 3]
                                    print('outdatasets:')
                                    print(outdatasets)

                                    # collect sizes of outloader sets
                                    out_sizes = []
                                    for outdataset in outdatasets:
                                        out_sizes.append(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))

                                    for idx_i, outdataset in enumerate(outdatasets):

                                        # get other indices:
                                        other_ind = [i for i in range(num_experts) if i != idx_i]

                                        # outloader needs to be limited to 5k samples for computational reasons
                                        # make sure each dataset is represented to full length up until 1667 samples (1/3 of 5k)
                                        if len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']) <= 1667:
                                            outloaderset = ConcatDataset([outloaderset, multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']])
                                        else:
                                            assert num_experts == 3  # otherwise this structure does not work
                                            # if all out sets are > 1667 --> all at most 1667
                                            if out_sizes[other_ind[0]] > 1667 and out_sizes[other_ind[1]] > 1667:
                                                outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(1667, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])
                                            # if only this set > 1667 --> fill up until 5k or length of dataset
                                            elif out_sizes[other_ind[0]] <= 1667 and out_sizes[other_ind[1]] <= 1667:
                                                rem_size = 5000 - out_sizes[other_ind[0]] - out_sizes[other_ind[1]]
                                                outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(rem_size, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])
                                            # if one other set is > 1667 --> 50/50 share of remaining space
                                            elif (out_sizes[other_ind[0]] <= 1667 and out_sizes[other_ind[1]] > 1667) or (out_sizes[other_ind[0]] > 1667 and out_sizes[other_ind[1]] <= 1667):
                                                # get id of the one set < 1667:
                                                small_id = other_ind[0] if out_sizes[other_ind[0]] <= 1667 else other_ind[1]
                                                # check out total remaining space
                                                rem_space_for_both_big_sets = int(5000 - out_sizes[small_id])
                                                # distribute remaining space 50/50
                                                space_for_this_set = int(rem_space_for_both_big_sets / 2)
                                                outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(space_for_this_set, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])


                                    outloader = DataLoader(outloaderset, batch_size=1, num_workers=2, shuffle=True)
                                else:
                                    print('iSUN/UCM is loadad as outloader.')
                                    outloader = md.get_data('5')

                                """# make sure inloader has the same length as outloader
                                if len(inloader) >= len(outloader):
                                    inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), len(outloader.dataset))), batch_size=1, shuffle=False)
                                else:
                                    outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), len(inloader.dataset))), batch_size=1, shuffle=True)"""

                                #inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), 2)), batch_size=1, shuffle=False)
                                #outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), 2)), batch_size=1, shuffle=True)

                        else:

                            # 1.: get data
                            print('Exp AI ' + str(exp.name[7]) + ' is tuned.')

                            # inloader is the standard val data inloader
                            inloader = DataLoader(multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_val_data'], num_workers=2, shuffle=False, batch_size=1)

                            # outloader is either pipe data or iSUN
                            if config.args.dyn_multi_tune_dataset == 'pipe':
                                _, outloader1, _ = md.get_data('0')
                                # gen AI set with same length as outloaders (approx.)
                                outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(1666, len(multi_exp_dataset['exp_AI_' + str(1) + '_val_data']))))
                                #outloaderset = Subset(outloader1.dataset, [i for i in range(100)])
                                outdatasets = [i for i in range(1, 4) if i != int(exp.name[7])]
                                print('outdatasets:')
                                print(outdatasets)

                                out_sizes = []
                                for outdataset in outdatasets:
                                    out_sizes.append(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))

                                # determine length of C-10
                                if out_sizes[0] <= 1667 and out_sizes[1] <= 1667:
                                    # if the other sets are both < 1667 --> fill up remaining space with C-10
                                    rem_space = 5000 - out_sizes[0] - out_sizes[1]
                                    outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(rem_space, len(outloader1.dataset))))
                                elif out_sizes[0] > 1667 and out_sizes[1] > 1667:
                                    # if both other sets are > 1667 --> limit all sets to 1667
                                    outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(1667, len(outloader1.dataset))))
                                elif (out_sizes[0] > 1667 and out_sizes[1] <= 1667) or (out_sizes[0] <= 1667 and out_sizes[1] > 1667):
                                    # if one other set is > 1667 --> 50/50 space
                                    # get id of the one set < 1667:
                                    small_id = 0 if out_sizes[0] <= 1667 else 1
                                    # check out total remaining space
                                    rem_space_for_both_big_sets = int(5000 - out_sizes[small_id])
                                    # distribute remaining space 50/50
                                    space_for_this_set = int(rem_space_for_both_big_sets / 2)
                                    outloaderset = Subset(outloader1.dataset, random.sample(range(len(outloader1.dataset)), min(space_for_this_set, len(outloader1.dataset))))

                                for outdataset in outdatasets:
                                    if len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']) <= 1667:
                                        outloaderset = ConcatDataset([outloaderset, multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']])
                                    elif out_sizes[0] > 1667 and out_sizes[1] > 1667:
                                        outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(1667, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])
                                    else:
                                        outloaderset = ConcatDataset([outloaderset, Subset(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'], random.sample(range(len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data'])), min(space_for_this_set, len(multi_exp_dataset['exp_AI_' + str(outdataset) + '_val_data']))))])


                                outloader = DataLoader(outloaderset, batch_size=1, num_workers=2, shuffle=False)
                            else:
                                print('iSUN/UCM is loadad as outloader.')
                                outloader = md.get_data('5')

                            """# make sure inloader has the same length as outloader
                            if len(inloader) >= len(outloader):
                                inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), len(outloader.dataset))), batch_size=1, shuffle=False)
                            else:
                                outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), len(inloader.dataset))), batch_size=1, shuffle=True)"""


                        # 2.: actual tuning
                        if config.args.dyn_MVP_test:
                            inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader.dataset)), min(50, len(inloader.dataset)))), batch_size=1, shuffle=False)
                            outloader = DataLoader(Subset(outloader.dataset, random.sample(range(len(outloader.dataset)), min(50, len(outloader.dataset)))), batch_size=1, shuffle=True)

                        if config.args.dyn_multi_comb_mech == 'maha':
                            if not hasattr(exp, 'trainloader'):
                                # go to train mode for train data (rest is already test)
                                if config.args.dyn_multi_transforms == 'adapted':
                                    train_data_copy = copy.deepcopy(ConcatDataset([multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_train_data'], multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_test_data']]))
                                    _ = utils.set_transform(train_data_copy, 'train')
                                else:
                                    train_data_copy = multi_exp_dataset['exp_AI_' + str(exp.name[7]) + '_train_data']

                                trainloader = DataLoader(train_data_copy, num_workers=2, shuffle=True, batch_size=64 if config.args.pipe_case == 'benchmark' else 32)

                                best_delta, best_eps, _, _ = maha.tune_dyn_maha_params(exp, inloader, outloader, trainloader=trainloader)
                            else:
                                best_delta, best_eps, _, _ = maha.tune_dyn_maha_params(exp, inloader, outloader)

                        elif config.args.dyn_multi_comb_mech == 'odin':
                            best_delta, best_eps, _, _ = odin.tune_dyn_odin_params(exp, inloader, outloader, 0.95)

                    config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                    config.args.selec_mech = config.args.dyn_multi_selec_mech

            # train gating model
            elif config.args.dyn_multi_comb_mech == 'gating':
                if gating_score < config.args.gating_trust_thresh:
                    print('gating model is trained.')

                    # create train and valloaders with respective targets
                    gen_AI_trainloader, gen_AI_valloader, _ = md.get_data('0')
                    gating_trainset = copy.deepcopy(gen_AI_trainloader.dataset)
                    gating_valset = copy.deepcopy(gen_AI_valloader.dataset)
                    exp1_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_train_data'])
                    exp1_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_val_data'])
                    exp2_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_train_data'])
                    exp2_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_val_data'])
                    exp3_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_train_data'])
                    exp3_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_val_data'])

                    # allocation mechanism 2
                    if config.args.pipe_case == 'benchmark':
                        _ = utils.set_target(gating_trainset, 0, 'CIFAR10')
                        _ = utils.set_target(gating_valset, 0, 'CIFAR10')
                        _ = utils.set_target(exp1_trainset, 1, 'SVHN')
                        _ = utils.set_target(exp1_valset, 1, 'SVHN')
                        _ = utils.set_target(exp2_trainset, 2, 'MNIST')
                        _ = utils.set_target(exp2_valset, 2, 'MNIST')
                        _ = utils.set_target(exp3_trainset, 3, 'FashionMNIST')
                        _ = utils.set_target(exp3_valset, 3, 'FashionMNIST')
                    else:
                        _ = utils.set_target(gating_trainset, 0, 'Euro_SAT_countryside')
                        _ = utils.set_target(gating_valset, 0, 'Euro_SAT_countryside')
                        _ = utils.set_target(exp1_trainset, 1, 'FMOW')
                        _ = utils.set_target(exp1_valset, 1, 'FMOW')
                        _ = utils.set_target(exp2_trainset, 2, 'AID')
                        _ = utils.set_target(exp2_valset, 2, 'AID')
                        _ = utils.set_target(exp3_trainset, 3, 'RESISC')
                        _ = utils.set_target(exp3_valset, 3, 'RESISC')

                    train_length = min(len(gating_trainset), len(exp1_trainset), len(exp2_trainset), len(exp3_trainset))
                    val_length = min(len(gating_valset), len(exp1_valset), len(exp2_valset), len(exp3_valset))

                    gating_trainset = Subset(gating_trainset, random.sample(range(len(gating_trainset)), train_length))
                    gating_valset = Subset(gating_valset, random.sample(range(len(gating_valset)), val_length))
                    exp1_trainset = Subset(exp1_trainset, random.sample(range(len(exp1_trainset)), train_length))
                    exp1_valset = Subset(exp1_valset, random.sample(range(len(exp1_valset)), val_length))
                    exp2_trainset = Subset(exp2_trainset, random.sample(range(len(exp2_trainset)), train_length))
                    exp2_valset = Subset(exp2_valset, random.sample(range(len(exp2_valset)), val_length))
                    exp3_trainset = Subset(exp3_trainset, random.sample(range(len(exp3_trainset)), train_length))
                    exp3_valset = Subset(exp3_valset, random.sample(range(len(exp3_valset)), val_length))

                    gating_tot_trainset = ConcatDataset([gating_trainset, exp1_trainset, exp2_trainset, exp3_trainset])
                    gating_tot_valset = ConcatDataset([gating_valset, exp1_valset, exp2_valset, exp3_valset])

                    # go to train mode for train data (rest is already test)
                    if config.args.dyn_multi_transforms == 'adapted':
                        train_data_copy = copy.deepcopy(gating_tot_trainset)
                        _ = utils.set_transform(train_data_copy, 'train')
                    else:
                        train_data_copy = gating_tot_trainset

                    gating_train_batch = config.args.gating_train_batch if config.args.pipe_case == 'benchmark' else config.args.sat_gating_train_batch

                    gating_trainloader = DataLoader(train_data_copy, num_workers=2, batch_size=gating_train_batch, shuffle=True)
                    gating_valloader = DataLoader(gating_tot_valset, num_workers=2, batch_size=1, shuffle=True)

                    targets = []
                    for data, target in gating_trainloader:
                        #print(target)
                        for i in range(len(target)):
                            targets.append(target[i].item())
                    target_set_train = list(set(targets))
                    targets = []
                    for data, target in gating_valloader:
                        #print(target)
                        for i in range(len(target)):
                            targets.append(target[i].item())
                    target_set_val = list(set(targets))

                    assert target_set_train == [0, 1, 2, 3]
                    assert target_set_val == [0, 1, 2, 3]

                    gating_trainloader.dataset.name = 'gating'
                    gating_valloader.dataset.name = 'gating'
                    print('Len of gating train data: ' + str(len(gating_tot_trainset)))
                    print('Len of gating val data: ' + str(len(gating_tot_valset)))

                    # train model
                    _, gating_val_acc = mb.train_dyn_gating_model(gating_trainloader, gating_valloader)
                    print('gating val acc: ' + str(gating_val_acc))

                    # SELEC MECH GATING (allocation mechanism 1)
                    if config.args.dyn_multi_selec_mech == 'gating':
                        print('gating model for selec mech is trained.')
                        print('the selec mech gating model is not checked for a minimum score, as its performance is at least as good as the previous gating model by design (only 2 classes, same data)')

                        # create train and valloaders with respective targets
                        gen_AI_trainloader, gen_AI_valloader, _ = md.get_data('0')
                        gating_trainset = copy.deepcopy(gen_AI_trainloader.dataset)
                        gating_valset = copy.deepcopy(gen_AI_valloader.dataset)
                        exp1_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_train_data'])
                        exp1_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(1) + '_val_data'])
                        exp2_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_train_data'])
                        exp2_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(2) + '_val_data'])
                        exp3_trainset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_train_data'])
                        exp3_valset = copy.deepcopy(multi_exp_dataset['exp_AI_' + str(3) + '_val_data'])

                        if config.args.pipe_case == 'benchmark':
                            _ = utils.set_target(gating_trainset, 0, 'CIFAR10')
                            _ = utils.set_target(gating_valset, 0, 'CIFAR10')
                            _ = utils.set_target(exp1_trainset, 1, 'SVHN')
                            _ = utils.set_target(exp1_valset, 1, 'SVHN')
                            _ = utils.set_target(exp2_trainset, 1, 'MNIST')
                            _ = utils.set_target(exp2_valset, 1, 'MNIST')
                            _ = utils.set_target(exp3_trainset, 1, 'FashionMNIST')
                            _ = utils.set_target(exp3_valset, 1, 'FashionMNIST')
                        else:
                            _ = utils.set_target(gating_trainset, 0, 'Euro_SAT_countryside')
                            _ = utils.set_target(gating_valset, 0, 'Euro_SAT_countryside')
                            _ = utils.set_target(exp1_trainset, 1, 'FMOW')
                            _ = utils.set_target(exp1_valset, 1, 'FMOW')
                            _ = utils.set_target(exp2_trainset, 1, 'AID')
                            _ = utils.set_target(exp2_valset, 1, 'AID')
                            _ = utils.set_target(exp3_trainset, 1, 'RESISC')
                            _ = utils.set_target(exp3_valset, 1, 'RESISC')

                        train_length = min(len(gating_trainset), len(exp1_trainset), len(exp2_trainset), len(exp3_trainset))
                        val_length = min(len(gating_valset), len(exp1_valset), len(exp2_valset), len(exp3_valset))

                        gating_trainset = Subset(gating_trainset, random.sample(range(len(gating_trainset)), train_length))
                        gating_valset = Subset(gating_valset, random.sample(range(len(gating_valset)), val_length))
                        exp1_trainset = Subset(exp1_trainset, random.sample(range(len(exp1_trainset)), train_length))
                        exp1_valset = Subset(exp1_valset, random.sample(range(len(exp1_valset)), val_length))
                        exp2_trainset = Subset(exp2_trainset, random.sample(range(len(exp2_trainset)), train_length))
                        exp2_valset = Subset(exp2_valset, random.sample(range(len(exp2_valset)), val_length))
                        exp3_trainset = Subset(exp3_trainset, random.sample(range(len(exp3_trainset)), train_length))
                        exp3_valset = Subset(exp3_valset, random.sample(range(len(exp3_valset)), val_length))

                        gating_tot_trainset = ConcatDataset([gating_trainset, exp1_trainset, exp2_trainset, exp3_trainset])
                        gating_tot_valset = ConcatDataset([gating_valset, exp1_valset, exp2_valset, exp3_valset])

                        # go to train mode for train data (rest is already test)
                        if config.args.dyn_multi_transforms == 'adapted':
                            train_data_copy = copy.deepcopy(gating_tot_trainset)
                            _ = utils.set_transform(train_data_copy, 'train')
                        else:
                            train_data_copy = gating_tot_trainset

                        gating_trainloader = DataLoader(train_data_copy, num_workers=2, batch_size=config.args.gating_train_batch, shuffle=True)
                        gating_valloader = DataLoader(gating_tot_valset, num_workers=2, batch_size=1, shuffle=True)

                        targets = []
                        for data, target in gating_trainloader:
                            #print(target)
                            for i in range(len(target)):
                                targets.append(target[i].item())
                        target_set_train = list(set(targets))
                        targets = []
                        for data, target in gating_valloader:
                            #print(target)
                            for i in range(len(target)):
                                targets.append(target[i].item())
                        target_set_val = list(set(targets))

                        assert target_set_train == [0, 1]
                        assert target_set_val == [0, 1]

                        gating_trainloader.dataset.name = 'gating'
                        gating_valloader.dataset.name = 'gating'
                        print('Len of gating train data: ' + str(len(gating_tot_trainset)))
                        print('Len of gating val data: ' + str(len(gating_tot_valset)))

                        # train model
                        _, gating_val_acc = mb.train_dyn_gating_model(gating_trainloader, gating_valloader, type='selec')
                        print('gating selec mech val acc: ' + str(gating_val_acc))

                elif counter > 0 and gating_score >= config.args.gating_trust_thresh:
                    print('gating model is already trained and trusted.')

                config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                config.args.selec_mech = config.args.selec_mech

            elif config.args.dyn_multi_comb_mech == 'unc_thresh':
                config.args.pipe_tune_dataset = config.args.dyn_multi_tune_dataset
                config.args.selec_mech = config.args.dyn_multi_selec_mech
                raise NotImplementedError

            # use all experts
            experts = [trained_experts['exp_AI_' + str(id+1)] for id in range(num_experts)]

            # generate pipeline data
            # correct size and noise was set during generation of batches
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)
            pipeloader, pipe_true_labels_tot, _ = md.gen_pipe_data(config.args.pipe_size, config.args.pipe_noise)
            _ = utils.set_transform(pipeloader.dataset, 'test')

            # allocation mechanism 1
            predloader, pred_indices, forwardloader, forward_indices = select_instances(gen_AI, pipeloader)
            batch_forward_indices.append(forward_indices)
            # classify instances and get gen AI accuracy
            pred_step1 = pipe_pred(gen_AI, predloader)
            if config.args.selec_mech == 'odin' or config.args.selec_mech == 'maha':
                batch_true_labels_step1 = list(np.array(pipe_true_labels_tot)[pred_indices['ind']])
            else:
                batch_true_labels_step1 = list(np.array(pipe_true_labels_tot)[pred_indices])
            batch_true_labels_step2 = list(np.array(pipe_true_labels_tot)[forward_indices])
            step1_perf.append(pipe_eval(pred_step1, batch_true_labels_step1)[metric])

            # allocation mechanism 2
            pred_step2, batch_human_effort = call_experts(experts, forwardloader, batch_true_labels_step2, forward_indices)
            human_effort.append(batch_human_effort['#_inst']/len(pipe_true_labels_tot))
            pred_pipe_wH = pred_step1 + pred_step2
            pipe_true_labels_wH = batch_true_labels_step1 + batch_true_labels_step2
            pred_pipe_woH = pred_step1 + list(np.array(pred_step2)[batch_human_effort['no_human_ind_step2']])
            pipe_true_labels_woH = batch_true_labels_step1 + list(np.array(batch_true_labels_step2)[batch_human_effort['no_human_ind_step2']])

            # get final accuracy
            pipe_perf.append(pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[metric])
            config.args.dyn_pipe = False
            matrix = rs.allocation_matrix([gen_AI] + experts, pipe_true_labels_woH, pred_pipe_woH, tot_true_labels_for_dyn_pipe=batch_true_labels_tot)
            config.args.dyn_pipe = True

            # get expert accuracies
            for id in range(num_experts):
                if id == 0:
                    if config.args.pipe_case == 'benchmark':
                        ind_of_exp_pred = [True if i.endswith('-SVHN') else False for i in pred_pipe_woH]
                    elif config.args.pipe_case == 'sat':
                        exp_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                        ind_of_exp_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
                    tot_ind_of_exp_pred = tot_ind_of_exp_pred + sum(ind_of_exp_pred)
                elif id == 1:
                    if config.args.pipe_case == 'benchmark':
                        ind_of_exp_pred = [True if i.endswith('-MNIST') else False for i in pred_pipe_woH]
                    elif config.args.pipe_case == 'sat':
                        exp_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                        ind_of_exp_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
                    tot_ind_of_exp_pred = tot_ind_of_exp_pred + sum(ind_of_exp_pred)
                else:
                    if config.args.pipe_case == 'benchmark':
                        ind_of_exp_pred = [True if i.endswith('-FMNIST') else False for i in pred_pipe_woH]
                    elif config.args.pipe_case == 'sat':
                        exp_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                        ind_of_exp_pred = [True if i in exp_classes else False for i in pred_pipe_woH]
                    tot_ind_of_exp_pred = tot_ind_of_exp_pred + sum(ind_of_exp_pred)

                exp_pred = list(np.array(pred_pipe_woH)[ind_of_exp_pred])
                exp_true = list(np.array(pipe_true_labels_woH)[ind_of_exp_pred])
                exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'].append(pipe_eval(exp_pred, exp_true)[metric])
                exp_coverage['exp_AI_' + str(id+1) + '_coverage'].append((len(exp_pred)/len(batch_true_labels_tot)))

            #### BASELINES ####
            print('baselines are computed. May take a while..')
            # gen AI for all
            print('')
            print('gen AI for all baseline:')
            gen_AI_pred = pipe_pred(gen_AI, pipeloader)
            gen_AI_only.append(pipe_eval(gen_AI_pred, pipe_true_labels_tot)[0])

            # gen AI + HITL according to selec mech
            print('')
            print('gen AI + HITL according to selec mech baseline:')
            gen_AI_HITL_selec_pred = pred_step1 + batch_true_labels_step2
            gen_AI_HITL_afterselec.append(pipe_eval(gen_AI_HITL_selec_pred, batch_true_labels_step1+batch_true_labels_step2)[0])

            # gen AI + HITL according to optimal allocation
            print('')
            print('gen AI + HITL according to optimal allocation baseline:')
            if config.args.pipe_case == 'benchmark':
                gen_AI_pred = pipe_pred(gen_AI, DataLoader(Subset(pipeloader.dataset, [i for i, label in enumerate(pipe_true_labels_tot) if label.endswith('CIFAR10')]),
                                                           num_workers=2, shuffle=False, batch_size=1))
                gen_AI_trues = list(np.array(pipe_true_labels_tot)[[i for i, label in enumerate(pipe_true_labels_tot) if label.endswith('CIFAR10')]])
                HITL_trues = list(np.array(pipe_true_labels_tot)[[i for i, label in enumerate(pipe_true_labels_tot) if not label.endswith('CIFAR10')]])
                gen_AI_HITL.append(pipe_eval(gen_AI_pred+HITL_trues, gen_AI_trues+HITL_trues)[0])
            else:
                classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
                gen_AI_pred = pipe_pred(gen_AI, DataLoader(Subset(pipeloader.dataset, [i for i, label in enumerate(pipe_true_labels_tot) if label in classes]),
                                                           num_workers=2, shuffle=False, batch_size=1))
                gen_AI_trues = list(np.array(pipe_true_labels_tot)[[i for i, label in enumerate(pipe_true_labels_tot) if label in classes]])
                HITL_trues = list(np.array(pipe_true_labels_tot)[[i for i, label in enumerate(pipe_true_labels_tot) if label in classes]])
                gen_AI_HITL.append(pipe_eval(gen_AI_pred+HITL_trues, gen_AI_trues+HITL_trues)[0])

            # gen AI + perfect exp allocation according selec mech
            print('')
            print('gen AI + perfect exp allocation (incl. HITL) according to selec mech baseline:')
            human_pred_true = batch_true_labels_step2
            pred_tot = pred_step1
            true_tot = batch_true_labels_step1
            for exp in experts:
                if config.args.pipe_case == 'benchmark':
                    if exp.name[7] == '1':
                        title = '-SVHN'
                    elif exp.name[7] == '2':
                        title = '-MNIST'
                    elif exp.name[7] == '3':
                        title = '-FMNIST'
                    pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(forwardloader.dataset, [i for i, label in enumerate(batch_true_labels_step2) if label.endswith(title)]),
                                                                    num_workers=2, shuffle=False, batch_size=1))
                    true_tot = true_tot + list(np.array(batch_true_labels_step2)[[i for i, label in enumerate(batch_true_labels_step2) if label.endswith(title)]])
                    # remaining samples for human:
                    human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label.endswith(title)]))
                else:
                    if exp.name[7] == '1':
                        title_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                    elif exp.name[7] == '2':
                        title_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                    elif exp.name[7] == '3':
                        title_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                    pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(forwardloader.dataset, [i for i, label in enumerate(batch_true_labels_step2) if label in title_classes]),
                                                                    num_workers=2, shuffle=False, batch_size=1))
                    true_tot = true_tot + list(np.array(batch_true_labels_step2)[[i for i, label in enumerate(batch_true_labels_step2) if label in title_classes]])
                    # remaining samples for human:
                    human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in title_classes]))
            perf_exp_allocation_afterselec.append(pipe_eval(pred_tot+human_pred_true, true_tot+human_pred_true)[0])

            # gen AI + perfect exp allocation to optimal allocation
            print('')
            print('gen AI + perfect exp allocation (incl. HITL) according to optimal allocation baseline:')
            pred_tot = []
            true_tot = []
            human_pred_true = batch_true_labels_tot
            exps = experts + [gen_AI]
            for exp in exps:
                if config.args.pipe_case == 'benchmark':
                    if hasattr(exp, 'trainloader'):
                        title = '-CIFAR10'
                    else:
                        if exp.name[7] == '1':
                            title = '-SVHN'
                        elif exp.name[7] == '2':
                            title = '-MNIST'
                        elif exp.name[7] == '3':
                            title = '-FMNIST'
                    pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(pipeloader.dataset, [i for i, label in enumerate(pipe_true_labels_tot) if label.endswith(title)]),
                                                                    num_workers=2, shuffle=False, batch_size=1))
                    true_tot = true_tot + list(np.array(pipe_true_labels_tot)[[i for i, label in enumerate(pipe_true_labels_tot) if label.endswith(title)]])
                    # remaining samples for human:
                    human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label.endswith(title)]))
                if hasattr(exp, 'trainloader'):
                    title_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
                else:
                    if exp.name[7] == '1':
                        title_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                    elif exp.name[7] == '2':
                        title_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                    elif exp.name[7] == '3':
                        title_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(pipeloader.dataset, [i for i, label in enumerate(pipe_true_labels_tot) if label in title_classes]),
                                                                num_workers=2, shuffle=False, batch_size=1))
                true_tot = true_tot + list(np.array(pipe_true_labels_tot)[[i for i, label in enumerate(pipe_true_labels_tot) if label in title_classes]])
                # remaining samples for human:
                human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in title_classes]))
            perf_exp_allocation.append(pipe_eval(pred_tot+human_pred_true, true_tot+human_pred_true)[0])


            # results of final batch
            print('Gen AI acc. on claimed instances: ' + str(step1_perf[idx+1]))
            print('Pipe acc: ' + str(pipe_perf[idx+1]))
            print('Human effort: ' + str(human_effort[idx+1]))
            print('allocation matrix:')
            print(matrix)
            for id in range(num_experts):
                print('Expert AI ' + str(id+1) + ' (last) available dataset sizes: ')
                print('trainset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'][-1]))
                print('valset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'][-1]))
                print('testset size: ' + str(dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'][-1]))
                print('Expert AI ' + str(id+1) + ' acc on claimed instances: ')
                print(exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'][idx+1])
                if not config.args.dyn_MVP_test:
                    print('Expert AI ' + str(id+1) + ' acc on standard test set: ')
                    print(exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'][idx+1])
                print('Expert AI ' + str(id+1) + ' coverage: ')
                print(exp_coverage['exp_AI_' + str(id+1) + '_coverage'][idx+1])
            print('')


            # track results
            # mlflow
            if run_name != '':
                mlflow.set_experiment(experiment_name=exp_name)
                run_name_now = str('BATCH_' + str(idx+2) + '_' + run_name)
                mlflow.start_run(run_name=run_name_now)

                mlflow.log_metric('gen_AI_claimed_acc', step1_perf[-1])
                mlflow.log_metric('pipe_performance', pipe_perf[-1])
                mlflow.log_metric('human_effort', human_effort[-1])
                mlflow.log_metric('gen_AI_only', gen_AI_only[-1])
                #mlflow.log_metric('gen_AI_improvement', gen_AI_improvement[-1])
                mlflow.log_metric('perf_exp_allocation', perf_exp_allocation[-1])
                mlflow.log_metric('perf_exp_allocation_afterselec', perf_exp_allocation_afterselec[-1])
                mlflow.log_metric('gen_AI_HITL', gen_AI_HITL[-1])
                mlflow.log_metric('gen_AI_HITL_afterselec', gen_AI_HITL_afterselec[-1])

                mlflow.log_params({'pipe_case': config.args.pipe_case})

                # log odin/maha params
                if config.args.pipe_case == 'benchmark':
                    odin_maha_params = {
                        'gen_AI_odin_delta_pipe': config.args.c10_wideresnet_odin_delta_tpr95_pipe,
                        'gen_AI_odin_eps_pipe': config.args.c10_wideresnet_odin_eps_tpr95_pipe,
                        'gen_AI_odin_delta_ext': config.args.c10_wideresnet_odin_delta_tpr95_iSUN,
                        'gen_AI_odin_eps_ext': config.args.c10_wideresnet_odin_eps_tpr95_iSUN,
                        'gen_AI_maha_delta_pipe': config.args.c10_wideresnet_maha_delta_pipe_wFMNIST,
                        'gen_AI_maha_eps_pipe': config.args.c10_wideresnet_maha_eps_pipe_wFMNIST,
                    }
                else:
                    odin_maha_params = {
                        'gen_AI_odin_delta_pipe': config.args.eurosat_odin_delta_tpr95,
                        'gen_AI_odin_eps_pipe': config.args.eurosat_odin_eps_tpr95,
                        'gen_AI_odin_delta_ext': config.args.eurosat_odin_delta_UCM,
                        'gen_AI_odin_eps_ext': config.args.eurosat_odin_eps_UCM,
                        'gen_AI_maha_delta_pipe': config.args.eurosat_maha_delta,
                        'gen_AI_maha_eps_pipe': config.args.eurosat_maha_eps,
                    }

                odin_maha_params['exp_AI_1_odin_delta_pipe'] = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe
                odin_maha_params['exp_AI_1_odin_eps_pipe'] = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe
                odin_maha_params['exp_AI_1_odin_delta_ext'] = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_ext
                odin_maha_params['exp_AI_1_odin_eps_ext'] = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_ext
                odin_maha_params['exp_AI_1_maha_delta_pipe'] = config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe
                odin_maha_params['exp_AI_1_maha_eps_pipe'] = config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe

                odin_maha_params['exp_AI_2_odin_delta_pipe'] = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe
                odin_maha_params['exp_AI_2_odin_eps_pipe'] = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe
                odin_maha_params['exp_AI_2_odin_delta_ext'] = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_ext
                odin_maha_params['exp_AI_2_odin_eps_ext'] = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_ext
                odin_maha_params['exp_AI_2_maha_delta_pipe'] = config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe
                odin_maha_params['exp_AI_2_maha_eps_pipe'] = config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe

                odin_maha_params['exp_AI_3_odin_delta_pipe'] = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe
                odin_maha_params['exp_AI_3_odin_eps_pipe'] = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe
                odin_maha_params['exp_AI_3_odin_delta_ext'] = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_ext
                odin_maha_params['exp_AI_3_odin_eps_ext'] = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_ext
                odin_maha_params['exp_AI_3_maha_delta_pipe'] = config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe
                odin_maha_params['exp_AI_3_maha_eps_pipe'] = config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe

                mlflow.log_params(odin_maha_params)

                for id in range(num_experts):
                    mlflow.log_metric('trust_scores_exp_AI_' + str(id+1), trust_scores['exp_AI_' + str(id+1) + '_trustscores'][-1])
                    mlflow.log_metric('exp_claimed_acc_exp_AI_' + str(id+1), exp_claimed_acc['exp_AI_' + str(id+1) + '_claimed_acc'][-1])
                    if not config.args.dyn_MVP_test:
                        mlflow.log_metric('exp_test_acc_exp_AI_' + str(id+1), exp_test_acc['exp_AI_' + str(id+1) + '_test_acc'][-1])
                    mlflow.log_metric('exp_coverage_exp_AI_' + str(id+1), exp_coverage['exp_AI_' + str(id+1) + '_coverage'][-1])
                    mlflow.log_metric('trust_batchid_exp_AI_' + str(id+1), trust_batchid_exp['exp_AI_' + str(id+1) + '_trust_batchid'])
                    mlflow.log_metric('trainset_sizes_exp_AI_' + str(id+1), dataset_sizes['exp_AI_' + str(id+1) + '_trainset_size'][-1])
                    mlflow.log_metric('valset_sizes_exp_AI_' + str(id+1), dataset_sizes['exp_AI_' + str(id+1) + '_valset_size'][-1])
                    mlflow.log_metric('testset_sizes_exp_AI_' + str(id+1), dataset_sizes['exp_AI_' + str(id+1) + '_testset_size'][-1])

                # log allocation matrix
                for i in range(matrix.shape[0] - 1):
                    for j in range(matrix.shape[1] - 1):
                        mlflow.log_metric(str(i) + '/' + str(j), matrix.iloc[i, j])

                mlflow.end_run()

            # create copy of models in result folder
            for exp in experts:
                model_path = config.args.pipe_root + 'results/' + config.args.pipe_case +'_'+ config.args.domain + '_' + config.args.pipe_type + '/models/' + exp.name + '_' + config.args.selec_mech + config.args.comb_mech + '_lastRun{}.pickle'.format(config.args.ablation_study)
                filehandler = open(model_path, 'wb')
                pickle.dump(exp, filehandler)

        # create total result dictionary
        result = {
            'gen_AI_claimed_acc': step1_perf,
            'pipe_performance': pipe_perf,
            'human_effort': human_effort,
            'trust_scores': trust_scores,
            'exp_claimed_acc': exp_claimed_acc,
            'exp_test_acc': exp_test_acc,
            'exp_coverage': exp_coverage,
            'trust_batchid': trust_batchid_exp,
            'dataset_sizes': dataset_sizes,
            'gen_AI_only': gen_AI_only,
            'gen_AI_improvement': gen_AI_improvement,
            'perf_exp_allocation': perf_exp_allocation,
            'perf_exp_allocation_afterselec': perf_exp_allocation_afterselec,
            'gen_AI_HITL': gen_AI_HITL,
            'gen_AI_HITL_afterselec': gen_AI_HITL_afterselec,
        }

        return result


    # DYNAMIC, SINGLE-DOMAIN
    if config.args.domain == 'single':

        # iterate through data batches
        for idx, (train_batch, batch_true_labels_tot) in enumerate(train_batches):

            # enforce reproducibility
            main_seed = config.args.main_seed
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)

            # if resume: Data sets need to be rebuilt, as they are never saved for computational reasons
            if idx < start_batch:

                if idx < len(batch_forward_indices):

                    # allocation mechanism 1
                    forward_dataset = Subset(train_batch.dataset, batch_forward_indices[idx])
                    batch_true_labels_step2 = list(np.array(batch_true_labels_tot)[batch_forward_indices[idx]])
                    no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat

                    # allocate only weak samples to expert
                    # allocation based on human expert
                    print('Expert allocation of not predicted instances..')

                    print('.. based on HITL.')
                    exp_allocation = np.zeros((len(forward_dataset), 1))
                    for pos, i in enumerate(batch_true_labels_step2):
                        if i in gen_AI.trainloader.dataset.classes[no_strong_classes:]:
                            exp_allocation[pos, 0] = 1
                        else:
                            exp_allocation[pos, 0] = 99
                    exp_indices = list(np.where(exp_allocation == 1)[0])
                    exp_dataset = Subset(forward_dataset, exp_indices)
                    if config.args.dyn_multi_split == 'random':
                        split = [int(0.9*len(exp_dataset)), int(0.1*len(exp_dataset))]
                    else:
                        raise NotImplementedError
                    if split[0] + split[1] != len(exp_dataset):
                        split[0] = split[0] + 1

                    #### DATASET REBUILD ####
                    batch_trainset, batch_valset = random_split(exp_dataset, split, torch.Generator().manual_seed(config.args.main_seed))
                    gen_AI_train_data = ConcatDataset([gen_AI_train_data, batch_trainset])
                    gen_AI_test_data = ConcatDataset([gen_AI_test_data, batch_valset])
                    dataset_sizes['single_trainset_size'].append(len(gen_AI_train_data))
                    dataset_sizes['single_testset_size'].append(len(gen_AI_test_data))

                    print('Expert trainset size: ' + str(dataset_sizes['single_trainset_size'][-1]))
                    print('Expert testset size: ' + str(dataset_sizes['single_testset_size'][-1]))

            # DYNAMIC, SINGLE-DOMAIN AIITL-SYSTEM FROM SCRATCH
            else:

                print('Batch no. ' + str(idx+1) + ':')

                if config.args.dyn_multi_transforms == 'adapted':
                    # go to eval mode
                    _ = utils.set_transform(train_batch.dataset, 'test')

                # allocation mechanism 1
                predloader, pred_indices, forwardloader, forward_indices = select_instances(gen_AI, train_batch)
                batch_forward_indices.append(forward_indices)

                # EXPERT CREATION #
                if score < trust_thresh:

                    print('Expert creation..')
                    batch_true_labels_step2 = list(np.array(batch_true_labels_tot)[forward_indices])
                    no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat

                    # allocate only weak samples to expert
                    # allocation based on human expert
                    print('Expert allocation of not predicted instances..')

                    print('.. based on HITL.')
                    exp_allocation = np.zeros((len(forward_indices), 1))
                    for pos, i in enumerate(batch_true_labels_step2):
                        if i in gen_AI.trainloader.dataset.classes[no_strong_classes:]:
                            exp_allocation[pos, 0] = 1
                        else:
                            exp_allocation[pos, 0] = 99
                    exp_indices = list(np.where(exp_allocation == 1)[0])
                    exp_dataset = Subset(forwardloader.dataset, exp_indices)
                    if config.args.dyn_multi_split == 'random':
                        split = [int(0.9*len(exp_dataset)), int(0.1*len(exp_dataset))]
                    else:
                        raise NotImplementedError
                    if split[0] + split[1] != len(exp_dataset):
                        split[0] = split[0] + 1

                    # create datasets
                    batch_trainset, batch_valset = random_split(exp_dataset, split, torch.Generator().manual_seed(config.args.main_seed))
                    gen_AI_train_data = ConcatDataset([gen_AI_train_data, batch_trainset])
                    gen_AI_test_data = ConcatDataset([gen_AI_test_data, batch_valset])
                    dataset_sizes['single_trainset_size'].append(len(gen_AI_train_data))
                    dataset_sizes['single_testset_size'].append(len(gen_AI_test_data))

                    # adjust pre-processing
                    if config.args.dyn_multi_transforms == 'adapted':
                        train_data_copy = copy.deepcopy(gen_AI_train_data)
                        _ = utils.set_transform(train_data_copy, 'train')
                    else:
                        train_data_copy = copy.deepcopy(gen_AI_train_data)
                    val_data_copy = copy.deepcopy(gen_AI_test_data)

                    if config.args.pipe_case == 'benchmark':
                        batch_size = config.args.c10_wideresnet_train_batch
                    else:
                        batch_size = config.args.eurosat_train_batch

                    # adjust targets of train data to the number of classes trained on
                    _ = utils.reset_target(train_data_copy, no_strong_classes, 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'ImageFolder')
                    _ = utils.reset_target(val_data_copy, no_strong_classes, 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'ImageFolder')

                    # create train data and train expert
                    trainloader = DataLoader(train_data_copy, num_workers=2, shuffle=True, batch_size=batch_size)
                    trainloader.dataset.classes = gen_AI.trainloader.dataset.classes[no_strong_classes:]
                    trainloader.dataset.name = 'dyn_single'
                    valloader = DataLoader(val_data_copy, num_workers=2, shuffle=True, batch_size=1)
                    valloader.dataset.name = 'dyn_single'
                    network, train_acc, _ = mb.dyn_expert_creation(trainloader, name=gen_AI_exp_name)
                    print('Train acc: ' + str(train_acc))
                    test_acc = mb.test(network, valloader)
                    print('Test acc: ' + str(test_acc))

                    # check whether expert can be trusted
                    if test_acc < trust_thresh:
                        print('Expert is not trusted, yet. Further training needed. All instances are handled by HITL.')
                    else:
                        print('Expert is trusted now! Future batches will consult expert first.')
                        trust_batchid_exp['single'] = idx+1
                    trust_scores['single'].append(test_acc)

                ## TUNING (Softmax)
                if trust_batchid_exp['single'] == (idx+1):
                    print('uncertainty thresholds are tuned on pipe data (using existing, old data) for both gen AI and the new expert for future batches.')
                    print('seed in script (for control): ' + str(config.args.main_seed))
                    utils.set_seed(config.args.main_seed)
                    # tuning data needs to be a data batch consisting of both strong and weak samples and correct labels (--> last forwardloader)
                    valloader = forwardloader
                    valtruelabels = []
                    for _, target in valloader:
                        valtruelabels.append(gen_AI.trainloader.dataset.classes[target.item()])

                    # acc target for gen AI: try to achieve same performance on weak data together with HITL as on strong data
                    acc_target = [0.95, 0.95] if idx == 0 else [gen_AI_strong_acc, max(exp_test_acc_strong['single'])]
                    th.tune_all_gen_AI_thresholds_dyn([gen_AI, network], valloader, valtruelabels, acc_target=acc_target)

                # further deployment
                if score >= trust_thresh:
                    print('Inference with expert deployment..')
                else:
                    print('Inference with HITL..')
                pipe_true_labels_step1 = list(np.array(batch_true_labels_tot)[pred_indices])
                pipe_true_labels_step2 = list(np.array(batch_true_labels_tot)[forward_indices])
                pred_step1 = pipe_pred(gen_AI, predloader)
                step1_perf.append(pipe_eval(pred_step1, pipe_true_labels_step1)[metric])

                if score >= trust_thresh:

                    # allocation mechanism 2 (essentially the same as allocation mechanism 1)


                    thresh_cache = config.args.conf_thresh_singledom
                    thresh_cache_sat = config.args.conf_thresh_singledom_sat
                    config.args.conf_thresh_singledom = config.args.conf_thresh_singledom_exp
                    config.args.conf_thresh_singledom_sat = config.args.conf_thresh_singledom_exp
                    predloader2, pred_indices2, forwardloader2, forward_indices2 = select_instances(network, forwardloader)
                    config.args.conf_thresh_singledom = thresh_cache
                    config.args.conf_thresh_singledom_sat = thresh_cache_sat

                    # classify samples
                    pred_step11 = pipe_pred(network, predloader2)
                    pipe_true_labels_step11 = list(np.array(pipe_true_labels_step2)[pred_indices2])
                    exp_coverage['single'].append((len(predloader2.dataset)/len(batch_true_labels_tot)))

                    pred_step22 = list(np.array(pipe_true_labels_step2)[forward_indices2])
                    pred_pipe_wH2 = pred_step1 + pred_step11 + pred_step22
                    pipe_true_labels_wH2 = pipe_true_labels_step1 + pipe_true_labels_step11 + pred_step22
                    exp_claimed_acc['single'].append(pipe_eval(pred_step11, pipe_true_labels_step11)[metric])

                    # get final accuracy, matrix and human effort
                    pipe_perf.append(pipe_eval(pred_pipe_wH2, pipe_true_labels_wH2)[metric])
                    human_effort.append(len(forward_indices2) / len(batch_true_labels_tot))


                else:
                    # allocation mechanism 2 (human expert labels all samples)
                    pred_step2 = list(np.array(batch_true_labels_tot)[forward_indices])
                    human_effort.append(len(forward_indices) / len(batch_true_labels_tot))
                    exp_coverage['single'].append(0)
                    pred_pipe_wH = pred_step1 + pred_step2
                    pipe_true_labels_wH = pipe_true_labels_step1 + pred_step2
                    pipe_perf.append(pipe_eval(pred_pipe_wH, pipe_true_labels_wH)[metric])
                    exp_claimed_acc['single'].append(0)
                    if (idx+1) == len(train_batches):
                        print('WARNING: Trust threshold was never reached. Final test batch deploys expert AI at current quality.')

                # matrix
                if score >= trust_thresh:
                    network_list = [gen_AI, network]
                    matrix = rs.allocation_matrix(network_list, pipe_true_labels_step1+pipe_true_labels_step11, pred_step1+pred_step11, batch_true_labels_tot)
                else:
                    network_list = [gen_AI]
                    matrix = rs.allocation_matrix(network_list, pipe_true_labels_step1, pred_step1, batch_true_labels_tot)


                score = test_acc
                # do not keep a lower trust score, when already trusted due to early stopping
                if score < trust_thresh and trust_batchid_exp['single'] < 99:
                    score = 1

                if score < trust_thresh:
                    # check for early stopping (starting after patience iterations)
                    if idx >= patience and trust_batchid_exp['single'] != (idx+1) and config.args.ablation_study != 'no_trust_early_stop':
                        base_score = trust_scores['single'][int(-1*patience-1)]
                        deltas = []
                        for i in range((patience)):
                            deltas.append(abs(trust_scores['single'][-1*(i+1)] - base_score))
                        if sum([True if d <= max_delta else False for d in deltas]) == patience:

                            # early stopping!
                            print('Training is stopped early based on non-increasing trust_score.')
                            score = 1
                            trust_batchid_exp['single'] = (idx+1)
                            if trust_batchid_exp['single'] == (idx+1):
                                print('uncertainty thresholds are tuned on pipe data (using existing, old data) for both gen AI and the new expert for future batches.')
                                print('seed in script (for control): ' + str(config.args.main_seed))
                                utils.set_seed(config.args.main_seed)
                                # tuning data needs to be a data batch consisting of both strong and weak samples and correct labels (--> last forwardloader)
                                valloader = forwardloader
                                valtruelabels = []
                                for _, target in valloader:
                                    valtruelabels.append(gen_AI.trainloader.dataset.classes[target.item()])

                                acc_target = [0.95, 0.95] if idx == 0 else [gen_AI_strong_acc, max(exp_test_acc_strong['single'])]  # acc target is to achieve strong acc performance on whole set with HITL
                                th.tune_all_gen_AI_thresholds_dyn([gen_AI, network], valloader, valtruelabels, acc_target=acc_target)

                #### BASELINES ####
                print('baselines are computed. May take a while..')
                # gen AI for all
                print('')
                print('gen AI for all baseline:')
                gen_AI_pred = pipe_pred(gen_AI, train_batch)
                gen_AI_only.append(pipe_eval(gen_AI_pred, batch_true_labels_tot)[0])

                # gen AI + HITL according to selec mech
                print('')
                print('gen AI + HITL according to selec mech baseline:')
                gen_AI_HITL_selec_pred = pred_step1 + pipe_true_labels_step2
                gen_AI_HITL_afterselec.append(pipe_eval(gen_AI_HITL_selec_pred, pipe_true_labels_step1+pipe_true_labels_step2)[0])

                # gen AI + HITL according to optimal allocation
                print('')
                print('gen AI + HITL according to optimal allocation baseline:')
                classes = gen_AI.trainloader.dataset.classes
                no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat
                gen_AI_pred = pipe_pred(gen_AI, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label in classes[:no_strong_classes]]),
                                                           num_workers=2, shuffle=False, batch_size=1))
                gen_AI_trues = list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in classes[:no_strong_classes]]])
                HITL_trues = list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in classes[no_strong_classes:]]])
                gen_AI_HITL.append(pipe_eval(gen_AI_pred+HITL_trues, gen_AI_trues+HITL_trues)[0])

                # gen AI + perfect exp allocation according selec mech
                print('')
                print('gen AI + perfect exp allocation (incl. HITL) according to selec mech baseline:')
                human_pred_true = pipe_true_labels_step2
                pred_tot = pred_step1
                true_tot = pipe_true_labels_step1
                prev_trusted_exp = []
                if score >= trust_thresh and trust_batchid_exp['single'] < (idx+1):
                    prev_trusted_exp.append(network)
                for exp in prev_trusted_exp:
                    pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(forwardloader.dataset, [i for i, label in enumerate(pipe_true_labels_step2) if label in classes[no_strong_classes:]]),
                                                                    num_workers=2, shuffle=False, batch_size=1))
                    true_tot = true_tot + list(np.array(pipe_true_labels_step2)[[i for i, label in enumerate(pipe_true_labels_step2) if label in classes[no_strong_classes:]]])
                    # remaining samples for human:
                    human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in classes[no_strong_classes:]]))
                perf_exp_allocation_afterselec.append(pipe_eval(pred_tot+human_pred_true, true_tot+human_pred_true)[0])

                # gen AI + perfect exp allocation to optimal allocation
                print('')
                print('gen AI + perfect exp allocation (incl. HITL) according to optimal allocation baseline:')
                pred_tot = []
                true_tot = []
                human_pred_true = batch_true_labels_tot
                exps = prev_trusted_exp + [gen_AI]
                for exp in exps:
                    if hasattr(exp, 'trainloader'):
                        pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label in classes[:no_strong_classes]]),
                                                                        num_workers=2, shuffle=False, batch_size=1))
                        true_tot = true_tot + list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in classes[:no_strong_classes]]])
                        # remaining samples for human:
                        human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in classes[:no_strong_classes]]))
                    else:
                        pred_tot = pred_tot + pipe_pred(exp, DataLoader(Subset(train_batch.dataset, [i for i, label in enumerate(batch_true_labels_tot) if label in classes[no_strong_classes:]]),
                                                                        num_workers=2, shuffle=False, batch_size=1))
                        true_tot = true_tot + list(np.array(batch_true_labels_tot)[[i for i, label in enumerate(batch_true_labels_tot) if label in classes[no_strong_classes:]]])
                        # remaining samples for human:
                        human_pred_true = list(np.delete(human_pred_true, [i for i, label in enumerate(human_pred_true) if label in classes[no_strong_classes:]]))

                perf_exp_allocation.append(pipe_eval(pred_tot+human_pred_true, true_tot+human_pred_true)[0])

                # get test accuracy of trained expert
                print('Accuracy of exp on standard test set is computed.')
                _, _, std_testloader = md.get_data('0')
                _, _, _, _, _, std_testloader_strong = md.gen_strong_weak_genAIdata(config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat)
                std_testloader_copy = copy.deepcopy(std_testloader)
                std_testloader_strong_copy = copy.deepcopy(std_testloader_strong)
                _ = utils.set_transform(std_testloader_copy.dataset, 'test')
                _ = utils.set_transform(std_testloader_strong_copy.dataset, 'test')

                std_test_true = []
                for i in range(0, len(std_testloader_copy)):
                    std_test_true.append(gen_AI.trainloader.dataset.classes[std_testloader_copy.dataset[i][1]])
                std_test_pred = pipe_pred(network, std_testloader_copy)
                exp_test_acc['single'].append(pipe_eval(std_test_pred, std_test_true)[0])

                print('Accuracy of exp on strong test set is computed.')
                std_teststrong_true = []
                for i in range(0, len(std_testloader_strong_copy)):
                    std_teststrong_true.append(gen_AI.trainloader.dataset.classes[std_testloader_strong_copy.dataset[i][1]])
                std_teststrong_pred = pipe_pred(network, std_testloader_strong_copy)
                exp_test_acc_strong['single'].append(pipe_eval(std_teststrong_pred, std_teststrong_true)[0])

                if idx == 0:
                    print('Accuracy of gen AI on strong test set is computed.')
                    _, _, std_testloader_strong, _, _, _ = md.gen_strong_weak_genAIdata(config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat)
                    _, _, std_testloader = md.get_data('0')
                    gen_AI_strong_acc = mb.test(gen_AI, std_testloader_strong)

                    print('Accuracy of gen AI on std test set is computed.')
                    gen_AI_std_acc = mb.test(gen_AI, std_testloader)


                # print results
                print('Gen AI acc. on claimed instances: ' + str(step1_perf[idx]))
                print('Pipe acc: ' + str(pipe_perf[idx]))
                print('Human effort: ' + str(human_effort[idx]))
                print('')
                print('Allocation matrix:')
                print(matrix)
                print('')
                print('Expert dataset sizes:')
                print('trainset size: ' + str(dataset_sizes['single_trainset_size'][-1]))
                print('testset size: ' + str(dataset_sizes['single_testset_size'][-1]))
                print('Expert coverage: ' + str(exp_coverage['single'][idx]))
                print('Expert claimed accuracy: ' + str(exp_claimed_acc['single'][idx]))
                print('Expert standard test accuracy: ' + str(exp_test_acc['single'][idx]))
                print('Expert strong test accuracy: ' + str(exp_test_acc_strong['single'][idx]))
                print('')

                # track results if part of 'run all pipelines'
                # mlflow
                if run_name != '':
                    mlflow.set_experiment(experiment_name=exp_name)
                    run_name_now = str('BATCH_' + str(idx+1) + '_' + run_name)
                    mlflow.start_run(run_name=run_name_now)

                    mlflow.log_metric('gen_AI_claimed_acc', step1_perf[-1])
                    mlflow.log_metric('pipe_performance', pipe_perf[-1])
                    mlflow.log_metric('human_effort', human_effort[-1])
                    mlflow.log_metric('gen_AI_only', gen_AI_only[-1])
                    #mlflow.log_metric('gen_AI_improvement', gen_AI_improvement[-1])
                    mlflow.log_metric('perf_exp_allocation', perf_exp_allocation[-1])
                    mlflow.log_metric('perf_exp_allocation_afterselec', perf_exp_allocation_afterselec[-1])
                    mlflow.log_metric('gen_AI_HITL', gen_AI_HITL[-1])
                    mlflow.log_metric('gen_AI_HITL_afterselec', gen_AI_HITL_afterselec[-1])

                    mlflow.log_params({'pipe_case': config.args.pipe_case})

                    # track unc threshs
                    # log odin/maha params
                    if config.args.pipe_case == 'benchmark':
                        unc_thresh_params = {
                            'gen_AI_conf_thresh': config.args.conf_thresh_singledom
                        }
                    else:
                        unc_thresh_params = {
                            'gen_AI_conf_thresh': config.args.conf_thresh_singledom_sat
                        }

                    unc_thresh_params['exp_AI_conf_thresh'] = config.args.conf_thresh_singledom_exp

                    mlflow.log_params(unc_thresh_params)

                    mlflow.log_metric('trust_scores_exp_AI', trust_scores['single'][-1])
                    mlflow.log_metric('exp_claimed_acc_exp_AI', exp_claimed_acc['single'][-1])
                    mlflow.log_metric('exp_test_acc', exp_test_acc['single'][-1])
                    mlflow.log_metric('exp_test_acc_strong', exp_test_acc_strong['single'][-1])
                    mlflow.log_metric('gen_AI_test_acc', gen_AI_std_acc)
                    mlflow.log_metric('gen_AI_test_acc_strong', gen_AI_strong_acc)
                    mlflow.log_metric('exp_coverage', exp_coverage['single'][-1])
                    mlflow.log_metric('trust_batchid_exp_AI', trust_batchid_exp['single'])
                    mlflow.log_metric('trainset_sizes_exp_AI', dataset_sizes['single_trainset_size'][-1])
                    mlflow.log_metric('testset_sizes_exp_AI', dataset_sizes['single_testset_size'][-1])

                    # log allocation matrix
                    for i in range(matrix.shape[0] - 1):
                        for j in range(matrix.shape[1] - 1):
                            mlflow.log_metric(str(i) + '/' + str(j), matrix.iloc[i, j])

                    mlflow.end_run()

                # save checkpoint
                if config.args.resume == '':
                    checkpoint_save = config.args.pipe_root + config.args.dyn_checkpoint
                utils.save_checkpoint({
                    'train_batch': idx + 1,
                    'step1_perf': step1_perf,
                    'pipe_perf': pipe_perf,
                    'human_effort': human_effort,
                    'batch_forward_ind': batch_forward_indices,

                    'dyn_single_train_epoch': config.args.c10_epochs if config.args.pipe_case == 'benchmark' else config.args.eurosat_epochs,
                    'trust_expert': score >= trust_thresh,
                    'trust_scores': trust_scores['single'],
                    'exp_claimed_acc': exp_claimed_acc['single'],
                    'exp_test_acc': exp_test_acc['single'],
                    'exp_test_acc_strong': exp_test_acc['single'],
                    'genAI_test_acc': gen_AI_std_acc,
                    'genAI_test_acc_strong': gen_AI_strong_acc,
                    'exp_coverage': exp_coverage['single'],
                    'gen_AI_bench_thresh': config.args.conf_thresh_singledom,
                    'gen_AI_sat_thresh': config.args.conf_thresh_singledom_sat,
                    'gen_AI_exp_thresh': config.args.conf_thresh_singledom_exp,
                }, checkpoint=checkpoint_save, filename='checkpoint_dyn_single.pth.tar')

        # create total result dictionary
        result = {
            'gen_AI_claimed_acc': step1_perf,
            'pipe_performance': pipe_perf,
            'human_effort': human_effort,
            'trust_scores': trust_scores,
            'exp_claimed_acc': exp_claimed_acc,
            'exp_test_acc': exp_test_acc,
            'exp_coverage': exp_coverage,
            'trust_batchid_exp': trust_batchid_exp,
            'dataset_sizes': dataset_sizes
        }

        # create copy of models in result folder
        model_path = config.args.pipe_root + 'results/' + config.args.pipe_case +'_'+ config.args.domain +'_'+ config.args.pipe_type + '/models/' + 'expert_lastRun{}.pickle'.format(config.args.ablation_study)
        filehandler = open(model_path, 'wb')
        pickle.dump(network, filehandler)

        return result
