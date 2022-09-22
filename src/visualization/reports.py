# this code is intended to provide evaluation reports

# general libraries
import copy
import numpy as np
import pandas as pd
import src.config
import src.data.make_data as md
import src.utils.model_building as mb
import src.utils.pipe as pipe
from torch.utils.data import DataLoader, Subset
import random
from typing import Tuple, Sequence, Any, Dict, Callable
import src.utils.utils as utils

# get config
config = src.config.cfg



def allocation_matrix(network_list: Sequence[object],
                      true_labels: Sequence[str],
                      pred: Sequence[str],
                      tot_true_labels_for_dyn_pipe: Sequence[str] = []) -> pd.DataFrame:
    """
    Computes the allocation matrix of data samples in the AIITL-system, whose i-th row and j-th column entry indicates the number of samples with true label belonging to i-th network and allocated label belonging to j-th network..

    :param network_list: sequence of network objects of the AIITL-system
    :param true_labels: sequence of total true labels
    :param pred: sequence of total labels predicted
    :return: pandas DataFrame
    """

    # copy needed to avoid destruction of original object
    true_labels_copy = copy.deepcopy(true_labels)
    pred_copy = copy.deepcopy(pred)
    tot_true_labels_for_dyn_pipe_copy = copy.deepcopy(tot_true_labels_for_dyn_pipe)

    print(len(tot_true_labels_for_dyn_pipe_copy))

    # intialize matrix
    if config.args.domain == 'multi':
        matrix = np.zeros((len(network_list), len(network_list)+1))
    else:
        if config.args.dyn_pipe:
            matrix = np.zeros((2, 3))
        else:
            matrix = np.zeros((2, 2))


    if config.args.domain == 'multi':

        # adjust class names of sat case
        if config.args.pipe_case == 'sat':
            genAI_classes = network_list[0].trainloader.dataset.classes
            exp1_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
            exp2_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
            exp3_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
            genAI_name = network_list[0].trainloader.dataset.name
            exp1_name = 'FMOW'
            exp2_name = 'AID'
            exp3_name = 'RESISC'

            for i in range(len(pred_copy)):
                if pred_copy[i] in genAI_classes:
                    pred_copy[i] = pred_copy[i] + '-' + genAI_name
                elif pred_copy[i] in exp1_classes:
                    pred_copy[i] = pred_copy[i] + '-' + exp1_name
                elif pred_copy[i] in exp2_classes:
                    pred_copy[i] = pred_copy[i] + '-' + exp2_name
                elif pred_copy[i] in exp3_classes:
                    pred_copy[i] = pred_copy[i] + '-' + exp3_name
                else:
                    raise NotImplementedError

            for i in range(len(true_labels_copy)):
                if true_labels_copy[i] in genAI_classes:
                    true_labels_copy[i] = true_labels_copy[i] + '-' + genAI_name
                elif true_labels_copy[i] in exp1_classes:
                    true_labels_copy[i] = true_labels_copy[i] + '-' + exp1_name
                elif true_labels_copy[i] in exp2_classes:
                    true_labels_copy[i] = true_labels_copy[i] + '-' + exp2_name
                elif true_labels_copy[i] in exp3_classes:
                    true_labels_copy[i] = true_labels_copy[i] + '-' + exp3_name
                else:
                    raise NotImplementedError

            for i in range(len(tot_true_labels_for_dyn_pipe_copy)):
                if tot_true_labels_for_dyn_pipe_copy[i] in genAI_classes:
                    tot_true_labels_for_dyn_pipe_copy[i] = tot_true_labels_for_dyn_pipe_copy[i] + '-' + genAI_name
                elif tot_true_labels_for_dyn_pipe_copy[i] in exp1_classes:
                    tot_true_labels_for_dyn_pipe_copy[i] = tot_true_labels_for_dyn_pipe_copy[i] + '-' + exp1_name
                elif tot_true_labels_for_dyn_pipe_copy[i] in exp2_classes:
                    tot_true_labels_for_dyn_pipe_copy[i] = tot_true_labels_for_dyn_pipe_copy[i] + '-' + exp2_name
                elif tot_true_labels_for_dyn_pipe_copy[i] in exp3_classes:
                    tot_true_labels_for_dyn_pipe_copy[i] = tot_true_labels_for_dyn_pipe_copy[i] + '-' + exp3_name
                else:
                    raise NotImplementedError

        # for each network
        for i in range(len(network_list)):

            # collect true label
            if hasattr(network_list[i], 'trainloader'):
                true_label = network_list[i].trainloader.dataset.name
            else:
                if config.args.pipe_case == 'benchmark':
                    if network_list[i].name == 'exp_AI_1_dyn':
                        true_label = 'SVHN'
                    elif network_list[i].name == 'exp_AI_2_dyn':
                        true_label = 'MNIST'
                    elif network_list[i].name == 'exp_AI_3_dyn':
                        true_label = 'FMNIST'
                elif config.args.pipe_case == 'sat':
                    if network_list[i].name == 'exp_AI_1_dyn':
                        true_label = 'FMOW'
                    elif network_list[i].name == 'exp_AI_2_dyn':
                        true_label = 'AID'
                    elif network_list[i].name == 'exp_AI_3_dyn':
                        true_label = 'RESISC'

            # collect allocated network
            for j in range(len(network_list)+1):
                # classic matrix entries
                if j != (len(network_list)):
                    if hasattr(network_list[j], 'trainloader'):
                        allocated_label = network_list[j].trainloader.dataset.name
                    else:
                        if config.args.pipe_case == 'benchmark':
                            if network_list[j].name == 'exp_AI_1_dyn':
                                allocated_label = 'SVHN'
                            elif network_list[j].name == 'exp_AI_2_dyn':
                                allocated_label = 'MNIST'
                            elif network_list[j].name == 'exp_AI_3_dyn':
                                allocated_label = 'FMNIST'
                        elif config.args.pipe_case == 'sat':
                            if network_list[j].name == 'exp_AI_1_dyn':
                                allocated_label = 'FMOW'
                            elif network_list[j].name == 'exp_AI_2_dyn':
                                allocated_label = 'AID'
                            elif network_list[j].name == 'exp_AI_3_dyn':
                                allocated_label = 'RESISC'

                    # compute matrix entry
                    ind = [True if x.endswith('-' + true_label) else False for idx, x in enumerate(true_labels_copy)]
                    matrix[i][j] = int(sum([True if x.endswith('-' + allocated_label) else False for x in list(np.array(pred_copy)[ind])]))

                # sums
                else:
                    if not config.args.dyn_pipe:
                        if config.args.ablation_study != 'smaller_OOD_share_v2':
                            if i == 0:
                                matrix[i][j] = int(config.args.pipe_size) - np.sum(matrix[i, :j])
                            else:
                                matrix[i][j] = int((config.args.pipe_noise[i-1])*config.args.pipe_size) - np.sum(matrix[i, :j])
                        else:
                            if i == 0:
                                matrix[i][j] = int(config.args.pipe_size) - np.sum(matrix[i, :j])
                            else:
                                matrix[i][j] = int((config.args.pipe_noise[i-1])*config.args.smaller_OOD_share_v2_batch_size) - np.sum(matrix[i, :j])
                    else:
                        if i == 0:
                            matrix[i][j] = sum([a.endswith('-CIFAR10') or a.endswith('Euro_SAT_countryside') for a in tot_true_labels_for_dyn_pipe_copy]) - np.sum(matrix[i, :j])
                        elif i == 1:
                            matrix[i][j] = sum([a.endswith('-SVHN') or a.endswith('FMOW_utilities') or a.endswith('FMOW') for a in tot_true_labels_for_dyn_pipe_copy]) - np.sum(matrix[i, :j])
                        elif i == 2:
                            matrix[i][j] = sum([a.endswith('-MNIST') or a.endswith('FMOW_public') or a.endswith('AID') for a in tot_true_labels_for_dyn_pipe_copy]) - np.sum(matrix[i, :j])
                        elif i == 3:
                            matrix[i][j] = sum([a.endswith('-FMNIST') or a.endswith('FMOW_private') or a.endswith('RESISC') for a in tot_true_labels_for_dyn_pipe_copy]) - np.sum(matrix[i, :j])

        # concatenate final matrix
        matrix = np.concatenate((matrix, np.sum(matrix, axis=0).reshape((1, len(network_list)+1))), axis=0)
        matrix = np.concatenate((matrix, np.sum(matrix, axis=1).reshape((len(network_list)+1, 1))), axis=1)
        matrix = np.rint(matrix)


    else:

        # pre-process data
        no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat
        classes = network_list[0].trainloader.dataset.classes
        # mark samples according to strong and weak samples
        for i in range(len(pred_copy)):
            if pred_copy[i] in classes[:no_strong_classes]:
                pred_copy[i] = pred_copy[i] + '-' + 'strong'
            else:
                pred_copy[i] = pred_copy[i] + '-' + 'weak'
        for i in range(len(true_labels_copy)):
            if true_labels_copy[i] in classes[:no_strong_classes]:
                true_labels_copy[i] = true_labels_copy[i] + '-' + 'strong'
            else:
                true_labels_copy[i] = true_labels_copy[i] + '-' + 'weak'
        for i in range(len(tot_true_labels_for_dyn_pipe_copy)):
            if tot_true_labels_for_dyn_pipe_copy[i] in classes[:no_strong_classes]:
                tot_true_labels_for_dyn_pipe_copy[i] = tot_true_labels_for_dyn_pipe_copy[i] + '-' + 'strong'
            else:
                tot_true_labels_for_dyn_pipe_copy[i] = tot_true_labels_for_dyn_pipe_copy[i] + '-' + 'weak'


        # collect true label
        for i in range(np.shape(matrix)[0]):
            true_label = 'strong' if i == 0 else 'weak'

            # collect allocated label
            for j in range(np.shape(matrix)[1]):
                # classic entry
                if j != (np.shape(matrix)[1] - 1):
                    allocated_label = 'strong' if j==0 else 'weak'
                    ind = [True if x.endswith('-' + true_label) else False for idx, x in enumerate(true_labels_copy)]
                    matrix[i][j] = int(sum([True if x.endswith('-' + allocated_label) else False for x in list(np.array(pred_copy)[ind])]))
                # sums
                else:
                    if i == 0:
                        matrix[i][j] = sum([a.endswith('-strong') for a in tot_true_labels_for_dyn_pipe_copy]) - np.sum(matrix[i, :j])
                    elif i == 1:
                        matrix[i][j] = sum([a.endswith('-weak') for a in tot_true_labels_for_dyn_pipe_copy]) - np.sum(matrix[i, :j])

        # concatenate final matrix
        if config.args.dyn_pipe:
            if config.args.domain == 'single':
                matrix = np.concatenate((matrix, np.sum(matrix, axis=1).reshape((2, 1))), axis=1)
                matrix = np.concatenate((matrix, np.sum(matrix, axis=0).reshape((1, 4))), axis=0)
            else:
                matrix = np.concatenate((matrix, np.sum(matrix, axis=0).reshape((1, len(network_list)+1))), axis=0)
                matrix = np.concatenate((matrix, np.sum(matrix, axis=1).reshape((len(network_list)+1, 1))), axis=1)
        else:
            if config.args.domain == 'single':
                matrix = np.concatenate((matrix, np.sum(matrix, axis=1).reshape((2, 1))), axis=1)
                matrix = np.concatenate((matrix, np.sum(matrix, axis=0).reshape((1, 3))), axis=0)
            else:
                matrix = np.concatenate((matrix, np.sum(matrix, axis=0).reshape((1, len(network_list)+1))), axis=0)
                matrix = np.concatenate((matrix, np.sum(matrix, axis=1).reshape((len(network_list)+1, 1))), axis=1)

        matrix = np.rint(matrix)

    # create output
    if config.args.domain == 'multi':
        index = ['Gen_AI', 'Exp_AI_1', 'Exp_AI_2', 'Exp_AI_3']
        columns = ['Gen_AI', 'Exp_AI_1', 'Exp_AI_2', 'Exp_AI_3']
    else:
        index = ['Gen_AI_strong', 'Gen_AI_weak']
        columns = ['Gen_AI', 'Gen_AI_exp'] if config.args.dyn_pipe else ['Gen_AI']
    output = pd.DataFrame(data=matrix, columns=columns + ['HITL', 'Total'],
                          index=index + ['Total'], dtype='int32')

    return output