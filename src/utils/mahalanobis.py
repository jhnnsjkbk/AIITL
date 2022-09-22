# this code is intended to provide all functions needed for the Maha allocation mechanism
# major parts of the code are based on https://github.com/pokaxpoka/deep_Mahalanobis_detector

# general libraries
import os.path
import time
import src.utils.utils as utils
import torch
import numpy as np
import src.config
import src.data.make_data as md
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import sklearn.covariance
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import Tuple, Any, Dict, Callable, Sequence
from torch import Tensor


# get config
config = src.config.cfg


def maha_mean_cov(network: object,
                  num_classes: int,
                  feature_list: Sequence[Tensor],
                  trainloader: DataLoader) -> Tuple[Sequence, Sequence]:
    """
    Compute sample mean and precision (inverse of covariance)
    Code is based on `"deep_Mahalanobis_detector" <https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/90c2105e78c6f76a2801fc4c1cb1b84f4ff9af63/lib_generation.py#L45>`_

    :param network: Network object
    :param num_classes: number of classes
    :param feature_list: Sequence of intermediate forward outputs
    :param trainloader: DataLoader of training data
    :return: sample_class_mean, precision
    """

    # paths to save or load mean and precision
    if hasattr(network, 'trainloader'):
        if config.args.pipe_case == 'benchmark':
            mean_path = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_' + 'wideresnet' + (config.args.ablation_study if config.args.ablation_study != 'smaller_OOD_share_v2' else '') + '_mean.pickle'
            cov_path = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_' + 'wideresnet' + (config.args.ablation_study if config.args.ablation_study != 'smaller_OOD_share_v2' else '') + '_cov.pickle'
        elif config.args.pipe_case == 'sat':
            mean_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + (config.args.ablation_study if config.args.ablation_study != 'smaller_OOD_share_v2' else '') + '_mean.pickle'
            cov_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + (config.args.ablation_study if config.args.ablation_study != 'smaller_OOD_share_v2' else '') + '_cov.pickle'
    else:
        mean_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/MAHA/' + network.name + config.args.ablation_study + (('_' + 'wideresnet') if config.args.pipe_case == 'benchmark' else '') + '_mean.pickle'
        cov_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/MAHA/' + network.name + config.args.ablation_study + (('_' + 'wideresnet') if config.args.pipe_case == 'benchmark' else '') + '_cov.pickle'

    # check whether mean_cov already exists
    if os.path.isfile(mean_path) and os.path.isfile(cov_path):
        print('Training data class means and covariance are loaded.')
        mean_file = open(mean_path, 'rb')
        sample_class_mean = pickle.load(mean_file)
        cov_file = open(cov_path, 'rb')
        precision = pickle.load(cov_file)

        return sample_class_mean, precision

    # compute mean and precision
    else:
        print('Training data class means and covariance are computed.')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.zeros(num_classes)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in trainloader:
            total += data.size(0)
            data = data.to(device)
            output, out_features = network.feature_list(data)

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                            = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1

        # get means
        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        # get precision (covariance)
        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        # save
        mean_file = open(mean_path, 'wb')
        pickle.dump(sample_class_mean, mean_file)
        cov_file = open(cov_path, 'wb')
        pickle.dump(precision, cov_file)

        time.sleep(5)

        return sample_class_mean, precision


def get_maha_layer_score(network: object,
                         testloader: DataLoader,
                         num_classes: int,
                         sample_mean: Sequence,
                         precision: Sequence,
                         layer_index: int,
                         magnitude: float) -> Sequence[float]:
    """
    Compute the Maha score for the considered layer.
    Code is based on `"deep_Mahalanobis_detector" <https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/90c2105e78c6f76a2801fc4c1cb1b84f4ff9af63/lib_generation.py#L45>`_

    :param network: Network object
    :param testloader: DataLoader of test dataset
    :param num_classes: Number of classes
    :param sample_mean: calculated training mean
    :param precision: calculated training covariance
    :param layer_index: Number of layers considered
    :param magnitude: parameter for input perturbation
    :return: Maha score
    """

    print('Mahalanobis scores are computed for layer ' + str(layer_index))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network.eval()
    Mahalanobis = []

    for data, target in testloader:

        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # get output of respective layer
        out_features = network.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # initialization
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        # compute Mahalanobis score
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = torch.add(data.data, -magnitude, gradient)
        tempInputs.requires_grad = False
        noise_out_features = network.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

    return Mahalanobis


def tune_maha_params(network: object,
                     tpr_target: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Tunes the hyperparameters of the considered network model

    :param network: Network object
    :param tpr_target: Targeted TPR, common sense is 0.95
    :return: best_delta, best_eps, best_tnr, tpr
    """

    # initialization
    num_layers = len(network.feature_list_sizes())
    num_classes = len(np.unique(network.trainloader.dataset.classes))


    # reduce batch size temporarily
    batch_size_cache = utils.adjust_train_batch_size(network, 'reduce')

    # get inloader and outloader + trainloader
    if config.args.pipe_case == 'benchmark':
        # get data based on in-distribution dataset key
        trainloader, _, _ = md.get_data(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                  index(network.trainloader.dataset.name)])
        inloader, outloader = md.get_maha_tune_data(list(config.BENCHMARK_DATA_SETS.keys())[list(config.BENCHMARK_DATA_SETS.values()).
                                                index(network.trainloader.dataset.name)])
    elif config.args.pipe_case == 'sat':
        # get data based on in-distribution dataset key
        trainloader, _, _ = md.get_data(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                        index(network.trainloader.dataset.name)])
        inloader, outloader = md.get_maha_tune_data(list(config.SAT_DATA_SETS.keys())[list(config.SAT_DATA_SETS.values()).
                                                   index(network.trainloader.dataset.name)])

    # in distribution set to 1, according to paper and odin, but different to their git code
    y_true = np.concatenate((np.ones(len(inloader), dtype="int32"), np.zeros(len(outloader), dtype="int32")))

    # create two seperate validation sets, one for training a logistic regressor, one for finding and validating best delta
    if config.args.pipe_case == 'benchmark':
        assert len(inloader) == 5000
        inloader_lr = DataLoader(Subset(inloader.dataset, [i for i in range(5000) if i % 2 == 0]), batch_size=1, shuffle=True, num_workers=2)
        outloader_lr = DataLoader(Subset(outloader.dataset, [i for i in range(5000) if i % 2 == 0]), batch_size=1, shuffle=True, num_workers=2)
        y_true_lr = np.concatenate((np.ones(len(inloader_lr), dtype="int32"), np.zeros(len(outloader_lr), dtype="int32")))
        inloader_delta = DataLoader(Subset(inloader.dataset, [i for i in range(5000) if i % 2 == 1]), batch_size=1, shuffle=True, num_workers=2)
        outloader_delta = DataLoader(Subset(outloader.dataset, [i for i in range(5000) if i % 2 == 1]), batch_size=1, shuffle=True, num_workers=2)
        y_true_delta = np.concatenate((np.ones(len(inloader_delta), dtype="int32"), np.zeros(len(outloader_delta), dtype="int32")))
    elif config.args.pipe_case == 'sat':
        #assert len(inloader) == 1000
        inloader_lr = DataLoader(Subset(inloader.dataset, [i for i in range(len(inloader.dataset)) if i % 2 == 0]), batch_size=1, shuffle=True, num_workers=2)
        outloader_lr = DataLoader(Subset(outloader.dataset, [i for i in range(len(outloader.dataset)) if i % 2 == 0]), batch_size=1, shuffle=True, num_workers=2)
        y_true_lr = np.concatenate((np.ones(len(inloader_lr), dtype="int32"), np.zeros(len(outloader_lr), dtype="int32")))
        inloader_delta = DataLoader(Subset(inloader.dataset, [i for i in range(len(inloader.dataset)) if i % 2 == 1]), batch_size=1, shuffle=True, num_workers=2)
        outloader_delta = DataLoader(Subset(outloader.dataset, [i for i in range(len(outloader.dataset)) if i % 2 == 1]), batch_size=1, shuffle=True, num_workers=2)
        y_true_delta = np.concatenate((np.ones(len(inloader_delta), dtype="int32"), np.zeros(len(outloader_delta), dtype="int32")))


    # magnitude (eps) tuning
    sample_class_mean, precision = maha_mean_cov(network, num_classes, network.feature_list_sizes(), trainloader)
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    if network.trainloader.dataset.name == 'MNIST':
        delta_list = torch.linspace(-10, 100, 10000).tolist()
    elif network.trainloader.dataset.name == 'FMNIST':
        delta_list = torch.linspace(-10, 30, 10000).tolist()
    elif config.args.pipe_case == 'sat':
        if trainloader.dataset.name == 'RESISC':
            delta_list = torch.linspace(-100, 1000, 100000).tolist()
        else:
            delta_list = torch.linspace(-100, 30, 100000).tolist()
    else:
        delta_list = torch.linspace(-40, 10, 10000).tolist()
    best_tnr = 0
    for idx, magnitude in enumerate(m_list):

        # STEP 1: CREATE REGRESSOR ON LR VAL SPLIT
        print('Magnitude no. ' + str(idx+1) + ' out of ' + str(len(m_list)) + ' (' + str(magnitude) + ') is tried out.')
        print('Step1: Build regressor for respective magnitude.')
        maha_scores_tot_lr = np.zeros((y_true_lr.shape[0], num_layers))
        for layer in range(num_layers):
            maha_score_in_lr = get_maha_layer_score(network, inloader_lr, num_classes, sample_class_mean, precision, layer,
                                                 magnitude)
            maha_score_out_lr = get_maha_layer_score(network, outloader_lr, num_classes, sample_class_mean, precision, layer,
                                                  magnitude)
            maha_score_layer_tot_lr = np.concatenate((maha_score_in_lr, maha_score_out_lr), axis=0)
            maha_scores_tot_lr[:, layer] = maha_score_layer_tot_lr

        X_train_lr = maha_scores_tot_lr
        y_train_lr = y_true_lr
        if network.trainloader.dataset.name == 'MNIST':
            scaler = StandardScaler().fit(X_train_lr)
            X_train_lr = scaler.transform(X_train_lr)
            scaler_path = config.args.pipe_root + 'models/benchmark/MAHA/MNIST_scaler_' + 'wideresnet' + '_' + config.args.pipe_tune_dataset + '_wFMNIST.pickle'
            filehandler = open(scaler_path, 'wb')
            pickle.dump(scaler, filehandler)
        if network.trainloader.dataset.name == 'FMNIST':
            scaler = StandardScaler().fit(X_train_lr)
            X_train_lr = scaler.transform(X_train_lr)
            scaler_path = config.args.pipe_root + 'models/benchmark/MAHA/FMNIST_scaler_' + 'wideresnet' + '_' + config.args.pipe_tune_dataset + '.pickle'
            filehandler = open(scaler_path, 'wb')
            pickle.dump(scaler, filehandler)
        if config.args.pipe_case == 'sat':
            scaler = StandardScaler().fit(X_train_lr)
            X_train_lr = scaler.transform(X_train_lr)
            if network.trainloader.dataset.name != 'Euro_SAT_countryside':
                scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_scaler_' + config.args.pipe_tune_dataset + '.pickle'
            else:
                scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_scaler_' + config.args.pipe_tune_dataset + '.pickle'
            filehandler = open(scaler_path, 'wb')
            pickle.dump(scaler, filehandler)
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train_lr, y_train_lr)

        # STEP 2: COMPUTE SCORES FOR DELTA VAL SPLIT AND GET DELTA FOR TPR 95
        print('Step2: Find delta at TPR' + str(int(100*tpr_target)))
        maha_scores_tot_delta = np.zeros((y_true_delta.shape[0], num_layers))
        for layer in range(num_layers):
            maha_score_in_delta = get_maha_layer_score(network, inloader_delta, num_classes, sample_class_mean, precision, layer,
                                                    magnitude)
            maha_score_out_delta = get_maha_layer_score(network, outloader_delta, num_classes, sample_class_mean, precision, layer,
                                                     magnitude)
            maha_score_layer_tot_delta = np.concatenate((maha_score_in_delta, maha_score_out_delta), axis=0)
            maha_scores_tot_delta[:, layer] = maha_score_layer_tot_delta

        if network.trainloader.dataset.name == 'MNIST':
            maha_scores_tot_delta = scaler.transform(maha_scores_tot_delta)
        if network.trainloader.dataset.name == 'FMNIST':
            maha_scores_tot_delta = scaler.transform(maha_scores_tot_delta)
        if config.args.pipe_case == 'sat':
            maha_scores_tot_delta = scaler.transform(maha_scores_tot_delta)
        for layer in range(num_layers):
            maha_scores_tot_delta[:, layer] = maha_scores_tot_delta[:, layer] * lr.coef_[0][layer]
        maha_scores = list(np.sum(maha_scores_tot_delta, axis=1))

        for d in delta_list:
            # simulate prediction as would have resulted for a given delta
            y_pred = []
            for j in range(len(y_true_delta)):
                if maha_scores[j] > d:
                    y_pred.append(1)  # in distribution
                else:
                    y_pred.append(0)  # out distribution

            # get the delta that results in a TPR of 95% --> end d gridsearch
            tn, fp, fn, tp = confusion_matrix(y_true_delta, y_pred).ravel()
            tpr = tp/(tp+fn)
            if tpr <= tpr_target:  # smaller as starting with a small delta, at the beginning everything is labelled as 1 --> tpr will be 1
                tnr = tn/(tn+fp)
                break


        # end tuning
        if tnr > best_tnr:
            best_tnr = tnr
            best_delta = d
            best_eps = magnitude
            # save models
            if config.args.pipe_case == 'benchmark':
                if config.args.pipe_tune_dataset == 'iSUN':
                    filename_lr = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_' + 'wideresnet' + '_maha_lr' + '.pickle'
                elif config.args.pipe_tune_dataset == 'pipe':
                    filename_lr = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_' + 'wideresnet' + '_maha_lr_pipe' + '_wFMNIST' + '.pickle'
            elif config.args.pipe_case == 'sat':
                if network.trainloader.dataset.name != 'Euro_SAT_countryside':
                    filename_lr = config.args.pipe_root + '/models/sat/MAHA/' + network.trainloader.dataset.name + '_maha_lr' + '.pickle'
                else:
                    filename_lr = config.args.pipe_root + '/models/sat/MAHA/' + network.trainloader.dataset.name + '_maha_lr' + '.pickle'


            filehandler = open(filename_lr, 'wb')
            pickle.dump(lr, filehandler)

    # switch back batch size
    utils.adjust_train_batch_size(network, 'switch_back', batch_size_cache)


    return best_delta, best_eps, best_tnr, tpr


def tune_all_maha(networks: Sequence[object]) -> None:
    """
    Function to tune mahalanobis for all the networks considered.

    :param networks: Sequence of networks considered
    """

    # tuning needs to be activated
    if config.args.pipe_tune_thresholds:
        maha_results = pd.DataFrame(columns=[model.trainloader.dataset.name for model in networks], index=['best_delta', 'best_eps', 'best_tnr', 'tpr'])
        for model in networks:
            best_delta, best_eps, best_tnr, tpr = tune_maha_params(model, tpr_target=0.95)
            maha_result = [best_delta, best_eps, best_tnr, tpr]
            maha_results[model.trainloader.dataset.name] = maha_result
            print('MAHA results:')
            print(maha_results[model.trainloader.dataset.name])

            # set parameters to respective config arguments
            if config.args.pipe_case == 'benchmark':
                if config.args.pipe_tune_dataset == 'pipe':

                    if model.trainloader.dataset.name == 'CIFAR10':
                        config.args.c10_wideresnet_maha_eps_pipe_wFMNIST = best_eps
                        config.args.c10_wideresnet_maha_delta_pipe_wFMNIST = best_delta
                    elif model.trainloader.dataset.name == 'SVHN':
                        config.args.svhn_wideresnet_maha_eps_pipe_wFMNIST = best_eps
                        config.args.svhn_wideresnet_maha_delta_pipe_wFMNIST = best_delta
                    elif model.trainloader.dataset.name == 'MNIST':
                        config.args.mnist_wideresnet_maha_eps_pipe_wFMNIST = best_eps
                        config.args.mnist_wideresnet_maha_delta_pipe_wFMNIST = best_delta
                    elif model.trainloader.dataset.name == 'FMNIST':
                        config.args.fmnist_wideresnet_maha_eps_pipe = best_eps
                        config.args.fmnist_wideresnet_maha_delta_pipe = best_delta
            elif config.args.pipe_case == 'sat':
                if model.trainloader.dataset.name == 'Euro_SAT_countryside':
                    config.args.eurosat_maha_eps = best_eps
                    config.args.eurosat_maha_delta = best_delta
                elif model.trainloader.dataset.name == 'FMOW':
                    config.args.fmow_maha_eps = best_eps
                    config.args.fmow_maha_delta = best_delta
                elif model.trainloader.dataset.name == 'AID':
                    config.args.aid_maha_eps = best_eps
                    config.args.aid_maha_delta = best_delta
                elif model.trainloader.dataset.name == 'RESISC':
                    config.args.resisc_maha_eps = best_eps
                    config.args.resisc_maha_delta = best_delta
                else:
                    raise NotImplementedError
        


        print('MAHA results:')
        print(maha_results)
        print('')

    else:
        print('Maha tuning already conducted. Respective parameters are provided in config.py')



def tune_dyn_maha_params(network: object,
                         inloader: DataLoader,
                         outloader: DataLoader,
                         tpr_target: float = 0.95,
                         trainloader: DataLoader = None) -> Tuple[float, float, float, float]:
    """
    Function to tune mahalanobis for the network considered in the dynamic system.

    :param network: Network object
    :param inloader: DataLoader of in-distribution data (for tuning)
    :param outloader: Dataloader of out-of-distribtion data (for tuning)
    :param tpr_target: Desired TPR target
    :param trainloader: Dataloader of training data (for class mean, precision)
    :return: best_delta, best_eps, best_tnr, tpr
    """

    # initialize
    num_layers = len(network.feature_list_sizes())
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
        raise NotImplementedError


    # reduce batch size temporarily
    batch_size_cache = utils.adjust_train_batch_size(network, 'reduce')

    # get inloader and outloader + trainloader if desired
    if config.args.pipe_case == 'benchmark':
        working_batch_size = 64
    elif config.args.pipe_case == 'sat':
        working_batch_size = 32
    if trainloader == None:
        trainloader = DataLoader(network.trainloader.dataset, num_workers=2, shuffle=True, batch_size=working_batch_size)


    # if to be changed: dont forget to switch application in pipe.py as well (if 1, then ...)
    y_true = np.concatenate((np.ones(len(inloader), dtype="int32"), np.zeros(len(outloader), dtype="int32")))   # in distribution set to 1, according to paper and odin, but different to their git code

    # create two seperate validation sets, one for training a logistic regressor, one for finding and validating best delta
    inloader_lr = DataLoader(Subset(inloader.dataset, [i for i in range(len(inloader)) if i % 2 == 0]), batch_size=1, shuffle=True, num_workers=2)
    outloader_lr = DataLoader(Subset(outloader.dataset, [i for i in range(len(outloader)) if i % 2 == 0]), batch_size=1, shuffle=True, num_workers=2)
    y_true_lr = np.concatenate((np.ones(len(inloader_lr), dtype="int32"), np.zeros(len(outloader_lr), dtype="int32")))
    inloader_delta = DataLoader(Subset(inloader.dataset, [i for i in range(len(inloader)) if i % 2 == 1]), batch_size=1, shuffle=True, num_workers=2)
    outloader_delta = DataLoader(Subset(outloader.dataset, [i for i in range(len(outloader)) if i % 2 == 1]), batch_size=1, shuffle=True, num_workers=2)
    y_true_delta = np.concatenate((np.ones(len(inloader_delta), dtype="int32"), np.zeros(len(outloader_delta), dtype="int32")))

    # magnitude (eps) tuning
    sample_class_mean, precision = maha_mean_cov(network, num_classes, network.feature_list_sizes(), trainloader)
    if config.args.dyn_MVP_test:
        m_list = [0.0]
    else:
        m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    delta_list = torch.linspace(-100, 1000, 100000).tolist()
    best_tnr = 0
    for idx, magnitude in enumerate(m_list):

        # STEP 1: CREATE REGRESSOR ON LR VAL SPLIT
        print('Magnitude no. ' + str(idx+1) + ' out of ' + str(len(m_list)) + ' (' + str(magnitude) + ') is tried out.')
        print('Step1: Build regressor for respective magnitude.')
        maha_scores_tot_lr = np.zeros((y_true_lr.shape[0], num_layers))
        for layer in range(num_layers):
            maha_score_in_lr = get_maha_layer_score(network, inloader_lr, num_classes, sample_class_mean, precision, layer,
                                                    magnitude)
            maha_score_out_lr = get_maha_layer_score(network, outloader_lr, num_classes, sample_class_mean, precision, layer,
                                                     magnitude)
            maha_score_layer_tot_lr = np.concatenate((maha_score_in_lr, maha_score_out_lr), axis=0)
            maha_scores_tot_lr[:, layer] = maha_score_layer_tot_lr

        X_train_lr = maha_scores_tot_lr
        y_train_lr = y_true_lr
        if not hasattr(network, 'trainloader'):
            if network.name == 'exp_AI_2_dyn' and config.args.pipe_case == 'benchmark':
                scaler = StandardScaler().fit(X_train_lr)
                X_train_lr = scaler.transform(X_train_lr)
                scaler_path = config.args.pipe_root + 'models/benchmark/MAHA/exp_AI_2_scaler_' + config.args.ablation_study + 'wideresnet' + '_' + 'pipe' + '_wFMNIST' + '.pickle'
                filehandler = open(scaler_path, 'wb')
                pickle.dump(scaler, filehandler)
        if config.args.pipe_case == 'sat':
            scaler = StandardScaler().fit(X_train_lr)
            X_train_lr = scaler.transform(X_train_lr)
            if not hasattr(network, 'trainloader'):
                scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.name + '_DYN_scaler' + config.args.ablation_study + '.pickle'
            else:
                scaler_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_DYN_scaler' + config.args.ablation_study + '.pickle'
            filehandler = open(scaler_path, 'wb')
            pickle.dump(scaler, filehandler)
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train_lr, y_train_lr)

        # STEP 2: COMPUTE SCORES FOR DELTA VAL SPLIT AND GET DELTA FOR TPR 95
        print('Step2: Find delta at TPR' + str(int(100*tpr_target)))
        maha_scores_tot_delta = np.zeros((y_true_delta.shape[0], num_layers))
        for layer in range(num_layers):
            maha_score_in_delta = get_maha_layer_score(network, inloader_delta, num_classes, sample_class_mean, precision, layer,
                                                       magnitude)
            maha_score_out_delta = get_maha_layer_score(network, outloader_delta, num_classes, sample_class_mean, precision, layer,
                                                        magnitude)
            maha_score_layer_tot_delta = np.concatenate((maha_score_in_delta, maha_score_out_delta), axis=0)
            maha_scores_tot_delta[:, layer] = maha_score_layer_tot_delta

        if not hasattr(network, 'trainloader'):
            if network.name == 'exp_AI_2_dyn' and config.args.pipe_case == 'benchmark':
                maha_scores_tot_delta = scaler.transform(maha_scores_tot_delta)
        if config.args.pipe_case == 'sat':
            maha_scores_tot_delta = scaler.transform(maha_scores_tot_delta)
        for layer in range(num_layers):
            maha_scores_tot_delta[:, layer] = maha_scores_tot_delta[:, layer] * lr.coef_[0][layer]
        maha_scores = list(np.sum(maha_scores_tot_delta, axis=1))

        for d in delta_list:
            # simulate prediction as would have resulted for a given delta
            y_pred = []
            for j in range(len(y_true_delta)):
                if maha_scores[j] > d:
                    y_pred.append(1)  # in distribution
                else:
                    y_pred.append(0)  # out distribution

            # get the delta that results in a TPR of 95% --> end d gridsearch
            tn, fp, fn, tp = confusion_matrix(y_true_delta, y_pred).ravel()
            tpr = tp/(tp+fn)
            if tpr <= tpr_target:  # smaller as starting with a small delta, at the beginning everything is labelled as 1 --> tpr will be 1
                tnr = tn/(tn+fp)
                break

        # end tuning
        if tnr >= best_tnr:
            best_tnr = tnr
            best_delta = d
            best_eps = magnitude
            # save models
            if config.args.pipe_case == 'benchmark':
                if hasattr(network, 'trainloader'):
                    filename_lr = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_DYN' + 'wideresnet' + '_maha_lr_pipe' + '_wFMNIST' + config.args.ablation_study + '.pickle'
                else:
                    filename_lr = config.args.pipe_root + 'models/benchmark/MAHA/' + network.name + '_DYN' + 'wideresnet' + '_maha_lr_pipe' + '_wFMNIST' + config.args.ablation_study + '.pickle'
            elif config.args.pipe_case == 'sat':
                if hasattr(network, 'trainloader'):
                    filename_lr = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_DYN_maha_lr_pipe' + config.args.ablation_study + '.pickle'
                else:
                    filename_lr = config.args.pipe_root + 'models/sat/MAHA/' + network.name + '_DYN_maha_lr_pipe' + config.args.ablation_study +'.pickle'
            filehandler = open(filename_lr, 'wb')
            pickle.dump(lr, filehandler)

    # switch back batch size
    utils.adjust_train_batch_size(network, 'switch_back', batch_size_cache)

    # print results
    maha_results = pd.DataFrame(columns=['gen AI', 'EXP_AI_1', 'EXP_AI_2', 'EXP_AI_3'], index=['best_delta', 'best_eps', 'best_tnr', 'tpr'])
    maha_result = [best_delta, best_eps, best_tnr, tpr]
    if hasattr(network, 'trainloader'):
        key = network.trainloader.dataset.name
    else:
        key = network.name
    maha_results[key] = maha_result
    print('MAHA results:')
    print(maha_results[key])

    # set parameters to respective config arguments
    if hasattr(network, 'trainloader'):
        if network.trainloader.dataset.name == 'CIFAR10':
            config.args.c10_wideresnet_maha_delta_pipe_wFMNIST = best_delta
            config.args.c10_wideresnet_maha_eps_pipe_wFMNIST = best_eps
        elif network.trainloader.dataset.name == 'Euro_SAT_countryside':
            config.args.eurosat_maha_delta = best_delta
            config.args.eurosat_maha_eps = best_eps
    else:
        if network.name == 'exp_AI_1_dyn':
            config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe = best_delta
            config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe = best_eps
        elif network.name == 'exp_AI_2_dyn':
            config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe = best_delta
            config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe = best_eps
        elif network.name == 'exp_AI_3_dyn':
            config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe = best_delta
            config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe = best_eps

    return best_delta, best_eps, best_tnr, tpr
