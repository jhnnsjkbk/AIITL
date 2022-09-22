# this code is intended to provide everything needed to train and access models

# general libraries
import torch
import torch.nn as nn
import src.config
import src.data.make_data as md
import src.utils.utils as utils
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pickle
import torch.optim as optims
from src.models.wideresnet import WideResNet
from src.models.densenet import initialize_model, intermediate_forward, feature_list, feature_list_sizes
import src.models.resnet as resnet
import os.path
from torchvision import models
import copy
from typing import Tuple, Sequence, Any, Dict, Callable
from torch import Tensor

# get config
config = src.config.cfg


def train(network: object,
          trainloader: DataLoader,
          optim: object,
          loss_fn: Callable[[Sequence[Tensor], Sequence[Tensor]], float],
          valloader: DataLoader = None,
          download: bool = True,
          model_path: str = '') -> Any:

    """
    Train and save deep neural networks.

    :param network: Network object
    :param trainloader: DataLoader of training dataset
    :param optim: Optimizer object
    :param loss_fn: Loss function
    :param valloader: Optional DataLoader of validation dataset
    :param download: Defines if the model should be saved in the specified model_path
    :param model_path: path to save trained model in (file-name of model is created automatically)
    :return Trained network and further information (e.g., train and val loss)
    """

    # parameter for early stopping, if needed
    if config.args.dyn_pipe:
        max_delta = config.args.dyn_valloss_max_delta
        patience = config.args.dyn_valloss_patience
    if config.args.pipe_case == 'sat' and not config.args.dyn_pipe and trainloader.dataset.name != 'Euro_SAT_countryside':
        max_delta = config.args.dyn_valloss_max_delta
        patience = 15

    # initialize lists
    vallosses = []
    valacc = []
    trainacc = []
    trainlosses = []

    # initialize lr-scheduler and phase
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()
    if valloader == None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200, verbose=True)  # no lr scheduling based on val loss
        phases = ['train']
    else:
        if config.args.pipe_case == 'sat' and trainloader.dataset.name != 'Euro_SAT_countryside':
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.96, verbose=True)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True, patience=10)  # lr scheduling based on val loss
        phases = ['train', 'val']

    # get number of epochs and log interval
    epochs, log_interval_fix = utils.get_epochs_loginterval(trainloader)

    start_epoch = 0

    # Load checkpoint if desired
    if config.args.resume != '':
        print('')
        print('==> Resuming from checkpoint..')
        if os.path.isfile(config.args.pipe_root + config.args.resume):
            resume_file = config.args.pipe_root + config.args.resume
        else:
            resume_file = os.path.join(config.args.pipe_root + config.args.resume, 'checkpoint.pth.tar')
        assert os.path.isfile(resume_file), 'Error: no checkpoint directory found!'
        checkpoint_save = os.path.dirname(resume_file)
        checkpoint_info = torch.load(resume_file)
        start_epoch = checkpoint_info['epoch']
        vallosses = checkpoint_info['vallosses']
        trainlosses = checkpoint_info['trainlosses']
        trainacc = checkpoint_info['trainacc']
        valacc = checkpoint_info['valacc']
        running_loss = checkpoint_info['running_loss']
        correct = checkpoint_info['correct']
        total = checkpoint_info['total']
        network.load_state_dict(checkpoint_info['state_dict'])
        optim.load_state_dict(checkpoint_info['optimizer'])


    # loop over data several times
    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % int(epoch+1))

        for phase in phases:

            if phase == 'train':
                network.train()
                torch.set_grad_enabled(True)
                loader = trainloader
                log_interval = log_interval_fix
                print('Training loop..')
            else:
                network.eval()
                torch.set_grad_enabled(False)
                loader = valloader
                log_interval = max(1, len(valloader) - 1)
                print('Validation loop..')

            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optim.zero_grad()

                # forward + backward + optimize
                output = network(data)
                loss = loss_fn(output, target)
                if phase == 'train':
                    loss.backward()
                    optim.step()

                # print statistics/log
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                if batch_idx % log_interval == 0:
                    if phase == 'train':
                        utils.progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (running_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    else:
                        utils.progress_bar(batch_idx, len(loader), 'Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)' % (running_loss/(batch_idx+1), 100.*correct/total, correct, total))

            if phase == 'train':
                trainacc.append(correct/total)
                trainlosses.append(running_loss/(batch_idx+1))

            else:
                valacc.append(correct/total)

            # lr-scheduler steps
            if len(phases) == 1:
                if trainloader.dataset.name == 'FMNIST' or trainloader.dataset.name == 'global':
                    utils.adjust_learning_rate(optim, epoch)
                else:
                    scheduler.step()
            else:
                if phase == 'val':
                    if trainloader.dataset.name != 'FMNIST':
                        if scheduler.__class__.__name__ == 'StepLR':
                            scheduler.step()
                        else:
                            scheduler.step(running_loss/(batch_idx+1))
                    else:
                        utils.adjust_learning_rate(optim, epoch)


        # check for early stopping
        if config.args.train_early_stop:
            if (config.args.dyn_pipe) or \
                    (config.args.pipe_case == 'sat' and trainloader.dataset.name != 'Euro_SAT_countryside'):
                vallosses.append(running_loss/(batch_idx+1))
                if epoch >= patience:
                    base_score = vallosses[int(-1*patience-1)]
                    deltas = []
                    for i in range((patience)):
                        deltas.append(base_score - vallosses[-1*(i+1)])
                    if sum([True if d <= max_delta else False for d in deltas]) == patience:
                        # early stopping!
                        print('Training is stopped early due to non-decreasing val loss.')
                        break

        # save checkpoint
        if config.args.resume == '':
            checkpoint_save = config.args.pipe_root + config.args.checkpoint
        if download == False and model_path != '':
            filename = model_path + '_checkpoint.pth.tar'
        else:
            filename = 'checkpoint.pth.tar'
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'vallosses': vallosses,
            'trainlosses': trainlosses,
            'valacc': valacc,
            'trainacc': trainacc,
            'running_loss': running_loss,
            'correct': correct,
            'total': total,
            'state_dict': network.state_dict(),
            'optimizer': optim.state_dict()
        }, checkpoint=checkpoint_save, filename=filename)

    final_loss = running_loss/(batch_idx+1)
    final_acc = correct/total

    # duration
    now = time.time()
    dur = now - start
    # create automatic naming for saved model
    if download:
        model_path = model_path + '.pth'
        torch.save(network.state_dict(), model_path)

        print('Training finished. Duration: ' + utils.format_time(dur))
        return network, model_path, final_loss

    print('Training finished. Duration: ' + utils.format_time(dur))

    # needed for dynamic pipeline to avoid any errors
    torch.set_grad_enabled(True)

    if config.args.ablation_study == 'fmow_overfit':
        return network, _, vallosses, valacc, trainlosses, trainacc

    else:
        return network, final_acc


def test(network: object,
         testloader: DataLoader) -> float:
    """
    Tests a trained network model on the test data and returns accuracy

    :param network: Network object
    :param testloader: DataLoader of test dataset
    :return: accuracy on testset
    """

    # initialize values
    network.eval()
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get log interval
    _, log_interval = utils.get_epochs_loginterval(testloader)

    print('Testing started.')

    # no need for gradients
    # compute predictions
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = network(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % log_interval == 0:
                utils.progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    accuracy = 100 * correct / total

    print('Testing done.')

    if not config.args.dyn_pipe:
        print('Test accuracy: ' + str(accuracy) + '%')

    return accuracy/100


def build_train_test_net(network: object,
                         data: str,
                         optim: object,
                         loss_fn: Callable[[Sequence[Tensor], Sequence[Tensor]], float],
                         tune_params: bool = False,
                         model_path: str = '') -> Tuple[object, DataLoader, DataLoader]:
    """
    Acutal creation of models.
    Retrieves data, trains, tests and saves a specified model.

    :param network: Network object
    :param data: Key of dataset as indicated in config.py
    :param optim: Optimizer object
    :param loss_fn: Loss function
    :param model_path: Path to save trained ML model (file-name is created automatically)
    :return network, trainloader, testloader
    """

    # save parameters to model metadata
    optim_cache = optim
    loss_fn_cache = loss_fn
    network_params = network.parameters()

    # load data
    if config.args.domain == 'multi':
        if config.args.pipe_case == 'benchmark':
            data = md.get_data(data)
            trainloader = data[0]
            valloader = data[1]
            testloader = data[2]
        elif config.args.pipe_case == 'sat':
            data = md.get_data(data)
            trainloader = data[0]
            valloader = data[1]
            testloader = data[2]
    else:
        print("'Strong' DataLoader is loaded.")
        trainloader, _, testloader, _, _, _ = md.gen_strong_weak_genAIdata(config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat)
        trainloader.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'

    # create network
    network = network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    if device == 'cuda':
        network = torch.nn.DataParallel(network)
        cudnn.benchmark = True

    # train network
    if tune_params == False:
        if config.args.pipe_case == 'benchmark' and trainloader.dataset.name != 'SVHN':
            network, model_path_new, _ = train(network, trainloader, optim, loss_fn, model_path=model_path)
        elif config.args.pipe_case == 'benchmark' and trainloader.dataset.name == 'SVHN':
            network, model_path_new, _ = train(network, trainloader, optim, loss_fn, valloader=valloader, model_path=model_path)
        elif config.args.pipe_case == 'sat' and trainloader.dataset.name != 'Euro_SAT_countryside':
            network, model_path_new, _ = train(network, trainloader, optim, loss_fn, valloader=valloader, model_path=model_path)
        elif config.args.pipe_case == 'sat' and trainloader.dataset.name == 'Euro_SAT_countryside':
            network, model_path_new, _ = train(network, trainloader, optim, loss_fn, model_path=model_path)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # test network
    if config.args.domain == 'multi':
        test(network, testloader)

    # save parameters to model metadata
    network.trainloader = trainloader
    network.model_path = model_path_new
    network.loss_fn = loss_fn_cache
    network.optim = optim_cache

    # save network instance
    filename = model_path_new[:-4] + '.pickle'
    filehandler = open(filename, 'wb')
    pickle.dump(network, filehandler)

    return network, trainloader, testloader


def get_general_model() -> object:
    """
    Returns the general ML model (CIFAR-10 or EuroSAT) by either loading or training it.

    :return: Object of ML model
    """

    # get paths to load or save the trained model
    if config.args.pipe_case == 'benchmark':
        if config.args.ablation_study == 'resnet18_backbone':
            path_cifar10_load = config.args.pipe_root + 'models/benchmark/CIFAR10/CIFAR10_ResNet18.pickle'
            path_cifar10_save = config.args.pipe_root + 'models/benchmark/CIFAR10/CIFAR10_ResNet18'
        else:
            path_cifar10_load = config.args.pipe_root + 'models/benchmark/CIFAR10/CIFAR10_WideResNet.pickle'
            path_cifar10_save = config.args.pipe_root + 'models/benchmark/CIFAR10/CIFAR10_WideResNet'

        if config.args.domain == 'single':
            path_cifar10_load = config.args.pipe_root + 'models/benchmark/CIFAR10/CIFAR10_WideResNet_K' + str(config.args.dyn_single_no_strong_classes) + '.pickle'
            path_cifar10_save = config.args.pipe_root + 'models/benchmark/CIFAR10/CIFAR10_WideResNet_K' + str(config.args.dyn_single_no_strong_classes)

        # load
        if os.path.isfile(path_cifar10_load):
            print('CIFAR10 net is loaded (general AI).')
            if config.args.domain == 'single':
                print('The model is tailored to be strong on ' + str(config.args.dyn_single_no_strong_classes) + ' classes.')
            if torch.cuda.is_available():
                cifar10_net = utils.load_gpu_pickle(path_cifar10_load)
                if config.args.domain == 'single':
                    cifar10_net.trainloader.dataset.name = 'CIFAR10'
                    cifar10_net.trainloader.dataset.classes = ['airplane-CIFAR10', 'automobile-CIFAR10', 'bird-CIFAR10', 'cat-CIFAR10', 'deer-CIFAR10', 'dog-CIFAR10', 'frog-CIFAR10', 'horse-CIFAR10', 'ship-CIFAR10', 'truck-CIFAR10']
            else:
                raise NotImplementedError

        # train
        else:
            print('CIFAR10 net is built (general AI).')
            if config.args.domain == 'single':
                print('The model is tailored to be strong on ' + str(config.args.dyn_single_no_strong_classes) + ' classes.')
            if config.args.domain == 'multi':
                if config.args.ablation_study == 'resnet18_backbone':
                    cifar10_net = resnet.ResNet18()
                else:
                    cifar10_net = WideResNet(28, 10, 0.2, 10)
            else:
                cifar10_net = WideResNet(28, 10, 0.2, config.args.dyn_single_no_strong_classes)
            cifar10_net, _, _ = build_train_test_net(network=cifar10_net, data='0', optim=optims.SGD(cifar10_net.parameters(),
                                                                                                     lr=config.args.c10_sgd__lr,
                                                                                                     momentum=config.args.c10_sgd__mom,
                                                                                                     weight_decay=config.args.c10_sgd__wd),
                                                     loss_fn=nn.CrossEntropyLoss(), model_path=path_cifar10_save)

        return cifar10_net


    elif config.args.pipe_case == 'sat':

        # initialize maha functions
        models.densenet.DenseNet.intermediate_forward = intermediate_forward
        models.densenet.DenseNet.feature_list = feature_list
        models.densenet.DenseNet.feature_list_sizes = feature_list_sizes

        # get paths to load or save the trained model
        path_eurosat_load = config.args.pipe_root + 'models/sat/EUROSAT/EuroSatDenseNet.pickle'
        path_eurosat_save = config.args.pipe_root + 'models/sat/EUROSAT/EuroSatDenseNet'
        if config.args.domain == 'single':
            path_eurosat_load = config.args.pipe_root + 'models/sat/EUROSAT/EuroSatDenseNet_K' + str(config.args.dyn_single_no_strong_classes) + '.pickle'
            path_eurosat_save = config.args.pipe_root + 'models/sat/EUROSAT/EuroSatDenseNet_K' + str(config.args.dyn_single_no_strong_classes)

        # load
        if os.path.isfile(path_eurosat_load):
            print('EUROSAT net is loaded (general AI).')
            if config.args.domain == 'single':
                print('The model is tailored to be strong on ' + str(config.args.dyn_single_no_strong_classes_sat) + ' classes.')
            if torch.cuda.is_available():
                eurosat_net = utils.load_gpu_pickle(path_eurosat_load)
                if config.args.domain == 'single':
                    eurosat_net.trainloader.dataset.name = 'Euro_SAT_countryside'
                    eurosat_net.trainloader.dataset.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
            else:
                raise NotImplementedError

        # train
        else:
            print('EUROSAT net is built (general AI).')
            if config.args.domain == 'single':
                print('The model is tailored to be strong on ' + str(config.args.dyn_single_no_strong_classes_sat) + ' classes.')
            # initialize model
            model_name = "densenet"
            num_classes = 7
            if config.args.domain == 'single':
                num_classes = config.args.dyn_single_no_strong_classes_sat
            feature_extract = config.args.eurosat_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # set parameters to be updated
            params_to_update = model_ft.parameters()
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)

            # train, test, save model model
            eurosat_net, _, _ = build_train_test_net(network=model_ft, data='0', optim=optims.SGD(params_to_update,
                                                                                                  lr=config.args.eurosat_sgd__lr,
                                                                                                  momentum=config.args.eurosat_sgd__mom,
                                                                                                  weight_decay=config.args.eurosat_sgd__wd),
                                                     loss_fn=nn.CrossEntropyLoss(), model_path=path_eurosat_save, tune_params=False)


        return eurosat_net

    else:
        raise NotImplementedError


def get_expert_models() -> Tuple[object, object, object]:
    """
    Returns the ML models of the artificial experts by either loading or training it.

    :return: All three objects of the artificial experts
    """

    if config.args.pipe_case == 'benchmark':
        # get paths to load or save the trained model
        path_svhn_load = config.args.pipe_root + 'models/benchmark/SVHN/SVHN_WideResNet.pickle'
        path_svhn_save = config.args.pipe_root + 'models/benchmark/SVHN/SVHN_WideResNet'
        path_mnist_load = config.args.pipe_root + 'models/benchmark/MNIST/MNIST_WideResNet.pickle'
        path_mnist_save = config.args.pipe_root + 'models/benchmark/MNIST/MNIST_WideResNet'
        path_fmnist_load = config.args.pipe_root + 'models/benchmark/FMNIST/FMNIST_WideResNet.pickle'
        path_fmnist_save = config.args.pipe_root + 'models/benchmark/FMNIST/FMNIST_WideResNet'

        # load
        if os.path.isfile(path_svhn_load):
            print('SVHN net is loaded (expert AI 1).')
            if torch.cuda.is_available():
                svhn_net = utils.load_gpu_pickle(path_svhn_load)

            else:
                raise NotImplementedError
        # train
        else:
            print('SVHN net is built (expert AI 1).')

            svhn_net = WideResNet(28, 10, 0.2, 10)
            svhn_net, _, _ = build_train_test_net(network=svhn_net, data='1', optim=optims.SGD(svhn_net.parameters(),
                                                                                               lr=config.args.svhn_wideresnet_sgd__lr,
                                                                                               momentum=config.args.svhn_sgd__mom,
                                                                                               weight_decay=config.args.svhn_wideresnet_sgd__wd),
                                                  loss_fn=nn.CrossEntropyLoss(), model_path=path_svhn_save)


        # load
        if os.path.isfile(path_mnist_load):
            print('MNIST net is loaded (expert AI 2).')
            if torch.cuda.is_available():
                try:
                    mnist_net = utils.load_gpu_pickle(path_mnist_load)

                except AttributeError:
                    raise NotImplementedError

            else:
                raise NotImplementedError
        # train
        else:
            print('MNIST net is built (expert AI 2).')

            mnist_net = WideResNet(28, 10, 0.2, 10)
            mnist_net, _, _ = build_train_test_net(network=mnist_net, data='2', optim=optims.SGD(mnist_net.parameters(),
                                                                                                 lr=config.args.mnist_wideresnet_sgd__lr,
                                                                                                 momentum=config.args.mnist_sgd__mom,
                                                                                                 weight_decay=config.args.mnist_wideresnet_sgd__wd),
                                                      loss_fn=nn.CrossEntropyLoss(), model_path=path_mnist_save)

        # load
        if os.path.isfile(path_fmnist_load):
            print('FMNIST net is loaded (expert AI 3).')
            if torch.cuda.is_available():
                fmnist_net = utils.load_gpu_pickle(path_fmnist_load)
            else:
                raise NotImplementedError
        # train
        else:
            print('FMNIST net is built (expert AI 3).')
            fmnist_net = WideResNet(28, 10, 0.05, 10)
            fmnist_net, _, _ = build_train_test_net(network=fmnist_net, data='3', optim=optims.SGD(fmnist_net.parameters(),
                                                                                                       lr=config.args.fmnist_sgd__lr,
                                                                                                       momentum=config.args.fmnist_sgd__mom,
                                                                                                       weight_decay=config.args.fmnist_sgd__wd),
                                                      loss_fn=nn.CrossEntropyLoss(), model_path=path_fmnist_save)

        return svhn_net, mnist_net, fmnist_net


    elif config.args.pipe_case == 'sat':

        # initialize maha functions
        models.densenet.DenseNet.intermediate_forward = intermediate_forward
        models.densenet.DenseNet.feature_list = feature_list
        models.densenet.DenseNet.feature_list_sizes = feature_list_sizes

        # get paths to load or save the trained model
        path_fmow_load = config.args.pipe_root + 'models/sat/FMOW/FmowDenseNet.pickle'
        path_fmow_save = config.args.pipe_root + 'models/sat/FMOW/FmowDenseNet'

        # load
        if os.path.isfile(path_fmow_load):
            print('FMOW net is loaded (expert AI 1).')
            if torch.cuda.is_available():
                fmow_net = utils.load_gpu_pickle(path_fmow_load)
            else:
                raise NotImplementedError
        # train
        else:
            print('FMOW net is built (expert AI 1).')
            # initialize model
            model_name = "densenet"
            num_classes = 35
            feature_extract = config.args.fmow_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # set parameters to be updated
            params_to_update = model_ft.parameters()
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)

            # train, test, save model model
            fmow_net, _, _ = build_train_test_net(network=model_ft, data='1', optim=optims.Adam(params_to_update,
                                                                                                    lr=config.args.fmow_adam__lr,
                                                                                                    weight_decay=config.args.fmow_adam__wd),
                                                      loss_fn=nn.CrossEntropyLoss(), model_path=path_fmow_save, tune_params=False)


        # get paths to load or save the trained model
        path_aid_load = config.args.pipe_root + 'models/sat/AID/AidDenseNet.pickle'
        path_aid_save = config.args.pipe_root + 'models/sat/AID/AidDenseNet'

        # load
        if os.path.isfile(path_aid_load):
            print('AID net is loaded (expert AI 2).')
            if torch.cuda.is_available():
                aid_net = utils.load_gpu_pickle(path_aid_load)
            else:
                raise NotImplementedError
        # train
        else:
            print('AID net is built (expert AI 2).')

            # initialize model
            model_name = "densenet"
            num_classes = 24
            feature_extract = config.args.aid_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # set parameters to be updated
            params_to_update = model_ft.parameters()
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)

            # train, test, save model model
            aid_net, _, _ = build_train_test_net(network=model_ft, data='2', optim=optims.Adam(params_to_update,
                                                                                                   lr=config.args.aid_adam__lr,
                                                                                                   weight_decay=config.args.aid_adam__wd),
                                                     loss_fn=nn.CrossEntropyLoss(), model_path=path_aid_save, tune_params=False)

        # get paths to load or save the trained model
        path_resisc_load = config.args.pipe_root + 'models/sat/RESISC/RESISCDenseNet.pickle'
        path_resisc_save = config.args.pipe_root + 'models/sat/RESISC/RESISCDenseNet'

        # load
        if os.path.isfile(path_resisc_load):
            print('RESISC net is loaded (expert AI 3).')
            if torch.cuda.is_available():
                resisc_net = utils.load_gpu_pickle(path_resisc_load)
            else:
                raise NotImplementedError
        # train
        else:
            print('RESISC net is built (expert AI 3).')

            # initialize model
            model_name = "densenet"
            num_classes = 12
            feature_extract = config.args.resisc_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # set parameters to be updated
            params_to_update = model_ft.parameters()
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)

            # train, test, save model model
            resisc_net, _, _ = build_train_test_net(network=model_ft, data='3', optim=optims.Adam(params_to_update,
                                                                                               lr=config.args.resisc_adam__lr,
                                                                                               weight_decay=config.args.resisc_adam__wd),
                                                 loss_fn=nn.CrossEntropyLoss(), model_path=path_resisc_save, tune_params=False)

        return fmow_net, aid_net, resisc_net


def train_gating_model(type: str = 'selec_mech') -> None:
    """
    Function to train/tune and save the gating model.

    :param type: Allocation mechanism. A1 = 'selec_mech', A2 = 'comb_mech
    """

    # get paths to save the trained model
    if config.args.pipe_case == 'benchmark':
        if type == 'selec_mech':
            model_path = config.args.pipe_root + 'models/benchmark/GATING/C10vRestWFMNISTGatingModel.pickle'
        elif type == 'comb_mech':
            model_path = config.args.pipe_root + 'models/benchmark/GATING/SVHNvMNISTvFMNISTGatingModel.pickle'
        elif type == 'total':
            model_path = config.args.pipe_root + 'models/benchmark/GATING/C10vSVHNvMNISTvFMNISTGatingModel.pickle'
    elif config.args.pipe_case == 'sat':
        if type == 'selec_mech':
            model_path = config.args.pipe_root + 'models/sat/GATING/EUROSATvRestV2GatingModel.pickle'
        elif type == 'comb_mech':
            model_path = config.args.pipe_root + 'models/sat/GATING/EurosatvFMOWvAIDvRESISCModel.pickle'
        elif type == 'total':
            raise NotImplementedError

    # check whether it already exists
    if os.path.isfile(model_path):
        print('Gating model for ' + type + ' is already created.')

    # train
    else:
        print('Gating model for ' + type + ' is built.')

        # get data
        trainloader, valloader = md.get_gating_data(type)

        if config.args.domain == 'multi':
            # adjust transforms to be consistent between static and dynamic AIITL-system (ensure that the gating model of static AIITL-system is trained on the same data structure as gating model of dynamic AIITL-system)
            if config.args.dyn_multi_transforms == 'adapted':
                train_data_copy = copy.deepcopy(trainloader.dataset)
                val_data_copy = copy.deepcopy(valloader.dataset)
                _ = utils.set_transform(train_data_copy, 'train')
                _ = utils.set_transform(val_data_copy, 'test')
            else:
                train_data_copy = trainloader.dataset
                val_data_copy = valloader.dataset

            if config.args.pipe_case == 'benchmark':
                trainloader = DataLoader(train_data_copy, num_workers=2, batch_size=config.args.gating_train_batch, shuffle=True)
                valloader = DataLoader(val_data_copy, num_workers=2, batch_size=1, shuffle=True)
            else:
                trainloader = DataLoader(train_data_copy, num_workers=2, batch_size=config.args.sat_gating_train_batch, shuffle=True)
                valloader = DataLoader(val_data_copy, num_workers=2, batch_size=1, shuffle=True)

        # train model
        if config.args.pipe_case == 'benchmark':
            if type == 'selec_mech':
                network, _ = initialize_model('densenet', 2, False, use_pretrained=True)
                params_to_update = network.parameters()
            elif type == 'comb_mech':
                network, _ = initialize_model('densenet', 4, False, use_pretrained=True)
                params_to_update = network.parameters()
            elif type == 'total':
                network, _ = initialize_model('densenet', 4, False, use_pretrained=True)
                params_to_update = network.parameters()
        elif config.args.pipe_case == 'sat':
            if type == 'selec_mech':
                network, _ = initialize_model('densenet', 2, False, use_pretrained=True)
            elif type == 'comb_mech':
                network, _ = initialize_model('densenet', 4, False, use_pretrained=True)
            elif type == 'total':
                network, _ = initialize_model('densenet', 4, False, use_pretrained=True)
            # set parameters to be updated
            params_to_update = network.parameters()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network = network.to(device)
        if device == 'cuda':
            network = torch.nn.DataParallel(network)
            cudnn.benchmark = True

        if config.args.pipe_case == 'benchmark':
            network, _ = train(network=network,
                            trainloader=trainloader,
                            valloader=valloader,
                               optim=optims.Adam(params_to_update, lr=config.args.gating_adam__lr, weight_decay=config.args.gating_adam__wd),
                            loss_fn=nn.CrossEntropyLoss(),
                            download=False)
        elif config.args.pipe_case == 'sat':
            network, _ = train(network=network,
                            trainloader=trainloader,
                               valloader=valloader,
                            optim=optims.Adam(params_to_update, lr=config.args.sat_gating_adam__lr, weight_decay=config.args.sat_gating_adam__wd),
                            loss_fn=nn.CrossEntropyLoss(),
                            download=False)

        # save network instance
        filehandler = open(model_path, 'wb')
        pickle.dump(network, filehandler)


def train_dyn_gating_model(trainloader: DataLoader,
                           valloader: DataLoader,
                           type: str = 'tot') -> object:
    """
    Trainining/Tuning of the gating model in the dynamic AIITL-system.

    :param trainloader: DataLoader of training data
    :param valloader: DataLoader of validation data
    :param type: 'tot' for allocation mechanism 2; 'selec' for allocation mechanism 1
    :return: Gating model object, validation accuracy
    """

    # train model
    if type == 'tot':
        network, _ = initialize_model('densenet', 4, False, use_pretrained=True)
    elif type == 'selec':
        network, _ = initialize_model('densenet', 2, False, use_pretrained=True)
    # set parameters to be updated
    params_to_update = network.parameters()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    if device == 'cuda':
        network = torch.nn.DataParallel(network)
        cudnn.benchmark = True

    if config.args.pipe_case == 'benchmark':
        network, val_acc = train(network=network,
                           trainloader=trainloader,
                           valloader=valloader,
                           optim=optims.Adam(params_to_update, lr=config.args.gating_adam__lr, weight_decay=config.args.gating_adam__wd),
                           loss_fn=nn.CrossEntropyLoss(),
                           download=False)

    else:
        network, val_acc = train(network=network,
                                 trainloader=trainloader,
                                 valloader=valloader,
                                 optim=optims.Adam(params_to_update, lr=config.args.sat_gating_adam__lr, weight_decay=config.args.sat_gating_adam__wd),
                                 loss_fn=nn.CrossEntropyLoss(),
                                 download=False)

    # save network
    if type == 'tot':
        filename = config.args.pipe_root + 'models/' + config.args.pipe_case + '/GATING/dyn_pipe_gating{}.pickle'.format(config.args.ablation_study)
    elif type == 'selec':
        filename = config.args.pipe_root + 'models/' + config.args.pipe_case + '/GATING/selec_mech_dyn_pipe_gating{}.pickle'.format(config.args.ablation_study)
    filehandler = open(filename, 'wb')
    pickle.dump(network, filehandler)

    return network, val_acc


def train_global_model() -> None:
    """
    Trainining of the global model.
    """

    # get paths to save the trained model
    if config.args.pipe_case == 'benchmark':
         model_path = config.args.pipe_root + 'models/benchmark/GLOBAL/' + 'benchmark_global_wFMNIST.pickle'
    elif config.args.pipe_case == 'sat':
        model_path = config.args.pipe_root + 'models/sat/GLOBAL/' + 'sat_global_v2.pickle'
    # check whether model already exists
    if os.path.isfile(model_path):
        print('Global model is already created.')

    # train
    else:
        print('Global model is built.')
        # load data
        trainloader, valloader = md.get_global_model_data(flag='global')

        # create network
        if config.args.pipe_case == 'benchmark':
            global_net = WideResNet(28, 10, 0.2, 40)
        elif config.args.pipe_case == 'sat':
            model_name = "densenet"
            num_classes = 78
            feature_extract = config.args.sat_global_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
            global_net, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # set parameters to be updated
            params_to_update = global_net.parameters()
            if feature_extract:
                params_to_update = []
                for name, param in global_net.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        global_net = global_net.to(device)
        if device == 'cuda':
            global_net = torch.nn.DataParallel(global_net)
            cudnn.benchmark = True

        # train network
        if config.args.pipe_case == 'benchmark':
            global_net, _ = train(global_net, trainloader, optim=optims.SGD(global_net.parameters(),
                                                                             lr=config.args.global_sgd__lr,
                                                                             momentum=config.args.global_sgd__mom,
                                                                             weight_decay=config.args.global_sgd__wd),
                                  loss_fn=nn.CrossEntropyLoss(), download=False)
        elif config.args.pipe_case == 'sat':
            global_net, _ = train(global_net, trainloader, valloader=valloader, optim=optims.Adam(params_to_update,
                                                                            lr=config.args.sat_global_adam__lr,
                                                                            weight_decay=config.args.sat_global_adam__wd),
                                  loss_fn=nn.CrossEntropyLoss(), download=False)

        # save network instance
        filename = model_path
        filehandler = open(filename, 'wb')
        pickle.dump(global_net, filehandler)


def pred_global(gen_AI: object,
                exp_AI_1: object,
                exp_AI_2: object,
                exp_AI_3: object,
                pipeloader: DataLoader) -> Sequence[str]:
    """
    Returns the predictions of the global model

    :param gen_AI: general ML model object
    :param exp_AI_1: artificial expert 1 object
    :param exp_AI_2: artificial expert 2 object
    :param exp_AI_3: artificial expert 3 object
    :param pipeloader: DataLoader of dataset to be predicted by global model
    :return: Sequence of predictions
    """

    # get global model
    if config.args.pipe_case == 'benchmark':
        filehandler = open(config.args.pipe_root + 'models/benchmark/GLOBAL/benchmark_global_wFMNIST.pickle', 'rb')
        global_net = pickle.load(filehandler)
    elif config.args.pipe_case == 'sat':
        filehandler = open(config.args.pipe_root + 'models/sat/GLOBAL/sat_global_v2.pickle', 'rb')
        global_net = pickle.load(filehandler)

    print('Global model predictions are computed, may take some time..')
    pred = []

    # collect all classes
    global_classes = gen_AI.trainloader.dataset.classes + exp_AI_1.trainloader.dataset.classes + exp_AI_2.trainloader.dataset.classes + exp_AI_3.trainloader.dataset.classes

    # compute predictions
    global_net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pipeloader):
            data, target = data.to(device), target.to(device)
            output = global_net(data)
            _, predicted = torch.max(output.data, 1)
            pred.append(global_classes[predicted.item()])

    return pred


def dyn_expert_creation(trainloader: DataLoader,
                        name: str,
                        valloader: DataLoader = None) -> Tuple[object, float, Sequence[int]]:
    """
    Trains, saves and returns the artificial expert on the provided data.

    :param trainloader: DataLoader of training data for artificial expert
    :param name: Name of expert
    :param valloader: DataLoader of validation data for artificial expert
    :return: network, train_acc, indices of training data (not needed in current implementation)
    """

    # initialize model
    if config.args.domain == 'multi':
        if config.args.pipe_case == 'benchmark':
            if config.args.ablation_study == 'resnet18_backbone':
                network = resnet.ResNet18()
            else:
                network = WideResNet(28, 10, 0.2, 10)
        elif config.args.pipe_case == 'sat':
            network, _ = initialize_model('densenet', len(trainloader.dataset.classes), False, use_pretrained=True)
        else:
            raise NotImplementedError
    else:
        no_strong_classes = config.args.dyn_single_no_strong_classes if config.args.pipe_case == 'benchmark' else config.args.dyn_single_no_strong_classes_sat
        if config.args.pipe_case == 'benchmark':
            network = WideResNet(28, 10, 0.2, 10-no_strong_classes)
        elif config.args.pipe_case == 'sat':
            network, _ = initialize_model('densenet', 7-no_strong_classes, False, use_pretrained=True)
        else:
            raise NotImplementedError
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    if device == 'cuda':
        network = torch.nn.DataParallel(network)
        cudnn.benchmark = True

    # collect indices of training (not needed in current implementation)
    train_ind = [i for i in range(len(trainloader.dataset))]
    loadername = trainloader.dataset.name

    # train model
    config.args.checkpoint = 'models/' + config.args.pipe_case + '/DYN_PIPE/' + config.args.domain.upper()
    if config.args.domain == 'single':

        if config.args.pipe_case == 'benchmark':
            lr = config.args.c10_sgd__lr
            mom = config.args.c10_sgd__mom
            wd = config.args.c10_sgd__wd
        elif config.args.pipe_case == 'sat':
            lr = config.args.eurosat_sgd__lr
            mom = config.args.eurosat_sgd__mom
            wd = config.args.eurosat_sgd__wd

        network, train_acc = train(network, trainloader, optim=optims.SGD(network.parameters(),
                                                                    lr=lr,
                                                                    momentum=mom,
                                                                    weight_decay=wd),
                             loss_fn=nn.CrossEntropyLoss(), download=False, model_path=name)
    elif config.args.domain == 'multi':
        if config.args.pipe_case == 'benchmark':
            network, train_acc = train(network, trainloader, valloader=valloader, optim=optims.SGD(network.parameters(),
                                                                          lr=config.args.dyn_multi_sgd_lr,
                                                                          momentum=config.args.dyn_multi_sgd_mom,
                                                                          weight_decay=config.args.dyn_multi_sgd_wd),
                                   loss_fn=nn.CrossEntropyLoss(), download=False, model_path=name)
        else:
            network, train_acc = train(network, trainloader, valloader=valloader, optim=optims.Adam(network.parameters(),
                                                                                                   lr=config.args.dyn_multi_adam_lr_sat,
                                                                                                   weight_decay=config.args.dyn_multi_adam_wd_sat),
                                       loss_fn=nn.CrossEntropyLoss(), download=False, model_path=name)

    # save model
    network = network.module.to(device)
    #network.trainloader = trainloader
    network.name = trainloader.dataset.name
    model_path = config.args.pipe_root + 'models/' + config.args.pipe_case + '/DYN_PIPE/' + config.args.domain.upper() + '/' + name + '.pickle'
    filehandler = open(model_path, 'wb')
    pickle.dump(network, filehandler)

    return network, train_acc, train_ind
