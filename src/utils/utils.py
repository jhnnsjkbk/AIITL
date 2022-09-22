# this code is intended to provide supplementary functions.

# general libraries
import copy
import pandas
import pandas as pd
import torch
import random
import numpy as np
import src.config
import time
import pickle
import io
from torch.utils.data import DataLoader
import os
import src.data.transforms as transforms
from src.data.augmentations import RandAugment
from typing import Tuple, Sequence, Any, Dict


# get config
config = src.config.cfg



def set_seed(seed: int) -> None:
    """
    Sets the specified seed to torch, random and numpy seeds

    :param seed: Desired seed number
    """

    torch.manual_seed(seed)
    random.seed(seed)

    np.random.seed(seed)

def progress_bar(current: int,
                 total: int,
                 msg: str = None) -> None:
    """
    Creates and prints a nice progress bar for training ML models.

    :param current: current batch idx
    :param total: total no. of batch indices
    :param msg: format of message
    """

    total_bar_length = config.args.bar_length

    global last_time, begin_time
    if current == 0:
        last_time = time.time()
        begin_time = time.time()  # Reset for new bar.

    # get lengths
    cur_len = int(total_bar_length*current/total)
    rest_len = int(total_bar_length - cur_len) - 1

    # print progress arrows
    bar = ' ['
    for i in range(cur_len):
        bar += '='
    bar += '>'
    for i in range(rest_len):
        bar += '.'
    bar += ']'
    print(bar)

    # get timestamps
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    # add supplementary information at the end of the progress bar
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    print(msg)

    # Go back to the center of the bar.
    print('  Batch index/total batches: %d/%d' % (current+1, total))
    print('')


def format_time(seconds: int) -> str:
    """
    Format seconds in easier format [D, h, m, s, ms]

    :param seconds: seconds
    :return: time format
    """

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Unpickler(pickle.Unpickler):
    """
    Overwrite Unpickler find_class method to make downloading GPU-trained pickle models possible on CPU
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def find_class(self: object,
                   module: str,
                   name: str) -> Any:
        """
        overwrites pickle.Unpickler.find_class method.
        """

        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        else:
            return super().find_class(module, name)



def load_gpu_pickle(model_path: str) -> object:
    """
    loads a pickle model instance and prepares all necessary steps to run on cpu

    :param model_path: model path
    :return: network object ready to use
    """

    filehandler = open(model_path, 'rb')
    network = Unpickler(filehandler).load()

    # collect properties
    if not config.args.dyn_pipe:
        optim = network.optim
        loss_fn = network.loss_fn
        trainloader = network.trainloader
    else:
        name = network.name

    # account for use of DataParallel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if network.__class__.__name__ == 'DataParallel':
        network = network.module.to(device)
    else:
        network = network.to(device)

    # add properties
    if not config.args.dyn_pipe:
        network.optim = optim
        network.loss_fn = loss_fn
        network.trainloader = trainloader
    else:
        network.name = name

    return network


def get_mean_std(dataset: object) -> Tuple[Sequence, Sequence]:
    """
    Computes mean and std for a given dataset

    :param dataset: torchvision dataset
    :return: mean, std as lists with length = number of channels
    """

    mean = 0.
    std = 0.
    nb_samples = 0.
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    for data, target in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()


def adjust_mlflow_yaml(id: int) -> None:
    """
    Adjusts the very first line of the mlflow .yaml-file to enable local access to its information

    :param id: ID of experiment
    """
    line_number = 1
    replacement_line = 'artifact_location: file:///Users/Documents/03_ai-in-the-loop/' + config.args.mlflow_path + str(id) + '\n'
    file_name = '../' + config.args.mlflow_path + str(id) + '/meta.yaml'

    time.sleep(2)

    with open(file_name) as file:  # Safely open the file
        lines = file.readlines()
        lines[line_number-1] = replacement_line
    with open(file_name, 'w') as file:
        file.writelines(lines)

    time.sleep(2)


def save_mlflow_exp_id(id: int) -> None:
    """
    Saves the mlflow experiment ID to the specified log file.

    :param id: ID of experiment
    """

    # read in already used ids
    case_domain_pipe = config.args.pipe_case + '_' + config.args.domain + '_' + config.args.pipe_type
    mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs{}.log'.format(config.args.ablation_study)

    # if in repro mode, save used IDs in new file!
    if config.args.results_repro:
        mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO{}.log'.format(config.args.ablation_study)

    # collect all IDs
    if os.path.isfile(mlrunsID_log_file):
        with open(mlrunsID_log_file) as f:
            IDs = [int(x) for x in f]
    else:
        IDs = []

    # only add id if it is not added before
    if not id in IDs:
        IDs.append(id)

    # save all used IDs
    with open(mlrunsID_log_file, 'w') as f:
        for id in IDs:
            f.write(str(id))
            f.write('\n')


def save_checkpoint(state: Any,
                    checkpoint: str = 'checkpoint',
                    filename: str = 'checkpoint.pth.tar') -> None:
    """
    Saves the information of state in respective checkpoint object file

    :param state: Information to be saved, e.g., provided by Dictionary
    :param checkpoint: location of checkpoint
    :param filename: filename of checkpoint
    """

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def adjust_learning_rate(optimizer: object,
                         epoch: int) -> None:
    """
    Manual learning rate scheduler, which reduces the learning rate by factor 0.1 if epoch is 150 or 225.

    :param optimizer: Optimizer object
    :param epoch: number of epoch
    """

    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
    if epoch in [150, 225]:
        print('lr is adjusted..')
        lr = old_lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def print_dict(dictionary: Dict) -> None:
    """
    Function to print out all information of a Dictionary in a readable format.

    :param dictionary: Dictionary to be printed
    """

    # prints dictionary line-wise
    for x in dictionary:
        # title
        print(x + ':')
        # if dictionary: second title
        if isinstance(dictionary[x], dict):
            for y in dictionary[x]:
                print(y + ':')
                # values
                print(dictionary[x][y])
        else:
            # values
            print(dictionary[x])


def set_target(dataset: object,
               target: int,
               target_set: str) -> object:
    """
    Adjusts the target_transform function of complex dataset structures to change all targets to a single target.
    target_set is either 'CIFAR10', 'MNIST', 'SVHN' or 'FashionMNIST', or one of the sat case dataset names. The adjustment only takes place for this specific dataset.

    :param dataset: Dataset object
    :param target: Desired target for all samples of dataset
    :param target_set: Effected target dataset
    :return: adjusted dataset
    """

    # first prio: find target transform
    # if not, then find dataset
    # if not, then check for datasets and iterate

    # if has target transform: change target resepectively
    # for sat case, we need to identify the correct dataset by checking for the respective length manually
    if hasattr(dataset, 'target_transform'):
        if config.args.pipe_case == 'benchmark':
            if dataset.__class__.__name__ == target_set:
                dataset.target_transform = lambda a, x=target: x
        else:
            if dataset.__class__.__name__ == 'ImageFolder':
                if len(dataset) == 19000 and target_set == 'Euro_SAT_countryside':
                    dataset.target_transform = lambda a, x=target: x
                if (len(dataset) == 7606 or len(dataset) == 1170 or len(dataset) == 1057) and target_set == 'FMOW_utilities':
                    dataset.target_transform = lambda a, x=target: x
                if (len(dataset) == 38061 or len(dataset) == 5713 or len(dataset) == 5666) and target_set == 'FMOW_public':
                    dataset.target_transform = lambda a, x=target: x
                if (len(dataset) == 26254 or len(dataset) == 3829 or len(dataset) == 3889) and target_set == 'FMOW_private':
                    dataset.target_transform = lambda a, x=target: x
                if (len(dataset) == 41554 or len(dataset) == 6223 or len(dataset) == 6185) and target_set == 'FMOW':
                    dataset.target_transform = lambda a, x=target: x
                if len(dataset) == 7960 and target_set == 'AID':
                    dataset.target_transform = lambda a, x=target: x
                if len(dataset) == 8400 and target_set == 'RESISC':
                    dataset.target_transform = lambda a, x=target: x

    # if it does not yet have the target transform function, recursively call function again
    elif hasattr(dataset, 'dataset'):
        dataset = set_target(dataset.dataset, target, target_set)
    elif hasattr(dataset, 'datasets'):
        for i in range(len(dataset.datasets)):
            dataset_cache = set_target(dataset.datasets[i], target, target_set)

    return dataset


def reset_target(dataset: object,
                 subtractor: int,
                 target_set: str) -> object:
    """
    resets target_transform of complex dataset structures by subtracting the subtractor.
    Needed for single domain experiments to adjust the unknown data classes to make it trainable on class ids 0...x
    target_set is either 'CIFAR10', 'MNIST', 'SVHN' or 'FashionMNIST' OR 'ImageFolder' for sat case.

    :param dataset: Dataset object
    :param substractor: Subtractor for target
    :param target_set: Effected target dataset
    :return: adjusted dataset
    """

    # first prio: find target transform
    # if not, then find dataset
    # if not, then check for datasets and iterate

    # if has target transform: change target resepectively. Sat case can be identified via ImageFolder, as in single-domain no further differentiation between other datasets is requried
    if hasattr(dataset, 'target_transform'):
        if dataset.__class__.__name__ == target_set:
            dataset.target_transform = lambda a, x=subtractor: (a-x)

    # if it does not yet have the target transform function, recursively call function again
    elif hasattr(dataset, 'dataset'):
        dataset = reset_target(dataset.dataset, subtractor, target_set)
    elif hasattr(dataset, 'datasets'):
        for i in range(len(dataset.datasets)):
            dataset_cache = reset_target(dataset.datasets[i], subtractor, target_set)

    return dataset


def set_transform(dataset: object,
                  mode: str) -> object:
    """
    Adjusts the transform function of complex dataset structures. This is needed to simulate real-world behaviour,
    where data entering the system is pre-processed in a standardized format as it cannot be known beforehand.
    For training, the transforms would then be adjusted to a certain training transforms design.

    :param dataset: Dataset object
    :param mode: Mode of transforms 'train' or 'eval'
    :return: adjusted dataset
    """


    # first prio: find transform
    # if not, then find dataset
    # if not, then check for datasets and iterate

    # default transforms in following order:
    # gen AI train, gen AI test, expAI1 train, expAI2 test, etc.

    # assumption for dynamic pipeline: Every experts follows the same preprocessing steps as gen AI
    # except for normalizing

    if config.args.domain == 'multi':
        if config.args.pipe_case == 'benchmark':

            # definition of all transforms
            transform_source = [
                transforms.Compose([
                    transforms.Resize(32),
                    RandAugment(2, 14),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # gen AI normalizing is already known
                ]),
                transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # gen AI normalizing is already known
                ]),
                transforms.Compose([
                    transforms.Resize(32),
                    RandAugment(2, 14),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # experts follow a standard approach
                ]),
                transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # experts follow a standard approach
                ]),
                transforms.Compose([
                    transforms.Grayscale(3),  # make MNIST compatible to other data and models
                    transforms.Resize(32),
                    RandAugment(2, 14),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # experts follow a standard approach
                ]),
                transforms.Compose([
                    transforms.Grayscale(3),  # make MNIST compatible to other data and models
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # experts follow a standard approach
                ]),
                transforms.Compose([
                    transforms.Grayscale(3),  # make FMNIST compatible to other data and models
                    transforms.Resize(32),
                    RandAugment(2, 14),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # experts follow a standard approach
                ]),
                transforms.Compose([
                    transforms.Grayscale(3),  # make FMNIST compatible to other data and models
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # experts follow a standard approach
                ])
            ]

        else:
            transform_source = [
                transforms.Compose([transforms.Resize(224),
                                  transforms.RandomCrop(224, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            ]
    else:
        if config.args.pipe_case == 'benchmark':

            # definition of all transforms
            transform_source = [
                transforms.Compose([
                    transforms.Resize(32),
                    RandAugment(2, 14),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
            ]

        else:
            transform_source = [
                transforms.Compose([transforms.Resize(224),
                                    transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            ]

    # actual adjustments

    # if dataset has transform, adjust it respectively based on the defintions above
    if hasattr(dataset, 'target_transform'):
        if config.args.pipe_case == 'benchmark':
            if dataset.__class__.__name__ == 'CIFAR10':
                trans_fn = transform_source[0] if mode == 'train' else transform_source[1]
            elif dataset.__class__.__name__ == 'SVHN':
                trans_fn = transform_source[2] if mode == 'train' else transform_source[3]
            elif dataset.__class__.__name__ == 'MNIST':
                trans_fn = transform_source[4] if mode == 'train' else transform_source[5]
            elif dataset.__class__.__name__ == 'FashionMNIST':
                trans_fn = transform_source[6] if mode == 'train' else transform_source[7]
        else:
            if dataset.__class__.__name__ == 'ImageFolder':
                if dataset.name == 'Euro_SAT_countryside':
                    trans_fn = transform_source[0] if mode == 'train' else transform_source[1]
                if dataset.name == 'FMOW_utilities':
                    trans_fn = transform_source[2] if mode == 'train' else transform_source[3]
                if dataset.name == 'FMOW_public':
                    trans_fn = transform_source[4] if mode == 'train' else transform_source[5]
                if dataset.name == 'FMOW_private':
                    trans_fn = transform_source[6] if mode == 'train' else transform_source[7]
                if dataset.name == 'FMOW':
                    trans_fn = transform_source[2] if mode == 'train' else transform_source[3]
                if dataset.name == 'AID':
                    trans_fn = transform_source[4] if mode == 'train' else transform_source[5]
                if dataset.name == 'RESISC':
                    trans_fn = transform_source[6] if mode == 'train' else transform_source[7]
        dataset.transform = trans_fn

    # if it does not yet have the transform function, recursively call function again
    elif hasattr(dataset, 'dataset'):
        dataset = set_transform(dataset.dataset, mode)
    elif hasattr(dataset, 'datasets'):
        for i in range(len(dataset.datasets)):
            dataset_cache = set_transform(dataset.datasets[i], mode)

    return dataset


def get_dataset_structure(dataset: object,
                          concat_borders: Sequence[int] = [],
                          subset_inds: Sequence[int] = [],
                          datasets: Sequence[str] = [],
                          concat_count: int = 0,
                          print_tot_structure: bool = True) -> Tuple[object, Sequence, Sequence, Sequence]:
    """
    Splits dataset into its subsets and concatdatasets to print out its structure.

    :param dataset: Dataset object
    :param concat_borders: Previous index number, where the concat datasets begin and end
    :param subset_inds: Previous index number, where the subset datasets begin and end
    :param datasets: Sequence to collect all datasets identified
    :param concat_count: Counter of number of concat datasets
    :param print_tot_structure: Activate to print out the structure
    :return: dataset, concat_borders, subset_inds, datasets
    """

    # first prio: find final dataset by identfying the target transform
    # if not, then find dataset
    # if not, then check for datasets and iterate


    # if dataset has target transform, collect respective information of the dataset and change its target transform to the identifier key of the dataset to make it countable
    if hasattr(dataset, 'target_transform'):
        if print_tot_structure:
            print('DATA')
            print(dataset.transform)
            print('')
        if config.args.pipe_case == 'benchmark':
            if hasattr(dataset, 'split'):
                split = dataset.split
            elif dataset.train:
                split = 'train'
            else:
                split = 'test'
        elif config.args.pipe_case == 'sat':
            if dataset.name == 'Euro_SAT_countryside' or dataset.name == 'AID' or dataset.name == 'RESISC' or dataset.name == 'FMOW':
                if len(subset_inds[-1]) == 15200 or len(subset_inds[-1]) == 6368 or len(subset_inds[-1]) == 6720 or len(subset_inds[-1]) == 43169:
                    split = 'train'
                else:
                    split = 'test'
            else:
                raise NotImplementedError

        # adjust target transform
        datasets.append((dataset.__class__.__name__, split))
        if config.args.pipe_case == 'benchmark':
            if dataset.__class__.__name__ == 'CIFAR10':
                key = 0 if dataset.train else 1
            elif dataset.__class__.__name__ == 'SVHN':
                key = 2 if dataset.split == 'train' else 3
            elif dataset.__class__.__name__ == 'MNIST':
                key = 4 if dataset.train else 5
            elif dataset.__class__.__name__ == 'FashionMNIST':
                key = 6 if dataset.train else 7
            dataset.target_transform = lambda x, y=key: y
        elif config.args.pipe_case == 'sat':
            if dataset.name == 'Euro_SAT_countryside':
                key = 0 if len(subset_inds[-1]) == 15200 else 1
            elif dataset.name == 'FMOW_utilities':
                key = 2 if len(dataset) == 7606 else 3
            elif dataset.name == 'FMOW_public':
                key = 4 if len(dataset) == 38061 else 5
            elif dataset.name == 'FMOW_private':
                key = 6 if len(dataset) == 26254 else 7
            elif dataset.name == 'FMOW':
                key = 2 if len(subset_inds[-1]) == 43169 else 3
            elif dataset.name == 'AID':
                key = 4 if len(subset_inds[-1]) == 6368 else 5
            elif dataset.name == 'RESISC':
                key = 6 if len(subset_inds[-1]) == 6720 else 7
            dataset.target_transform = lambda x, y=key: y


    # if it does not yet have the transform function, recursively call function again and print out/collect information about structure
    elif hasattr(dataset, 'dataset'):
        if print_tot_structure:
            print('SUBSET')
            print('len of subset: ' + str(len(dataset)))
            print('len of indices: ' + str(len(dataset.indices)))
        subset_inds.append(dataset.indices)
        dataset, _, _, _ = get_dataset_structure(dataset.dataset, concat_borders, subset_inds, datasets, concat_count, print_tot_structure)
    elif hasattr(dataset, 'datasets'):
        concat_count = concat_count + 1
        if print_tot_structure:
            print('CONCAT no. ' + str(concat_count))
            print('no. of concat ' + str(concat_count) + ' datasets: ' + str(len(dataset.datasets)))
            print('concat no. ' + str(concat_count) + ' dataset borders:')
        borders = []
        for i in range(len(dataset.datasets)):
            if len(borders) == 0:
                borders.append(len(dataset.datasets[i]))
            else:
                value = borders[-1] + len(dataset.datasets[i])
                borders.append(value)
        concat_borders.append(borders)
        if print_tot_structure:
            print(borders)
        for i in range(len(dataset.datasets)):
            if print_tot_structure:
                print('(Concat' + str(concat_count) + ') i: ' + str(i))
            dataset_cache, _, _, _ = get_dataset_structure(dataset.datasets[i], concat_borders, subset_inds, datasets, concat_count, print_tot_structure)
        if print_tot_structure:
            print('CONCAT ' + str(concat_count) + ' ending')

    return dataset, concat_borders, subset_inds, datasets


def underlying_datasets(dataset: object) -> pandas.DataFrame:
    """
    Collects all information of the underlying datasets of a complex dataset consisting of several different datasets.

    :param dataset: Dataset object
    :return: Pandas DataFrame with all information
    """

    # translation:
    if config.args.pipe_case == 'benchmark':
        translation = np.array([
            ['CIFAR10_train', 0],
            ['CIFAR10_test', 1],
            ['SVHN_train', 2],
            ['SVHN_test', 3],
            ['MNIST_train', 4],
            ['MNIST_test', 5],
            ['FashionMNIST_train', 6],
            ['FashionMNIST_test', 7],
        ], dtype=object)
    elif config.args.pipe_case == 'sat':
        translation = np.array([
            ['Euro_SAT_countryside_train', 0],
            ['Euro_SAT_countryside_test', 1],
            ['FMOW_train', 2],
            ['FMOW_test', 3],
            ['AID_train', 4],
            ['AID_test', 5],
            ['RESISC_train', 6],
            ['RESISC_test', 7],
        ], dtype=object)


    dataset_copy = copy.deepcopy(dataset)

    # prepare data (adjust target transform to make dataset countable)
    _, _, _, _ = get_dataset_structure(dataset_copy, print_tot_structure=False)
    dataloader = DataLoader(dataset_copy, batch_size=1, shuffle=False, num_workers=2)

    # identify datasets
    targets = []
    for data, target in dataloader:
        targets.append(target.item())
    underlying_sets = [translation[i, 0] for i in targets]

    # collect information in DataFrame
    results = np.empty((len(list(set(underlying_sets))), 3), dtype=object)
    results[:, 0] = list(set(underlying_sets))
    for idx, element in enumerate(list(set(underlying_sets))):
        results[idx, 1] = underlying_sets.count(element)

    # how many different datasets are there? -> initialize lists for indices
    sets = set(list(results[:, 0]))
    indices = {}
    for set_i in sets:
        indices[set_i] = []

    # get all indices
    for i in range(len(dataset)):
        name, train_test, idx = get_original_idx(dataset, i)
        indices[name + '_' + train_test].append(idx)
        indices[name + '_' + train_test].sort()

    print(indices)

    # assign indices to results structure
    result_df = pd.DataFrame(results, columns=['dataset', 'no. of samples', 'indices'])

    return result_df



def get_original_idx(dataset: object,
                     idx: int) -> Tuple[str, str, int, str]:
    """
    Computes the original index in the original dataset for a given index of a complex dataset structure.
    This can be used to check whether a sample is from the original training or test dataset

    :param dataset: Dataset object
    :param idx: Considered index
    :return: Sequence of information about the index
    """

    # for sat case, we need a slightly different handling
    if dataset.__class__.__name__ == 'ImageFolder':
        if dataset.name == 'Euro_SAT_countryside' or dataset.name == 'AID' or dataset.name == 'RESISC' or dataset.name == 'FMOW':
            train_test = 'train/test'
        else:
            raise NotImplementedError
        name = dataset.name
        path = dataset.samples[idx][0]
        return (name, train_test, idx, path)


    # if it is the final dataset: collect and return information about name, train or test split and original index
    if hasattr(dataset, 'transform'):

        if config.args.pipe_case == 'benchmark':
            name = dataset.__class__.__name__
        elif config.args.pipe_case == 'sat':
            name = dataset.name

        path = dataset.samples[idx][0]

        if config.args.pipe_case == 'benchmark':
            # if this is the final set:
            if hasattr(dataset, 'train'):
                train_test = 'train' if dataset.train else 'test'
            else:
                train_test = 'train' if dataset.split == 'train' else 'test'
        elif config.args.pipe_case == 'sat':
            if dataset.name == 'Euro_SAT_countryside' or dataset.name == 'AID' or dataset.name == 'RESISC' or dataset.name == 'FMOW':
                train_test = 'train/test'
            else:
                raise NotImplementedError

        return (name, train_test, idx, path)

    # if it does not yet have the transform function, recursively call function again
    if hasattr(dataset, 'dataset'):

        # EXCEPTION: if this is the final subset, return information
        if hasattr(dataset.dataset, 'transform'):

            if config.args.pipe_case == 'benchmark':
                if hasattr(dataset.dataset, 'train'):
                    train_test = 'train' if dataset.dataset.train else 'test'
                else:
                    train_test = 'train' if dataset.dataset.split == 'train' else 'test'
            elif config.args.pipe_case == 'sat':
                # lengths of train datasets
                if len(dataset) == 15200 or len(dataset) == 7606 or len(dataset) == 38061 or len(dataset) == 26254 or len(dataset) == 6368 or len(dataset) == 41554 or len(dataset) == 6720 or len(dataset) == 43169:
                    train_test = 'train'
                else:
                    train_test = 'test'

            if config.args.pipe_case == 'benchmark':
                name = dataset.dataset.__class__.__name__
            elif config.args.pipe_case == 'sat':
                name = dataset.dataset.name

            path = dataset.dataset.samples[dataset.indices[idx]][0]

            return (name, train_test, dataset.indices[idx], path)

        # EXCEPTION: if this is the final subset, but it follows a concatset (case for FMOW)
        elif hasattr(dataset.dataset, 'datasets') and hasattr(dataset.dataset.datasets[0], 'name') and config.args.pipe_case == 'sat' and hasattr(dataset.dataset.datasets[0], 'transform'):
            if dataset.dataset.datasets[0].name == 'FMOW':
                if len(dataset.dataset.datasets) == 3:
                    if len(dataset) == 43169:
                        train_test = 'train'
                    else:
                        train_test = 'test'
                    name = dataset.dataset.datasets[0].name

                    # if FMOW train, val or test set
                    if dataset.indices[idx] < 41554:
                        path = dataset.dataset.datasets[0].samples[dataset.indices[idx]][0]
                    elif dataset.indices[idx] < (41554+6223):
                        path = dataset.dataset.datasets[1].samples[(dataset.indices[idx] - 41554)][0]
                    else:
                        path = dataset.dataset.datasets[2].samples[(dataset.indices[idx] - 41554 - 6223)][0]


                    return (name, train_test, dataset.indices[idx], path)


        # if it does not yet have the transform function, recursively call function again
        else:
            idx = dataset.indices[idx]
            return get_original_idx(dataset.dataset, idx)

    # if it does not yet have the transform function, recursively call function again
    elif hasattr(dataset, 'datasets'):

        borders = []
        for i in range(len(dataset.datasets)):
            if len(borders) == 0:
                borders.append(len(dataset.datasets[i]))
            else:
                value = borders[-1] + len(dataset.datasets[i])
                borders.append(value)

        for i, border in enumerate(borders):
            if idx < border:
                set_idx = i
                if i == 0:
                    idx = idx
                else:
                    idx = idx - borders[i-1]
                break

        return get_original_idx(dataset.datasets[set_idx], idx)

    else:
        raise NotImplementedError

def get_delta_eps(network: object,
                  mech: str) -> Tuple[int, int]:
    """
    returns the parameters for the respective allocation mechanism.

    :param network: considered network
    :param mech: allocation mechanism ('maha' or 'odin')
    :return: delta, eps
    """

    if mech == 'odin':

        # get odin parameters of network model
        if config.args.dyn_pipe:
            if config.args.pipe_tune_dataset == 'pipe':
                if hasattr(network, 'trainloader'):
                    if network.trainloader.dataset.name == 'CIFAR10':
                        delta = config.args.c10_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.c10_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'Euro_SAT_countryside':
                        delta = config.args.eurosat_odin_delta_tpr95
                        eps = config.args.eurosat_odin_eps_tpr95
                else:
                    if network.name == 'exp_AI_1_dyn':
                        delta = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe
                    elif network.name == 'exp_AI_2_dyn':
                        delta = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe
                    elif network.name == 'exp_AI_3_dyn':
                        delta = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe
            else:
                if hasattr(network, 'trainloader'):
                    if network.trainloader.dataset.name == 'CIFAR10':
                        delta = config.args.c10_wideresnet_odin_delta_tpr95_iSUN
                        eps = config.args.c10_wideresnet_odin_eps_tpr95_iSUN
                    elif network.trainloader.dataset.name == 'Euro_SAT_countryside':
                        delta = config.args.eurosat_odin_delta_UCM
                        eps = config.args.eurosat_odin_eps_UCM
                else:
                    if network.name == 'exp_AI_1_dyn':
                        delta = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_ext
                        eps = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_ext
                    elif network.name == 'exp_AI_2_dyn':
                        delta = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_ext
                        eps = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_ext
                    elif network.name == 'exp_AI_3_dyn':
                        delta = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_ext
                        eps = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_ext
        else:
            if config.args.pipe_case == 'benchmark':
                if config.args.pipe_tune_dataset == 'iSUN':
                    if network.trainloader.dataset.name == 'CIFAR10':
                        delta = config.args.c10_wideresnet_odin_delta_tpr95_iSUN
                        eps = config.args.c10_wideresnet_odin_eps_tpr95_iSUN
                    elif network.trainloader.dataset.name == 'SVHN':
                        delta = config.args.svhn_wideresnet_odin_delta_tpr95_iSUN
                        eps = config.args.svhn_wideresnet_odin_eps_tpr95_iSUN
                    elif network.trainloader.dataset.name == 'MNIST':
                        delta = config.args.mnist_wideresnet_odin_delta_tpr95_iSUN
                        eps = config.args.mnist_wideresnet_odin_eps_tpr95_iSUN
                    elif network.trainloader.dataset.name == 'FMNIST':
                        delta = config.args.fmnist_wideresnet_odin_delta_tpr95_iSUN
                        eps = config.args.fmnist_wideresnet_odin_eps_tpr95_iSUN
                elif config.args.pipe_tune_dataset == 'pipe':
                    if network.trainloader.dataset.name == 'CIFAR10':
                        delta = config.args.c10_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.c10_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'SVHN':
                        delta = config.args.svhn_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.svhn_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'MNIST':
                        delta = config.args.mnist_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.mnist_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'FMNIST':
                        delta = config.args.fmnist_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.fmnist_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'exp_AI_1_dyn':
                        delta = config.args.exp_AI_1_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.exp_AI_1_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'exp_AI_2_dyn':
                        delta = config.args.exp_AI_2_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.exp_AI_2_wideresnet_odin_eps_tpr95_pipe
                    elif network.trainloader.dataset.name == 'exp_AI_3_dyn':
                        delta = config.args.exp_AI_3_wideresnet_odin_delta_tpr95_pipe
                        eps = config.args.exp_AI_3_wideresnet_odin_eps_tpr95_pipe
            elif config.args.pipe_case == 'sat':
                if config.args.pipe_tune_dataset == 'pipe':
                    if network.trainloader.dataset.name == 'Euro_SAT_countryside':
                        delta = config.args.eurosat_odin_delta_tpr95
                        eps = config.args.eurosat_odin_eps_tpr95
                    elif network.trainloader.dataset.name == 'FMOW':
                        delta = config.args.fmow_odin_delta_tpr95
                        eps = config.args.fmow_odin_eps_tpr95
                    elif network.trainloader.dataset.name == 'AID':
                        delta = config.args.aid_odin_delta_tpr95
                        eps = config.args.aid_odin_eps_tpr95
                    elif network.trainloader.dataset.name == 'RESISC':
                        delta = config.args.resisc_odin_delta_tpr95
                        eps = config.args.resisc_odin_eps_tpr95
                else:
                    if network.trainloader.dataset.name == 'Euro_SAT_countryside':
                        delta = config.args.eurosat_odin_delta_UCM
                        eps = config.args.eurosat_odin_eps_UCM
                    elif network.trainloader.dataset.name == 'FMOW':
                        delta = config.args.fmow_odin_delta_UCM
                        eps = config.args.fmow_odin_eps_UCM
                    elif network.trainloader.dataset.name == 'AID':
                        delta = config.args.aid_odin_delta_UCM
                        eps = config.args.aid_odin_eps_UCM
                    elif network.trainloader.dataset.name == 'RESISC':
                        delta = config.args.resisc_odin_delta_UCM
                        eps = config.args.resisc_odin_eps_UCM

    elif mech == 'maha':

        # load respective params
        if config.args.dyn_pipe:
            if hasattr(network, 'trainloader'):
                if network.trainloader.dataset.name == 'CIFAR10':
                    delta = config.args.c10_wideresnet_maha_delta_pipe_wFMNIST
                    eps = config.args.c10_wideresnet_maha_eps_pipe_wFMNIST
                if network.trainloader.dataset.name == 'Euro_SAT_countryside':
                    eps = config.args.eurosat_maha_eps
                    delta = config.args.eurosat_maha_delta
            else:
                if network.name == 'exp_AI_1_dyn':
                    delta = config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe
                    eps = config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe
                elif network.name == 'exp_AI_2_dyn':
                    delta = config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe
                    eps = config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe
                elif network.name == 'exp_AI_3_dyn':
                    delta = config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe
                    eps = config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe
        else:
            if config.args.pipe_case == 'benchmark':
                if network.trainloader.dataset.name == 'CIFAR10':
                    eps = config.args.c10_wideresnet_maha_eps_pipe_wFMNIST
                    delta = config.args.c10_wideresnet_maha_delta_pipe_wFMNIST
                elif network.trainloader.dataset.name == 'SVHN':
                    eps = config.args.svhn_wideresnet_maha_eps_pipe_wFMNIST
                    delta = config.args.svhn_wideresnet_maha_delta_pipe_wFMNIST
                elif network.trainloader.dataset.name == 'MNIST':
                    eps = config.args.mnist_wideresnet_maha_eps_pipe_wFMNIST
                    delta = config.args.mnist_wideresnet_maha_delta_pipe_wFMNIST
                elif network.trainloader.dataset.name == 'FMNIST':
                    eps = config.args.fmnist_wideresnet_maha_eps_pipe
                    delta = config.args.fmnist_wideresnet_maha_delta_pipe
                elif network.trainloader.dataset.name == 'exp_AI_1_dyn':
                    delta = config.args.exp_AI_1_wideresnet_maha_delta_tpr95_pipe
                    eps = config.args.exp_AI_1_wideresnet_maha_eps_tpr95_pipe
                elif network.trainloader.dataset.name == 'exp_AI_2_dyn':
                    delta = config.args.exp_AI_2_wideresnet_maha_delta_tpr95_pipe
                    eps = config.args.exp_AI_2_wideresnet_maha_eps_tpr95_pipe
                elif network.trainloader.dataset.name == 'exp_AI_3_dyn':
                    delta = config.args.exp_AI_3_wideresnet_maha_delta_tpr95_pipe
                    eps = config.args.exp_AI_3_wideresnet_maha_eps_tpr95_pipe
            elif config.args.pipe_case == 'sat':
                if network.trainloader.dataset.name == 'Euro_SAT_countryside':
                    eps = config.args.eurosat_maha_eps
                    delta = config.args.eurosat_maha_delta
                elif network.trainloader.dataset.name == 'FMOW':
                    eps = config.args.fmow_maha_eps
                    delta = config.args.fmow_maha_delta
                elif network.trainloader.dataset.name == 'AID':
                    eps = config.args.aid_maha_eps
                    delta = config.args.aid_maha_delta
                elif network.trainloader.dataset.name == 'RESISC':
                    eps = config.args.resisc_maha_eps
                    delta = config.args.resisc_maha_delta
                else:
                    raise NotImplementedError

    else:
        raise NotImplementedError


    return delta, eps


def get_maha_models(network: object) -> object:
    """
    returns the log regression model needed for maha

    :param network: considered network
    :return: log regression model for Maha
    """

    if hasattr(network, 'trainloader'):
        if config.args.pipe_tune_dataset == 'iSUN':
            if config.args.pipe_case == 'benchmark':
                file_path = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_' + 'wideresnet' + '_maha_lr' + '.pickle'
        elif config.args.pipe_tune_dataset == 'pipe':
            if config.args.pipe_case == 'benchmark':
                if config.args.dyn_pipe:
                    file_path = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_DYN' + 'wideresnet' + '_maha_lr_pipe' + '_wFMNIST' + config.args.ablation_study + '.pickle'
                else:
                    file_path = config.args.pipe_root + 'models/benchmark/MAHA/' + network.trainloader.dataset.name + '_' + 'wideresnet' + '_maha_lr_pipe' + '_wFMNIST' + '.pickle'
            elif config.args.pipe_case == 'sat':
                if config.args.dyn_pipe:
                    file_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_DYN_maha_lr_pipe' + config.args.ablation_study + '.pickle'
                else:
                    if network.trainloader.dataset.name != 'Euro_SAT_countryside':
                        file_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_maha_lr' + '.pickle'
                    else:
                        file_path = config.args.pipe_root + 'models/sat/MAHA/' + network.trainloader.dataset.name + '_maha_lr' + '.pickle'

    else:
        if config.args.pipe_case == 'benchmark':
            file_path = config.args.pipe_root + 'models/benchmark/MAHA/' + network.name + '_DYN' + 'wideresnet' + '_maha_lr_pipe' + '_wFMNIST' + config.args.ablation_study + '.pickle'
        else:
            file_path = config.args.pipe_root + 'models/sat/MAHA/' + network.name + '_DYN_maha_lr_pipe' + config.args.ablation_study + '.pickle'

    print('')
    print('file path of current model/expert: ' + file_path)

    filehandler = open(file_path, 'rb')
    lr = pickle.load(filehandler)

    return lr


def adjust_train_batch_size(network: object,
                            mode: str,
                            cached_batchsize: int = 0) -> None:
    """
    adjusts the train batch size temporarily

    :param network: considered network
    :param mode: 'reduce' before computing maha mean/cov; 'switch_back' after computing maha mean/cov
    :param batch_size_cache: remembered batch size
    """

    if mode == 'reduce':
        # reduce batch size temporarily
        if config.args.pipe_case == 'benchmark':
            working_batch_size = 64
        elif config.args.pipe_case == 'sat':
            working_batch_size = 32
        if hasattr(network, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                if network.trainloader.dataset.name == config.BENCHMARK_DATA_SETS['0']:
                    batch_size_cache = config.args.c10_wideresnet_train_batch
                    config.args.c10_wideresnet_train_batch = working_batch_size
                elif network.trainloader.dataset.name == config.BENCHMARK_DATA_SETS['1']:
                    batch_size_cache = config.args.svhn_wideresnet_train_batch
                    config.args.svhn_wideresnet_train_batch = working_batch_size
                elif network.trainloader.dataset.name == config.BENCHMARK_DATA_SETS['2']:
                    batch_size_cache = config.args.mnist_wideresnet_train_batch
                    config.args.mnist_wideresnet_train_batch = working_batch_size
                elif network.trainloader.dataset.name == 'FMNIST':
                    batch_size_cache = config.args.fmnist_wideresnet_train_batch
                    config.args.fmnist_wideresnet_train_batch = working_batch_size
                elif network.trainloader.dataset.name == 'exp_AI_1_dyn' or \
                        network.trainloader.dataset.name == 'exp_AI_2_dyn' or \
                        network.trainloader.dataset.name == 'exp_AI_3_dyn':
                    batch_size_cache = config.args.dyn_multi_train_batch
                    config.args.dyn_multi_train_batch = working_batch_size
                else:
                    raise NotImplementedError

            elif config.args.pipe_case == 'sat':
                if network.trainloader.dataset.name == 'Euro_SAT_countryside':
                    batch_size_cache = config.args.eurosat_train_batch
                    config.args.eurosat_train_batch = working_batch_size
                elif network.trainloader.dataset.name == 'FMOW':
                    batch_size_cache = config.args.fmow_train_batch
                    config.args.fmow_train_batch = working_batch_size
                elif network.trainloader.dataset.name == 'AID':
                    batch_size_cache = config.args.aid_train_batch
                    config.args.aid_train_batch = working_batch_size
                elif network.trainloader.dataset.name == 'RESISC':
                    batch_size_cache = config.args.resisc_train_batch
                    config.args.resisc_train_batch = working_batch_size
                else:
                    raise NotImplementedError
        else:
            if config.args.pipe_case == 'benchmark':
                if network.name == 'exp_AI_1_dyn' or \
                        network.name == 'exp_AI_2_dyn' or \
                        network.name == 'exp_AI_3_dyn':
                    batch_size_cache = config.args.dyn_multi_train_batch
                    config.args.dyn_multi_train_batch = working_batch_size
            else:
                if network.name == 'exp_AI_1_dyn' or \
                        network.name == 'exp_AI_2_dyn' or \
                        network.name == 'exp_AI_3_dyn':
                    batch_size_cache = config.args.dyn_multi_train_batch_sat
                    config.args.dyn_multi_train_batch_sat = working_batch_size


        return batch_size_cache


    elif mode == 'switch_back':

        # switch back original batch size
        if hasattr(network, 'trainloader'):
            if config.args.pipe_case == 'benchmark':
                if network.trainloader.dataset.name == config.BENCHMARK_DATA_SETS['0']:
                    config.args.c10_wideresnet_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == config.BENCHMARK_DATA_SETS['1']:
                    config.args.svhn_wideresnet_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == config.BENCHMARK_DATA_SETS['2']:
                    config.args.mnist_wideresnet_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == 'FMNIST':
                    config.args.fmnist_wideresnet_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == 'exp_AI_1_dyn' or \
                        network.trainloader.dataset.name == 'exp_AI_2_dyn' or \
                        network.trainloader.dataset.name == 'exp_AI_3_dyn':
                    config.args.dyn_multi_train_batch = cached_batchsize
                else:
                    raise NotImplementedError

            elif config.args.pipe_case == 'sat':
                if network.trainloader.dataset.name == 'Euro_SAT_countryside':
                    config.args.eurosat_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == 'FMOW':
                    config.args.fmow_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == 'AID':
                    config.args.aid_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == 'RESISC':
                    config.args.resisc_train_batch = cached_batchsize
                elif network.trainloader.dataset.name == 'exp_AI_1_dyn' or \
                        network.trainloader.dataset.name == 'exp_AI_2_dyn' or \
                        network.trainloader.dataset.name == 'exp_AI_3_dyn':
                    config.args.dyn_multi_train_batch_sat = cached_batchsize
                else:
                    raise NotImplementedError
        else:
            if config.args.pipe_case == 'benchmark':
                if network.name == 'exp_AI_1_dyn' or \
                        network.name == 'exp_AI_2_dyn' or \
                        network.name == 'exp_AI_3_dyn':
                    config.args.dyn_multi_train_batch = cached_batchsize
            else:
                if network.name == 'exp_AI_1_dyn' or \
                        network.name == 'exp_AI_2_dyn' or \
                        network.name == 'exp_AI_3_dyn':
                    config.args.dyn_multi_train_batch_sat = cached_batchsize


def get_epochs_loginterval(trainloader: DataLoader) -> Tuple[int, int]:
    """
    returns the number of epochs and the log interval for the given trainloader.

    :param trainloader: DataLoader of training dataset)
    :return: epochs, log_interval
    """

    if config.args.pipe_case == 'benchmark':
        if trainloader.dataset.name == 'CIFAR10':
            epochs = config.args.c10_epochs
            log_interval_fix = config.args.c10_log_interval
        elif trainloader.dataset.name == 'SVHN':
            epochs = config.args.svhn_epochs
            log_interval_fix = config.args.svhn_log_interval
        elif trainloader.dataset.name == 'MNIST':
            epochs = config.args.mnist_epochs
            log_interval_fix = config.args.mnist_log_interval
        elif trainloader.dataset.name == 'FMNIST':
            epochs = config.args.fmnist_epochs
            log_interval_fix = config.args.fmnist_log_interval
        elif trainloader.dataset.name == 'gating':
            epochs = config.args.gating_epochs
            log_interval_fix = config.args.gating_log_interval
        elif trainloader.dataset.name == 'global':
            epochs = config.args.global_epochs
            log_interval_fix = config.args.global_log_interval
        elif trainloader.dataset.name == 'dyn_single':
            epochs = config.args.c10_epochs
            log_interval_fix = config.args.dyn_log_interval
        elif trainloader.dataset.name == 'exp_AI_1_dyn' or \
                trainloader.dataset.name == 'exp_AI_2_dyn' or \
                trainloader.dataset.name == 'exp_AI_3_dyn':
            epochs = config.args.dyn_multi_train_epochs
            log_interval_fix = config.args.dyn_log_interval
        else:
            raise NotImplementedError
    elif config.args.pipe_case == 'sat':
        if trainloader.dataset.name == 'Euro_SAT_countryside':
            epochs = config.args.eurosat_epochs
            log_interval_fix = config.args.eurosat_log_interval
        elif trainloader.dataset.name == 'FMOW':
            epochs = config.args.fmow_epochs
            log_interval_fix = config.args.fmow_log_interval
        elif trainloader.dataset.name == 'AID':
            epochs = config.args.aid_epochs
            log_interval_fix = config.args.aid_log_interval
        elif trainloader.dataset.name == 'RESISC':
            epochs = config.args.resisc_epochs
            log_interval_fix = config.args.resisc_log_interval
        elif trainloader.dataset.name == 'gating':
            epochs = config.args.sat_gating_epochs
            log_interval_fix = config.args.sat_gating_log_interval
        elif trainloader.dataset.name == 'global':
            epochs = config.args.sat_global_epochs
            log_interval_fix = config.args.sat_global_log_interval
        elif trainloader.dataset.name == 'dyn_single':
            epochs = config.args.eurosat_epochs
            log_interval_fix = config.args.dyn_log_interval
        elif trainloader.dataset.name == 'dyn_multi':
            epochs = config.args.dyn_multi_train_epochs_sat
            log_interval_fix = config.args.dyn_log_interval_sat
        elif trainloader.dataset.name == 'exp_AI_1_dyn' or \
                trainloader.dataset.name == 'exp_AI_2_dyn' or \
                trainloader.dataset.name == 'exp_AI_3_dyn':
            epochs = config.args.dyn_multi_train_epochs_sat
            log_interval_fix = config.args.dyn_log_interval
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return epochs, log_interval_fix
