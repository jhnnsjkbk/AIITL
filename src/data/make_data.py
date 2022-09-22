# The code in this file is intended to load and provide all kinds of data needed in the project

# general libraries
from torch.utils.data import DataLoader, Subset, ConcatDataset
import src.config
import random
from src.data.augmentations import RandAugment
import numpy as np
import src.data.transforms as transforms
import src.utils.utils as utils
import copy
from typing import Tuple, Sequence

# datasets
from torchvision.datasets import CIFAR10, SVHN, MNIST, CIFAR100, ImageFolder, FashionMNIST

# get config
config = src.config.cfg


def get_data(dataset: str = '0') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads and preprocesses the specified dataset and creates respective DataLoader.

    :param dataset: Key of dataset as indicated in config.py
    :return: trainloader, valloader, testloader
    """

    if config.args.pipe_case == 'benchmark':

        # initialize test/val split of test datasets
        path_processed = config.args.pipe_root + 'data/processed'
        evens = [num for num in range(0, 10000) if num % 2 == 0]
        not_evens = [i for i in range(10000) if i not in evens]
        fifth = [num for num in range(0, 25000) if num % 5 == 0]
        not_fifth = [i for i in range(26032) if i not in fifth]

        # get CIFAR10
        if dataset == '0':
            print(config.BENCHMARK_DATA_SETS[dataset] + ' data is retrieved.')

            # define transforms (Normalization params are mean/std and calculated beforehand)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_transform.transforms.insert(0, RandAugment(2, 14))
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            trainset = CIFAR10(root=path_processed, train=True, download=True, transform=train_transform)
            testset = CIFAR10(root=path_processed, train=False, download=True, transform=test_transform)
            # split testset into val und test data
            valset = Subset(testset, evens)
            testset = Subset(testset, not_evens)

            # adjust class names to respective dataset
            trainset.classes = [s + '-CIFAR10' for s in trainset.classes]
            valset.classes = trainset.classes
            testset.classes = trainset.classes

            # define name for later references
            trainset.name = config.BENCHMARK_DATA_SETS[dataset]
            valset.name = config.BENCHMARK_DATA_SETS[dataset]
            testset.name = config.BENCHMARK_DATA_SETS[dataset]

            # create dataloader for training ML models
            trainloader = DataLoader(trainset, batch_size=config.args.c10_wideresnet_train_batch, shuffle=True,
                                         num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.c10_test_batch, shuffle=False,
                                   num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.c10_test_batch, shuffle=False,
                                    num_workers=2)

        if dataset == '1':
            print(config.BENCHMARK_DATA_SETS[dataset] + ' data is retrieved.')

            # define transforms
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_transform.transforms.insert(0, RandAugment(2, 14))
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


            trainset = SVHN(root=path_processed, split='train', download=True, transform=train_transform)
            testset = SVHN(root=path_processed, split='test', download=True, transform=test_transform)
            # split testset into val und test data
            valset = Subset(testset, fifth)
            testset = Subset(testset, not_fifth)

            # adjust class names to respective dataset
            trainset.classes = [str(s) + '-SVHN' for s in list(set(trainset.labels))]
            valset.classes = trainset.classes
            testset.classes = trainset.classes

            # define name for later references
            trainset.name = config.BENCHMARK_DATA_SETS[dataset]
            valset.name = config.BENCHMARK_DATA_SETS[dataset]
            testset.name = config.BENCHMARK_DATA_SETS[dataset]

            # create dataloader for training ML models
            trainloader = DataLoader(trainset, batch_size=config.args.svhn_wideresnet_train_batch, shuffle=True,
                                         num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.svhn_test_batch, shuffle=False,
                                    num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.svhn_test_batch, shuffle=False,
                                    num_workers=2)

        if dataset == '2':
            print(config.BENCHMARK_DATA_SETS[dataset] + ' data is retrieved.')

            # define transforms
            train_transform = transforms.Compose([
                transforms.Grayscale(3),  # make MNIST compatible to other data and models
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_transform = transforms.Compose([
                transforms.Grayscale(3),  # make MNIST compatible to other data and models
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


            trainset = MNIST(root=path_processed, train=True, download=True, transform=train_transform)
            testset = MNIST(root=path_processed, train=False, download=True, transform=test_transform)
            # split testset into val und test data
            valset = Subset(testset, evens)
            testset = Subset(testset, not_evens)

            # adjust class names to respective dataset
            trainset.classes = [str(s)[0] + '-MNIST' for s in trainset.classes]
            valset.classes = trainset.classes
            testset.classes = trainset.classes

            # define name for later references
            trainset.name = config.BENCHMARK_DATA_SETS[dataset]
            valset.name = config.BENCHMARK_DATA_SETS[dataset]
            testset.name = config.BENCHMARK_DATA_SETS[dataset]

            # create dataloader for training ML models
            trainloader = DataLoader(trainset, batch_size=config.args.mnist_wideresnet_train_batch, shuffle=True,
                                         num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.mnist_test_batch, shuffle=False,
                                    num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.mnist_test_batch, shuffle=False,
                                    num_workers=2)

        if dataset == '3':
            print(config.BENCHMARK_DATA_SETS[dataset] + ' data is retrieved.')

            # define transforms (Normalization params are mean/std and calculated beforehand)
            train_transform = transforms.Compose([
                transforms.Grayscale(3),  # make FMNIST compatible to other data and models
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.2023, 0.2023, 0.2023)),
                transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3),
            ])
            test_transform = transforms.Compose([
                transforms.Grayscale(3),  # make FMNIST compatible to other data and models
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.2023, 0.2023, 0.2023))
            ])


            trainset = FashionMNIST(root=path_processed, train=True, download=True, transform=train_transform)
            testset = FashionMNIST(root=path_processed, train=False, download=True, transform=test_transform)
            # split testset into val und test data
            valset = Subset(testset, evens)
            testset = Subset(testset, not_evens)

            # adjust class names to respective dataset
            trainset.classes = [str(s) + '-FMNIST' for s in trainset.classes]
            valset.classes = trainset.classes
            testset.classes = trainset.classes

            # define name for later references
            trainset.name = config.BENCHMARK_DATA_SETS[dataset]
            valset.name = config.BENCHMARK_DATA_SETS[dataset]
            testset.name = config.BENCHMARK_DATA_SETS[dataset]

            # create dataloader for training ML models
            trainloader = DataLoader(trainset, batch_size=config.args.fmnist_wideresnet_train_batch, shuffle=True,
                                         num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.fmnist_test_batch, shuffle=False,
                                   num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.fmnist_test_batch, shuffle=False,
                                    num_workers=2)


        if dataset == '4':

            # define transforms (Normalization params are mean/std and calculated beforehand)
            stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])
            train_transform.transforms.insert(0, RandAugment(2, 14))
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ])

            trainset = CIFAR100(root=path_processed, train=True, download=True, transform=train_transform)
            testset = CIFAR100(root=path_processed, train=False, download=True, transform=test_transform)
            # split testset into val und test data
            valset = Subset(testset, evens)
            testset = Subset(testset, not_evens)

            # adjust class names to respective dataset
            trainset.classes = [str(s) + '-CIFAR100' for s in trainset.classes]
            valset.classes = trainset.classes
            testset.classes = trainset.classes

            # define name for later references
            trainset.name = config.BENCHMARK_DATA_SETS[dataset]
            valset.name = config.BENCHMARK_DATA_SETS[dataset]
            testset.name = config.BENCHMARK_DATA_SETS[dataset]

            # create dataloader for training ML models
            trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                                     num_workers=2)
            valloader = DataLoader(valset, batch_size=1, shuffle=False,
                                   num_workers=2)
            testloader = DataLoader(testset, batch_size=1, shuffle=False,
                                    num_workers=2)


    elif config.args.pipe_case == 'sat':

        if dataset == '0':
            print(config.SAT_DATA_SETS[dataset] + ' data is retrieved.')
            path = config.args.pipe_root + 'data/processed/SAT_CASE_v2/EUROSAT'

            # define transforms (Normalization params are mean/std and calculated beforehand)
            transform_train = transforms.Compose([transforms.Resize(224),
                                                  transforms.RandomCrop(224, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            transform_test = transforms.Compose([transforms.Resize(224),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            # load data splits
            comp_set = ImageFolder(path, transform=transforms.ToTensor())

            seed_cache = config.args.main_seed
            random.seed(24)  # we need a constant seed for this data split to avoid data leakage, when running results on different seeds

            # create train, val and test splits
            tot_ind = list(range(len(comp_set)))
            random.shuffle(tot_ind)
            train_ind = tot_ind[:int(0.8*len(comp_set))]
            val_ind = tot_ind[int(0.8*len(comp_set)) : int(0.9*len(comp_set))]
            test_ind = tot_ind[int(0.9*len(comp_set)):]

            random.seed(seed_cache)

            classes = comp_set.classes
            trainset = Subset(ImageFolder(path, transform=transform_train), train_ind)
            valset = Subset(ImageFolder(path, transform=transform_test), val_ind)
            testset = Subset(ImageFolder(path, transform=transform_test), test_ind)

            # adjust class names to respective dataset
            trainset.classes = classes
            valset.classes = classes
            testset.classes = classes

            # define name for later references
            trainset.name = config.SAT_DATA_SETS[dataset]
            valset.name = config.SAT_DATA_SETS[dataset]
            testset.name = config.SAT_DATA_SETS[dataset]
            trainset.dataset.name = config.SAT_DATA_SETS[dataset]
            valset.dataset.name = config.SAT_DATA_SETS[dataset]
            testset.dataset.name = config.SAT_DATA_SETS[dataset]

            # create DataLoaders
            trainloader = DataLoader(trainset, batch_size=config.args.eurosat_train_batch, shuffle=True, num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.eurosat_test_batch, shuffle=False, num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.eurosat_test_batch, shuffle=False, num_workers=2)

        elif dataset == '1':

            print(config.SAT_DATA_SETS[dataset] + ' data is retrieved.')
            path_train = config.args.pipe_root + 'data/processed/SAT_CASE_v2/FMOW/train'
            path_val = config.args.pipe_root + 'data/processed/SAT_CASE_v2/FMOW/val_id'
            path_test = config.args.pipe_root + 'data/processed/SAT_CASE_v2/FMOW/test_id'

            # define transforms (Normalization params are mean/std and calculated beforehand)
            transform_train = transforms.Compose([transforms.Resize(224),
                                                  transforms.RandomCrop(224, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.4208116829395294, 0.42075517773628235, 0.3973940908908844], [0.19225512444972992, 0.18699270486831665, 0.18694618344306946])])
            transform_test = transforms.Compose([transforms.Resize(224),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.4208116829395294, 0.42075517773628235, 0.3973940908908844], [0.19225512444972992, 0.18699270486831665, 0.18694618344306946])])

            # load data sets (the dataset is loaded twice, as we aim at first concatenating the original dataset and then draw train and val/test data from the two transforms structures, respectively)

            # TRAIN SET (all transforms to train)
            trainset = ImageFolder(path_train, transform=transform_train)
            valset = ImageFolder(path_val, transform=transform_train)
            testset = ImageFolder(path_test, transform=transform_train)
            compset_fortrain = ConcatDataset([trainset, valset, testset])

            # VAL/TEST SET (all transforms to test)
            trainset = ImageFolder(path_train, transform=transform_test)
            valset = ImageFolder(path_val, transform=transform_test)
            testset = ImageFolder(path_test, transform=transform_test)
            compset_forvaltest = ConcatDataset([trainset, valset, testset])

            seed_cache = config.args.main_seed
            random.seed(24)  # we need a constant seed for this data split to avoid data leakage, when running results on different seeds

            # create train, val and test splits
            tot_ind = list(range(len(compset_fortrain)))
            random.shuffle(tot_ind)
            train_ind = tot_ind[:int(0.8*len(compset_fortrain))]
            val_ind = tot_ind[int(0.8*len(compset_fortrain)) : int(0.9*len(compset_fortrain))]
            test_ind = tot_ind[int(0.9*len(compset_fortrain)):]

            random.seed(seed_cache)
            classes = trainset.classes

            trainset = Subset(compset_fortrain, train_ind)
            valset = Subset(compset_forvaltest, val_ind)
            testset = Subset(compset_forvaltest, test_ind)

            # adjust class names to respective dataset
            trainset.classes = classes
            valset.classes = classes
            testset.classes = classes

            # define name for later references
            trainset.name = config.SAT_DATA_SETS[dataset]
            valset.name = config.SAT_DATA_SETS[dataset]
            testset.name = config.SAT_DATA_SETS[dataset]

            # set names for refences in other functions (e.g., utils.set_transform)
            trainset.dataset.datasets[0].name = config.SAT_DATA_SETS[dataset]
            trainset.dataset.datasets[1].name = config.SAT_DATA_SETS[dataset]
            trainset.dataset.datasets[2].name = config.SAT_DATA_SETS[dataset]
            valset.dataset.datasets[0].name = config.SAT_DATA_SETS[dataset]
            valset.dataset.datasets[1].name = config.SAT_DATA_SETS[dataset]
            valset.dataset.datasets[2].name = config.SAT_DATA_SETS[dataset]
            testset.dataset.datasets[0].name = config.SAT_DATA_SETS[dataset]
            testset.dataset.datasets[1].name = config.SAT_DATA_SETS[dataset]
            testset.dataset.datasets[2].name = config.SAT_DATA_SETS[dataset]

            # create DataLoaders
            trainloader = DataLoader(trainset, batch_size=config.args.fmow_train_batch, shuffle=True, num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.fmow_test_batch, shuffle=False, num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.fmow_test_batch, shuffle=False, num_workers=2)


        elif dataset == '2':
            print(config.SAT_DATA_SETS[dataset] + ' data is retrieved.')
            path = config.args.pipe_root + 'data/processed/SAT_CASE_v2/AID'

            # define transforms (Normalization params are mean/std and calculated beforehand)
            transform_train = transforms.Compose([transforms.Resize(224),
                                                  transforms.RandomCrop(224, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.4088771939277649, 0.4172964096069336, 0.38169991970062256], [0.17088061571121216, 0.1579170674085617, 0.1539187729358673])])
            transform_test = transforms.Compose([transforms.Resize(224),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.4088771939277649, 0.4172964096069336, 0.38169991970062256], [0.17088061571121216, 0.1579170674085617, 0.1539187729358673])])

            # load data splits
            comp_set = ImageFolder(path, transform=transforms.ToTensor())

            seed_cache = config.args.main_seed
            random.seed(24)  # we need a constant seed for this data split to avoid data leakage, when running results on different seeds

            # create train, val and test splits
            tot_ind = list(range(len(comp_set)))
            random.shuffle(tot_ind)
            train_ind = tot_ind[:int(0.8*len(comp_set))]
            val_ind = tot_ind[int(0.8*len(comp_set)) : int(0.9*len(comp_set))]
            test_ind = tot_ind[int(0.9*len(comp_set)):]

            random.seed(seed_cache)
            classes = comp_set.classes

            trainset = Subset(ImageFolder(path, transform=transform_train), train_ind)
            valset = Subset(ImageFolder(path, transform=transform_test), val_ind)
            testset = Subset(ImageFolder(path, transform=transform_test), test_ind)

            # adjust class names to respective dataset
            trainset.classes = classes
            valset.classes = classes
            testset.classes = classes

            # define name for later references
            trainset.name = config.SAT_DATA_SETS[dataset]
            valset.name = config.SAT_DATA_SETS[dataset]
            testset.name = config.SAT_DATA_SETS[dataset]
            trainset.dataset.name = config.SAT_DATA_SETS[dataset]
            valset.dataset.name = config.SAT_DATA_SETS[dataset]
            testset.dataset.name = config.SAT_DATA_SETS[dataset]

            # create DataLoaders
            trainloader = DataLoader(trainset, batch_size=config.args.aid_train_batch, shuffle=True, num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.aid_test_batch, shuffle=False, num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.aid_test_batch, shuffle=False, num_workers=2)

        elif dataset == '3':
            print(config.SAT_DATA_SETS[dataset] + ' data is retrieved.')
            path = config.args.pipe_root + 'data/processed/SAT_CASE_v2/RESISC'

            # define transforms (Normalization params are mean/std and calculated beforehand)
            transform_train = transforms.Compose([transforms.Resize(224),
                                                  transforms.RandomCrop(224, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.37495875358581543, 0.3951551616191864, 0.37500184774398804], [0.16299572587013245, 0.15253905951976776, 0.1510869413614273])])
            transform_test = transforms.Compose([transforms.Resize(224),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.37495875358581543, 0.3951551616191864, 0.37500184774398804], [0.16299572587013245, 0.15253905951976776, 0.1510869413614273])])

            # load data splits
            comp_set = ImageFolder(path, transform=transforms.ToTensor())

            seed_cache = config.args.main_seed
            random.seed(24)  # we need a constant seed for this data split to avoid data leakage, when running results on different seeds

            # create train, val and test splits
            tot_ind = list(range(len(comp_set)))
            random.shuffle(tot_ind)
            train_ind = tot_ind[:int(0.8*len(comp_set))]
            val_ind = tot_ind[int(0.8*len(comp_set)) : int(0.9*len(comp_set))]
            test_ind = tot_ind[int(0.9*len(comp_set)):]

            random.seed(seed_cache)
            classes = comp_set.classes

            trainset = Subset(ImageFolder(path, transform=transform_train), train_ind)
            valset = Subset(ImageFolder(path, transform=transform_test), val_ind)
            testset = Subset(ImageFolder(path, transform=transform_test), test_ind)

            # adjust class names to respective dataset
            trainset.classes = classes
            valset.classes = classes
            testset.classes = classes

            # define name for later references
            trainset.name = config.SAT_DATA_SETS[dataset]
            valset.name = config.SAT_DATA_SETS[dataset]
            testset.name = config.SAT_DATA_SETS[dataset]
            trainset.dataset.name = config.SAT_DATA_SETS[dataset]
            valset.dataset.name = config.SAT_DATA_SETS[dataset]
            testset.dataset.name = config.SAT_DATA_SETS[dataset]

            # create DataLoaders
            trainloader = DataLoader(trainset, batch_size=config.args.resisc_train_batch, shuffle=True, num_workers=2)
            valloader = DataLoader(valset, batch_size=config.args.resisc_test_batch, shuffle=False, num_workers=2)
            testloader = DataLoader(testset, batch_size=config.args.resisc_test_batch, shuffle=False, num_workers=2)

    else:
        raise NotImplementedError

    if dataset == '5':

        # dataset for ODIN tuning on external data

        if config.args.pipe_case == 'benchmark':
            # get iSUN
            print('iSUN is loaded.')

            # define transforms (Normalization params are mean/std and calculated beforehand)
            transform_isun = transforms.Compose([
                transforms.Resize(32 if config.args.pipe_case == 'benchmark' else 224),
                transforms.ToTensor(),
                transforms.Normalize([0.4861, 0.4633, 0.4275], [0.2331, 0.2343, 0.2441])
            ])
            outset = ImageFolder(config.args.pipe_root + 'data/raw/iSUN/', transform=transform_isun)
            outloader = DataLoader(outset, batch_size=1, shuffle=True)

            return outloader

        else:
            # get UCMerced
            print('UCMerced is loaded.')

            # define transforms
            transform_ucm = transforms.Compose([
                transforms.Resize(32 if config.args.pipe_case == 'benchmark' else 224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            outset = ImageFolder(config.args.pipe_root + 'data/raw/UCMerced/', transform=transform_ucm)
            outloader = DataLoader(outset, batch_size=1, shuffle=True)

            return outloader

    return trainloader, valloader, testloader



def gen_pipe_data(size: int,
                  noise: Sequence[float] = [0.2, 0.2, 0.2],
                  split: str = 'test') -> Tuple[DataLoader, Sequence, Sequence]:
    """
    Generates data for the pipeline by concatenating the available datasets.

    :param size: Number of samples of general ML model data
    :param noise: Share of unknown data of the artificial experts; multiplicator of size
    :param split: Split used to draw pipeline data; 'test' or 'val'
    :return: DataLoader of pipeline data, list of true labels, list of indices of the Subsets used
    """



    print('pipeline data is generated based on ' + split + ' data.')

    # enforce reproducibility
    main_seed = config.args.main_seed

    # some checks
    if config.args.pipe_case == 'benchmark':
        if size > 5000:
            return print('Error: Maximum 5000 C10 instances can be used, as remaining data was already used as val data.')
        for noise_level in noise:
            if (noise_level * size) > 5000:
                return print('Error: Maximum 5000 instances can be used, as remaining data was already used as val data.')


    # gen. model data: get all data and real class labels
    if split == 'test':
        _, _, gen_AI_testloader = get_data('0')
    else:
        _, gen_AI_testloader, _ = get_data('0')
    gen_AI_testset = gen_AI_testloader.dataset
    gen_AI_classes = gen_AI_testset.classes
    gen_AI_truelabels = []
    for i in range(0, len(gen_AI_testset)):
        gen_AI_truelabels.append(gen_AI_classes[gen_AI_testset[i][1]])

    # exp1 data: get all data and real class labels
    if noise[0] != 0:
        if split == 'test':
            _, _, exp_AI_1_testloader = get_data('1')
        else:
            _, exp_AI_1_testloader, _ = get_data('1')
        exp_AI_1_testset = exp_AI_1_testloader.dataset
        exp_AI_1_classes = exp_AI_1_testset.classes
        exp_AI_1_truelabels = []
        for i in range(0, len(exp_AI_1_testset)):
            exp_AI_1_truelabels.append(exp_AI_1_classes[exp_AI_1_testset[i][1]])

    # exp2 data: get all data and real class labels
    if noise[1] != 0:
        if split == 'test':
            _, _, exp_AI_2_testloader = get_data('2')
        else:
            _, exp_AI_2_testloader, _ = get_data('2')
        exp_AI_2_testset = exp_AI_2_testloader.dataset
        exp_AI_2_classes = exp_AI_2_testset.classes
        exp_AI_2_truelabels = []
        for i in range(0, len(exp_AI_2_testset)):
            exp_AI_2_truelabels.append(exp_AI_2_classes[exp_AI_2_testset[i][1]])

    # exp3 data: get all data and real class labels
    if noise[2] != 0:
        if split == 'test':
            _, _, exp_AI_3_testloader = get_data('3')
        else:
            _, exp_AI_3_testloader, _ = get_data('3')
        exp_AI_3_testset = exp_AI_3_testloader.dataset
        exp_AI_3_classes = exp_AI_3_testset.classes
        exp_AI_3_truelabels = []
        for i in range(0, len(exp_AI_3_testset)):
            exp_AI_3_truelabels.append(exp_AI_3_classes[exp_AI_3_testset[i][1]])


    # create indices to draw Subsets with respective sizes
    print('seed in script (for control): ' + str(main_seed))
    utils.set_seed(main_seed)
    gen_AI_ind = random.sample(range(len(gen_AI_testloader)), size)
    if config.args.ablation_study == 'smaller_OOD_share_v2':  # adjust size for the ablation study to keep a batch of ~1000 samples (some rounding errors occur)
        size = config.args.smaller_OOD_share_v2_batch_size
    pipe_inds = [gen_AI_ind]
    if noise[0] != 0:
        exp_AI_1_ind = random.sample(range(len(exp_AI_1_testloader)), int(size*noise[0]))
        pipe_inds.append(exp_AI_1_ind)
    if noise[1] != 0:
        exp_AI_2_ind = random.sample(range(len(exp_AI_2_testloader)), int(size*noise[1]))
        pipe_inds.append(exp_AI_2_ind)
    if noise[2] != 0:
        exp_AI_3_ind = random.sample(range(len(exp_AI_3_testloader)), int(size*noise[2]))
        pipe_inds.append(exp_AI_3_ind)


    # create Subsets and respective true class labels
    gen_AI_subset = Subset(gen_AI_testset, gen_AI_ind)
    gen_AI_subset_truelabels = []
    for i in range(0, len(gen_AI_ind)):
        gen_AI_subset_truelabels.append(gen_AI_truelabels[gen_AI_ind[i]])

    if noise[0] != 0:
        exp_AI_1_subset = Subset(exp_AI_1_testset, exp_AI_1_ind)
        exp_AI_1_subset_truelabels = []
        for i in range(0, len(exp_AI_1_ind)):
            exp_AI_1_subset_truelabels.append(exp_AI_1_truelabels[exp_AI_1_ind[i]])

    if noise[0] != 0:
        exp_AI_2_subset = Subset(exp_AI_2_testset, exp_AI_2_ind)
        exp_AI_2_subset_truelabels = []
        for i in range(0, len(exp_AI_2_ind)):
            exp_AI_2_subset_truelabels.append(exp_AI_2_truelabels[exp_AI_2_ind[i]])

    if noise[2] != 0:
        exp_AI_3_subset = Subset(exp_AI_3_testset, exp_AI_3_ind)
        exp_AI_3_subset_truelabels = []
        for i in range(0, len(exp_AI_3_ind)):
            exp_AI_3_subset_truelabels.append(exp_AI_3_truelabels[exp_AI_3_ind[i]])


    # create complete dataset/dataloader incl. true class labels out of respective subsets
    pipe_dataset = gen_AI_subset
    pipe_true_labels = gen_AI_subset_truelabels
    for idx, noise_level in enumerate(noise):
        if noise_level != 0:
            if idx == 0:
                pipe_dataset = ConcatDataset([pipe_dataset, exp_AI_1_subset])
                pipe_true_labels = pipe_true_labels + exp_AI_1_subset_truelabels
            elif idx == 1:
                pipe_dataset = ConcatDataset([pipe_dataset, exp_AI_2_subset])
                pipe_true_labels = pipe_true_labels + exp_AI_2_subset_truelabels
            elif idx == 2:
                pipe_dataset = ConcatDataset([pipe_dataset, exp_AI_3_subset])
                pipe_true_labels = pipe_true_labels + exp_AI_3_subset_truelabels

    pipeloader = DataLoader(pipe_dataset, batch_size=1, num_workers=2)  # no shuffling to keep order of true labels


    return pipeloader, pipe_true_labels, pipe_inds


def get_odin_tune_data(inDataset: str = '0') -> Tuple[DataLoader, DataLoader]:
    """
    Provides the data to tune ODIN hyperparameters. The type of the out-of-distribution data for tuning is defined via args.pipe_tune_dataset.

    :param inDataset: Key of dataset representing in-distribution data
    :return: DataLoader of in-distribution data, DataLoader of out-of-distribution data for tuning
    """

    # inDataloader (val data)
    _, inDataloader, _ = get_data(inDataset)
    inData_testset = inDataloader.dataset

    # limit sample for sat case to reduce computing time
    if config.args.pipe_case == 'sat':
        inData_testset = Subset(inData_testset, random.sample(range(len(inDataloader)), min(len(inDataloader), 5000)))

    # create inDataloader
    inloader = DataLoader(inData_testset, batch_size=1, num_workers=2, shuffle=True)

    # create outDataloader
    if config.args.pipe_tune_dataset == 'iSUN':
        # outDataloader - iSUN
        transform_isun = transforms.Compose([
            transforms.Resize(len(inloader.dataset[0][0][0])),
            transforms.ToTensor(),
            transforms.Normalize([0.4861, 0.4633, 0.4275], [0.2331, 0.2343, 0.2441])
        ])
        outset = ImageFolder(config.args.pipe_root + 'data/raw/iSUN/', transform=transform_isun)
        outloader = DataLoader(outset, batch_size=1, shuffle=True)

    elif config.args.pipe_tune_dataset == 'UCM':
        assert config.args.pipe_case == 'sat'
        outloader = get_data('5')

    elif config.args.pipe_tune_dataset == 'pipe':

        # create dataset with 5000 instances from pipeline data

        # define "not" inDataset
        indataset_id = int(inDataset)
        outdatasets = [str(i) for i in range(4) if i != indataset_id]

        # get out datasets
        _, outDataloader1, _ = get_data(outdatasets[0])
        _, outDataloader2, _ = get_data(outdatasets[1])
        _, outDataloader3, _ = get_data(outdatasets[2])

        if config.args.pipe_case == 'benchmark':
            idx1 = [i for i in range(5000) if i % 3 == 0]
            idx2 = idx1
            idx2.pop(0)
            outloader = DataLoader(ConcatDataset([Subset(outDataloader1.dataset, idx1),
                                                  Subset(outDataloader2.dataset, idx1),
                                                  Subset(outDataloader3.dataset, idx2)]),
                                   batch_size=1, num_workers=2, shuffle=True)
        elif config.args.pipe_case == 'sat':

            # dataset size is limited to <= 5000 samples manually
            if inDataset == '0':
                outset = ConcatDataset([Subset(outDataloader1.dataset, random.sample(range(len(outDataloader1.dataset)), 3364)),
                                       outDataloader2.dataset,
                                        outDataloader3.dataset])
            elif inDataset == '1':
                outset = ConcatDataset([outDataloader1.dataset, outDataloader2.dataset, outDataloader3.dataset])
            elif inDataset == '2':
                outset = ConcatDataset([outDataloader1.dataset,
                                       Subset(outDataloader2.dataset, random.sample(range(len(outDataloader2.dataset)), 2260)),
                                        outDataloader3.dataset])
            else:
                outset = ConcatDataset([outDataloader1.dataset,
                                        Subset(outDataloader2.dataset, random.sample(range(len(outDataloader2.dataset)), 2304)),
                                        outDataloader3.dataset])

                outloader = DataLoader(outset, batch_size=1, num_workers=2, shuffle=True)

    return inloader, outloader

def get_maha_tune_data(inDataset: str = '0') -> Tuple[DataLoader, DataLoader]:
    """
    Provides the data to tune Maha hyperparameters.

    :param inDataset: Key of dataset representing in-distribution data
    :return: DataLoader of in-distribution data, DataLoader of out-of-distribution data for tuning
    """

    # inDataloader
    _, inDataloader, _ = get_data(inDataset)
    inloader = inDataloader

    # limit sample for sat case to reduce computing time
    if config.args.pipe_case == 'sat':
        inloader = DataLoader(Subset(inloader.dataset, random.sample(range(len(inloader)), min(len(inloader), 5000))),
                              batch_size=1, num_workers=2, shuffle=True)


    # create dataset with 5000 instances from pipeline data

    # define "not" inDataset
    indataset_id = int(inDataset)
    outdatasets = [str(i) for i in range(4) if i != indataset_id]

    # get out datasets
    _, outDataloader1, _ = get_data(outdatasets[0])
    _, outDataloader2, _ = get_data(outdatasets[1])
    _, outDataloader3, _ = get_data(outdatasets[2])

    if config.args.pipe_case == 'benchmark':
        idx1 = [i for i in range(5000) if i % 3 == 0]
        idx2 = [i for i in range(5000) if i % 3 == 0 if i != 0]
        outloader = DataLoader(ConcatDataset([Subset(outDataloader1.dataset, idx1),
                                              Subset(outDataloader2.dataset, idx1),
                                              Subset(outDataloader3.dataset, idx2)]),
                               batch_size=1, num_workers=2, shuffle=True)

    elif config.args.pipe_case == 'sat':

        # dataset size is limited to <= 5000 samples manually
        if inDataset == '0':
            outset = ConcatDataset([Subset(outDataloader1.dataset, random.sample(range(len(outDataloader1.dataset)), 3364)),
                                    outDataloader2.dataset,
                                    outDataloader3.dataset])
        elif inDataset == '1':
            outset = ConcatDataset([outDataloader1.dataset, outDataloader2.dataset, outDataloader3.dataset])
        elif inDataset == '2':
            outset = ConcatDataset([outDataloader1.dataset,
                                    Subset(outDataloader2.dataset, random.sample(range(len(outDataloader2.dataset)), 2260)),
                                    outDataloader3.dataset])
        else:
            outset = ConcatDataset([outDataloader1.dataset,
                                    Subset(outDataloader2.dataset, random.sample(range(len(outDataloader2.dataset)), 2304)),
                                    outDataloader3.dataset])

        outloader = DataLoader(outset, batch_size=1, num_workers=2, shuffle=True)

    return inloader, outloader


def get_gating_data(type: str = 'selec_mech') -> Tuple[DataLoader, DataLoader]:
    """
    Provides DataLoaders of training and validation data for the gating models.

    :param type: Type of allocation mechanism; 'selec_mech' = A1, 'comb_mech' = A2
    :return: training and validation data
    """

    # gen model data
    gen_AI_data1, gen_AI_data2, _ = get_data('0')
    gen_AI_trainset = copy.deepcopy(gen_AI_data1.dataset)  # copies are needed to avoid problems with overwriting the targets below
    gen_AI_valset = copy.deepcopy(gen_AI_data2.dataset)

    # expdata
    exp_AI_1_data1, exp_AI_1_data2, _ = get_data('1')
    exp_AI_1_trainset = copy.deepcopy(exp_AI_1_data1.dataset)
    exp_AI_1_valset = copy.deepcopy(exp_AI_1_data2.dataset)

    exp_AI_2_data1, exp_AI_2_data2, _ = get_data('2')
    exp_AI_2_trainset = copy.deepcopy(exp_AI_2_data1.dataset)
    exp_AI_2_valset = copy.deepcopy(exp_AI_2_data2.dataset)

    exp_AI_3_data1, exp_AI_3_data2, _ = get_data('3')
    exp_AI_3_trainset = copy.deepcopy(exp_AI_3_data1.dataset)
    exp_AI_3_valset = copy.deepcopy(exp_AI_3_data2.dataset)

    # adjust targets for gating model and create gating datasets
    if type == 'selec_mech':
        if config.args.pipe_case == 'benchmark':
            gen_AI_trainset.target_transform = lambda x: 1
            gen_AI_valset.dataset.target_transform = lambda x: 1
            exp_AI_1_trainset.target_transform = lambda x: 0
            exp_AI_1_valset.dataset.target_transform = lambda x: 0
            exp_AI_2_trainset.target_transform = lambda x: 0
            exp_AI_2_valset.dataset.target_transform = lambda x: 0
            exp_AI_3_trainset.target_transform = lambda x: 0
            exp_AI_3_valset.dataset.target_transform = lambda x: 0
        elif config.args.pipe_case == 'sat':
            gen_AI_trainset.dataset.target_transform = lambda x: 1
            gen_AI_valset.dataset.target_transform = lambda x: 1
            exp_AI_1_trainset.dataset.datasets[0].target_transform = lambda x: 0
            exp_AI_1_trainset.dataset.datasets[1].target_transform = lambda x: 0
            exp_AI_1_trainset.dataset.datasets[2].target_transform = lambda x: 0
            exp_AI_1_valset.dataset.datasets[0].target_transform = lambda x: 0
            exp_AI_1_valset.dataset.datasets[1].target_transform = lambda x: 0
            exp_AI_1_valset.dataset.datasets[2].target_transform = lambda x: 0
            exp_AI_2_trainset.dataset.target_transform = lambda x: 0
            exp_AI_2_valset.dataset.target_transform = lambda x: 0
            exp_AI_3_trainset.dataset.target_transform = lambda x: 0
            exp_AI_3_valset.dataset.target_transform = lambda x: 0

    elif type == 'comb_mech':
        if config.args.pipe_case == 'benchmark':
            gen_AI_trainset.target_transform = lambda x: 0
            gen_AI_valset.dataset.target_transform = lambda x: 0
            exp_AI_1_trainset.target_transform = lambda x: 1
            exp_AI_1_valset.dataset.target_transform = lambda x: 1
            exp_AI_2_trainset.target_transform = lambda x: 2
            exp_AI_2_valset.dataset.target_transform = lambda x: 2
            exp_AI_3_trainset.target_transform = lambda x: 3
            exp_AI_3_valset.dataset.target_transform = lambda x: 3
        elif config.args.pipe_case == 'sat':
            gen_AI_trainset.dataset.target_transform = lambda x: 0
            gen_AI_valset.dataset.target_transform = lambda x: 0
            exp_AI_1_trainset.dataset.datasets[0].target_transform = lambda x: 1
            exp_AI_1_trainset.dataset.datasets[1].target_transform = lambda x: 1
            exp_AI_1_trainset.dataset.datasets[2].target_transform = lambda x: 1
            exp_AI_1_valset.dataset.datasets[0].target_transform = lambda x: 1
            exp_AI_1_valset.dataset.datasets[1].target_transform = lambda x: 1
            exp_AI_1_valset.dataset.datasets[2].target_transform = lambda x: 1
            exp_AI_2_trainset.dataset.target_transform = lambda x: 2
            exp_AI_2_valset.dataset.target_transform = lambda x: 2
            exp_AI_3_trainset.dataset.target_transform = lambda x: 3
            exp_AI_3_valset.dataset.target_transform = lambda x: 3

    # concat datasets
    trainset = ConcatDataset([gen_AI_trainset, exp_AI_1_trainset, exp_AI_2_trainset, exp_AI_3_trainset])
    valset = ConcatDataset([gen_AI_valset, exp_AI_1_valset, exp_AI_2_valset, exp_AI_3_valset])

    # create DataLoaders
    if config.args.pipe_case == 'benchmark':
        trainloader = DataLoader(trainset, batch_size=config.args.gating_train_batch, num_workers=2, shuffle=True)
        valloader = DataLoader(valset, batch_size=1, num_workers=2, shuffle=True)
    elif config.args.pipe_case == 'sat':
        trainloader = DataLoader(trainset, batch_size=config.args.sat_gating_train_batch, num_workers=2, shuffle=True)
        valloader = DataLoader(valset, batch_size=1, num_workers=2, shuffle=True)

    trainloader.dataset.name = 'gating'
    valloader.dataset.name = 'gating'

    return trainloader, valloader


def get_global_model_data(flag: str = None) -> Tuple[DataLoader, DataLoader]:
    """
    Provides DataLoaders of training and validation data for the global model.

    :param flag: Can be used to differentiate between intended use of the global model (not needed for regular global model)
    :return: training and validation data
    """

    # collect target mapping (map targets of all datasets after each other to provide an overview)
    if config.args.pipe_case == 'benchmark':
        target_mapping = np.empty((40, 2), dtype=object)
    elif config.args.pipe_case == 'sat':
        target_mapping = np.empty((78, 2), dtype=object)

    # gen_AI data: get data, collect target mapping (no renaming of targets needed)
    gen_AI_trainloader, gen_AI_valloader, _ = get_data('0')
    gen_AI_train_dataset = copy.deepcopy(gen_AI_trainloader.dataset)  # copies are needed to avoid problems with overwriting the targets below
    gen_AI_val_dataset = copy.deepcopy(gen_AI_valloader.dataset)
    gen_AI_classes = gen_AI_trainloader.dataset.classes
    target_mapping[:len(gen_AI_classes), 0] = gen_AI_classes
    if config.args.pipe_case == 'benchmark':
        target_mapping[:len(gen_AI_classes), 1] = list(range(10))
    elif config.args.pipe_case == 'sat':
        target_mapping[:len(gen_AI_classes), 1] = list(range(7))


    # exp_AI_1 data: get data, collect target mapping, rename targets respectively
    exp_AI_1_trainloader, exp_AI_1_valloader, _ = get_data('1')
    exp_AI_1_train_dataset = copy.deepcopy(exp_AI_1_trainloader.dataset)
    exp_AI_1_val_dataset = copy.deepcopy(exp_AI_1_valloader.dataset)
    exp_AI_1_classes = exp_AI_1_trainloader.dataset.classes
    target_mapping[len(gen_AI_classes):len(gen_AI_classes)+len(exp_AI_1_classes), 0] = exp_AI_1_classes
    if config.args.pipe_case == 'benchmark':
        target_mapping[len(gen_AI_classes):len(gen_AI_classes)+len(exp_AI_1_classes), 1] = list(range(10, 20))
        exp_AI_1_train_dataset.labels = exp_AI_1_train_dataset.labels + 10
        exp_AI_1_val_dataset.dataset.target_transform = lambda x: x+10
    elif config.args.pipe_case == 'sat':
        target_mapping[len(gen_AI_classes):len(gen_AI_classes)+len(exp_AI_1_classes), 1] = list(range(7, 42))
        exp_AI_1_train_dataset.dataset.datasets[0].target_transform = lambda x: x+7
        exp_AI_1_train_dataset.dataset.datasets[1].target_transform = lambda x: x+7
        exp_AI_1_train_dataset.dataset.datasets[2].target_transform = lambda x: x+7
        exp_AI_1_val_dataset.dataset.datasets[0].target_transform = lambda x: x+7
        exp_AI_1_val_dataset.dataset.datasets[1].target_transform = lambda x: x+7
        exp_AI_1_val_dataset.dataset.datasets[2].target_transform = lambda x: x+7

    # exp_AI_2 data: get data, collect target mapping, rename targets respectively
    exp_AI_2_trainloader, exp_AI_2_valloader, _ = get_data('2')
    exp_AI_2_train_dataset = copy.deepcopy(exp_AI_2_trainloader.dataset)
    exp_AI_2_val_dataset = copy.deepcopy(exp_AI_2_valloader.dataset)
    exp_AI_2_classes = exp_AI_2_trainloader.dataset.classes
    target_mapping[len(gen_AI_classes)+len(exp_AI_1_classes):len(gen_AI_classes)+len(exp_AI_1_classes)+len(exp_AI_2_classes), 0] = exp_AI_2_classes
    if config.args.pipe_case == 'benchmark':
        target_mapping[len(gen_AI_classes)+len(exp_AI_1_classes):len(gen_AI_classes)+len(exp_AI_1_classes)+len(exp_AI_2_classes), 1] = list(range(20, 30))
        exp_AI_2_train_dataset.targets = exp_AI_2_train_dataset.targets + 20
        exp_AI_2_val_dataset.dataset.target_transform = lambda x: x+20
    elif config.args.pipe_case == 'sat':
        target_mapping[len(gen_AI_classes)+len(exp_AI_1_classes):len(gen_AI_classes)+len(exp_AI_1_classes)+len(exp_AI_2_classes), 1] = list(range(42, 66))
        exp_AI_2_train_dataset.dataset.target_transform = lambda x: x+42
        exp_AI_2_val_dataset.dataset.target_transform = lambda x: x+42

    # exp_AI_3 data: get data, collect target mapping, rename targets respectively
    exp_AI_3_trainloader, exp_AI_3_valloader, _ = get_data('3')
    exp_AI_3_train_dataset = copy.deepcopy(exp_AI_3_trainloader.dataset)
    exp_AI_3_val_dataset = copy.deepcopy(exp_AI_3_valloader.dataset)
    exp_AI_3_classes = exp_AI_3_trainloader.dataset.classes
    target_mapping[len(gen_AI_classes)+len(exp_AI_1_classes)+len(exp_AI_2_classes):, 0] = exp_AI_3_classes
    if config.args.pipe_case == 'benchmark':
        target_mapping[len(gen_AI_classes)+len(exp_AI_1_classes)+len(exp_AI_2_classes):, 1] = list(range(30, 40))
        exp_AI_3_train_dataset.targets = exp_AI_3_train_dataset.targets + 30
        exp_AI_3_val_dataset.dataset.target_transform = lambda x: x+30
    elif config.args.pipe_case == 'sat':
        target_mapping[len(gen_AI_classes)+len(exp_AI_1_classes)+len(exp_AI_2_classes):, 1] = list(range(66, 78))
        exp_AI_3_train_dataset.dataset.target_transform = lambda x: x+66
        exp_AI_3_val_dataset.dataset.target_transform = lambda x: x+66

    # create DataLoaders
    if config.args.pipe_case == 'benchmark':
        batch_size = config.args.global_train_batch
    elif config.args.pipe_case == 'sat':
        batch_size = config.args.sat_global_train_batch
    trainloader = DataLoader(ConcatDataset([gen_AI_train_dataset, exp_AI_1_train_dataset, exp_AI_2_train_dataset, exp_AI_3_train_dataset]),
                                 batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(ConcatDataset([gen_AI_val_dataset, exp_AI_1_val_dataset, exp_AI_2_val_dataset, exp_AI_3_val_dataset]),
                             batch_size=1, shuffle=False, num_workers=2)
    trainloader.dataset.name = 'global'
    valloader.dataset.name = 'global'

    # adjust sizes based on flag
    if flag == 'global':
        train_share = 1
        val_share = 1
        trainloader = DataLoader(Subset(trainloader.dataset, random.sample(range(len(trainloader.dataset)),
                                                                           int(train_share*len(trainloader.dataset)))),
                                 batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = DataLoader(Subset(valloader.dataset, random.sample(range(len(valloader.dataset)),
                                                                       int(val_share*len(valloader.dataset)))),
                               batch_size=1, shuffle=False, num_workers=2)

        trainloader.dataset.name = 'global'


    return trainloader, valloader


def gen_strong_weak_genAIdata(no_strong_classes: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Provides training, validation and testdata for the single-domain systems, based on the desired class split into known and unknown classes.
    By default, the first x classes of the dataset are drawn as known.

    :param no_strong_classes: Number of known classes of general ML model in single-domain systems
    :return: trainloader_known, valloader_known, testloader_known, trainloader_unknown, valloader_unknown, testloader_unknown
    """

    # get total data
    trainloader_tot, valloader_tot, testloader_tot = get_data('0')
    tot_classes = len(trainloader_tot.dataset.classes)
    assert tot_classes - no_strong_classes >= 2

    # reproduction
    print('seed in script (for control): ' + str(config.args.main_seed))
    utils.set_seed(config.args.main_seed)

    strong_classes = list(range(no_strong_classes))
    
    # create batch size 1 loaders for better handling below
    trainloader_tot = DataLoader(trainloader_tot.dataset, batch_size=1)
    valloader_tot = DataLoader(valloader_tot.dataset, batch_size=1)
    testloader_tot = DataLoader(testloader_tot.dataset, batch_size=1)


    # collect respective indices of known and unknown data
    train_strong_ind = []
    train_weak_ind = []
    for idx, (data, target) in enumerate(trainloader_tot):
        if target.item() in strong_classes:
            train_strong_ind.append(idx)
        else:
            train_weak_ind.append(idx)
    assert len(train_weak_ind) + len(train_strong_ind) == len(trainloader_tot.dataset)

    val_strong_ind = []
    val_weak_ind = []
    for idx, (data, target) in enumerate(valloader_tot):
        if target.item() in strong_classes:
            val_strong_ind.append(idx)
        else:
            val_weak_ind.append(idx)
    assert len(val_weak_ind) + len(val_strong_ind) == len(valloader_tot.dataset)

    test_strong_ind = []
    test_weak_ind = []
    for idx, (data, target) in enumerate(testloader_tot):
        if target.item() in strong_classes:
            test_strong_ind.append(idx)
        else:
            test_weak_ind.append(idx)
    assert len(test_weak_ind) + len(test_strong_ind) == len(testloader_tot.dataset)


    # create respective DataLoaders based on the respective Subsets
    if config.args.pipe_case == 'sat':
        train_batch_size = config.args.eurosat_train_batch
    elif config.args.pipe_case == 'benchmark':
        train_batch_size = config.args.c10_wideresnet_train_batch
    trainloader_known = DataLoader(Subset(trainloader_tot.dataset, train_strong_ind), batch_size=train_batch_size, shuffle=False, num_workers=2)
    trainloader_unknown = DataLoader(Subset(trainloader_tot.dataset, train_weak_ind), batch_size=train_batch_size, shuffle=False, num_workers=2)
    valloader_known = DataLoader(Subset(valloader_tot.dataset, val_strong_ind), batch_size=1, shuffle=False, num_workers=2)
    valloader_unknown = DataLoader(Subset(valloader_tot.dataset, val_weak_ind), batch_size=1, shuffle=False, num_workers=2)
    testloader_known = DataLoader(Subset(testloader_tot.dataset, test_strong_ind), batch_size=1, shuffle=False, num_workers=2)
    testloader_unknown = DataLoader(Subset(testloader_tot.dataset, test_weak_ind), batch_size=1, shuffle=False, num_workers=2)

    # rename datasets for later references
    trainloader_known.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'
    trainloader_unknown.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'
    valloader_known.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'
    valloader_unknown.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'
    testloader_known.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'
    testloader_unknown.dataset.name = 'CIFAR10' if config.args.pipe_case == 'benchmark' else 'Euro_SAT_countryside'

    # add classes for later references
    trainloader_known.dataset.classes = trainloader_tot.dataset.classes
    trainloader_unknown.dataset.classes = trainloader_tot.dataset.classes
    valloader_known.dataset.classes = trainloader_tot.dataset.classes
    valloader_unknown.dataset.classes = trainloader_tot.dataset.classes
    testloader_known.dataset.classes = trainloader_tot.dataset.classes
    testloader_unknown.dataset.classes = trainloader_tot.dataset.classes

    return trainloader_known, valloader_known, testloader_known, trainloader_unknown, valloader_unknown, testloader_unknown


def gen_train_batches(domain: str = 'single') -> Sequence:
    """
    Provides the data batches for the dynamic systems. Each batch consists of a list with the DataLoader. For multi-domain, it further consists of the
    true labels and list of indications about whether the data sample is from the original training or validation distribution.

    :param domain: Considered domain; 'single' or 'multi'
    :return: Sequence of batches
    """

    print(domain + ' domain batches for dynamic pipeline are created.')

    if domain == 'single':

        # collect remaining data (not yet used for any training or tuning in single domain)
        if config.args.pipe_case == 'benchmark':
            no_strong_classes = config.args.dyn_single_no_strong_classes
            classes = ['airplane-CIFAR10', 'automobile-CIFAR10', 'bird-CIFAR10', 'cat-CIFAR10', 'deer-CIFAR10', 'dog-CIFAR10', 'frog-CIFAR10', 'horse-CIFAR10', 'ship-CIFAR10', 'truck-CIFAR10']
        elif config.args.pipe_case == 'sat':
            no_strong_classes = config.args.dyn_single_no_strong_classes_sat
            classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
        _, valloader_strong, testloader_strong, trainloader_weak, valloader_weak, testloader_weak = gen_strong_weak_genAIdata(no_strong_classes)
        rem_strong_data = ConcatDataset([valloader_strong.dataset, testloader_strong.dataset])
        rem_weak_data = ConcatDataset([trainloader_weak.dataset, valloader_weak.dataset, testloader_weak.dataset])

        no_batches = config.args.dyn_single_no_batches

        # specify design of batches
        if config.args.dyn_batch_design == 'independent':
            raise NotImplementedError
        elif config.args.dyn_batch_design == 'dependent':  # for single-domain, the static AIITL-system just selects the first batch of the single-domain batches

            # create data batches, put it into dataloaders

            tot_strong_ind = list(range(len(rem_strong_data)))
            tot_weak_ind = list(range(len(rem_weak_data)))

            # shuffle indices
            random.shuffle(tot_strong_ind)
            tot_strong_ind = np.array(tot_strong_ind)
            random.shuffle(tot_weak_ind)
            tot_weak_ind = np.array(tot_weak_ind)

            # draw constant number of samples out of shuffled total indices for each batch
            batches = []
            range_split = range(0, no_batches)
            for i in range_split:
                tot_strong_subsetind = list(tot_strong_ind[i*int(len(tot_strong_ind)/no_batches):(i+1)*int(len(tot_strong_ind)/no_batches)])
                gen_AI_strong_set = Subset(rem_strong_data, tot_strong_subsetind)
                tot_weak_subsetind = list(tot_weak_ind[i*int(len(tot_weak_ind)/no_batches):(i+1)*int(len(tot_weak_ind)/no_batches)])
                gen_AI_weak_set = Subset(rem_weak_data, tot_weak_subsetind)
                batch_set = ConcatDataset([gen_AI_strong_set, gen_AI_weak_set])
                batch_loader = DataLoader(batch_set, num_workers=2, shuffle=False, batch_size=1)

                # collect true labels
                batch_true_labels = []
                for _, target in batch_loader:
                    batch_true_labels.append(classes[target.item()])
                batches.append([batch_loader, batch_true_labels])

            # make sure all classes are present in each batch
            for i in range(len(batches)):
                targets = []
                for data, target in batches[i][0]:
                    targets.append(target.item())
                targets = list(set(targets))
                if config.args.pipe_case == 'benchmark':
                    assert len(targets) == 10
                elif config.args.pipe_case == 'sat':
                    assert len(targets) == 7

            print('Avg. batch size: ' + str(len(batches[0][0].dataset)))
            print('no. of batches: ' + str(no_batches))

            return batches


    if domain == 'multi':

        # Expert data: use train + valdata, but use valdata as hold out set for later OOD tuning etc during inference.
        # C-10: remaining test + valdata, but use valdata as hold out set for later OOD tuning etc.
        # Not included in batches for safety reasons (avoid any leakage, will be accessed directly from "get_data()" if needed)

        # adjust pipe size and noise manually to distribute all available data over the number of batches and facilitate same data batch design for static system
        if config.args.dyn_batch_design == 'independent':
            raise NotImplementedError
        elif config.args.dyn_batch_design == 'dependent':
            no_batches = config.args.dyn_no_batches
            if config.args.dyn_MVP_test:
                no_batches = 1000
            if config.args.pipe_case == 'benchmark':
                config.args.pipe_size = int(5000/(no_batches+1))
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [78257, 65000, 65000]]
            elif config.args.pipe_case == 'sat':
                config.args.pipe_size = int(1900/(no_batches+1))
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [48565, 7164, 7560]]
        elif config.args.dyn_batch_design == 'independent_recurring_genAI':  # adjust size and noise to account for desired share of OOD data
            no_batches = config.args.independent_recurring_genAI_dyn_no_batches
            if config.args.pipe_case == 'benchmark':
                config.args.pipe_size = int(sum([int(i/no_batches) for i in [78257, 65000, 65000]])/config.args.smaller_OOD_share)
                assert config.args.pipe_size <= 5000
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [78257, 65000, 65000]]
            elif config.args.pipe_case == 'sat':
                config.args.pipe_size = int(sum([int(i/no_batches) for i in [48565, 7164, 7560]])/config.args.smaller_OOD_share)
                assert config.args.pipe_size <= 1900
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [48565, 7164, 7560]]


        # get total test data of general ML model to be used in the batches
        main_seed = config.args.main_seed  # enforce reproducibility
        print('seed in script (for control): ' + str(main_seed))
        utils.set_seed(main_seed)
        _, _, gen_AI_testloader = get_data('0')
        gen_AI_truelabels = []
        for _, target in gen_AI_testloader:
            gen_AI_truelabels.append(gen_AI_testloader.dataset.classes[target.item()])

        # get general model data and split into available data (gen_AI_test_ind_foruse)
        # and data that will be used in final test batch (gen_AI_testpipe_ind), that is to be compared for the static system
        print('seed in script (for control): ' + str(main_seed))
        utils.set_seed(main_seed)  # enforce reproducibility
        if config.args.dyn_batch_design == 'independent':
            raise NotImplementedError
        elif config.args.dyn_batch_design == 'dependent':
            gen_AI_testpipe_ind = random.sample(range(len(gen_AI_testloader)), config.args.pipe_size)  # this is the same list of indices as when the static AIITL-system is called
            gen_AI_test_ind_foruse = []
            for i in range(len(gen_AI_testloader)):
                if i not in gen_AI_testpipe_ind:
                    gen_AI_test_ind_foruse.append(i)
            assert (len(gen_AI_testpipe_ind) + len(gen_AI_test_ind_foruse)) == len(gen_AI_testloader)
        elif config.args.dyn_batch_design == 'independent_recurring_genAI':
            # use same data as for test batch in every batch
            gen_AI_test_ind_foruse = random.sample(range(len(gen_AI_testloader)), config.args.pipe_size)
        gen_AI_truelabels = list(np.array(gen_AI_truelabels)[gen_AI_test_ind_foruse])


        # get expert training data and true labels
        exp_AI_1, exp_AI_1_val, _ = get_data('1')
        exp_AI_1_truelabels = []
        for i in range(0, len(exp_AI_1.dataset)):
            exp_AI_1_truelabels.append(exp_AI_1.dataset.classes[exp_AI_1.dataset[i][1]])
        for i in range(0, len(exp_AI_1_val.dataset)):
            exp_AI_1_truelabels.append(exp_AI_1_val.dataset.classes[exp_AI_1_val.dataset[i][1]])

        exp_AI_2, exp_AI_2_val, _ = get_data('2')
        exp_AI_2_truelabels = []
        for i in range(0, len(exp_AI_2.dataset)):
            exp_AI_2_truelabels.append(exp_AI_2.dataset.classes[exp_AI_2.dataset[i][1]])
        for i in range(0, len(exp_AI_2_val.dataset)):
            exp_AI_2_truelabels.append(exp_AI_2_val.dataset.classes[exp_AI_2_val.dataset[i][1]])

        exp_AI_3, exp_AI_3_val, _ = get_data('3')
        exp_AI_3_truelabels = []
        for i in range(0, len(exp_AI_3.dataset)):
            exp_AI_3_truelabels.append(exp_AI_3.dataset.classes[exp_AI_3.dataset[i][1]])
        for i in range(0, len(exp_AI_3_val.dataset)):
            exp_AI_3_truelabels.append(exp_AI_3_val.dataset.classes[exp_AI_3_val.dataset[i][1]])


        # create dataset of total remaining instances (not used yet)
        rem_genAI_data = Subset(gen_AI_testloader.dataset, gen_AI_test_ind_foruse)
        rem_exp_AI_1_data = ConcatDataset([exp_AI_1.dataset, exp_AI_1_val.dataset])
        rem_exp_AI_2_data = ConcatDataset([exp_AI_2.dataset, exp_AI_2_val.dataset])
        rem_exp_AI_3_data = ConcatDataset([exp_AI_3.dataset, exp_AI_3_val.dataset])

        # shuffle indices
        tot_gen_AI_ind = list(range(len(rem_genAI_data)))
        tot_exp_AI_1_ind = list(range(len(rem_exp_AI_1_data)))
        tot_exp_AI_2_ind = list(range(len(rem_exp_AI_2_data)))
        tot_exp_AI_3_ind = list(range(len(rem_exp_AI_3_data)))
        random.shuffle(tot_gen_AI_ind)
        tot_gen_AI_ind = np.array(tot_gen_AI_ind)
        random.shuffle(tot_exp_AI_1_ind)
        tot_exp_AI_1_ind = np.array(tot_exp_AI_1_ind)
        random.shuffle(tot_exp_AI_2_ind)
        tot_exp_AI_2_ind = np.array(tot_exp_AI_2_ind)
        random.shuffle(tot_exp_AI_3_ind)
        tot_exp_AI_3_ind = np.array(tot_exp_AI_3_ind)


        # create data batches, put it into dataloaders, collect indicator for train or validation data origin
        # train set: 0, val set: 1, gen AI set: -1
        batches = []
        if config.args.dyn_MVP_test:
            range_split = range(0, 2)
        else:
            range_split = range(0, no_batches)
        for i in range_split:
            gen_AI_subsetind = list(tot_gen_AI_ind[i*int(len(tot_gen_AI_ind)/no_batches):(i+1)*int(len(tot_gen_AI_ind)/no_batches)])
            gen_AI_set = Subset(rem_genAI_data, gen_AI_subsetind)
            gen_AI_set_truelabels = list(np.array(gen_AI_truelabels)[gen_AI_subsetind])
            gen_AI_set_type = [-1] * len(gen_AI_set_truelabels)
            # if recurring gen AI, adjust selection
            if config.args.dyn_batch_design == 'independent_recurring_genAI':
                gen_AI_set = rem_genAI_data
                gen_AI_set_truelabels = gen_AI_truelabels
                gen_AI_set_type = [-1] * len(gen_AI_set_truelabels)

            exp_AI_1_subsetind = list(tot_exp_AI_1_ind[i*int(len(tot_exp_AI_1_ind)/no_batches):(i+1)*int(len(tot_exp_AI_1_ind)/no_batches)])
            if config.args.dyn_batch_design == 'independent_recurring_genAI':  # if recurring gen AI, adjust selection to ensure correct number of unknown data samples
                exp_AI_1_subsetind = list(tot_exp_AI_1_ind[i*int(config.args.pipe_noise[0]*config.args.pipe_size):(i+1)*int(config.args.pipe_noise[0]*config.args.pipe_size)])
            exp_AI_1_set = Subset(rem_exp_AI_1_data, exp_AI_1_subsetind)
            exp_AI_1_set_truelabels = list(np.array(exp_AI_1_truelabels)[exp_AI_1_subsetind])

            exp_AI_2_subsetind = list(tot_exp_AI_2_ind[i*int(len(tot_exp_AI_2_ind)/no_batches):(i+1)*int(len(tot_exp_AI_2_ind)/no_batches)])
            if config.args.dyn_batch_design == 'independent_recurring_genAI':  # if recurring gen AI, adjust selection to ensure correct number of unknown data samples
                exp_AI_2_subsetind = list(tot_exp_AI_2_ind[i*int(config.args.pipe_noise[1]*config.args.pipe_size):(i+1)*int(config.args.pipe_noise[1]*config.args.pipe_size)])
            exp_AI_2_set = Subset(rem_exp_AI_2_data, exp_AI_2_subsetind)
            exp_AI_2_set_truelabels = list(np.array(exp_AI_2_truelabels)[exp_AI_2_subsetind])

            exp_AI_3_subsetind = list(tot_exp_AI_3_ind[i*int(len(tot_exp_AI_3_ind)/no_batches):(i+1)*int(len(tot_exp_AI_3_ind)/no_batches)])
            if config.args.dyn_batch_design == 'independent_recurring_genAI':  # if recurring gen AI, adjust selection to ensure correct number of unknown data samples
                exp_AI_3_subsetind = list(tot_exp_AI_3_ind[i*int(config.args.pipe_noise[2]*config.args.pipe_size):(i+1)*int(config.args.pipe_noise[2]*config.args.pipe_size)])
            exp_AI_3_set = Subset(rem_exp_AI_3_data, exp_AI_3_subsetind)
            exp_AI_3_set_truelabels = list(np.array(exp_AI_3_truelabels)[exp_AI_3_subsetind])


            # for the sat case, the idx shuffle is repeated until the first batch consists of all classes (design requirement, to simulate perfect human to expert allocation of labelled unknown data)
            if i == 0 and config.args.pipe_case == 'sat' and not config.args.dyn_MVP_test:

                # FMOW
                fmow_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                missing_fmow_class = list(set(fmow_classes) - set(exp_AI_1_set_truelabels))
                seed_i = 1
                while len(missing_fmow_class) > 0:
                    #print('reshuffeling..')
                    # shuffle again
                    random.seed(seed_i)
                    random.shuffle(tot_exp_AI_1_ind)
                    tot_exp_AI_1_ind = np.array(tot_exp_AI_1_ind)
                    # assign again
                    exp_AI_1_subsetind = list(tot_exp_AI_1_ind[i*int(len(tot_exp_AI_1_ind)/no_batches):(i+1)*int(len(tot_exp_AI_1_ind)/no_batches)])
                    if config.args.dyn_batch_design == 'independent_recurring_genAI':
                        exp_AI_1_subsetind = list(tot_exp_AI_1_ind[i*int(config.args.pipe_noise[0]*config.args.pipe_size):(i+1)*int(config.args.pipe_noise[0]*config.args.pipe_size)])
                    exp_AI_1_set = Subset(rem_exp_AI_1_data, exp_AI_1_subsetind)
                    exp_AI_1_set_truelabels = list(np.array(exp_AI_1_truelabels)[exp_AI_1_subsetind])
                    # check again
                    missing_fmow_class = list(set(fmow_classes) - set(exp_AI_1_set_truelabels))
                    if len(missing_fmow_class) == 0:
                        print('random_idx for fmow idx shuffling: ' + str(seed_i))
                    else:
                        seed_i += 1

                # AID
                aid_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                missing_aid_class = list(set(aid_classes) - set(exp_AI_2_set_truelabels))
                seed_i = 1
                while len(missing_aid_class) > 0:
                    #print('reshuffeling..')
                    # shuffle again
                    random.seed(seed_i)
                    random.shuffle(tot_exp_AI_2_ind)
                    tot_exp_AI_2_ind = np.array(tot_exp_AI_2_ind)
                    # assign again
                    exp_AI_2_subsetind = list(tot_exp_AI_2_ind[i*int(len(tot_exp_AI_2_ind)/no_batches):(i+1)*int(len(tot_exp_AI_2_ind)/no_batches)])
                    if config.args.dyn_batch_design == 'independent_recurring_genAI':
                        exp_AI_2_subsetind = list(tot_exp_AI_2_ind[i*int(config.args.pipe_noise[1]*config.args.pipe_size):(i+1)*int(config.args.pipe_noise[1]*config.args.pipe_size)])
                    exp_AI_2_set = Subset(rem_exp_AI_2_data, exp_AI_2_subsetind)
                    exp_AI_2_set_truelabels = list(np.array(exp_AI_2_truelabels)[exp_AI_2_subsetind])
                    # check again
                    missing_aid_class = list(set(aid_classes) - set(exp_AI_2_set_truelabels))
                    if len(missing_aid_class) == 0:
                        print('random_idx for aid idx shuffling: ' + str(seed_i))
                    else:
                        seed_i += 1

                # RESISC
                resisc_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                missing_resisc_class = list(set(resisc_classes) - set(exp_AI_3_set_truelabels))
                seed_i = 1
                while len(missing_resisc_class) > 0:

                    #print('reshuffeling..')
                    # shuffle again
                    random.seed(seed_i)
                    random.shuffle(tot_exp_AI_3_ind)
                    tot_exp_AI_3_ind = np.array(tot_exp_AI_3_ind)
                    # assign again
                    exp_AI_3_subsetind = list(tot_exp_AI_3_ind[i*int(len(tot_exp_AI_3_ind)/no_batches):(i+1)*int(len(tot_exp_AI_3_ind)/no_batches)])
                    if config.args.dyn_batch_design == 'independent_recurring_genAI':
                        exp_AI_3_subsetind = list(tot_exp_AI_3_ind[i*int(config.args.pipe_noise[2]*config.args.pipe_size):(i+1)*int(config.args.pipe_noise[2]*config.args.pipe_size)])
                    exp_AI_3_set = Subset(rem_exp_AI_3_data, exp_AI_3_subsetind)
                    exp_AI_3_set_truelabels = list(np.array(exp_AI_3_truelabels)[exp_AI_3_subsetind])
                    # check again
                    missing_resisc_class = list(set(resisc_classes) - set(exp_AI_3_set_truelabels))
                    if len(missing_resisc_class) == 0:
                        print('random_idx for resisc idx shuffling: ' + str(seed_i))
                    else:
                        seed_i += 1

                random.seed(config.args.main_seed)

            # collect indicator for train or validation data origin
            # train set: 0, val set: 1, gen AI set: -1
            exp_AI_1_set_type = [0 if i < len(exp_AI_1.dataset) else 1 for i in exp_AI_1_subsetind]
            exp_AI_2_set_type = [0 if i < len(exp_AI_2.dataset) else 1 for i in exp_AI_2_subsetind]
            exp_AI_3_set_type = [0 if i < len(exp_AI_3.dataset) else 1 for i in exp_AI_3_subsetind]


            # create outputs
            batch_set = ConcatDataset([gen_AI_set, exp_AI_1_set, exp_AI_2_set, exp_AI_3_set])
            batch_true_labels = gen_AI_set_truelabels + exp_AI_1_set_truelabels + exp_AI_2_set_truelabels + exp_AI_3_set_truelabels
            batch_set_types = gen_AI_set_type + exp_AI_1_set_type + exp_AI_2_set_type + exp_AI_3_set_type
            batch_loader = DataLoader(batch_set, num_workers=2, shuffle=False, batch_size=1)
            batches.append([batch_loader, batch_true_labels, batch_set_types])


        # make sure all expert classes are present in the first batch (HITL needs to allocate to experts in first batch)
        for idx, i in enumerate(range(len(batches))):
            if config.args.pipe_case == 'benchmark':
                if config.args.dyn_MVP_test:
                    pass
                else:
                    if idx == 0:
                        assert sum([True if label.endswith('-SVHN') else False for label in list(set(batches[i][1]))]) == 10
                        assert sum([True if label.endswith('-MNIST') else False for label in list(set(batches[i][1]))]) == 10
                        assert sum([True if label.endswith('-FMNIST') else False for label in list(set(batches[i][1]))]) == 10
            elif config.args.pipe_case == 'sat':
                if config.args.dyn_MVP_test:
                    pass
                else:
                    exp1_classes = ['amusement_park', 'archaeological_site', 'border_checkpoint', 'burial_site', 'construction_site', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'lighthouse', 'military_facility', 'nuclear_powerplant', 'oil_or_gas_facility', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'prison', 'race_track', 'runway', 'shopping_mall', 'smokestack', 'solar_farm', 'space_facility', 'surface_mine', 'swimming_pool', 'toll_booth', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
                    exp2_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Industrial', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Port', 'RailwayStation', 'Resort', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
                    exp3_classes = ['airplane', 'basketball_court', 'circular_farmland', 'cloud', 'island', 'mobile_home_park', 'palace', 'roundabout', 'sea_ice', 'ship', 'snowberg', 'tennis_court']
                    gen_AI_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River', 'SeaLake']
                    if idx == 0:
                        assert sum([True if label in exp1_classes else False for label in list(set(batches[i][1]))]) == 35
                        assert sum([True if label in exp2_classes else False for label in list(set(batches[i][1]))]) == 24
                        assert sum([True if label in exp3_classes else False for label in list(set(batches[i][1]))]) == 12



        print('Avg. batch size: ' + str(len(batches[0][0].dataset)))
        print('no. of batches: ' + str(no_batches))

        return batches
