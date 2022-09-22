# this code is intended to provide all arguments for the command line

import argparse


class Config:
    def __init__(self):

        # instantiation:
        # config = Config()
        # example_param = config.args.example_param

        self.parser = argparse.ArgumentParser()

        # some general settings
        self.parser.add_argument('-main_seed', default=24, type=int,
                                 help='seed to make torch, random and numpy reproducible')
        self.parser.add_argument('-bar_length', default=65, type=int,
                                 help='length of bar for training progress')
        self.parser.add_argument('-c', '--checkpoint', default='models', type=str, metavar='PATH',
                                 help='path to save checkpoint')
        self.parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                                 help='path to latest checkpoint (default: none)')

        ### BENCHMARK CASE ###

        # CIFAR10 model building
        self.parser.add_argument('-c10_wideresnet_train_batch', default=256, type=int,
                                 help='batch size of training data of CIFAR-10 model')
        self.parser.add_argument('-c10_test_batch', default=1, type=int,
                                 help='batch size of test data of CIFAR-10 model')
        self.parser.add_argument('-c10_epochs', default=200, type=int,
                                 help='epochs of CIFAR-10 model')
        self.parser.add_argument('-c10_sgd__lr', default=0.1, type=float,
                                 help='Learning rate of SGD optimizer of CIFAR-10 model')
        self.parser.add_argument('-c10_sgd__mom', default=0.9, type=float,
                                 help='Momentum of SGD optimizer of CIFAR-10 model')
        self.parser.add_argument('-c10_sgd__wd', default=1e-4, type=float,
                                 help='Weight decay of SGD optimizer of CIFAR-10 model')
        self.parser.add_argument('-c10_log_interval', default=195, type=int,
                                 help='Log interval of CIFAR-10 model')
        self.parser.add_argument('-c10_wideresnet_odin_eps_tpr95_iSUN', default=0.001600, type=float,
                                 help='ODIN eps tuned on iSUN for CIFAR-10 model')
        self.parser.add_argument('-c10_wideresnet_odin_delta_tpr95_iSUN', default=0.101029, type=float,
                                 help='ODIN delta tuned on iSUN for CIFAR-10 model')
        self.parser.add_argument('-c10_wideresnet_odin_eps_tpr95_pipe', default=0.001600, type=float,
                                 help='ODIN eps tuned on pipe data for CIFAR-10 model')
        self.parser.add_argument('-c10_wideresnet_odin_delta_tpr95_pipe', default=0.101029, type=float,
                                 help='ODIN delta tuned on pipe data for CIFAR-10 model')
        self.parser.add_argument('-c10_wideresnet_maha_eps_pipe_wFMNIST', default=0.000000, type=float,
                                 help='Maha eps tuned on pipe data for CIFAR-10 model')
        self.parser.add_argument('-c10_wideresnet_maha_delta_pipe_wFMNIST', default=-4.336433, type=float,
                                 help='Maha delta tuned on pipe data for CIFAR-10 model')

        # SVHN model building
        self.parser.add_argument('-svhn_wideresnet_train_batch', default=256, type=int,
                                 help='batch size of training data of SVHN model')
        self.parser.add_argument('-svhn_test_batch', default=1, type=int,
                                 help='batch size of test data of SVHN model')
        self.parser.add_argument('-svhn_epochs', default=100, type=int,
                                 help='epochs of SVHN model')
        self.parser.add_argument('-svhn_wideresnet_sgd__lr', default=0.1, type=float,
                                 help='Learning rate of SGD optimizer of SVHN model')
        self.parser.add_argument('-svhn_sgd__mom', default=0.9, type=float,
                                 help='Momentum of SGD optimizer of SVHN model')
        self.parser.add_argument('-svhn_wideresnet_sgd__wd', default=1e-4, type=float,
                                 help='Weight decay of SGD optimizer of SVHN model')
        self.parser.add_argument('-svhn_log_interval', default=195, type=int,
                                 help='Log interval of SVHN model')
        self.parser.add_argument('-svhn_wideresnet_odin_eps_tpr95_iSUN', default=0.003200, type=float,
                                 help='ODIN eps tuned on iSUN for SVHN model')
        self.parser.add_argument('-svhn_wideresnet_odin_delta_tpr95_iSUN', default=0.100626, type=float,
                                 help='ODIN delta tuned on iSUN for SVHN model')
        self.parser.add_argument('-svhn_wideresnet_odin_eps_tpr95_pipe', default=0.004000, type=float,
                                 help='ODIN eps tuned on pipe data for SVHN model')
        self.parser.add_argument('-svhn_wideresnet_odin_delta_tpr95_pipe', default=0.100648, type=float,
                                 help='ODIN delta tuned on pipe data for SVHN model')
        self.parser.add_argument('-svhn_wideresnet_maha_eps_pipe_wFMNIST', default=0.000000, type=float,
                                 help='Maha eps tuned on pipe data for SVHN model')
        self.parser.add_argument('-svhn_wideresnet_maha_delta_pipe_wFMNIST', default=-4.671067, type=float,
                                 help='Maha delta tuned on pipe data for SVHN model')

        # MNIST model building
        self.parser.add_argument('-mnist_wideresnet_train_batch', default=256, type=int,
                                 help='batch size of training data of MNIST model')
        self.parser.add_argument('-mnist_test_batch', default=1, type=int,
                                 help='batch size of test data of MNIST model')
        self.parser.add_argument('-mnist_epochs', default=20, type=int,
                                 help='epochs of MNIST model')
        self.parser.add_argument('-mnist_wideresnet_sgd__lr', default=0.1, type=float,
                                 help='Learning rate of SGD optimizer of MNIST model')
        self.parser.add_argument('-mnist_sgd__mom', default=0.9, type=float,
                                 help='Momentum of SGD optimizer of MNIST model')
        self.parser.add_argument('-mnist_wideresnet_sgd__wd', default=1e-4, type=float,
                                 help='Weight decay of SGD optimizer of MNIST model')
        self.parser.add_argument('-mnist_log_interval', default=195, type=int,
                                 help='Log interval of MNIST model')
        self.parser.add_argument('-mnist_wideresnet_odin_eps_tpr95_iSUN', default=0.000400, type=float,
                                 help='ODIN eps tuned on iSUN for MNIST model')
        self.parser.add_argument('-mnist_wideresnet_odin_delta_tpr95_iSUN', default=0.100999, type=float,
                                 help='ODIN delta tuned on iSUN for MNIST model')
        self.parser.add_argument('-mnist_wideresnet_odin_eps_tpr95_pipe', default=0.000400, type=float,
                                 help='ODIN eps tuned on pipe data for MNIST model')
        self.parser.add_argument('-mnist_wideresnet_odin_delta_tpr95_pipe', default=0.100999, type=float,
                                 help='ODIN delta tuned on pipe data for MNIST model')
        self.parser.add_argument('-mnist_wideresnet_maha_eps_pipe_wFMNIST', default=0.000000, type=float,
                                 help='Maha eps tuned on pipe data for MNIST model')
        self.parser.add_argument('-mnist_wideresnet_maha_delta_pipe_wFMNIST', default=11.694169, type=float,
                                 help='Maha delta tuned on pipe data for MNIST model')

        # FMNIST model building
        self.parser.add_argument('-fmnist_wideresnet_train_batch', default=128, type=int,
                                 help='batch size of training data of FashionMNIST model')
        self.parser.add_argument('-fmnist_test_batch', default=1, type=int,
                                 help='batch size of test data of FashionMNIST model')
        self.parser.add_argument('-fmnist_epochs', default=300, type=int,
                                 help='epochs of FashionMNIST model')
        self.parser.add_argument('-fmnist_sgd__lr', default=0.1, type=float,
                                 help='Learning rate of SGD optimizer of FashionMNIST model')
        self.parser.add_argument('-fmnist_sgd__mom', default=0.9, type=float,
                                 help='Momentum of SGD optimizer of FashionMNIST model')
        self.parser.add_argument('-fmnist_sgd__wd', default=5e-4, type=float,
                                 help='Weight decay of SGD optimizer of FashionMNIST model')
        self.parser.add_argument('-fmnist_log_interval', default=460, type=int,
                                 help='Log interval of FashionMNIST model')
        self.parser.add_argument('-fmnist_wideresnet_odin_eps_tpr95_iSUN', default=0.002800, type=float,
                                 help='ODIN eps tuned on iSUN for FashionMNIST model')
        self.parser.add_argument('-fmnist_wideresnet_odin_delta_tpr95_iSUN', default=0.100790, type=float,
                                 help='ODIN delta tuned on iSUN for FashionMNIST model')
        self.parser.add_argument('-fmnist_wideresnet_odin_eps_tpr95_pipe', default=0.004000, type=float,
                                 help='ODIN eps tuned on pipe data for FashionMNIST model')
        self.parser.add_argument('-fmnist_wideresnet_odin_delta_tpr95_pipe', default=0.100819, type=float,
                                 help='ODIN delta tuned on pipe data for FashionMNIST model')
        self.parser.add_argument('-fmnist_wideresnet_maha_eps_pipe', default=0.000000, type=float,
                                 help='Maha eps tuned on pipe data for FashionMNIST model')
        self.parser.add_argument('-fmnist_wideresnet_maha_delta_pipe', default=25.779577, type=float,
                                 help='Maha delta tuned on pipe data for FashionMNIST model')

        # Gating model building
        self.parser.add_argument('-gating_train_batch', default=256, type=int,
                                 help='Batch size of training data of gating model')
        self.parser.add_argument('-gating_epochs', default=3, type=int,
                                 help='Epochs of gating model')
        self.parser.add_argument('-gating_adam__lr', default=1e-4, type=float,
                                 help='Learning rate of Adam optimizer of gating model')
        self.parser.add_argument('-gating_adam__wd', default=0, type=float,
                                 help='Weight decay of Adam optimizer of gating model')
        self.parser.add_argument('-gating_log_interval', default=390, type=int,
                                 help='Log interval of gating model')

        # Global model building
        self.parser.add_argument('-global_train_batch', default=256, type=int,
                                 help='Batch size of training data of global model')
        self.parser.add_argument('-global_epochs', default=300, type=int,
                                 help='Epochs of global model')
        self.parser.add_argument('-global_sgd__lr', default=0.1, type=float,
                                 help='Learning rate of SGD optimizer of global model')
        self.parser.add_argument('-global_sgd__wd', default=0.9, type=float,
                                 help='Weight decay of SGD optimizer of global model')
        self.parser.add_argument('-global_sgd__mom', default=1e-4, type=float,
                                 help='Momentum of SGD optimizer of global model')
        self.parser.add_argument('-global_log_interval', default=390, type=int,
                                 help='Log interval of global model')

        ### SATELLITE CASE ###

        # EuroSAT_countryside model building
        self.parser.add_argument('-eurosat_train_batch', default=64, type=int,
                                 help='batch size of training data of EuroSat model')
        self.parser.add_argument('-eurosat_test_batch', default=1, type=int,
                                 help='batch size of test data of EuroSat model')
        self.parser.add_argument('-eurosat_epochs', default=20, type=int,
                                 help='Epochs of EuroSat model')
        self.parser.add_argument('-eurosat_sgd__lr', default=0.001, type=float,
                                 help='Learning rate of SGD optimizer of EuroSat model')
        self.parser.add_argument('-eurosat_sgd__mom', default=0.9, type=float,
                                 help='Momentum of SGD optimizer of EuroSat model')
        self.parser.add_argument('-eurosat_sgd__wd', default=1e-4, type=float,
                                 help='Weight decay of SGD optimizer of EuroSat model')
        self.parser.add_argument('-eurosat_log_interval', default=118, type=int,
                                 help='log interval of EuroSat model')
        self.parser.add_argument('-eurosat_feat_ex', default=False, type=bool,
                                 help='Feature extracting activation of EuroSat model')
        self.parser.add_argument('-eurosat_odin_delta_tpr95', default=0.144081, type=float,
                                 help='ODIN delta tuned on pipe data of EuroSat model')
        self.parser.add_argument('-eurosat_odin_eps_tpr95', default=0.002200, type=float,
                                 help='ODIN eps tuned on pipe data of EuroSat model')
        self.parser.add_argument('-eurosat_odin_delta_UCM', default=0.143901, type=float,
                                 help='ODIN delta tuned on UCM of EuroSat model')
        self.parser.add_argument('-eurosat_odin_eps_UCM', default=0.000800, type=float,
                                 help='ODIN eps tuned on UCM of EuroSat model')
        self.parser.add_argument('-eurosat_maha_delta', default=15.447654, type=float,
                                 help='Maha delta tuned on pipe data of EuroSat model')
        self.parser.add_argument('-eurosat_maha_eps', default=0.00000, type=float,
                                 help='Maha eps tuned on pipe data of EuroSat model')
        
        # FMOW model building
        self.parser.add_argument('-fmow_train_batch', default=64, type=int,
                                 help='batch size of training data of fmow model')
        self.parser.add_argument('-fmow_test_batch', default=1, type=int,
                                 help='batch size of test data of fmow model')
        self.parser.add_argument('-fmow_epochs', default=50, type=int,
                                 help='Epochs of fmow model')
        self.parser.add_argument('-fmow_adam__lr', default=1e-4, type=float,
                                 help='Learning rate of Adam optimizer of fmow model')
        self.parser.add_argument('-fmow_adam__wd', default=0, type=float,
                                 help='Weight decay of Adam optimizer of fmow model')
        self.parser.add_argument('-fmow_log_interval', default=148, type=int,
                                 help='log interval of fmow model')
        self.parser.add_argument('-fmow_feat_ex', default=False, type=bool,
                                 help='Feature extracting activation of fmow model')
        self.parser.add_argument('-fmow_odin_delta_tpr95', default=0.028814, type=float,
                                 help='ODIN delta tuned on pipe data of fmow model')
        self.parser.add_argument('-fmow_odin_eps_tpr95', default=0.000000, type=float,
                                 help='ODIN eps tuned on pipe data of fmow model')
        self.parser.add_argument('-fmow_odin_delta_UCM', default=0.028784, type=float,
                                 help='ODIN delta tuned on UCM of fmow model')
        self.parser.add_argument('-fmow_odin_eps_UCM', default=0.000000, type=float,
                                 help='ODIN eps tuned on UCM of fmow model')
        self.parser.add_argument('-fmow_maha_delta', default=4.086840, type=float,
                                 help='Maha delta tuned on pipe data of fmow model')
        self.parser.add_argument('-fmow_maha_eps', default=0.000000, type=float,
                                 help='Maha eps tuned on pipe data of fmow model')

        # AID model building
        self.parser.add_argument('-aid_train_batch', default=64, type=int,
                                 help='batch size of training data of aid model')
        self.parser.add_argument('-aid_test_batch', default=1, type=int,
                                 help='batch size of test data of aid model')
        self.parser.add_argument('-aid_epochs', default=50, type=int,
                                 help='Epochs of aid model')
        self.parser.add_argument('-aid_adam__lr', default=1e-4, type=float,
                                 help='Learning rate of Adam optimizer of aid model')
        self.parser.add_argument('-aid_adam__wd', default=0, type=float,
                                 help='Weight decay of Adam optimizer of aid model')
        self.parser.add_argument('-aid_log_interval', default=148, type=int,
                                 help='log interval of aid model')
        self.parser.add_argument('-aid_feat_ex', default=False, type=bool,
                                 help='Feature extracting activation of aid model')
        self.parser.add_argument('-aid_odin_delta_tpr95', default=0.042017, type=float,
                                 help='ODIN delta tuned on pipe data of aid model')
        self.parser.add_argument('-aid_odin_eps_tpr95', default=0.000000, type=float,
                                 help='ODIN eps tuned on pipe data of aid model')
        self.parser.add_argument('-aid_odin_delta_UCM', default=0.042017, type=float,
                                 help='ODIN delta tuned on UCM of aid model')
        self.parser.add_argument('-aid_odin_eps_UCM', default=0.000000, type=float,
                                 help='ODIN eps tuned on UCM of aid model')
        self.parser.add_argument('-aid_maha_delta', default=17.010269, type=float,
                                 help='Maha delta tuned on pipe data of aid model')
        self.parser.add_argument('-aid_maha_eps', default=0.010000, type=float,
                                 help='Maha eps tuned on pipe data of aid model')

        # RESISC model building
        self.parser.add_argument('-resisc_train_batch', default=64, type=int,
                                 help='batch size of training data of resisc model')
        self.parser.add_argument('-resisc_test_batch', default=1, type=int,
                                 help='batch size of test data of resisc model')
        self.parser.add_argument('-resisc_epochs', default=50, type=int,
                                 help='Epochs of resisc model')
        self.parser.add_argument('-resisc_adam__lr', default=1e-4, type=float,
                                 help='Learning rate of Adam optimizer of resisc model')
        self.parser.add_argument('-resisc_adam__wd', default=0, type=float,
                                 help='Weight decay of Adam optimizer of resisc model')
        self.parser.add_argument('-resisc_log_interval', default=148, type=int,
                                 help='log interval of resisc model')
        self.parser.add_argument('-resisc_feat_ex', default=False, type=bool,
                                 help='Feature extracting activation of resisc model')
        self.parser.add_argument('-resisc_odin_delta_tpr95', default=0.084009, type=float,
                                 help='ODIN delta tuned on pipe data of resisc model')
        self.parser.add_argument('-resisc_odin_eps_tpr95', default=0.000000, type=float,
                                 help='ODIN eps tuned on pipe data of resisc model')
        self.parser.add_argument('-resisc_odin_delta_UCM', default=0.084009, type=float,
                                 help='ODIN delta tuned on UCM of resisc model')
        self.parser.add_argument('-resisc_odin_eps_UCM', default=0.000000, type=float,
                                 help='ODIN eps tuned on UCM of resisc model')
        self.parser.add_argument('-resisc_maha_delta', default=39.07439, type=float,
                                 help='Maha delta tuned on pipe data of resisc model')
        self.parser.add_argument('-resisc_maha_eps', default=0.000000, type=float,
                                 help='Maha eps tuned on pipe data of resisc model')

        # satgating
        # Gating model building
        self.parser.add_argument('-sat_gating_train_batch', default=32, type=int,
                                 help='Batch size of training data of sat gating model')
        self.parser.add_argument('-sat_gating_epochs', default=2, type=int,
                                 help='Epochs of sat gating model')
        self.parser.add_argument('-sat_gating_adam__lr', default=1e-4, type=float,
                                 help='Learning rate of Adam optimizer of sat gating model')
        self.parser.add_argument('-sat_gating_adam__wd', default=0, type=float,
                                 help='Weight decay of Adam optimizer of sat gating model')
        self.parser.add_argument('-sat_gating_log_interval', default=250, type=int,
                                 help='Log interval of sat gating model')

        # Global model building
        self.parser.add_argument('-sat_global_train_batch', default=64, type=int,
                                 help='Batch size of training data of sat global model')
        self.parser.add_argument('-sat_global_epochs', default=50, type=int,
                                 help='Epochs of sat global model')
        self.parser.add_argument('-sat_global_adam__lr', default=1e-4, type=float,
                                 help='Learning rate of  optimizer of sat global model')
        self.parser.add_argument('-sat_global_adam__mom', default=0, type=float,
                                 help='Momentum of SGD optimizer of sat global model')
        self.parser.add_argument('-sat_global_log_interval', default=250, type=int,
                                 help='Log interval of sat global model')
        self.parser.add_argument('-sat_global_feat_ex', default=False, type=bool,
                                 help='Feature extracting activation of sat global model')

        # pipeline parameters
        self.parser.add_argument('-pipe_root', default='../ai-in-the-loop/', type=str,
                                 help='root directory of project')
        self.parser.add_argument('-pipe_case', required=True, default='benchmark', type=str, choices=['benchmark', 'sat'],
                                 help='experiment type')
        self.parser.add_argument('-pipe_tune_dataset', default='pipe', type=str, choices=['iSUN', 'pipe', 'UCM'],
                                 help='tuning dataset for allocation mechanisms')
        self.parser.add_argument('--domain', required=True, default='multi', type=str, choices=['multi', 'single'],
                                 help='domain of system')
        self.parser.add_argument('-pipe_type', default='static', required=True, type=str, choices=['static', 'dynamic'],
                                 help='System type')
        self.parser.add_argument('-no_seeds', default=1, type=int,
                                 help='Number of seeds desired for experiments')
        self.parser.add_argument('-mlflow_path', default='ai-in-the-loop/mlruns/', type=str,
                                 help='Path of saving mlflow experiments')
        self.parser.add_argument('-train_early_stop', default=True, type=bool,
                                 help='activation of early stopping for training')
        self.parser.add_argument('-pipe_size', default=0, type=int,
                                 help='Number of general ML model data samples.')
        self.parser.add_argument('-pipe_noise', nargs='+', default=[0, 0, 0], type=float,
                                 help='share of unknown data p. artificial expert as multiplicator of size')
        self.parser.add_argument('-pipe_tune_thresholds', action='store_true',
                                 help='Activate tuning of allocation mechanisms')
        self.parser.add_argument('-pipe_HITL_same_effort', default=False, type=bool,
                                 help='Include HITL with same effort as AIITL-system baselin')
        self.parser.add_argument('-global_model_HITL_same_effort', default=False, type=bool,
                                 help='Include baseline of global model with HITL of same effort as AIITL system')
        self.parser.add_argument('-pipe_odin_tpr', default=0.95, type=float, choices=[0.95],
                                 help='desired ODIN tpr while tuning')
        self.parser.add_argument('-pipe_maha_tpr', default=0.95, type=float, choices=[0.95],
                                 help='desired Maha tpr while tuning')
        self.parser.add_argument('-exp_name', default='', type=str,
                                 help='name of mlflow experiment')
        self.parser.add_argument('-results_repro', action='store_true',
                                 help='Activate reproduction of results')
        self.parser.add_argument('-ablation_study', default='', type=str, choices=['', 'smaller_OOD_share', 'no_trust_early_stop', 'resnet18_backbone', 'trust_thresholds', 'unc_vs_OOD_share', 'fmow_overfit', 'smaller_OOD_share_v2'],
                                 help="Ablation study. '' for default results")
        self.parser.add_argument('-smaller_OOD_share', default=0.5, type=float,
                                 help="smaller OOD share considered in ablation study 'smaller_OOD_share'")
        self.parser.add_argument('-smaller_OOD_share_v2_batch_size', default=1000, type=float,
                                 help="batch size considered in ablation study 'smaller_OOD_share_v2'")

        # Uncertainty mechanism
        self.parser.add_argument('-unc_mech', default='S', type=str, choices=['S'],
                                 help='S: Softmax')
        self.parser.add_argument('-softmax_dim', default=1, type=int,
                                 help='Dimension of Softmax')

        # Uncertainty score
        self.parser.add_argument('-unc_score', default='conf', type=str,
                                 choices=['conf'],
                                 help='Uncertainty score metric')
        self.parser.add_argument('-conf_thresh_benchmark_wideresnet_pipe_wFMNIST', default=0.992, type=float,
                                 help='Softmax threshold of CIFAR-10 model in multi-domain system')
        self.parser.add_argument('-conf_thresh_sat', default=0.92, type=float,
                                 help='Softmax threshold of EuroSat model in multi-domain system')
        self.parser.add_argument('-conf_thresh_singledom', default=0, type=float,
                                 help='Softmax threshold of CIFAR-10 model in single-domain system')
        self.parser.add_argument('-conf_thresh_singledom_exp', default=0, type=float,
                                 help='Softmax threshold of artificial expert model in single-domain system')
        self.parser.add_argument('-conf_thresh_singledom_sat', default=0, type=float,
                                 help='Softmax threshold of Eurosat model in single-domain system')

        # instance selection mechanism
        self.parser.add_argument('-selec_mech', default='unc_thresh', type=str,
                                 choices=['unc_thresh', 'unc_bottom_x', 'odin', 'odin_ext', 'gating', 'maha'],
                                 help='Allocation mechanism 1 (also referred to as selection mechanism)')
        self.parser.add_argument('-odin_temper', default=1000, type=int,
                                 help='Temperature scaling of OIDN')
        self.parser.add_argument('-bottom_x', default=1, type=int,
                                 help='Bottom x number of samples considered')

        # Expert combination mechanism
        self.parser.add_argument('-comb_mech', default='min_uncertain', type=str,
                                 choices=['min_uncertain', 'odin', 'odin_ext', 'gating', 'maha'],
                                 help='Allocation mechanism 2 (also referred to as combination mechanism)')

        # dynamic pipeline
        self.parser.add_argument('-dyn_pipe', default=False, type=bool,
                                 help='Activate environment of dynamic AIITL-system')
        self.parser.add_argument('-dyn_num_experts', default=3, type=int,
                                 help='Number of experts desired in dynamic AIITL-system')
        self.parser.add_argument('-dyn_trust_thresh', default=0.95, type=float,
                                 help='Trust threshold of artificial experts in dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_train_batch', default=128, type=int,
                                 help='Training batch size in dynamic AIITL-system of benchmark case')
        self.parser.add_argument('-dyn_multi_train_epochs', default=300, type=int,
                                 help='Training epochs in dynamic AIITL-system of benchmark case')
        self.parser.add_argument('-dyn_multi_sgd_lr', default=0.1, type=float,
                                 help='Learning rate of SGD optimizer in dynamic AIITL-system of benchmark case')
        self.parser.add_argument('-dyn_multi_sgd_mom', default=0.9, type=float,
                                 help='Momentum of SGD optimizer in dynamic AIITL-system of benchmark case')
        self.parser.add_argument('-dyn_multi_sgd_wd', default=1e-4, type=float,
                                 help='Weight decay of SGD optimizer in dynamic AIITL-system of benchmark case')
        self.parser.add_argument('-dyn_multi_train_batch_sat', default=64, type=int,
                                 help='Training batch size in dynamic AIITL-system of sat case')
        self.parser.add_argument('-dyn_multi_train_epochs_sat', default=50, type=int,
                                 help='Training epochs in dynamic AIITL-system of sat case')
        self.parser.add_argument('-dyn_multi_adam_lr_sat', default=1e-4, type=float,
                                 help='Learning rate of Adam optimizer in dynamic AIITL-system of sat case')
        self.parser.add_argument('-dyn_multi_adam_wd_sat', default=0, type=float,
                                 help='Weight decay of Adam optimizer in dynamic AIITL-system of sat case')
        self.parser.add_argument('-dyn_log_interval', default=150, type=int,
                                 help='Log interval in dynamic AIITL-system')
        self.parser.add_argument('-dyn_checkpoint', default='models', type=str,
                                 help='Directory of checkpoint in dynamic AIITL-system')
        self.parser.add_argument('-dyn_resume', default='', type=str,
                                 help='File path of resume storage in dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_resume_expAI1', default=False, type=bool,
                                 help='Activate resums of training of the first artificial expert in dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_resume_expAI2', default=False, type=bool,
                                 help='Activate resums of training of the second artificial expert in dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_resume_expAI3', default=False, type=bool,
                                 help='Activate resums of training of the third artificial expert in dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_comb_mech', default='odin', type=str, choices=['odin', 'maha', 'gating', 'unc_thresh'],
                                 help='Allocation mechanism 2 throughout dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_selec_mech', default='odin', type=str, choices=['odin', 'maha', 'gating'],
                                 help='Allocation mechanism 1 throughout dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_tune_dataset', default='iSUN', type=str, choices=['pipe', 'iSUN', 'UCM'],
                                 help='Tuning dataset throughout dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_tune_pot', default='when_trusted', type=str, choices=['when_trusted', 'when_trusted_allforallo', 'when_all_trusted', 'when_min_valset_size'],
                                 help='Tuning point of time in dynamic AIITL-system')
        self.parser.add_argument('-dyn_trust_max_delta', default=0.01, type=float,
                                 help='Maximum delta for plateau of early stopping of trusting artificial experts in dynamic AIITL-system')
        self.parser.add_argument('-dyn_trust_patience', default=3, type=int,
                                 help='Patience (number of batches) for plateau of early stopping of trusting artificial experts in dynamic AIITL-system')
        self.parser.add_argument('-dyn_valloss_max_delta', default=0.02, type=float,
                                 help='Maximum delta for plateau of early stopping of training artificial experts in dynamic AIITL-system')
        self.parser.add_argument('-dyn_valloss_patience', default=20, type=int,
                                 help='Patience (number of epochs) for plateau of early stopping of training artificial experts in dynamic AIITL-system')
        self.parser.add_argument('-dyn_multi_split', default='train_val', type=str, choices=['random', 'train_val'],
                                 help="Split of data in dynamic AIITL-system. 'Random' refers to random split, 'train_val' refers to original split")
        self.parser.add_argument('-dyn_multi_transforms', default='adapted', type=str, choices=['original', 'adapted'],
                                 help="Type of preprocessing in dynamic AIITL-system. 'adapted' refers to more realistic scenario with consistent pre-processing steps for all unknown data")
        self.parser.add_argument('-dyn_MVP_test', action='store_true',
                                 help='Start MVP testing')
        self.parser.add_argument('-gating_trust_thresh', default=0.95, type=float,
                                 help='Trust thresh of gating model in dynamic AIITL-system. The gating model is designed with a trust thresh to ensure a certain quality, while maha/odin are retrained and improved every time a new artificial expert is added')
        self.parser.add_argument('-dyn_multi_finalbatch', default=True, type=bool,
                                 help='Include the final test batch in dynamic, multi-domain AIITL-system')
        self.parser.add_argument('-dyn_no_batches', default=30, type=int,
                                 help='Number of batches in dynamic, multi-domain AIITL-system')
        self.parser.add_argument('-independent_recurring_genAI_dyn_no_batches', default=90, type=int,
                                 help='Number of batches when recurrent general ML model samples are needed in dynamic AIITL-system')
        self.parser.add_argument('-dyn_single_no_batches', default=30, type=int,
                                 help='Number of baches in dynamic, single-domain AIITL-system')
        self.parser.add_argument('-dyn_single_no_strong_classes', default=6, type=int,
                                 help='Number of known classes in dynamic, single-domain AIITL-system of benchmark case')
        self.parser.add_argument('-dyn_single_no_strong_classes_sat', default=4, type=int,
                                 help='Number of known classes in dynamic, single-domain AIITL-system of sat case')
        self.parser.add_argument('-dyn_batch_design', default='dependent', type=str, choices=['dependent', 'independent', 'independent_recurring_genAI'],
                                 help='choose whether the final static pipeline batch should be included (dependent) or not (independent) or recurrent general ML samples are needed (independent_recurring_genAI).')


        self.parser.add_argument('-exp_AI_1_wideresnet_odin_delta_tpr95_pipe', default=0, type=float,
                                 help='ODIN delta tuned on pipe data of artificial expert 1 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_1_wideresnet_odin_eps_tpr95_pipe', default=0, type=float,
                                 help='ODIN eps tuned on pipe data of artificial expert 1 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_2_wideresnet_odin_delta_tpr95_pipe', default=0, type=float,
                                 help='ODIN delta tuned on pipe data of artificial expert 2 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_2_wideresnet_odin_eps_tpr95_pipe', default=0, type=float,
                                 help='ODIN eps tuned on pipe data of artificial expert 2 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_3_wideresnet_odin_delta_tpr95_pipe', default=0, type=float,
                                 help='ODIN delta tuned on pipe data of artificial expert 3 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_3_wideresnet_odin_eps_tpr95_pipe', default=0, type=float,
                                 help='ODIN eps tuned on pipe data of artificial expert 3 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_1_wideresnet_odin_delta_tpr95_ext', default=0, type=float,
                                 help='ODIN delta tuned on ext data of artificial expert 1 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_1_wideresnet_odin_eps_tpr95_ext', default=0, type=float,
                                 help='ODIN eps tuned on ext data of artificial expert 1 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_2_wideresnet_odin_delta_tpr95_ext', default=0, type=float,
                                 help='ODIN delta tuned on ext data of artificial expert 2 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_2_wideresnet_odin_eps_tpr95_ext', default=0, type=float,
                                 help='ODIN eps tuned on ext data of artificial expert 2 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_3_wideresnet_odin_delta_tpr95_ext', default=0, type=float,
                                 help='ODIN delta tuned on ext data of artificial expert 3 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_3_wideresnet_odin_eps_tpr95_ext', default=0, type=float,
                                 help='ODIN eps tuned on ext data of artificial expert 3 in dynamic AIITL-system')

        self.parser.add_argument('-exp_AI_1_wideresnet_maha_delta_tpr95_pipe', default=0, type=float,
                                 help='Maha delta of artificial expert 1 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_1_wideresnet_maha_eps_tpr95_pipe', default=0, type=float,
                                 help='Maha eps of artificial expert 1 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_2_wideresnet_maha_delta_tpr95_pipe', default=0, type=float,
                                 help='Maha delta of artificial expert 2 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_2_wideresnet_maha_eps_tpr95_pipe', default=0, type=float,
                                 help='Maha eps of artificial expert 2 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_3_wideresnet_maha_delta_tpr95_pipe', default=0, type=float,
                                 help='Maha delta of artificial expert 3 in dynamic AIITL-system')
        self.parser.add_argument('-exp_AI_3_wideresnet_maha_eps_tpr95_pipe', default=0, type=float,
                                 help='Maha eps of artificial expert 3 in dynamic AIITL-system')

        # plots
        self.parser.add_argument('-softmax_thresholds', action='store_true',
                                 help='Plot softmax thresholds')
        self.parser.add_argument('-utility_graph', action='store_true',
                                 help='Plot utility graph')
        self.parser.add_argument('-acc_cov_table', action='store_true',
                                 help='Create acc_cov table')
        self.parser.add_argument('-exp_learning_graph', action='store_true',
                                 help='Plot expert related metrics')
        self.parser.add_argument('-allocation_matrix', action='store_true',
                                 help='Create allocation matrix')
        self.parser.add_argument('-selec_comb_mechs', action='store_true',
                                 help='Create overview of all allocation mechanism combinations')
        self.parser.add_argument('-selec_comb_mechs_baselines', action='store_true',
                                 help='Create overview of baselines of all allocation mechanism combinations')
        self.parser.add_argument('-allocation_system', action='store_true',
                                 help='Create overview of allocation system')
        self.parser.add_argument('--saliency', action='store_true',
                                 help='Create saliency maps of selected images')
        self.parser.add_argument('-datasets', action='store_true',
                                 help='Create overview of one image per class for datasets of specific case')
        self.parser.add_argument('-sub_case', default='', type=str,
                                 help="Sub case considered in plots. In the static system, 'allo1/allo2' is needed, in dynamic only 'allo' is needed")
        self.parser.add_argument('-utility_a', default=0, type=float,
                                 help='Alpha of utility')
        self.parser.add_argument('-utility_b', default=0, type=float,
                                 help='beta of utility')
        self.parser.add_argument('-incl_finalbatch', default=True, type=bool,
                                 help='Include final batch in plot')
        self.parser.add_argument('-no_finalbatch', action='store_true',
                                 help='Exclude final batch in plot')
        self.parser.add_argument('-batch_no', default=0, type=int,
                                 help='Batch no considered, e.g., for allocation matrix')
        self.parser.add_argument('-fig_size', nargs='+', default=[8, 4], type=float,
                                 help='Size of figure. Array of x and y shape')
        self.parser.add_argument('-fmow_overfit_type', type=str, choices=['acc', 'loss'],
                                 help='Plot type of fmow overfit')
        self.parser.add_argument('-smaller_OOD_plot', type=str, choices=['odin', 'system'],
                                 help='Alpha of utility')
        self.parser.add_argument('-mlruns_ID', default=0, type=int,
                                 help='mlruns ID considered')


        self.args = self.parser.parse_args()

        # data set keys
        self.BENCHMARK_DATA_SETS = {
            '0': 'CIFAR10',
            '1': 'SVHN',
            '2': 'MNIST',
            '3': 'FMNIST'
        }

        self.SAT_DATA_SETS = {
            '0': 'Euro_SAT_countryside',
            '1': 'FMOW',
            '2': 'AID',
            '3': 'RESISC',
        }

# create project-wide config instance
cfg = Config()