# this code is intended to run entire results, also for ablation studies

# all libraries
import all_pipes_dyn, all_pipes_stat
import src.utils.pipe as pipe
import src.config
import random
import numpy as np
import mlflow
import src.utils.model_building as mb
import src.data.make_data as md
import src.utils.utils as utils
from torch.utils.data import Subset, ConcatDataset, DataLoader
import os


def main():

    # get config
    config = src.config.cfg

    # initialize setting
    case_domain_pipe = config.args.pipe_case + '_' + config.args.domain + '_' + config.args.pipe_type
    case_domain = config.args.pipe_case + '_' + config.args.domain + '_'

    # specify config params for ablation studies if needed
    if config.args.ablation_study != '':
        print('WARNING: You are running the {} ablation study'.format(config.args.ablation_study))
        if config.args.ablation_study == 'smaller_OOD_share':
            config.args.dyn_batch_design = 'independent_recurring_genAI'
            config.args.dyn_trust_patience = int(0.1 * config.args.independent_recurring_genAI_dyn_no_batches)
        elif config.args.ablation_study == 'no_trust_early_stop':
            # the implementation is handled inside pipe.py
            pass
        elif config.args.ablation_study == 'resnet18_backbone':
            # the implementation is handled inside model_building.py
            # set respective initial ODIN_ext params (overwrite wideresnet params)
            config.args.c10_wideresnet_odin_eps_tpr95_iSUN = 0.004000
            config.args.c10_wideresnet_odin_delta_tpr95_iSUN = 0.100949
        elif config.args.ablation_study == 'trust_thresholds':
            # the implementation is handled inside all_pipes_dyn.py
            pass
        elif config.args.ablation_study == 'unc_vs_OOD_share':
            # code is below, only conducted for multi-domain, static pipeline in benchmark case
            assert config.args.pipe_case == 'benchmark'
            assert config.args.pipe_type == 'static'
            assert config.args.domain == 'multi'
            pass
        elif config.args.ablation_study == 'fmow_overfit':
            # code is below, only conducted for the following settings
            assert config.args.pipe_case == 'sat'
            assert config.args.pipe_type == 'static'
            assert config.args.domain == 'multi'
            pass


    # specify mlruns location
    config.args.mlflow_path = 'ai-in-the-loop/results/' + case_domain_pipe + '/mlruns/'
    mlflow.set_tracking_uri('file:///home/dawe/ai-in-the-loop/results/' + case_domain_pipe + '/mlruns')


    # draw and save seeds and run scripts for specific number of times indicated by args.no_seeds (default 1)
    if not config.args.results_repro:

        if config.args.pipe_type == 'static':

            if config.args.ablation_study != 'unc_vs_OOD_share' and config.args.ablation_study != 'fmow_overfit':

                for i in range(config.args.no_seeds):

                    # read in already used seeds for specific case_domain_pipe setting
                    seed_log_file = '../ai-in-the-loop/results/{}/overview/{}_seeds.log'.format(case_domain_pipe, case_domain_pipe)
                    if os.path.isfile(seed_log_file):
                        with open(seed_log_file) as f:
                            seeds = [int(x) for x in f]
                    else:
                        seeds = []

                    # if no ablation study: draw seed if static pipeline, in multi use seeds drawn for static pipeline
                    if config.args.ablation_study == '':
                        seed = int(random.randint(1, 999999))
                        while seed in seeds:
                            # repeat draw until new seed is drawn
                            seed = int(random.randint(1, 999999))

                        seeds.append(seed)
                    else:
                        assert len(seeds) != 0, 'ERROR: Run standard results before ablation study!'
                        seed = seeds[i]

                    # print seed
                    print('')
                    print('######################')
                    if config.args.ablation_study == '':
                        print('seed no. ' + str(len(seeds)) + ': ' + str(seed))
                    else:
                        print('seed no. ' + str(i+1) + ': ' + str(seed))
                    print('######################')
                    print('')
                    print('')
                    print('')
                    print('')

                    # save all used seeds
                    if config.args.ablation_study == '':
                        with open(seed_log_file, 'w') as f:
                            for seed in seeds:
                                f.write(str(seed))
                                f.write('\n')

                    # run script for respective seed
                    config.args.main_seed = seed
                    if config.args.pipe_type == 'static':
                        all_pipes_stat.main()
                    elif config.args.pipe_type == 'dynamic':
                        all_pipes_dyn.main()

            # run study about impact of unknown data on uncertainty
            elif config.args.ablation_study == 'unc_vs_OOD_share':

                # set experiment
                if config.args.exp_name == '':
                    exp_name = 'run_' + config.args.domain + '_static_pipeline_' + config.args.pipe_case + '_OOD_share_analysis'
                else:
                    exp_name = config.args.exp_name
                mlflow.set_experiment(experiment_name=exp_name)
                exp = mlflow.get_experiment_by_name(exp_name)
                utils.adjust_mlflow_yaml(exp.experiment_id)
                utils.save_mlflow_exp_id(exp.experiment_id)
                print('Exp. ID: ' + str(exp.experiment_id))
                mlflow.start_run(run_name='OOD_share_analysis')

                # run analysis for first static seed only
                # collect static seeds
                static_seed_log_file = '../ai-in-the-loop/results/{}static/overview/{}static_seeds.log'.format(case_domain, case_domain)
                if os.path.isfile(static_seed_log_file):
                    with open(static_seed_log_file) as f:
                        static_seeds = [int(x) for x in f]
                else:
                    print('ERROR: Dynamic pipeline not possible, as no static seeds are run, yet! Please run static pipelines first.')
                    raise NotImplementedError

                # create batch of 1000k CIFAR-10 + SVHN data samples and compute uncertainty with increasing SVHN (OOD) share
                gen_AI = mb.get_general_model()
                _, _, cifar10 = md.get_data('0')
                _, _, svhn = md.get_data('1')
                _, _, mnist = md.get_data('2')
                _, _, fmnist = md.get_data('3')
                _, _, cifar100 = md.get_data('4')
                config.args.main_seed = static_seeds[0]
                print('')
                print('######################')
                print('seed: ' + str(static_seeds[0]))
                print('######################')
                print('')
                print('')
                print('')
                print('')
                cifar10 = Subset(cifar10.dataset, random.sample(range(len(cifar10.dataset)), 1000))
                svhn = Subset(svhn.dataset, random.sample(range(len(svhn.dataset)), 1000))
                mnist = Subset(mnist.dataset, random.sample(range(len(mnist.dataset)), 1000))
                fmnist = Subset(fmnist.dataset, random.sample(range(len(fmnist.dataset)), 1000))
                cifar100 = Subset(cifar100.dataset, random.sample(range(len(cifar100.dataset)), 1000))

                confidences = []

                # loop over unknown datasets
                for j in range(4):
                    if j == 0:
                        dataset = svhn
                    elif j == 1:
                        dataset = mnist
                    elif j == 2:
                        dataset = fmnist
                    elif j == 3:
                        dataset = cifar100

                    # loop over different shares of unknown data
                    for i in range(101):
                        print('OOD share: {}/{}'.format(i/100, 100))
                        config.args.main_seed = static_seeds[0]
                        # i indicates the OOD share
                        ood_share = round(i/100, 2)
                        id_share = round(1-ood_share, 2)
                        cifar10_index = list(range(len(cifar10)))[:int(id_share*len(cifar10))]
                        dataset_index = list(range(len(dataset)))[:int(ood_share*len(dataset))]
                        print('len cifar10: ' + str(len(cifar10_index)))
                        print('len out dataset: ' + str(len(dataset_index)))
                        batch_dataset = ConcatDataset([Subset(cifar10, cifar10_index), Subset(dataset, dataset_index)])
                        assert len(batch_dataset) == 1000

                        batch_loader = DataLoader(batch_dataset, num_workers=2, batch_size=1, shuffle=True)

                        # compute Softmax confidence
                        confidences.append(np.mean(pipe.get_unc_score(pipe.get_Softmax_prob(gen_AI, batch_loader), 'conf')))
                        print('avg. batch confidence: ' + str(confidences[-1]))
                        mlflow.log_metric(str(str(j) + '_ood_share_' + str(ood_share)), confidences[-1])

                mlflow.end_run()


            # run study about overfitting of the FMOW model
            elif config.args.ablation_study == 'fmow_overfit':

                # some checks
                assert config.args.pipe_case == 'sat'
                assert config.args.pipe_type == 'static'
                assert config.args.domain == 'multi'

                # experiment name
                if config.args.exp_name == '':
                    exp_name = 'run_' + config.args.domain + '_static_pipeline_' + config.args.pipe_case + '_fMoW_Overfit'
                else:
                    exp_name = config.args.exp_name
                mlflow.set_experiment(experiment_name=exp_name)
                exp = mlflow.get_experiment_by_name(exp_name)
                utils.adjust_mlflow_yaml(exp.experiment_id)
                utils.save_mlflow_exp_id(exp.experiment_id)
                print('Exp. ID: ' + str(exp.experiment_id))

                # run analysis for first static seed only
                # collect static seeds
                static_seed_log_file = '../ai-in-the-loop/results/{}static/overview/{}static_seeds.log'.format(case_domain, case_domain)
                if os.path.isfile(static_seed_log_file):
                    with open(static_seed_log_file) as f:
                        static_seeds = [int(x) for x in f]
                else:
                    print('ERROR: Dynamic pipeline not possible, as no static seeds are run, yet! Please run static pipelines first.')
                    raise NotImplementedError
                config.args.main_seed = static_seeds[0]
                print('')
                print('######################')
                print('seed: ' + str(static_seeds[0]))
                print('######################')
                print('')
                print('')
                print('')
                print('')

                # training process

                # get data and modules
                trainloader, valloader, _ = md.get_data('1')
                model_name = "densenet"
                num_classes = 35
                feature_extract = config.args.fmow_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
                import src.models.densenet as dn
                import torch
                import torch.optim as optims
                import torch.nn as nn
                import torch.backends.cudnn as cudnn
                model_ft, input_size = dn.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

                # set parameters to be updated
                params_to_update = model_ft.parameters()
                if feature_extract:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad:
                            params_to_update.append(param)
                network = model_ft
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                network = network.to(device)
                if device == 'cuda':
                    network = torch.nn.DataParallel(network)
                    cudnn.benchmark = True

                # training
                _, _, vallosses, valacc, trainlosses, trainacc = mb.train(
                    network=network,
                    trainloader=trainloader,
                    valloader=valloader,
                    optim=optims.Adam(params_to_update,
                                      lr=config.args.fmow_adam__lr,
                                      weight_decay=config.args.fmow_adam__wd),
                    loss_fn=nn.CrossEntropyLoss(),
                    download=False,
                )

                # track results
                for i in range(len(vallosses)):
                    mlflow.start_run(run_name='epoch_{}'.format(str(i+1)))
                    mlflow.log_metric('trainloss', trainlosses[i])
                    mlflow.log_metric('trainacc', trainacc[i])
                    mlflow.log_metric('valloss', vallosses[i])
                    mlflow.log_metric('valacc', valacc[i])
                    mlflow.end_run()

        # dynamic AIITL-system
        else:
            # read in already used seeds for specific case_domain_pipe setting
            seed_log_file = '../ai-in-the-loop/results/{}/overview/{}_seeds.log'.format(case_domain_pipe, case_domain_pipe)
            if os.path.isfile(seed_log_file):
                with open(seed_log_file) as f:
                    seeds = [int(x) for x in f]
            else:
                seeds = []

            # collect static seeds
            static_seed_log_file = '../ai-in-the-loop/results/{}static/overview/{}static_seeds.log'.format(case_domain, case_domain)
            if os.path.isfile(static_seed_log_file):
                with open(static_seed_log_file) as f:
                    static_seeds = [int(x) for x in f]
            else:
                print('ERROR: Dynamic pipeline not possible, as no static seeds are run, yet! Please run static pipelines first.')
                raise NotImplementedError

            # collect the seeds that are not run, yet. In case of ablation study, results are always run on first seed
            if config.args.ablation_study == '':
                static_seeds_to_run = [static_seed for static_seed in static_seeds if static_seed not in seeds]
            else:
                static_seeds_to_run = [seeds[0]]

            assert len(seeds) != 0, 'ERROR: Run standard results before ablation study!'

            # check whether run configuration (args.no_seeds) fits into the number of seeds to run
            assert len(static_seeds_to_run) >= config.args.no_seeds, 'ERROR: Mismatch between remaining static seeds to run and desired no of seeds to run defined by parser'
            if len(static_seeds_to_run) > config.args.no_seeds:
                print('WARNING: There are more seeds to run than desired by config.args.no_seeds! By default only the first x seeds are taken. {} seed(s) is/are desired.'.format(config.args.no_seeds))
                static_seeds_to_run = static_seeds_to_run[:config.args.no_seeds]
                assert len(static_seeds_to_run) == config.args.no_seeds

            # run dynamic system for all seeds to be run on
            for idx, static_seed in enumerate(static_seeds_to_run):

                if config.args.ablation_study == '':
                    # save all used seeds
                    seeds.append(static_seed)
                    with open(seed_log_file, 'w') as f:
                        for seed in seeds:
                            f.write(str(seed))
                            f.write('\n')

                print('')
                print('######################')
                if config.args.ablation_study == '':
                    print('seed no. ' + str(len(seeds)) + ': ' + str(static_seed))
                else:
                    print('seed no. ' + str(idx+1) + ': ' + str(static_seed))
                print('######################')
                print('')
                print('')
                print('')
                print('')

                # run script for respective seed (if not run, yet) and add to seed file
                config.args.main_seed = static_seed
                if config.args.pipe_type == 'static':
                    all_pipes_stat.main()
                elif config.args.pipe_type == 'dynamic':
                    all_pipes_dyn.main()


    # run results in REPRO mode
    else:

        print('')
        print('######################')
        print('CAUTION: You are running results in reproduction mode.')
        print('######################')
        print('')

        # read in already used seeds for specific case_domain_pipe setting
        seed_log_file = '../ai-in-the-loop/results/{}/overview/{}_seeds.log'.format(case_domain_pipe, case_domain_pipe)
        if os.path.isfile(seed_log_file):
            with open(seed_log_file) as f:
                seeds = [int(x) for x in f]

            # only repro for the existing first seed
            # This ensures, that reproduction is only done for the results of the Master's Thesis (no further seeds available)
            seeds = [seeds[0]]
        else:
            print('ERROR: Reproduction not possible, as no seeds are saved, yet! Please run in non-repro mode first.')
            raise NotImplementedError

        # run system for all seeds
        for seed in seeds:
            print('')
            print('######################')
            print('seed no. ' + str(len(seeds)) + ': ' + str(seed))
            print('######################')
            print('')
            print('')
            print('')
            print('')

            if config.args.ablation_study != 'unc_vs_OOD_share' and config.args.ablation_study != 'fmow_overfit':
                # run script for respective seed
                config.args.main_seed = seed
                if config.args.pipe_type == 'static':
                    all_pipes_stat.main()
                elif config.args.pipe_type == 'dynamic':
                    all_pipes_dyn.main()

            # reproduction of study on impact of unknown data
            elif config.args.ablation_study == 'unc_vs_OOD_share':

                # experiment
                if config.args.exp_name == '':
                    exp_name = 'run_' + config.args.domain + '_static_pipeline_' + config.args.pipe_case + '_OOD_share_analysis_REPRO'
                else:
                    exp_name = config.args.exp_name
                mlflow.set_experiment(experiment_name=exp_name)
                exp = mlflow.get_experiment_by_name(exp_name)
                utils.adjust_mlflow_yaml(exp.experiment_id)
                utils.save_mlflow_exp_id(exp.experiment_id)
                print('Exp. ID: ' + str(exp.experiment_id))
                mlflow.start_run(run_name='OOD_share_analysis')

                # create batch of 1000k CIFAR-10 + SVHN data samples and compute uncertainty with increasing SVHN (OOD) share
                gen_AI = mb.get_general_model()
                _, _, cifar10 = md.get_data('0')
                _, _, svhn = md.get_data('1')
                config.args.main_seed = seed
                print('')
                print('######################')
                print('seed: ' + str(seed))
                print('######################')
                print('')
                print('')
                print('')
                print('')
                cifar10 = Subset(cifar10.dataset, random.sample(range(len(cifar10.dataset)), 1000))
                svhn = Subset(svhn.dataset, random.sample(range(len(svhn.dataset)), 1000))

                confidences = []

                # get uncertainties for all shares of unknown data
                for i in range(101):
                    print('OOD share: {}/{}'.format(i/100, 100))
                    config.args.main_seed = seed
                    # i indicates the OOD share
                    ood_share = round(i/100, 2)
                    id_share = round(1-ood_share, 2)
                    cifar10_index = list(range(len(cifar10)))[:int(id_share*len(cifar10))]
                    print('len cifar10: ' + str(len(cifar10_index)))
                    print('len svhn: ' + str(len(svhn_index)))
                    svhn_index = list(range(len(svhn)))[:int(ood_share*len(svhn))]
                    batch_dataset = ConcatDataset([Subset(cifar10, cifar10_index), Subset(svhn, svhn_index)])
                    assert len(batch_dataset) == 1000

                    batch_loader = DataLoader(batch_dataset, num_workers=2, batch_size=1, shuffle=True)

                    # compute Softmax confidence
                    confidences.append(np.mean(pipe.get_unc_score(pipe.get_Softmax_prob(gen_AI, batch_loader), 'conf')))
                    print('avg. batch confidence: ' + str(confidences[-1]))
                    mlflow.log_metric(str('ood_share_' + str(ood_share)), confidences[-1])

                mlflow.end_run()


            # reproduction of study on overfitting of FMOW model
            elif config.args.ablation_study == 'fmow_overfit':

                # some checks
                assert config.args.pipe_case == 'sat'
                assert config.args.pipe_type == 'static'
                assert config.args.domain == 'multi'

                # experiment
                if config.args.exp_name == '':
                    exp_name = 'run_' + config.args.domain + '_static_pipeline_' + config.args.pipe_case + '_fMoW_Overfit_REPRO'
                else:
                    exp_name = config.args.exp_name
                mlflow.set_experiment(experiment_name=exp_name)
                exp = mlflow.get_experiment_by_name(exp_name)
                utils.adjust_mlflow_yaml(exp.experiment_id)
                utils.save_mlflow_exp_id(exp.experiment_id)
                print('Exp. ID: ' + str(exp.experiment_id))

                # print seed
                config.args.main_seed = seed
                print('')
                print('######################')
                print('seed: ' + str(seed))
                print('######################')
                print('')
                print('')
                print('')
                print('')

                # training process

                # get data
                trainloader, valloader, _ = md.get_data('1')
                model_name = "densenet"
                num_classes = 35
                feature_extract = config.args.fmow_feat_ex  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
                import src.models.densenet as dn
                import torch
                import torch.optim as optims
                import torch.nn as nn
                import torch.backends.cudnn as cudnn
                model_ft, input_size = dn.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

                # set parameters to be updated
                params_to_update = model_ft.parameters()
                if feature_extract:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad:
                            params_to_update.append(param)
                network = model_ft
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                network = network.to(device)
                if device == 'cuda':
                    network = torch.nn.DataParallel(network)
                    cudnn.benchmark = True

                # training
                _, _, vallosses, valacc, trainlosses, trainacc = mb.train(
                    network=network,
                    trainloader=trainloader,
                    valloader=valloader,
                    optim=optims.Adam(params_to_update,
                                      lr=config.args.fmow_adam__lr,
                                      weight_decay=config.args.fmow_adam__wd),
                    loss_fn=nn.CrossEntropyLoss(),
                    download=False,
                )

                # track results
                for i in range(len(vallosses)):
                    mlflow.start_run(run_name='epoch_{}'.format(str(i+1)))
                    mlflow.log_metric('trainloss', trainlosses[i])
                    mlflow.log_metric('trainacc', trainacc[i])
                    mlflow.log_metric('valloss', vallosses[i])
                    mlflow.log_metric('valacc', valacc[i])
                    mlflow.end_run()

if __name__ == "__main__":
    main()

