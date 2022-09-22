# this code is intended to create plots and tables shown in Master's Thesis.

# all libraries
import pandas as pd
import src.utils.model_building as mb
import src.config
import src.visualization.reports as rs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.font_manager as fm
import mlflow
import math
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.image as mpimg
import src.utils.utils as utils
from textwrap import wrap
import random

def main():

    # get config
    config = src.config.cfg
    if config.args.no_finalbatch:
        config.args.incl_finalbatch = False

    # get setting
    case_domain_pipe = config.args.pipe_case + '_' + config.args.domain + '_' + config.args.pipe_type

    # get colors of colormap
    cmap = cm.get_cmap('viridis')
    start = 0.2
    stop = 0.8
    if config.args.utility_graph:
        if config.args.ablation_study == '':
            if config.args.domain == 'single':
                n = 1
            else:
                n = 3
        elif config.args.ablation_study == 'no_trust_early_stop':
            n = 2
        elif config.args.ablation_study == 'trust_thresholds':
            n = 6
        elif config.args.ablation_study == 'resnet18_backbone':
            n = 2
    elif config.args.exp_learning_graph:
        if config.args.domain == 'single':
            n = 1
        else:
            n = 3
    elif config.args.softmax_thresholds:
        n = 3
    elif config.args.ablation_study == 'fmow_overfit':
        n = 2
    elif config.args.ablation_study == 'smaller_OOD_share_v2':
        n = 2
    else:
        n = 4
    colors = cmap(np.linspace(start, stop, n))

    # specify mlruns location
    mlflow.set_tracking_uri('file:///home/dawe/ai-in-the-loop/results/' + case_domain_pipe + '/mlruns')

    # if no ablation study
    if config.args.ablation_study == '':

        # collect mlrunsIDs available
        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO.log'
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs.log'

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError

        # Impact of Softmax thresholds
        if config.args.softmax_thresholds:
            assert config.args.domain == 'single'

            print('')
            print('Single domain softmax threshold plot is created for pipe case specified by parser.')

            # collect metrics in numpy array
            LEN_OF_PLOT = int(len([i/1000 for i in range(950,  1001)]))
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')
            pipe_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_cov = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_cov = np.zeros((LEN_OF_PLOT, len(IDs)))
            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))
                pipe_acc[:, i] = results[results['tags.mlflow.runName'] != 'baselines']['metrics.pipe_acc_wH']
                human_cov[:, i] = results[results['tags.mlflow.runName'] != 'baselines']['metrics.human_effort']
                genAI_cov[:, i] = results[results['tags.mlflow.runName'] != 'baselines']['metrics.gen_AI_coverage']

            # average (does not have an effect for single ID consideration as intended by default)
            pipe_acc_avg = list(np.mean(pipe_acc, axis=1))
            pipe_acc_avg.reverse()
            human_cov_avg = list(np.mean(human_cov, axis=1))
            human_cov_avg.reverse()
            genAI_cov_avg = list(np.mean(genAI_cov, axis=1))
            genAI_cov_avg.reverse()

            # plot
            x = [i/1000 for i in range(950,  1001)]
            plt.rcParams["font.family"] = "cmr10"
            plt.plot(x, pipe_acc_avg, label='Accuracy', color=colors[0], linewidth=2, linestyle='solid')
            plt.plot(x, human_cov_avg, label='Human coverage', color=colors[1], linewidth=2, linestyle='solid')
            plt.plot(x, genAI_cov_avg, label='General model coverage', color=colors[2], linewidth=2, linestyle='solid')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Softmax threshold', fontsize=16)
            plt.ylabel('Accuracy / coverage', fontsize=16)
            #plt.title('Development of pipeline metrics in single domain, ' + config.args.pipe_case + ' case')
            plt.legend(fontsize=16, loc=8, ncol=3,
                       bbox_to_anchor=(0.5, -0.4)
                       )
            plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_Softmax_thresholds{}.pdf'.format(('REPRO' if config.args.results_repro else '')), bbox_inches='tight')


        # Development of utility
        if config.args.utility_graph:
            assert config.args.pipe_type == 'dynamic'

            print('')
            print('Utility plot is created for {} case and utility with params a = {}; b = {}.'.format(config.args.pipe_case, config.args.utility_a, config.args.utility_b))

            # collect metrics in numpy array
            if config.args.domain == 'multi':
                LEN_OF_PLOT = config.args.dyn_no_batches + (1 if config.args.incl_finalbatch else 0)
            else:
                LEN_OF_PLOT = config.args.dyn_single_no_batches
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')
            if config.args.domain == 'single':
                pipe_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
                human_effort = np.zeros((LEN_OF_PLOT, len(IDs)))
                pipe_acc_utility = np.zeros((LEN_OF_PLOT, len(IDs)))
            else:
                pipe_acc_gating = np.zeros((LEN_OF_PLOT, len(IDs)))
                human_effort_gating = np.zeros((LEN_OF_PLOT, len(IDs)))
                pipe_acc_utility_gating = np.zeros((LEN_OF_PLOT, len(IDs)))

                pipe_acc_odin = np.zeros((LEN_OF_PLOT, len(IDs)))
                human_effort_odin = np.zeros((LEN_OF_PLOT, len(IDs)))
                pipe_acc_utility_odin = np.zeros((LEN_OF_PLOT, len(IDs)))

                pipe_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
                human_effort_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
                pipe_acc_utility_maha = np.zeros((LEN_OF_PLOT, len(IDs)))

            genAI_only = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_only_utility = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_HITL = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_HITL_human_effort = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_HITL_utility = np.zeros((LEN_OF_PLOT, len(IDs)))

            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))
                if config.args.domain == 'single':
                    pipe_acc[:, i] = results[~results['tags.mlflow.runName'].str.startswith('PARAMS')]['metrics.pipe_performance']
                    human_effort[:, i] = results[~results['tags.mlflow.runName'].str.startswith('PARAMS')]['metrics.human_effort']
                    pipe_acc_utility[:, i] = config.args.utility_a * pipe_acc[:, i] - config.args.utility_b * human_effort[:, i]
                else:
                    if config.args.incl_finalbatch:
                        pipe_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.pipe_performance']
                        human_effort_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.human_effort']
                        pipe_acc_utility_gating[:, i] = config.args.utility_a * pipe_acc_gating[:, i] - config.args.utility_b * human_effort_gating[:, i]

                        pipe_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin'))]['metrics.pipe_performance']
                        human_effort_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin'))]['metrics.human_effort']
                        pipe_acc_utility_odin[:, i] = config.args.utility_a * pipe_acc_odin[:, i] - config.args.utility_b * human_effort_odin[:, i]

                        pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.pipe_performance']
                        human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.human_effort']
                        pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]
                    else:
                        pipe_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                        human_effort_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                        pipe_acc_utility_gating[:, i] = config.args.utility_a * pipe_acc_gating[:, i] - config.args.utility_b * human_effort_gating[:, i]

                        pipe_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                        human_effort_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                        pipe_acc_utility_odin[:, i] = config.args.utility_a * pipe_acc_odin[:, i] - config.args.utility_b * human_effort_odin[:, i]

                        pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                        human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                        pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]

                if config.args.domain == 'multi':
                    if not config.args.incl_finalbatch:
                        genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_only']
                        genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                        genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_HITL']
                        genAI_HITL_human_effort[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                        genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]
                    else:
                        genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.gen_AI_only']
                        genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                        genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.gen_AI_HITL']
                        genAI_HITL_human_effort[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                        genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]
                else:
                    genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS'))]['metrics.gen_AI_only']
                    genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                    genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS'))]['metrics.gen_AI_HITL']
                    genAI_HITL_human_effort[:, i] = ([(800)/(1000)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(266)/(338)] * LEN_OF_PLOT)
                    genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]

            # average (does not have an effect for single ID consideration as intended by default)
            if config.args.domain == 'single':
                pipe_acc_utility_avg = list(np.mean(pipe_acc_utility, axis=1))
                pipe_acc_utility_avg.reverse()
            else:
                pipe_acc_utility_gating_avg = list(np.mean(pipe_acc_utility_gating, axis=1))
                pipe_acc_utility_gating_avg.reverse()
                pipe_acc_utility_odin_avg = list(np.mean(pipe_acc_utility_odin, axis=1))
                pipe_acc_utility_odin_avg.reverse()
                pipe_acc_utility_maha_avg = list(np.mean(pipe_acc_utility_maha, axis=1))
                pipe_acc_utility_maha_avg.reverse()

            genAI_only_utility_avg = list(np.mean(genAI_only_utility, axis=1))
            genAI_only_utility_avg.reverse()
            genAI_HITL_utility_avg = list(np.mean(genAI_HITL_utility, axis=1))
            genAI_HITL_utility_avg.reverse()

            # plot
            x = [i+1 for i in range(LEN_OF_PLOT)]
            plt.rcParams["font.family"] = "cmr10"
            if config.args.domain == 'single':
                if (config.args.pipe_case == 'benchmark' and config.args.utility_b == 0.5) or \
                        (config.args.pipe_case == 'sat' and config.args.utility_b == 0.75):
                    plt.plot(x, pipe_acc_utility_avg, label='AIITL-system: Softmax', color=colors[0], linestyle='solid', marker='o' , linewidth=2)
                else:
                    plt.plot(x, pipe_acc_utility_avg, label='AIITL-system: Softmax', color=colors[0], linestyle='solid', marker='o', markersize=4)
            else:
                if (config.args.pipe_case == 'benchmark' and config.args.utility_b == 0.5) or \
                        (config.args.pipe_case == 'sat' and config.args.utility_b == 0.75):
                    plt.plot(x, pipe_acc_utility_gating_avg, label='AIITL-system: Gating', color=colors[0], linestyle='solid', marker='o', linewidth=2)
                    plt.plot(x, pipe_acc_utility_maha_avg, label='AIITL-system: Maha', color=colors[1], linestyle='solid', marker='^', linewidth=2)
                    plt.plot(x, pipe_acc_utility_odin_avg, label='AIITL-system: ODIN', color=colors[2], linestyle='solid', marker='D', linewidth=2)
                else:
                    plt.plot(x, pipe_acc_utility_gating_avg, label='AIITL-system: Gating', color=colors[0], linestyle='solid', marker='o', markersize=3)
                    plt.plot(x, pipe_acc_utility_maha_avg, label='AIITL-system: Maha', color=colors[1], linestyle='solid', marker='^', markersize=3)
                    plt.plot(x, pipe_acc_utility_odin_avg, label='AIITL-system: ODIN', color=colors[2], linestyle='solid', marker='D', markersize=3)
            if (config.args.pipe_case == 'benchmark' and config.args.utility_b == 0.5) or \
                    (config.args.pipe_case == 'sat' and config.args.utility_b == 0.75):
                plt.plot(x, genAI_HITL_utility_avg, label='HITL-system', color='black', linestyle='solid', linewidth=2)
                plt.plot(x, genAI_only_utility_avg, label='General model', color='black', linestyle='dashed', linewidth=2)
            else:
                plt.plot(x, genAI_HITL_utility_avg, label='HITL-system', color='black', linestyle='solid')
                plt.plot(x, genAI_only_utility_avg, label='General model', color='black', linestyle='dashed')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Batch no.', fontsize=16)
            plt.rc('axes', unicode_minus=False)
            if (config.args.pipe_case == 'benchmark' and config.args.utility_b == 0.5) or \
                    (config.args.pipe_case == 'sat' and config.args.utility_b == 0.75):
                plt.ylabel(r'Utility ($\alpha$={}, $\beta$={})'.format(config.args.utility_a, config.args.utility_b), fontsize=16)
            """elif (config.args.pipe_case == 'benchmark' and config.args.utility_b != 0.5) or \
                    (config.args.pipe_case == 'sat' and config.args.utility_b != 0.75):
                plt.ylabel('Utility', fontsize=16)"""
            if config.args.domain == 'multi':
                if (config.args.pipe_case == 'benchmark' and config.args.utility_b == 0.5) or (config.args.pipe_case == 'sat' and config.args.utility_b == 0.75):
                    plt.legend(fontsize=16, loc=8, ncol=3,
                               bbox_to_anchor=(0.5, -0.5)
                           )
                if config.args.utility_b == 99:
                    plt.legend(fontsize=16, loc=8, ncol=3,
                               bbox_to_anchor=(0.5, -2)
                               )
            else:
                if config.args.pipe_case == 'benchmark':
                    if config.args.utility_b == 0.5:
                        plt.legend(fontsize=16, loc=8, ncol=3,
                                   bbox_to_anchor=(0.5, -0.4)
                                   )
                    if config.args.utility_b == 99:
                        plt.legend(fontsize=16, loc=8, ncol=3,
                                   bbox_to_anchor=(0.5, -2)
                                   )
                else:
                    if config.args.utility_b == 0.75:
                        plt.legend(fontsize=16, loc=8, ncol=3,
                                   bbox_to_anchor=(0.5, -0.4)
                                   )
                    if config.args.utility_b == 99:
                        raise NotImplementedError
            plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_utility_scores_{}_{}{}.pdf'.format(config.args.utility_a, config.args.utility_b, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')


        # Overview of coverages and accuracy
        if config.args.acc_cov_table:
            assert config.args.pipe_type == 'dynamic'

            print('')
            print('Accuracy/Coverage table is created for {} case and {} domain.'.format(config.args.pipe_case, config.args.domain))
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')

            # collect metrics in numpy array, arranged in desired table format, to be transformed to latex later
            if config.args.domain == 'multi':
                NO_COLUMNS = 7
                NO_ROWS = 6
                index = ['Gen. model cov.', 'Exp. 1 cov.', 'Exp. 2 cov.', 'Exp. 3 cov.', 'Human cov.', 'Accuracy']
                columns = [
                    [' ', 'Gating', 'Maha', 'ODIN', 'Gen. model', 'HITL-system', 'Perfect coverage'],
                    ['1', '31', '31', '31', '31', '31', '31']
                ]
                columns = list(zip(*columns))
                columns = pd.MultiIndex.from_tuples(columns, names=['Allo. mech.', 'Batch'])
                final_batch = config.args.dyn_no_batches + (1 if config.args.incl_finalbatch else 0)
            else:
                NO_COLUMNS = 5
                NO_ROWS = 4
                index = ['Gen. model cov.', 'Exp. 1 cov.', 'Human cov.', 'Accuracy']
                columns = [
                    [' ', 'Softmax', 'Gen. model', 'HITL-system', 'Perfect coverage'],
                    ['1', '31', '31', '31', '31']
                ]
                columns = list(zip(*columns))
                columns = pd.MultiIndex.from_tuples(columns, names=['Allo. mech.', 'Batch'])
                final_batch = config.args.dyn_single_no_batches

            acc_cov_data = np.zeros((NO_ROWS, NO_COLUMNS))

            if config.args.domain == 'multi':
                genAI_cov = np.zeros((1, len(IDs)))
                expAI_1_cov = np.zeros((1, len(IDs)))
                expAI_2_cov = np.zeros((1, len(IDs)))
                expAI_3_cov = np.zeros((1, len(IDs)))
                human_cov = np.zeros((1, len(IDs)))
                pipe_acc = np.zeros((1, len(IDs)))
            else:
                genAI_cov = np.zeros((1, len(IDs)))
                expAI_1_cov = np.zeros((1, len(IDs)))
                human_cov = np.zeros((1, len(IDs)))
                pipe_acc = np.zeros((1, len(IDs)))

            # collect values
            for col in range(NO_COLUMNS-1):
                for i, id in enumerate(IDs):
                    results = mlflow.search_runs(experiment_ids=str(id))
                    if config.args.domain == 'multi':
                        if col in [0]:
                            batch_no = 1
                        else:
                            batch_no = final_batch

                        if col in [0, 1]:
                            subcase = 'gating'
                        elif col in [2]:
                            subcase = 'maha'
                        else:
                            subcase = 'odin'

                        # if no baselines
                        if col in [0, 1, 2, 3]:
                            expAI_1_cov[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.exp_coverage_exp_AI_1']
                            expAI_2_cov[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.exp_coverage_exp_AI_2']
                            expAI_3_cov[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.exp_coverage_exp_AI_3']
                            human_cov[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.human_effort']
                            pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.pipe_performance']
                            genAI_cov[:, i] = 1 - human_cov[:, i] - expAI_1_cov[:, i] - expAI_2_cov[:, i] - expAI_3_cov[:, i]
                        # if gen AI only
                        elif col in [4]:
                            expAI_1_cov[:, i] = 0
                            expAI_2_cov[:, i] = 0
                            expAI_3_cov[:, i] = 0
                            human_cov[:, i] = 0
                            pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.gen_AI_only']
                            genAI_cov[:, i] = 1
                        # if gen AI + HITL
                        else:
                            expAI_1_cov[:, i] = 0
                            expAI_2_cov[:, i] = 0
                            expAI_3_cov[:, i] = 0
                            genAI_cov[:, i] = (161/7101) if config.args.pipe_case == 'benchmark' else (61/2169)
                            pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_') & (results['tags.mlflow.runName'].str.contains(subcase))]['metrics.gen_AI_HITL']
                            human_cov[:, i] = 1 - genAI_cov[:, i]

                    else:
                        if col in [0]:
                            batch_no = 1
                        else:
                            batch_no = final_batch

                        # if no baselines
                        if col in [0, 1]:
                            expAI_1_cov[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_')]['metrics.exp_coverage']
                            human_cov[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_')]['metrics.human_effort']
                            pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_')]['metrics.pipe_performance']
                            genAI_cov[:, i] = 1 - human_cov[:, i] - expAI_1_cov[:, i]
                        # if gen AI only
                        elif col in [2]:
                            expAI_1_cov[:, i] = 0
                            human_cov[:, i] = 0
                            pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_')]['metrics.gen_AI_only']
                            genAI_cov[:, i] = 1
                        # if gen AI + HITL
                        else:
                            expAI_1_cov[:, i] = 0
                            genAI_cov[:, i] = (200/1000) if config.args.pipe_case == 'benchmark' else (72/338)
                            human_cov[:, i] = 1 - genAI_cov[:, i]
                            pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.contains('_' + str(batch_no) + '_')]['metrics.gen_AI_HITL']

                # average (does not have an effect for single ID consideration as intended by default) and add to table
                if config.args.domain == 'single':
                    expAI_1_cov_avg = np.mean(expAI_1_cov)
                    human_cov_avg = np.mean(human_cov)
                    pipe_acc_avg = np.mean(pipe_acc)
                    genAI_cov_avg = np.mean(genAI_cov)

                    acc_cov_data[0, col] = genAI_cov_avg
                    acc_cov_data[1, col] = expAI_1_cov_avg
                    acc_cov_data[2, col] = human_cov_avg
                    acc_cov_data[3, col] = pipe_acc_avg

                else:
                    expAI_1_cov_avg = np.mean(expAI_1_cov)
                    expAI_2_cov_avg = np.mean(expAI_2_cov)
                    expAI_3_cov_avg = np.mean(expAI_3_cov)
                    human_cov_avg = np.mean(human_cov)
                    pipe_acc_avg = np.mean(pipe_acc)
                    genAI_cov_avg = np.mean(genAI_cov)

                    acc_cov_data[0, col] = genAI_cov_avg
                    acc_cov_data[1, col] = expAI_1_cov_avg
                    acc_cov_data[2, col] = expAI_2_cov_avg
                    acc_cov_data[3, col] = expAI_3_cov_avg
                    acc_cov_data[4, col] = human_cov_avg
                    acc_cov_data[5, col] = pipe_acc_avg

            if config.args.domain == 'single':
                acc_cov_data[0, -1] = (200/800) if config.args.pipe_case == 'benchmark' else (72/338)
                acc_cov_data[1, -1] = 1 - acc_cov_data[0, -1]
                acc_cov_data[2, -1] = 0
                acc_cov_data[3, -1] = 0
            else:
                acc_cov_data[0, -1] = (161/7101) if config.args.pipe_case == 'benchmark' else (61/2169)
                acc_cov_data[1, -1] = (2608/7101) if config.args.pipe_case == 'benchmark' else (1618/2169)
                acc_cov_data[2, -1] = (2166/7101) if config.args.pipe_case == 'benchmark' else (238/2169)
                acc_cov_data[3, -1] = 1 - acc_cov_data[0, -1] - acc_cov_data[1, -1] - acc_cov_data[2, -1]
                acc_cov_data[4, -1] = 0
                acc_cov_data[5, -1] = 0

            # table output
            output = pd.DataFrame(acc_cov_data, index=index, columns=columns)
            output = output.round(2)
            print(output)
            print('')
            print('')
            print(output.to_latex(bold_rows=True,
                                  multicolumn_format='c',
                                  caption='Overview of the coverages of the general ML model, the artificial experts and the human experts as well as the system accuracy for the {}-domain, {} system of the {} case.'.format(config.args.domain, config.args.pipe_type, config.args.pipe_case),
                                  label=case_domain_pipe + '_acc_cov'))
            output.to_csv('../ai-in-the-loop/results/{}/logs/{}_acc_cov.csv'.format(case_domain_pipe, case_domain_pipe))


        # expert related metrics
        if config.args.exp_learning_graph:
            assert config.args.pipe_type == 'dynamic'

            print('')
            print('Expert learning plot is created for {} case.'.format(config.args.pipe_case))

            # collect metrics in numpy array
            if config.args.domain == 'multi':
                LEN_OF_PLOT = config.args.dyn_no_batches + (1 if config.args.incl_finalbatch else 0)
            else:
                LEN_OF_PLOT = config.args.dyn_single_no_batches
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')
            if config.args.domain == 'single':
                genAI_strong_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_strong_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_trust_scores = np.zeros((LEN_OF_PLOT, len(IDs)))
            else:
                expAI_1_strong_acc_gating = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_1_strong_acc_odin = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_1_strong_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_2_strong_acc_gating = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_2_strong_acc_odin = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_2_strong_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_3_strong_acc_gating = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_3_strong_acc_odin = np.zeros((LEN_OF_PLOT, len(IDs)))
                expAI_3_strong_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))

            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))
                if config.args.domain == 'single':
                    genAI_strong_acc[:, i] = results[~results['tags.mlflow.runName'].str.startswith('PARAMS')]['metrics.gen_AI_test_acc_strong']
                    expAI_strong_acc[:, i] = results[~results['tags.mlflow.runName'].str.startswith('PARAMS')]['metrics.exp_test_acc_strong']
                    expAI_trust_scores[:, i] = results[~results['tags.mlflow.runName'].str.startswith('PARAMS')]['metrics.trust_scores_exp_AI']
                else:
                    if config.args.incl_finalbatch:
                        expAI_1_strong_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.trust_scores_exp_AI_' + str(1)]
                        expAI_1_strong_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin'))]['metrics.trust_scores_exp_AI_' + str(1)]
                        expAI_1_strong_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_scores_exp_AI_' + str(1)]
                        expAI_2_strong_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.trust_scores_exp_AI_' + str(2)]
                        expAI_2_strong_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin'))]['metrics.trust_scores_exp_AI_' + str(2)]
                        expAI_2_strong_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_scores_exp_AI_' + str(2)]
                        expAI_3_strong_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.trust_scores_exp_AI_' + str(3)]
                        expAI_3_strong_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin'))]['metrics.trust_scores_exp_AI_' + str(3)]
                        expAI_3_strong_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_scores_exp_AI_' + str(3)]
                    else:
                        expAI_1_strong_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(1)]
                        expAI_1_strong_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(1)]
                        expAI_1_strong_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(1)]
                        expAI_2_strong_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(2)]
                        expAI_2_strong_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(2)]
                        expAI_2_strong_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(2)]
                        expAI_3_strong_acc_gating[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(3)]
                        expAI_3_strong_acc_odin[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('odin')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(3)]
                        expAI_3_strong_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_scores_exp_AI_' + str(3)]

            # average (does not have an effect for single ID consideration as intended by default)
            if config.args.domain == 'single':
                genAI_strong_acc = list(np.mean(genAI_strong_acc, axis=1))
                genAI_strong_acc.reverse()
                expAI_strong_acc = list(np.mean(expAI_strong_acc, axis=1))
                expAI_strong_acc.reverse()
                expAI_trust_scores = list(np.mean(expAI_trust_scores, axis=1))
                expAI_trust_scores.reverse()
            else:
                expAI_1_strong_acc_gating = list(np.mean(expAI_1_strong_acc_gating, axis=1))
                expAI_1_strong_acc_gating.reverse()
                expAI_1_strong_acc_odin = list(np.mean(expAI_1_strong_acc_odin, axis=1))
                expAI_1_strong_acc_odin.reverse()
                expAI_1_strong_acc_maha = list(np.mean(expAI_1_strong_acc_maha, axis=1))
                expAI_1_strong_acc_maha.reverse()
                expAI_2_strong_acc_gating = list(np.mean(expAI_2_strong_acc_gating, axis=1))
                expAI_2_strong_acc_gating.reverse()
                expAI_2_strong_acc_odin = list(np.mean(expAI_2_strong_acc_odin, axis=1))
                expAI_2_strong_acc_odin.reverse()
                expAI_2_strong_acc_maha = list(np.mean(expAI_2_strong_acc_maha, axis=1))
                expAI_2_strong_acc_maha.reverse()
                expAI_3_strong_acc_gating = list(np.mean(expAI_3_strong_acc_gating, axis=1))
                expAI_3_strong_acc_gating.reverse()
                expAI_3_strong_acc_odin = list(np.mean(expAI_3_strong_acc_odin, axis=1))
                expAI_3_strong_acc_odin.reverse()
                expAI_3_strong_acc_maha = list(np.mean(expAI_3_strong_acc_maha, axis=1))
                expAI_3_strong_acc_maha.reverse()

            # trust threshold
            trust_thresh = 0.9 if (config.args.pipe_case == 'sat' and config.args.pipe_type == 'dynamic' and config.args.domain == 'single') else 0.95
            trust_thresh_list = [trust_thresh] * LEN_OF_PLOT

            # plot
            x = [i+1 for i in range(LEN_OF_PLOT)]
            
            # initialize multi domain labels
            multi_label_exp1 = 'Exp. 1 (SVHN)' if config.args.pipe_case == 'benchmark' else 'Exp. 1 (fMoW)'
            multi_label_exp2 = 'Exp. 2 (MNIST)' if config.args.pipe_case == 'benchmark' else 'Exp. 2 (AID)'
            multi_label_exp3 = 'Exp. 3 (Fashion-MNIST)' if config.args.pipe_case == 'benchmark' else 'Exp. 3 (RESISC)'

            plt.rcParams["font.family"] = "cmr10"
            if config.args.domain == 'single':
                plt.plot(x, expAI_trust_scores, label='Artificial expert', color=colors[0], linestyle='solid', marker='o' , linewidth=2)
                plt.plot(x, genAI_strong_acc, label='General model', color='black', linestyle='solid', linewidth=2)
                plt.plot(x, trust_thresh_list, label='Trust threshold', color='black', linestyle='dashed', linewidth=2)
                #plt.plot(x, expAI_strong_acc, label='Artificial exp.: testset', color=colors[0], linestyle='dashed', linewidth=2)
            else:
                if config.args.sub_case == 'gating':
                    plt.plot(x, expAI_1_strong_acc_gating, label=multi_label_exp1, color=colors[0], linestyle='solid', marker='o', markersize=3)
                    plt.plot(x, expAI_2_strong_acc_gating, label=multi_label_exp2, color=colors[1], linestyle='solid', marker='^', markersize=3)
                    plt.plot(x, expAI_3_strong_acc_gating, label=multi_label_exp3, color=colors[2], linestyle='solid', marker='D', markersize=3)
                    plt.plot(x, trust_thresh_list, label='Trust threshold', color='black', linestyle='dashed')
                elif config.args.sub_case == 'maha':
                    plt.plot(x, expAI_1_strong_acc_maha, label=multi_label_exp1, color=colors[0], linestyle='solid', marker='o', markersize=3)
                    plt.plot(x, expAI_2_strong_acc_maha, label=multi_label_exp2, color=colors[1], linestyle='solid', marker='^', markersize=3)
                    plt.plot(x, expAI_3_strong_acc_maha, label=multi_label_exp3, color=colors[2], linestyle='solid', marker='D', markersize=3)
                    plt.plot(x, trust_thresh_list, label='Trust threshold', color='black', linestyle='dashed')
                elif config.args.sub_case == 'odin':
                    plt.plot(x, expAI_1_strong_acc_odin, label=multi_label_exp1, color=colors[0], linestyle='solid', marker='o', markersize=3)
                    plt.plot(x, expAI_2_strong_acc_odin, label=multi_label_exp2, color=colors[1], linestyle='solid', marker='^', markersize=3)
                    plt.plot(x, expAI_3_strong_acc_odin, label=multi_label_exp3, color=colors[2], linestyle='solid', marker='D', markersize=3)
                    plt.plot(x, trust_thresh_list, label='Trust threshold', color='black', linestyle='dashed')
            plt.xlabel('Batch no.', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            if config.args.domain == 'multi':
                if config.args.sub_case == 'maha' or config.args.sub_case == 'odin':
                    ax = plt.gca()
                    ax.axes.yaxis.set_ticklabels([])
                if config.args.pipe_case == 'sat':
                    plt.ylim(0.1, 1)
                else:
                    plt.ylim(0.75, 1)
            if config.args.domain == 'single':
                plt.ylabel('Trust scores', fontsize=16)
            if config.args.domain == 'single':
                plt.legend(fontsize=16, loc=8, ncol=3,
                       bbox_to_anchor=(0.5, -0.4)
                       )
            if config.args.domain == 'multi':
                if config.args.utility_b == 99:
                    config.args.fig_size = [8, 4]
                    plt.legend(fontsize=16, loc=8, ncol=2,
                               bbox_to_anchor=(0.5, -0.5)
                               )
            plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
            if config.args.domain == 'single':
                plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_expert_related_metrics{}.pdf'.format(('REPRO' if config.args.results_repro else '')), bbox_inches='tight')
            else:
                plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_{}_related_metrics{}.pdf'.format(config.args.sub_case, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')


        # allocation matrix
        if config.args.allocation_matrix:
            assert config.args.domain == 'multi'

            print('')
            print('Allocation matrix is created for {} case, {} pipeline, {} batch no, {} subcase and {} mlruns ID.'.format(config.args.pipe_case, config.args.pipe_type, config.args.batch_no, config.args.sub_case, IDs[0]))

            matrix = np.zeros((9, 6))
            results = mlflow.search_runs(experiment_ids=str(IDs[0]))

            # slice out row with respective metrics (specified by batch no and subcase)
            if config.args.pipe_type == 'static':
                metrics = results[results['tags.mlflow.runName'] == config.args.sub_case]
            else:
                metrics = results[(results['tags.mlflow.runName'].str.contains(config.args.sub_case)) & (results['tags.mlflow.runName'].str.contains('_' + str(config.args.batch_no) + '_'))]

            for i in range(np.shape(matrix)[0]):
                for j in range(np.shape(matrix)[1]):
                    if i < 4 and j < 5:
                        # regular allocation
                        matrix[i, j] = int(metrics['metrics.{}/{}'.format(i, j)])
                    elif i < 4 and j >= 5:
                        # sum of row
                        matrix[i, j] = np.sum(matrix[i,:5], dtype=int)
                    elif i == 4:
                        # sum of column
                        matrix[i, j] = np.sum(matrix[:4, j], dtype=int)
                    elif i == 5 and j < 5:
                        # coverage
                        matrix[i, j] = matrix[i-1, j] / matrix[i-1, -1]
                    elif i == 6 and j < 5:
                        # desired coverage
                        if j == 4:
                            matrix[i, j] = 0.0
                        else:
                            matrix[i, j] = matrix[j, 5] / matrix[i-2, -1]
                    elif i == 7 and j < 5:
                        # acc on claimed data
                        if j == 4:
                            matrix[i, j] = 1.0
                        elif j == 0:
                            matrix[i, j] = metrics['metrics.gen_AI_claimed_acc']
                        else:
                            if config.args.pipe_type == 'dynamic':
                                matrix[i, j] = metrics['metrics.exp_claimed_acc_exp_AI_{}'.format(j)]
                            else:
                                matrix[i, j] = metrics['metrics.Exp_AI_{}_claimed_acc'.format(j)]
                    elif i == 8 and j < 4:
                        if config.args.pipe_type == 'dynamic':
                            if j == 0:
                                if config.args.pipe_case == 'benchmark':
                                    matrix[i, j] = 0.9644
                                else:
                                    matrix[i, j] = 0.9652631578947368
                            else:
                                matrix[i, j] = metrics['metrics.exp_test_acc_exp_AI_{}'.format(j)]
                        else:
                            if j == 0:
                                if config.args.pipe_case == 'benchmark':
                                    matrix[i, j] = 0.9644
                                else:
                                    matrix[i, j] = 0.9652631578947368
                            else:
                                matrix[i, j] = results[results['tags.mlflow.runName'] == 'baselines']['metrics.exp_AI_{}_acc'.format(j)]
                    elif i == 8 and j == 4:
                        matrix[i, j] = 1.0

            # output
            matrix = pd.DataFrame(matrix, index=['Gen. model', 'Exp. 1', 'Exp. 2', 'Exp. 3', 'Total', 'Coverage', 'Perfect coverage', 'Claimed data accuracy', 'Testset accuracy'],
                                  columns=['Gen. model', 'Exp. 1', 'Exp. 2', 'Exp. 3', 'Human exp.', 'Total'])
            matrix = matrix.round(2)
            print(matrix)
            print('')
            print('')
            print(matrix.to_latex(bold_rows=True,
                                    caption='Allocation matrix with origins in rows and assignments in columns for the {}-domain, {} system of the {} case.'.format(config.args.domain, config.args.pipe_type, config.args.pipe_case),
                                    label=case_domain_pipe + '_allomatrix'))


        # overview of combinations of allocation mechanisms
        if config.args.selec_comb_mechs:
            assert config.args.pipe_type == 'static'
            assert config.args.domain == 'multi'

            print('')
            print('Combinations of selec and comb mechs table is created for {} case and {} domain.'.format(config.args.pipe_case, config.args.domain))
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')

            output = np.zeros((5, 15))

            # collect runs, add needed utility score, pivot table, average over all pivot table
            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))

                # compute needed utility
                utility = (config.args.utility_a * np.array(results['metrics.pipe_acc_wH']) - config.args.utility_b * np.array(results['metrics.human_effort'])).ravel()
                results['metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b)] = utility
                # get data
                results = pd.pivot_table(results, values=['metrics.human_effort', 'metrics.pipe_acc_wH', 'metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b)],
                               columns=['params.Comb_mech'], index=['params.Selec_mech'])
                index = results.index
                results.columns = results.columns.swaplevel(0,1)
                results = results.sort_index(axis=1)

                # get data in correct order
                results = np.array(results.values)[:, [ 1,  0,  2,
                                                        4,  3,  5,
                                                        10,  9, 11,
                                                        13, 12, 14,
                                                        7,  6,  8,]]

                # average iteratively (no effect in default setting)
                if i == 0:
                    output = results
                else:
                    for j in range(np.shape(output)[0]):
                        for k in range(np.shape(output)[1]):
                            output[j, k] = (output[j, k]*(i) + results[j, k])/(i+1)

            # cast back into pivot table format
            tuples = [(       'Gating',   'metrics.pipe_acc_wH'),
                      (       'Gating',  'metrics.human_effort'),
                      (       'Gating', 'metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b)),
                      (         'Maha',   'metrics.pipe_acc_wH'),
                      (         'Maha',  'metrics.human_effort'),
                      (         'Maha', 'metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b)),
                      ('ODIN',   'metrics.pipe_acc_wH'),
                      ('ODIN',  'metrics.human_effort'),
                      ('ODIN', 'metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b)),
                      (         'ODIN\_ext',   'metrics.pipe_acc_wH'),
                      (         'ODIN\_ext',  'metrics.human_effort'),
                      (         'ODIN\_ext', 'metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b)),
                      (     'Softmax',   'metrics.pipe_acc_wH'),
                      (     'Softmax',  'metrics.human_effort'),
                      (     'Softmax', 'metrics.utility_{}_{}'.format(config.args.utility_a, config.args.utility_b))]
            columns = pd.MultiIndex.from_tuples(tuples, names=['Comb_mech', 'metric'])
            output = pd.DataFrame(output, index=index, columns=columns)

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', -1)

            # create output
            output = output.round(2)
            print(output)
            print('')
            print('')
            print(output.to_latex(bold_rows=True,
                                  multicolumn_format='c',
                                  caption='Overview of accuracy, human coverage and utility ($\alpha={}$, $\beta={}$) for all allocation mechanism combinations of the {}-domain, {} system of the {} case.'.format(config.args.utility_a, config.args.utility_b, config.args.domain, config.args.pipe_type, config.args.pipe_case),
                                  label=case_domain_pipe + '_combos'))
            output.to_csv('../ai-in-the-loop/results/{}/logs/{}_combos.csv'.format(case_domain_pipe, case_domain_pipe))


        # Baselines for all allocation mechanisms
        if config.args.selec_comb_mechs_baselines:
            assert config.args.pipe_type == 'static'
            assert config.args.domain == 'multi'

            print('')
            print('Baselines for {} case and {} domain are created.'.format(config.args.pipe_case, config.args.domain))
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')

            output = np.zeros((5, 3))
            full_auto_LB = np.zeros((1, len(IDs)))
            full_auto_random = np.zeros((1, len(IDs)))
            full_auto_global = np.zeros((1, len(IDs)))
            full_auto_perfect = np.zeros((1, len(IDs)))
            hitl_UB = np.zeros((1, len(IDs)))

            # collect metrics
            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))

                full_auto_LB[:, i] = results[results['tags.mlflow.runName'].str.startswith('baselines')]['metrics.full_auto_LB']
                full_auto_random[:, i] = results[results['tags.mlflow.runName'].str.startswith('baselines')]['metrics.full_auto_random']
                full_auto_global[:, i] = results[results['tags.mlflow.runName'].str.startswith('baselines')]['metrics.full_auto_global']
                full_auto_perfect[:, i] = results[results['tags.mlflow.runName'].str.startswith('baselines')]['metrics.full_auto_UB']
                hitl_UB[:, i] = results[results['tags.mlflow.runName'].str.startswith('baselines')]['metrics.hitl_UB']

            # average (does not have an effect for single ID consideration as intended by default)
            output[0, 0] = np.mean(full_auto_LB)
            output[1, 0] = np.mean(full_auto_random)
            output[2, 0] = np.mean(full_auto_global)
            output[3, 0] = np.mean(full_auto_perfect)
            output[4, 0] = np.mean(hitl_UB)

            # remaining stats
            output[0, 1] = 0.0
            output[1, 1] = 0.0
            output[2, 1] = 0.0
            output[3, 1] = 0.0
            output[4, 1] = ((7101-161)/7101) if config.args.pipe_case == 'benchmark' else ((2169-61)/2169)

            output[0, 2] = config.args.utility_a * output[0, 0] - config.args.utility_b * output[0, 1]
            output[1, 2] = config.args.utility_a * output[1, 0] - config.args.utility_b * output[1, 1]
            output[2, 2] = config.args.utility_a * output[2, 0] - config.args.utility_b * output[2, 1]
            output[3, 2] = config.args.utility_a * output[3, 0] - config.args.utility_b * output[3, 1]
            output[4, 2] = config.args.utility_a * output[4, 0] - config.args.utility_b * output[4, 1]

            # create output
            output = pd.DataFrame(output, index=['General model', 'Random expert allocation', 'Global model', 'Perfect expert allocation', 'HITL-system'],
                                  columns=['Accuracy', 'Human coverage', 'Utility'])

            output = output.round(2)
            print(output)
            print('')
            print('')
            print(output.to_latex(bold_rows=True,
                                  multicolumn_format='c',
                                  caption='Baselines',
                                  label=case_domain_pipe + '_baselines'))


        # print allocation system results
        if config.args.allocation_system:
            assert config.args.pipe_type == 'static'
            assert config.args.domain == 'multi'

            print('')
            print('Allocation System for {} case and {} domain is created.'.format(config.args.pipe_case, config.args.domain))
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')

            output = np.zeros((1, 3))
            pipe_acc = np.zeros((1, len(IDs)))
            human_cov = np.zeros((1, len(IDs)))

            # collect metrics
            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))

                pipe_acc[:, i] = results[results['tags.mlflow.runName'].str.startswith('allo')]['metrics.pipe_acc_wH']
                human_cov[:, i] = results[results['tags.mlflow.runName'].str.startswith('allo')]['metrics.human_effort']

            # average (does not have an effect for single ID consideration as intended by default)
            output[0, 0] = np.mean(pipe_acc)
            output[0, 1] = np.mean(human_cov)

            # remaining stats
            output[0, 2] = config.args.utility_a * output[0, 0] - config.args.utility_b * output[0, 1]

            output = pd.DataFrame(output, index=['Allocation_system'],
                                  columns=['Accuracy', 'Human coverage', 'Utility'])

            # create output
            output = output.round(2)
            print(output)
            print('')
            print('')
            print(output.to_latex(bold_rows=True,
                                  multicolumn_format='c',
                                  caption='Overview of accuracy, human coverage and utility ($\alpha={}$, $\beta={}$) for the allocation system of the {}-domain, {} system of the {} case.'.format(config.args.utility_a, config.args.utility_b, config.args.domain, config.args.pipe_type, config.args.pipe_case),
                                  label=case_domain_pipe + '_allo_system'))


        if config.args.saliency:

            # we only plot saliency maps for satellite case
            assert config.args.pipe_case == 'sat'
            assert config.args.domain == 'multi'

            # load models
            gen_AI = mb.get_general_model()
            if config.args.pipe_type == 'static':
                exp_AI_1, exp_AI_2, exp_AI_3 = mb.get_expert_models()
            else:
                config.args.dyn_pipe = True
                print('exp1 is loaded')
                exp_AI_1 = utils.load_gpu_pickle(config.args.pipe_root + 'results/sat_multi_dynamic/models/exp_AI_1_dyn_mahamaha_lastRun.pickle')
                print('exp2 is loaded')
                exp_AI_2 = utils.load_gpu_pickle(config.args.pipe_root + 'results/sat_multi_dynamic/models/exp_AI_2_dyn_mahamaha_lastRun.pickle')
                print('exp3 is loaded')
                exp_AI_3 = utils.load_gpu_pickle(config.args.pipe_root + 'results/sat_multi_dynamic/models/exp_AI_3_dyn_mahamaha_lastRun.pickle')
            models = [gen_AI, exp_AI_1, exp_AI_2, exp_AI_3]

            # set up transform of images
            transform_ = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

            #file = open('../ai-in-the-loop/results/sat_multi_dynamic/logs/correct_files.txt', 'r')
            #images = file.read().split("\n")
            #images = images[:-1]

            # manual selection
            eurosat_class = 'Pasture'
            eurosat_class_no = '14071'
            fmow_class = 'electric_substation'
            fmow_class_no = '75164'
            aid_class = 'Center'
            aid_class_no = '49'
            resisc_class = 'ship'
            resisc_class_no = '453'

            #Eurosat_image = config.args.pipe_root + 'data/raw/EUROSAT/' + eurosat_class + '/' + eurosat_class + '_' + eurosat_class_no + '.jpg'
            Eurosat_image = config.args.pipe_root + 'data/processed/SAT_CASE_v2/EUROSAT/' + eurosat_class + '/' + eurosat_class_no + '.png'
            FMOW_image = config.args.pipe_root + 'data/processed/SAT_CASE_v2/FMOW/train/' + fmow_class + '/' + fmow_class_no + '.png'
            AID_image = config.args.pipe_root + 'data/processed/SAT_CASE_v2/AID/' + aid_class + '/' + aid_class.lower() + '_' + aid_class_no + '.jpg'
            RESISC_image = config.args.pipe_root + 'data/processed/SAT_CASE_v2/RESISC/' + resisc_class + '/' + resisc_class + '_' + resisc_class_no + '.jpg'

            images = [Eurosat_image, FMOW_image, AID_image, RESISC_image]
            fig, ax = plt.subplots(4, 5)

            # loop through all models
            for model_id, model in enumerate(models):
                for pic_id, pic in enumerate(images):

                    print(pic_id)

                    if 'EUROSAT' in pic:
                        pic.replace('processed/SAT_CASE_v2','raw')

                    #img_class = pic.split('/')[-2]

                    # Open the image file
                    image = Image.open(pic)
                    # Transforms the image
                    image = transform_(image)
                    # Reshape the image (because the model use 4-dimensional tensor (batch_size, channel, width, height))
                    image = image.reshape(1, 3, 224, 224)
                    # Set the device for the image
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    image = image.to(device)
                    # Set the requires_grad_ to the image for retrieving gradients
                    image.requires_grad_()

                    model = model.to(device)
                    model.eval()

                    # Retrieve output from the image
                    output = model(image)
                    # Catch the output
                    output_idx = output.argmax()
                    output_max = output[0, output_idx]
                    # Do backpropagation to get the derivative of the output based on the image
                    output_max.backward()

                    # Retrieve the saliency map and also pick the maximum value from channels on each pixel.
                    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
                    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
                    saliency = saliency.reshape(224, 224)

                    # Visualize the image and the saliency map
                    ax[pic_id][model_id+1].imshow(saliency.cpu(), cmap='viridis')
                    #ax[pic_id][model_id+1].axis('off')
                    ax[pic_id][model_id+1].set_xticks([])
                    ax[pic_id][model_id+1].set_yticks([])
                    ax[pic_id][model_id+1].set_xticklabels([])
                    ax[pic_id][model_id+1].set_yticklabels([])

                    if pic_id == 0:
                        ax[pic_id][0].set_title('Original', fontsize=9, family='cmr10')
                        ax[pic_id][1].set_title('\n'.join(wrap('Gen. model (EuroSAT)', 10)), fontsize=9, family='cmr10')
                        ax[pic_id][2].set_title('\n'.join(wrap('Art. expert 1 (fMoW)', 13)), fontsize=9, family='cmr10')
                        ax[pic_id][3].set_title('\n'.join(wrap('Art. expert 2 (AID)', 13)), fontsize=9, family='cmr10')
                        ax[pic_id][4].set_title('\n'.join(wrap('Art. expert 3 (RESISC)', 13)), fontsize=9, family='cmr10')

                    # original image
                    if model_id == 0:
                        if pic_id == 0:
                            eurosat_class_no = '1062'
                            Eurosat_image = config.args.pipe_root + 'data/raw/EUROSAT/' + eurosat_class + '/' + eurosat_class + '_' + eurosat_class_no + '.jpg'
                            pic = Eurosat_image
                        img = mpimg.imread(pic)
                        ax[pic_id][0].imshow(img)
                        #ax[pic_id][0].axis('off')

                        if pic_id == 0:
                            ax[pic_id][0].set_ylabel('EuroSAT/\n' + eurosat_class.replace('_', ' '),fontsize=9, family='cmr10')
                        elif pic_id == 1:
                            ax[pic_id][0].set_ylabel('fMoW/\n' + 'Electric Substation',fontsize=9, family='cmr10')
                        elif pic_id == 2:
                            ax[pic_id][0].set_ylabel('AID/\n' + aid_class.replace('_', ' '),fontsize=9, family='cmr10')
                        else:
                            ax[pic_id][0].set_ylabel('RESISC/\n' + 'Ship',fontsize=9, family='cmr10')

                        """if 'EUROSAT' in pic:
                            ax[pic_id][0].set_ylabel('EuroSAT/\n' + img_class.replace('_', ' '),fontsize=4, family='cmr10')
                        elif 'FMOW' in pic:
                            ax[pic_id][0].set_ylabel('FMOW/\n' + img_class.replace('_', ' '),fontsize=4, family='cmr10')
                        elif 'AID' in pic:
                            ax[pic_id][0].set_ylabel('AID/\n' + img_class.replace('_', ' '),fontsize=4, family='cmr10')
                        elif 'RESISC' in pic:
                            ax[pic_id][0].set_ylabel('RESISC/\n' + img_class.replace('_', ' '),fontsize=4, family='cmr10')"""

                        ax[pic_id][0].set_xticks([])
                        ax[pic_id][0].set_yticks([])
                        ax[pic_id][0].set_xticklabels([])
                        ax[pic_id][0].set_yticklabels([])

                    if pic_id == (model_id):
                        for axis_type in ['top','bottom','left','right']:
                            ax[pic_id][model_id+1].spines[axis_type].set_linewidth(3)
                            ax[pic_id][model_id+1].spines[axis_type].set_color('forestgreen')

            plt.tight_layout()
            plt.subplots_adjust(wspace=.001)
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_saliencymap' + '.pdf', bbox_inches='tight')

        if config.args.datasets:

            assert config.args.pipe_case == 'sat'
            assert config.args.domain == 'multi'

            # get paths of a random picture per class and per dataset
            eurosat_path = '../ai-in-the-loop/data/raw/EUROSAT'
            eurosat_classes = os.listdir(eurosat_path)
            fmow_path = '../ai-in-the-loop/data/processed/SAT_CASE_v2/FMOW/train'
            fmow_classes = os.listdir(fmow_path)
            aid_path = '../ai-in-the-loop/data/processed/SAT_CASE_v2/AID'
            aid_classes = os.listdir(aid_path)
            resisc_path = '../ai-in-the-loop/data/processed/SAT_CASE_v2/RESISC'
            resisc_classes = os.listdir(resisc_path)

            gen_dataset_samples = []
            exp1_dataset_samples = []
            exp2_dataset_samples = []
            exp3_dataset_samples = []


            for i in eurosat_classes:
                imgs = os.listdir(eurosat_path + '/' + i)
                rand_img_ind = random.randint(0, len(imgs))
                rand_img = imgs[rand_img_ind]
                rand_img = eurosat_path + '/' + i + '/' + rand_img
                gen_dataset_samples.append(rand_img)

            for i in fmow_classes:
                imgs = os.listdir(fmow_path + '/' + i)
                rand_img_ind = random.randint(0, len(imgs))
                rand_img = imgs[rand_img_ind]
                rand_img = fmow_path + '/' + i + '/' + rand_img
                exp1_dataset_samples.append(rand_img)

            for i in aid_classes:
                imgs = os.listdir(aid_path + '/' + i)
                rand_img_ind = random.randint(0, len(imgs))
                rand_img = imgs[rand_img_ind]
                rand_img = aid_path + '/' + i + '/' + rand_img
                exp2_dataset_samples.append(rand_img)

            for i in resisc_classes:
                imgs = os.listdir(resisc_path + '/' + i)
                rand_img_ind = random.randint(0, len(imgs))
                rand_img = imgs[rand_img_ind]
                rand_img = resisc_path + '/' + i + '/' + rand_img
                exp3_dataset_samples.append(rand_img)

            # EUROSAT -> 7

            nrow=5
            ncol=2

            fig, ax = plt.subplots(nrow, ncol,
                                   gridspec_kw=dict(wspace=0.0, hspace=0.0,
                                                    top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                                    left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)
                                                    ),
                                   figsize=(ncol + 1, nrow + 1),)
            counter = 0

            for i in range(len(ax[0])):
                for j in range(len(ax)):
                    if counter >= 7:
                        #ax[j][i].axis('off')
                        ax[j][i].set_visible(False)
                    else:
                        pic = gen_dataset_samples[counter]
                        print(pic)
                        img = Image.open(pic)
                        #img = mpimg.imread(pic)
                        ax[j][i].imshow(img)
                        ax[j][i].set_title(eurosat_classes[counter].replace('_', ' '), fontsize=5, family='cmr10', y=0.00, color='black', backgroundcolor= 'whitesmoke')
                        ax[j][i].set_xticks([])
                        ax[j][i].set_yticks([])
                        ax[j][i].set_xticklabels([])
                        ax[j][i].set_yticklabels([])

                        counter += 1

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_dataset_EUROSAT.pdf', bbox_inches='tight')


            # FMOW -> 35 classes
            nrow=5
            ncol=7
            fig, ax = plt.subplots(nrow, ncol,
                                   gridspec_kw=dict(wspace=0.0, hspace=0.0,
                                                    top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                                    left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
                                   figsize=(ncol + 1, nrow + 1),)
            counter = 0

            for i in range(len(ax[0])):
                for j in range(len(ax)):
                    if counter >= 35:
                        #ax[j][i].axis('off')
                        ax[j][i].set_visible(False)
                    else:
                        pic = exp1_dataset_samples[counter]
                        print(pic)
                        img = Image.open(pic)
                        #img = mpimg.imread(pic)
                        ax[j][i].imshow(img)
                        ax[j][i].set_title(fmow_classes[counter].replace('_', ' '), fontsize=5, family='cmr10', y=0.00, color='black', backgroundcolor= 'white')
                        ax[j][i].set_xticks([])
                        ax[j][i].set_yticks([])
                        ax[j][i].set_xticklabels([])
                        ax[j][i].set_yticklabels([])

                        counter += 1

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_dataset_FMOW.pdf', bbox_inches='tight')


            # AID -> 24
            nrow=4
            ncol=6
            fig, ax = plt.subplots(nrow, ncol,
                                   gridspec_kw=dict(wspace=0.0, hspace=0.0,
                                                    top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                                    left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
                                   figsize=(ncol + 1, nrow + 1),)
            counter = 0

            for i in range(len(ax[0])):
                for j in range(len(ax)):
                    if counter >= 24:
                        #ax[j][i].axis('off')
                        ax[j][i].set_visible(False)
                    else:
                        pic = exp2_dataset_samples[counter]
                        print(pic)
                        img = Image.open(pic)
                        #img = mpimg.imread(pic)
                        ax[j][i].imshow(img)
                        ax[j][i].set_title(aid_classes[counter].replace('_', ' '), fontsize=5, family='cmr10', y=0.00, color='black', backgroundcolor= 'white')
                        ax[j][i].set_xticks([])
                        ax[j][i].set_yticks([])
                        ax[j][i].set_xticklabels([])
                        ax[j][i].set_yticklabels([])

                        counter += 1

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_dataset_AID.pdf', bbox_inches='tight')


            # RESISC -> 12
            nrow=4
            ncol=3
            fig, ax = plt.subplots(nrow, ncol,
                                   gridspec_kw=dict(wspace=0.0, hspace=0.0,
                                                    top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                                    left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
                                   figsize=(ncol + 1, nrow + 1),)
            counter = 0

            for i in range(len(ax[0])):
                for j in range(len(ax)):
                    if counter >= 12:
                        #ax[j][i].axis('off')
                        ax[j][i].set_visible(False)
                    else:
                        pic = exp3_dataset_samples[counter]
                        print(pic)
                        img = Image.open(pic)
                        #img = mpimg.imread(pic)
                        ax[j][i].imshow(img)
                        ax[j][i].set_title(resisc_classes[counter].replace('_', ' '), fontsize=5, family='cmr10', y=0.00, color='black', backgroundcolor= 'white')
                        ax[j][i].set_xticks([])
                        ax[j][i].set_yticks([])
                        ax[j][i].set_xticklabels([])
                        ax[j][i].set_yticklabels([])

                        counter += 1

            #plt.tight_layout()
            #plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_dataset_RESISC.pdf', bbox_inches='tight')







    # development of utility for ablation study smaller_OOD_share
    elif config.args.ablation_study == 'smaller_OOD_share':

        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO{}.log'.format(config.args.ablation_study)
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs{}.log'.format(config.args.ablation_study)

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError


        if config.args.utility_graph:
            # utility graph maha/maha (no comparison)
            assert config.args.pipe_type == 'dynamic'
            assert config.args.domain == 'multi'

            print('')
            print('Utility plot is created for {} case and utility with params a = {}; b = {}.'.format(config.args.pipe_case, config.args.utility_a, config.args.utility_b))

            # collect metrics in numpy array
            LEN_OF_PLOT = config.args.independent_recurring_genAI_dyn_no_batches + (1 if config.args.incl_finalbatch else 0)
            print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
            print('')
            pipe_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha = np.zeros((LEN_OF_PLOT, len(IDs)))

            genAI_only = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_only_utility = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_HITL = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_HITL_human_effort = np.zeros((LEN_OF_PLOT, len(IDs)))
            genAI_HITL_utility = np.zeros((LEN_OF_PLOT, len(IDs)))

            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))
                if config.args.incl_finalbatch:
                    pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.pipe_performance']
                    human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.human_effort']
                    pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]
                else:
                    pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]

                if not config.args.incl_finalbatch:
                    genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_only']
                    genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                    genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_HITL']
                    genAI_HITL_human_effort[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                    genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]
                else:
                    genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.gen_AI_only']
                    genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                    genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.gen_AI_HITL']
                    genAI_HITL_human_effort[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                    genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]

            # average (does not have an effect for single ID consideration as intended by default)
            pipe_acc_utility_maha_avg = list(np.mean(pipe_acc_utility_maha, axis=1))
            pipe_acc_utility_maha_avg.reverse()

            genAI_only_utility_avg = list(np.mean(genAI_only_utility, axis=1))
            genAI_only_utility_avg.reverse()
            genAI_HITL_utility_avg = list(np.mean(genAI_HITL_utility, axis=1))
            genAI_HITL_utility_avg.reverse()

            # plot
            x = [i+1 for i in range(LEN_OF_PLOT)]
            plt.rcParams["font.family"] = "cmr10"
            plt.plot(x, pipe_acc_utility_maha_avg, label='33% OOD-share', color='royalblue')
            plt.plot(x, genAI_only_utility_avg, label='General model', color='black', linestyle='dashed')
            plt.plot(x, genAI_HITL_utility_avg, label='HITL-system', color='black', linestyle='solid')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Batch no.', fontsize=16)
            plt.ylabel(r'Utility ($\alpha$={}, $\beta$={})'.format(config.args.utility_a, config.args.utility_b), fontsize=16)
            #plt.title('Development of utility with a = {}; b = {} for {} domain'.format(config.args.utility_a, config.args.utility_b, config.args.domain))
            plt.legend(fontsize=16, loc=8,
                       #bbox_to_anchor=(1.6, 0)
                       )
            plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_utility_scores_{}_{}{}{}.pdf'.format(config.args.utility_a, config.args.utility_b, config.args.ablation_study, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')

        else:
            raise NotImplementedError


    # further ablation studies
    elif config.args.ablation_study == 'no_trust_early_stop' or config.args.ablation_study == 'trust_thresholds' or config.args.ablation_study == 'resnet18_backbone':

        # plots needed:
        # utility graph maha/maha compared to conventional maha/maha utility graph

        assert config.args.utility_graph or config.args.acc_cov_table
        assert config.args.pipe_type == 'dynamic'
        assert config.args.domain == 'multi'

        #### FIRST: collect old/conventional maha/maha data ####

        # collect metrics in numpy array
        # collect mlrunsIDs for averaging
        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO.log'
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs.log'

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError


        # get data
        LEN_OF_PLOT = config.args.dyn_no_batches + (1 if config.args.incl_finalbatch else 0)
        print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
        print('')
        pipe_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
        human_effort_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
        pipe_acc_utility_maha = np.zeros((LEN_OF_PLOT, len(IDs)))

        genAI_only = np.zeros((LEN_OF_PLOT, len(IDs)))
        genAI_only_utility = np.zeros((LEN_OF_PLOT, len(IDs)))
        genAI_HITL = np.zeros((LEN_OF_PLOT, len(IDs)))
        genAI_HITL_human_effort = np.zeros((LEN_OF_PLOT, len(IDs)))
        genAI_HITL_utility = np.zeros((LEN_OF_PLOT, len(IDs)))
        genAI_claimed_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp1_claimed_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp2_claimed_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp3_claimed_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        genAI_claimed_cov = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp1_claimed_cov = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp2_claimed_cov = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp3_claimed_cov = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp1_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp2_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp3_acc = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp1_trust_id = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp2_trust_id = np.zeros((LEN_OF_PLOT, len(IDs)))
        exp3_trust_id = np.zeros((LEN_OF_PLOT, len(IDs)))

        for i, id in enumerate(IDs):
            results = mlflow.search_runs(experiment_ids=str(id))
            if config.args.incl_finalbatch:
                pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.pipe_performance']
                human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.human_effort']
                pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]
            else:
                pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]

            if not config.args.incl_finalbatch:
                genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_only']
                genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_HITL']
                genAI_HITL_human_effort[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]
            else:
                genAI_only[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.gen_AI_only']
                genAI_only_utility[:, i] = (config.args.utility_a * genAI_only[:, i]).ravel()
                genAI_HITL[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('gating'))]['metrics.gen_AI_HITL']
                genAI_HITL_human_effort[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                genAI_HITL_utility[:, i] = config.args.utility_a * genAI_HITL[:, i] - config.args.utility_b * genAI_HITL_human_effort[:, i]


            if config.args.incl_finalbatch:
                if config.args.ablation_study == 'resnet18_backbone':
                    genAI_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.gen_AI_claimed_acc']
                    exp1_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_claimed_acc_exp_AI_1']
                    exp2_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_claimed_acc_exp_AI_2']
                    exp3_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_claimed_acc_exp_AI_3']
                    genAI_claimed_cov[:, i] = (results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.0/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.1/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.2/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.3/0']) / 7101
                    exp1_claimed_cov[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_coverage_exp_AI_1']
                    exp2_claimed_cov[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_coverage_exp_AI_2']
                    exp3_claimed_cov[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_coverage_exp_AI_3']
                if config.args.ablation_study == 'no_trust_early_stop':
                    exp1_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_test_acc_exp_AI_1']
                    exp2_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_test_acc_exp_AI_2']
                    exp3_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_test_acc_exp_AI_3']
                    exp1_trust_id[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_batchid_exp_AI_1']
                    exp2_trust_id[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_batchid_exp_AI_2']
                    exp3_trust_id[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_batchid_exp_AI_3']
            else:
                if config.args.ablation_study == 'resnet18_backbone':
                    genAI_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_claimed_acc']
                    exp1_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_claimed_acc_exp_AI_1']
                    exp2_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_claimed_acc_exp_AI_2']
                    exp3_claimed_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_claimed_acc_exp_AI_3']
                    genAI_claimed_cov[:, i] = (results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.0/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.1/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.2/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.3/0']) / 7101
                    exp1_claimed_cov[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_coverage_exp_AI_1']
                    exp2_claimed_cov[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_coverage_exp_AI_2']
                    exp3_claimed_cov[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_coverage_exp_AI_3']
                if config.args.ablation_study == 'no_trust_early_stop':
                    exp1_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_test_acc_exp_AI_1']
                    exp2_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_test_acc_exp_AI_2']
                    exp3_acc[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_test_acc_exp_AI_3']
                    exp1_trust_id[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_batchid_exp_AI_1']
                    exp2_trust_id[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_batchid_exp_AI_2']
                    exp3_trust_id[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_batchid_exp_AI_3']


        # average (does not have an effect for single ID consideration as intended by default)
        pipe_acc_utility_maha_avg = list(np.mean(pipe_acc_utility_maha, axis=1))
        pipe_acc_utility_maha_avg.reverse()
        genAI_only_utility_avg = list(np.mean(genAI_only_utility, axis=1))
        genAI_only_utility_avg.reverse()
        genAI_HITL_utility_avg = list(np.mean(genAI_HITL_utility, axis=1))
        genAI_HITL_utility_avg.reverse()
        if config.args.ablation_study == 'resnet18_backbone':
            genAI_claimed_acc = list(np.mean(genAI_claimed_acc, axis=1))
            genAI_claimed_acc.reverse()
            exp1_claimed_acc = list(np.mean(exp1_claimed_acc, axis=1))
            exp1_claimed_acc.reverse()
            exp2_claimed_acc = list(np.mean(exp2_claimed_acc, axis=1))
            exp2_claimed_acc.reverse()
            exp3_claimed_acc = list(np.mean(exp3_claimed_acc, axis=1))
            exp3_claimed_acc.reverse()
            genAI_claimed_cov = list(np.mean(genAI_claimed_cov, axis=1))
            genAI_claimed_cov.reverse()
            exp1_claimed_cov = list(np.mean(exp1_claimed_cov, axis=1))
            exp1_claimed_cov.reverse()
            exp2_claimed_cov = list(np.mean(exp2_claimed_cov, axis=1))
            exp2_claimed_cov.reverse()
            exp3_claimed_cov = list(np.mean(exp3_claimed_cov, axis=1))
            exp3_claimed_cov.reverse()
        if config.args.ablation_study == 'no_trust_early_stop':
            exp1_acc = list(np.mean(exp1_acc, axis=1))
            exp1_acc.reverse()
            exp2_acc = list(np.mean(exp2_acc, axis=1))
            exp2_acc.reverse()
            exp3_acc = list(np.mean(exp3_acc, axis=1))
            exp3_acc.reverse()
            exp1_trust_id = list(np.mean(exp1_trust_id, axis=1))
            exp1_trust_id.reverse()
            exp2_trust_id = list(np.mean(exp2_trust_id, axis=1))
            exp2_trust_id.reverse()
            exp3_trust_id = list(np.mean(exp3_trust_id, axis=1))
            exp3_trust_id.reverse()


        #### SECOND: COLLECT SAME DATA FOR ABLATION STUDY

        # collect metrics in numpy array
        # collect mlrunsIDs for averaging
        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO{}.log'.format(config.args.ablation_study)
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs{}.log'.format(config.args.ablation_study)

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError

        # get data
        LEN_OF_PLOT = config.args.dyn_no_batches + (1 if config.args.incl_finalbatch else 0)
        print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
        print('')

        if config.args.ablation_study == 'no_trust_early_stop' or config.args.ablation_study == 'resnet18_backbone':

            pipe_acc_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha = np.zeros((LEN_OF_PLOT, len(IDs)))
            if config.args.ablation_study == 'resnet18_backbone':
                genAI_only_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                genAI_only_utility_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                genAI_HITL_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                genAI_HITL_human_effort_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                genAI_HITL_utility_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                genAI_claimed_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp1_claimed_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp2_claimed_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp3_claimed_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                genAI_claimed_cov_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp1_claimed_cov_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp2_claimed_cov_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp3_claimed_cov_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
            if config.args.ablation_study == 'no_trust_early_stop':
                exp1_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp2_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp3_acc_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp1_trust_id_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp2_trust_id_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))
                exp3_trust_id_ABLATION = np.zeros((LEN_OF_PLOT, len(IDs)))

            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))
                if config.args.incl_finalbatch:
                    pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.pipe_performance']
                    human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.human_effort']
                    pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]
                    if config.args.ablation_study == 'resnet18_backbone':
                        genAI_only_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.gen_AI_only']
                        genAI_only_utility_ABLATION[:, i] = (config.args.utility_a * genAI_only_ABLATION[:, i]).ravel()
                        genAI_HITL_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.gen_AI_HITL']
                        genAI_HITL_human_effort_ABLATION[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                        genAI_HITL_utility_ABLATION[:, i] = config.args.utility_a * genAI_HITL_ABLATION[:, i] - config.args.utility_b * genAI_HITL_human_effort_ABLATION[:, i]
                        genAI_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.gen_AI_claimed_acc']
                        exp1_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_claimed_acc_exp_AI_1']
                        exp2_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_claimed_acc_exp_AI_2']
                        exp3_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_claimed_acc_exp_AI_3']
                        genAI_claimed_cov_ABLATION[:, i] = (results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.0/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.1/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.2/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.3/0']) / 7101
                        exp1_claimed_cov_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_coverage_exp_AI_1']
                        exp2_claimed_cov_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_coverage_exp_AI_2']
                        exp3_claimed_cov_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_coverage_exp_AI_3']
                    if config.args.ablation_study == 'no_trust_early_stop':
                        exp1_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_test_acc_exp_AI_1']
                        exp2_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_test_acc_exp_AI_2']
                        exp3_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.exp_test_acc_exp_AI_3']
                        exp1_trust_id_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_batchid_exp_AI_1']
                        exp2_trust_id_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_batchid_exp_AI_2']
                        exp3_trust_id_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha'))]['metrics.trust_batchid_exp_AI_3']
                else:
                    pipe_acc_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha[:, i] = config.args.utility_a * pipe_acc_maha[:, i] - config.args.utility_b * human_effort_maha[:, i]
                    if config.args.ablation_study == 'resnet18_backbone':
                        genAI_only_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_only']
                        genAI_only_utility_ABLATION[:, i] = (config.args.utility_a * genAI_only_ABLATION[:, i]).ravel()
                        genAI_HITL_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_HITL']
                        genAI_HITL_human_effort_ABLATION[:, i] = ([(7101-161)/(7101)] * LEN_OF_PLOT) if config.args.pipe_case == 'benchmark' else ([(2169-61)/(2169)] * LEN_OF_PLOT)
                        genAI_HITL_utility_ABLATION[:, i] = config.args.utility_a * genAI_HITL_ABLATION[:, i] - config.args.utility_b * genAI_HITL_human_effort_ABLATION[:, i]
                        genAI_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.gen_AI_claimed_acc']
                        exp1_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_claimed_acc_exp_AI_1']
                        exp2_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_claimed_acc_exp_AI_2']
                        exp3_claimed_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_claimed_acc_exp_AI_3']
                        genAI_claimed_cov_ABLATION[:, i] = (results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.0/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.1/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.2/0'] + results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.3/0']) / 7101
                        exp1_claimed_cov_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_coverage_exp_AI_1']
                        exp2_claimed_cov_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_coverage_exp_AI_2']
                        exp3_claimed_cov_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_coverage_exp_AI_3']
                    if config.args.ablation_study == 'no_trust_early_stop':
                        exp1_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_test_acc_exp_AI_1']
                        exp2_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_test_acc_exp_AI_2']
                        exp3_acc_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.exp_test_acc_exp_AI_3']
                        exp1_trust_id_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_batchid_exp_AI_1']
                        exp2_trust_id_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_batchid_exp_AI_2']
                        exp3_trust_id_ABLATION[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.trust_batchid_exp_AI_3']

            # average (does not have an effect for single ID consideration as intended by default)
            pipe_acc_utility_maha_avg_ABLATION = list(np.mean(pipe_acc_utility_maha, axis=1))
            pipe_acc_utility_maha_avg_ABLATION.reverse()
            if config.args.ablation_study == 'resnet18_backbone':
                genAI_only_utility_avg_ABLATION = list(np.mean(genAI_only_utility_ABLATION, axis=1))
                genAI_only_utility_avg_ABLATION.reverse()
                genAI_HITL_utility_avg_ABLATION = list(np.mean(genAI_HITL_utility_ABLATION, axis=1))
                genAI_HITL_utility_avg_ABLATION.reverse()
                genAI_claimed_acc_ABLATION = list(np.mean(genAI_claimed_acc_ABLATION, axis=1))
                genAI_claimed_acc_ABLATION.reverse()
                exp1_claimed_acc_ABLATION = list(np.mean(exp1_claimed_acc_ABLATION, axis=1))
                exp1_claimed_acc_ABLATION.reverse()
                exp2_claimed_acc_ABLATION = list(np.mean(exp2_claimed_acc_ABLATION, axis=1))
                exp2_claimed_acc_ABLATION.reverse()
                exp3_claimed_acc_ABLATION = list(np.mean(exp3_claimed_acc_ABLATION, axis=1))
                exp3_claimed_acc_ABLATION.reverse()
                genAI_claimed_cov_ABLATION = list(np.mean(genAI_claimed_cov_ABLATION, axis=1))
                genAI_claimed_cov_ABLATION.reverse()
                exp1_claimed_cov_ABLATION = list(np.mean(exp1_claimed_cov_ABLATION, axis=1))
                exp1_claimed_cov_ABLATION.reverse()
                exp2_claimed_cov_ABLATION = list(np.mean(exp2_claimed_cov_ABLATION, axis=1))
                exp2_claimed_cov_ABLATION.reverse()
                exp3_claimed_cov_ABLATION = list(np.mean(exp3_claimed_cov_ABLATION, axis=1))
                exp3_claimed_cov_ABLATION.reverse()
            if config.args.ablation_study == 'no_trust_early_stop':
                exp1_acc_ABLATION = list(np.mean(exp1_acc_ABLATION, axis=1))
                exp1_acc_ABLATION.reverse()
                exp2_acc_ABLATION = list(np.mean(exp2_acc_ABLATION, axis=1))
                exp2_acc_ABLATION.reverse()
                exp3_acc_ABLATION = list(np.mean(exp3_acc_ABLATION, axis=1))
                exp3_acc_ABLATION.reverse()
                exp1_trust_id_ABLATION = list(np.mean(exp1_trust_id_ABLATION, axis=1))
                exp1_trust_id_ABLATION.reverse()
                exp2_trust_id_ABLATION = list(np.mean(exp2_trust_id_ABLATION, axis=1))
                exp2_trust_id_ABLATION.reverse()
                exp3_trust_id_ABLATION = list(np.mean(exp3_trust_id_ABLATION, axis=1))
                exp3_trust_id_ABLATION.reverse()


            # concat as of batch no. that returns a trusted expert based on early stopping (manually looked up)
            concat = True
            concat_batch_no = 9
            if concat and config.args.ablation_study == 'no_trust_early_stop':
                pipe_acc_utility_maha_avg_ABLATION = pipe_acc_utility_maha_avg[:(concat_batch_no-1)] + pipe_acc_utility_maha_avg_ABLATION[(concat_batch_no-1):]


            if config.args.utility_graph:
                # plot
                x = [i+1 for i in range(LEN_OF_PLOT)]
                plt.rcParams["font.family"] = "cmr10"
                if config.args.ablation_study == 'no_trust_early_stop':
                    plt.plot(x, pipe_acc_utility_maha_avg_ABLATION, label='Excl. early stopping', color=colors[0], marker='o' , linewidth=2)
                    plt.plot(x, pipe_acc_utility_maha_avg, label='Incl. early stopping', color=colors[1], marker='^', linewidth=2)
                else:
                    plt.plot(x, pipe_acc_utility_maha_avg_ABLATION, label='ResNet-18', color=colors[0], marker='o' , linewidth=2)
                    plt.plot(x, pipe_acc_utility_maha_avg, label='Wide-ResNet-28-10', color=colors[1], marker='^' , linewidth=2)
                plt.plot(x, genAI_HITL_utility_avg, label='HITL-system', color='black', linestyle='solid')
                plt.plot(x, genAI_only_utility_avg, label='General model', color='black', linestyle='dashed')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel('Batch no.', fontsize=16)
                plt.ylabel(r'Utility ($\alpha$={}, $\beta$={})'.format(config.args.utility_a, config.args.utility_b), fontsize=16)
                plt.legend(fontsize=16, loc=8, ncol=2,
                           bbox_to_anchor=(0.5, -0.5)
                           )
                plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
                plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_utility_scores_{}_{}{}{}.pdf'.format(config.args.utility_a, config.args.utility_b, config.args.ablation_study, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')

            # supplementary information
            elif config.args.acc_cov_table:

                if config.args.ablation_study == 'resnet18_backbone':
                    final_data = np.zeros((4,4))
                    final_data[0,0] = genAI_claimed_acc_ABLATION[-1]
                    final_data[1,0] = exp1_claimed_acc_ABLATION[-1]
                    final_data[2,0] = exp2_claimed_acc_ABLATION[-1]
                    final_data[3,0] = exp3_claimed_acc_ABLATION[-1]
                    final_data[0,1] = genAI_claimed_cov_ABLATION[-1]
                    final_data[1,1] = exp1_claimed_cov_ABLATION[-1]
                    final_data[2,1] = exp2_claimed_cov_ABLATION[-1]
                    final_data[3,1] = exp3_claimed_cov_ABLATION[-1]
                    final_data[0,2] = genAI_claimed_acc[-1]
                    final_data[1,2] = exp1_claimed_acc[-1]
                    final_data[2,2] = exp2_claimed_acc[-1]
                    final_data[3,2] = exp3_claimed_acc[-1]
                    final_data[0,3] = genAI_claimed_cov[-1]
                    final_data[1,3] = exp1_claimed_cov[-1]
                    final_data[2,3] = exp2_claimed_cov[-1]
                    final_data[3,3] = exp3_claimed_cov[-1]

                    output = pd.DataFrame(final_data, index=['CIFAR-10', 'SVHN (1)', 'MNIST(2)', 'Fashion-MNIST (3)'], columns=['Accuracy_RN-18', 'Coverage_RN-18', 'Accuracy_WRN-28-10', 'Coverage_WRN-28-10'])

                    output = output.round(2)
                    print(output)
                    print('')
                    print('')
                    print(output.to_latex(bold_rows=True,
                                          multicolumn_format='c',
                                          caption='TBD',
                                          label='TBD'))


                elif config.args.ablation_study == 'no_trust_early_stop':
                    final_data = np.zeros((3,4))
                    final_data[0,0] = exp1_trust_id_ABLATION[-1]
                    final_data[1,0] = exp2_trust_id_ABLATION[-1]
                    final_data[2,0] = exp3_trust_id_ABLATION[-1]
                    final_data[0,1] = exp1_acc_ABLATION[-1]
                    final_data[1,1] = exp2_acc_ABLATION[-1]
                    final_data[2,1] = exp3_acc_ABLATION[-1]

                    final_data[0,2] = exp1_trust_id[-1]
                    final_data[1,2] = exp2_trust_id[-1]
                    final_data[2,2] = exp3_trust_id[-1]
                    final_data[0,3] = exp1_acc[-1]
                    final_data[1,3] = exp2_acc[-1]
                    final_data[2,3] = exp3_acc[-1]                    

                    output = pd.DataFrame(final_data, index=['SVHN (1)', 'MNIST(2)', 'Fashion-MNIST (3)'], columns=['Trust batch no. / excl.', 'acc / excl.', 'Trust batch no. / incl.', 'acc / incl.'])

                    output = output.round(2)
                    print(output)
                    print('')
                    print('')
                    print(output.to_latex(bold_rows=True,
                                          multicolumn_format='c',
                                          caption='TBD',
                                          label='TBD'))

            else:
                raise NotImplementedError


        # further ablation study
        elif config.args.ablation_study == 'trust_thresholds':

            # intialize
            pipe_acc_maha_7 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_7 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_7 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_75 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_75 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_75 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_8 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_8 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_8 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_85 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_85 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_85 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_9 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_9 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_9 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_91 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_91 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_91 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_92 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_92 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_92 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_93 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_93 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_93 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_maha_94 = np.zeros((LEN_OF_PLOT, len(IDs)))
            human_effort_maha_94 = np.zeros((LEN_OF_PLOT, len(IDs)))
            pipe_acc_utility_maha_94 = np.zeros((LEN_OF_PLOT, len(IDs)))

            # get data
            for i, id in enumerate(IDs):
                results = mlflow.search_runs(experiment_ids=str(id))
                if config.args.incl_finalbatch:
                    pipe_acc_maha_7[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.7/'))]['metrics.pipe_performance']
                    human_effort_maha_7[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.7/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_7[:, i] = config.args.utility_a * pipe_acc_maha_7[:, i] - config.args.utility_b * human_effort_maha_7[:, i]
                    pipe_acc_maha_75[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.75/'))]['metrics.pipe_performance']
                    human_effort_maha_75[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.75/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_75[:, i] = config.args.utility_a * pipe_acc_maha_75[:, i] - config.args.utility_b * human_effort_maha_75[:, i]
                    pipe_acc_maha_8[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.8/'))]['metrics.pipe_performance']
                    human_effort_maha_8[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.8/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_8[:, i] = config.args.utility_a * pipe_acc_maha_8[:, i] - config.args.utility_b * human_effort_maha_8[:, i]
                    pipe_acc_maha_85[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.85/'))]['metrics.pipe_performance']
                    human_effort_maha_85[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.85/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_85[:, i] = config.args.utility_a * pipe_acc_maha_85[:, i] - config.args.utility_b * human_effort_maha_85[:, i]
                    pipe_acc_maha_9[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.9/'))]['metrics.pipe_performance']
                    human_effort_maha_9[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.9/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_9[:, i] = config.args.utility_a * pipe_acc_maha_9[:, i] - config.args.utility_b * human_effort_maha_9[:, i]
                    pipe_acc_maha_91[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.91/'))]['metrics.pipe_performance']
                    human_effort_maha_91[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.91/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_91[:, i] = config.args.utility_a * pipe_acc_maha_91[:, i] - config.args.utility_b * human_effort_maha_91[:, i]
                    pipe_acc_maha_92[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.92/'))]['metrics.pipe_performance']
                    human_effort_maha_92[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.92/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_92[:, i] = config.args.utility_a * pipe_acc_maha_92[:, i] - config.args.utility_b * human_effort_maha_92[:, i]
                    pipe_acc_maha_93[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.93/'))]['metrics.pipe_performance']
                    human_effort_maha_93[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.93/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_93[:, i] = config.args.utility_a * pipe_acc_maha_93[:, i] - config.args.utility_b * human_effort_maha_93[:, i]
                    pipe_acc_maha_94[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.94/'))]['metrics.pipe_performance']
                    human_effort_maha_94[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.94/'))]['metrics.human_effort']
                    pipe_acc_utility_maha_94[:, i] = config.args.utility_a * pipe_acc_maha_94[:, i] - config.args.utility_b * human_effort_maha_94[:, i]
                else:
                    pipe_acc_maha_7[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.7/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_7[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.7/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_7[:, i] = config.args.utility_a * pipe_acc_maha_7[:, i] - config.args.utility_b * human_effort_maha_7[:, i]
                    pipe_acc_maha_75[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.75/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_75[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.75/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_75[:, i] = config.args.utility_a * pipe_acc_maha_75[:, i] - config.args.utility_b * human_effort_maha_75[:, i]
                    pipe_acc_maha_8[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.8/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_8[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.8/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_8[:, i] = config.args.utility_a * pipe_acc_maha_8[:, i] - config.args.utility_b * human_effort_maha_8[:, i]
                    pipe_acc_maha_85[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.85/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_85[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.85/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_85[:, i] = config.args.utility_a * pipe_acc_maha_85[:, i] - config.args.utility_b * human_effort_maha_85[:, i]
                    pipe_acc_maha_9[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.9/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_9[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.9/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_9[:, i] = config.args.utility_a * pipe_acc_maha_9[:, i] - config.args.utility_b * human_effort_maha_9[:, i]
                    pipe_acc_maha_91[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.91/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_91[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.91/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_91[:, i] = config.args.utility_a * pipe_acc_maha_91[:, i] - config.args.utility_b * human_effort_maha_91[:, i]
                    pipe_acc_maha_92[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.92/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_92[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.92/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_92[:, i] = config.args.utility_a * pipe_acc_maha_92[:, i] - config.args.utility_b * human_effort_maha_92[:, i]
                    pipe_acc_maha_93[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.93/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_93[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.93/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_93[:, i] = config.args.utility_a * pipe_acc_maha_93[:, i] - config.args.utility_b * human_effort_maha_93[:, i]
                    pipe_acc_maha_94[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.94/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.pipe_performance']
                    human_effort_maha_94[:, i] = results[(~results['tags.mlflow.runName'].str.startswith('PARAMS')) & (results['tags.mlflow.runName'].str.contains('maha')) & (results['tags.mlflow.runName'].str.contains('/0.94/')) & (~results['tags.mlflow.runName'].str.contains('_' + str((config.args.dyn_no_batches+1)) + '_'))]['metrics.human_effort']
                    pipe_acc_utility_maha_94[:, i] = config.args.utility_a * pipe_acc_maha_94[:, i] - config.args.utility_b * human_effort_maha_94[:, i]


            # average (does not have an effect for single ID consideration as intended by default)
            pipe_acc_utility_maha_avg_ABLATION_7 = list(np.mean(pipe_acc_utility_maha_7, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_7.reverse()
            pipe_acc_utility_maha_avg_ABLATION_75 = list(np.mean(pipe_acc_utility_maha_75, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_75.reverse()
            pipe_acc_utility_maha_avg_ABLATION_8 = list(np.mean(pipe_acc_utility_maha_8, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_8.reverse()
            pipe_acc_utility_maha_avg_ABLATION_85 = list(np.mean(pipe_acc_utility_maha_85, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_85.reverse()
            pipe_acc_utility_maha_avg_ABLATION_9 = list(np.mean(pipe_acc_utility_maha_9, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_9.reverse()
            pipe_acc_utility_maha_avg_ABLATION_91 = list(np.mean(pipe_acc_utility_maha_91, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_91.reverse()
            pipe_acc_utility_maha_avg_ABLATION_92 = list(np.mean(pipe_acc_utility_maha_92, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_92.reverse()
            pipe_acc_utility_maha_avg_ABLATION_93 = list(np.mean(pipe_acc_utility_maha_93, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_93.reverse()
            pipe_acc_utility_maha_avg_ABLATION_94 = list(np.mean(pipe_acc_utility_maha_94, axis=1))
            pipe_acc_utility_maha_avg_ABLATION_94.reverse()

            # plot
            x = [i+1 for i in range(LEN_OF_PLOT)]
            plt.rcParams["font.family"] = "cmr10"
            #plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_7, label='0.7', color=colors[0], linestyle='dashed', dashes=(7.5, 1), linewidth=2)
            #plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_75, label='0.75', color=colors[1], linestyle='dashed', dashes=(5, 1), linewidth=2)
            #plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_8, label='0.8', color=colors[2], linestyle='dashed', dashes=(2.5, 1), linewidth=2)
            #plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_85, label='0.85', color=colors[3], linestyle='dotted', linewidth=2)
            plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_9, label='0.90', color=colors[0], linestyle='dashed', dashes=(7.5, 1), linewidth=2)
            plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_91, label='0.91', color=colors[1], linestyle='dashed', dashes=(5, 1), linewidth=2)
            plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_92, label='0.92', color=colors[2], linestyle='dashed', dashes=(2.5, 1), linewidth=2)
            plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_93, label='0.93', color=colors[3], linestyle='dotted', linewidth=2)
            plt.plot(x, pipe_acc_utility_maha_avg_ABLATION_94, label='0.94', color=colors[4], linestyle='dashdot', linewidth=2)
            plt.plot(x, pipe_acc_utility_maha_avg, label='0.95 (default)', color=colors[5], linestyle='solid', linewidth=2)
            #plt.plot(x, genAI_only_utility_avg, label='Gen. AI only', color='black', linestyle='--')
            #plt.plot(x, genAI_HITL_utility_avg, label='Gen. AI + HITL', color='black', )
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Batch no.', fontsize=16)
            plt.ylabel(r'Utility ($\alpha$={}, $\beta$={})'.format(config.args.utility_a, config.args.utility_b), fontsize=16)
            #plt.title('Development of utility with a = {}; b = {} for {} domain'.format(config.args.utility_a, config.args.utility_b, config.args.domain))
            plt.legend(fontsize=16, loc=8, ncol=3,
                       bbox_to_anchor=(0.5, -0.5)
                       )
            plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
            plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_utility_scores_{}_{}{}{}.pdf'.format(config.args.utility_a, config.args.utility_b, config.args.ablation_study, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')

        else:
            raise NotImplementedError


    # study on impact of unknown data on uncertainty
    elif config.args.ablation_study == 'unc_vs_OOD_share':

        assert config.args.pipe_case == 'benchmark'
        assert config.args.pipe_type == 'static'
        assert config.args.domain == 'multi'

        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO{}.log'.format(config.args.ablation_study)
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs{}.log'.format(config.args.ablation_study)

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError

        print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
        print('')

        # get data
        svhn_confidences = np.zeros((101, len(IDs)))
        mnist_confidences = np.zeros((101, len(IDs)))
        fmnist_confidences = np.zeros((101, len(IDs)))
        cifar100_confidences = np.zeros((101, len(IDs)))

        for i, id in enumerate(IDs):
            results = mlflow.search_runs(experiment_ids=str(id))
            for k in range(4):  # all 4 OOD sets
                for j in range(101):
                    ood_share = round(j/100, 2)
                    if k == 0:
                        svhn_confidences[j, i] = results['metrics.{}_ood_share_'.format(k) + str(ood_share)]
                    elif k == 1:
                        mnist_confidences[j, i] = results['metrics.{}_ood_share_'.format(k) + str(ood_share)]
                    elif k == 2:
                        fmnist_confidences[j, i] = results['metrics.{}_ood_share_'.format(k) + str(ood_share)]
                    elif k == 3:
                        cifar100_confidences[j, i] = results['metrics.{}_ood_share_'.format(k) + str(ood_share)]

        svhn_confidences = list(np.mean(svhn_confidences, axis=1))
        mnist_confidences = list(np.mean(mnist_confidences, axis=1))
        fmnist_confidences = list(np.mean(fmnist_confidences, axis=1))
        cifar100_confidences = list(np.mean(cifar100_confidences, axis=1))

        # check whether everythin is collected
        assert len(svhn_confidences) == 101
        assert len(mnist_confidences) == 101
        assert len(fmnist_confidences) == 101
        assert len(cifar100_confidences) == 101

        # plot
        x = [i for i in range(len(svhn_confidences))]
        plt.rcParams["font.family"] = "cmr10"
        plt.plot(x, cifar100_confidences, label='CIFAR-100', color=colors[0], linestyle='solid', linewidth=2)
        plt.plot(x, mnist_confidences, label='MNIST', color=colors[1], linestyle='dashed', linewidth=2)
        plt.plot(x, fmnist_confidences, label='Fashion-MNIST', color=colors[2], linestyle='dotted', linewidth=2)
        plt.plot(x, svhn_confidences, label='SVHN', color=colors[3], linestyle='dashdot', linewidth=2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Share of unknown data [%]', fontsize=16)
        plt.rcParams["font.family"] = "cmr10"
        plt.ylabel('Confidence (Softmax)', fontsize=16)
        #plt.ylim(0, 1)
        #plt.title('Model confidence depending on share of unknown data')
        plt.legend(fontsize=16, loc=8, ncol=4,
                   bbox_to_anchor=(0.5, -0.4)
                   )
        plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
        plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + 'OOD_share_analysis{}.pdf'.format(('REPRO' if config.args.results_repro else '')), bbox_inches='tight')


    # study on overfitting of FMOW model
    elif config.args.ablation_study == 'fmow_overfit':

        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO{}.log'.format(config.args.ablation_study)
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs{}.log'.format(config.args.ablation_study)

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError

        print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
        print('')

        assert len(IDs) == 1
        id = IDs[0]
        results = mlflow.search_runs(experiment_ids=str(id))

        # get data
        vallosses = list(results['metrics.valloss'])
        valacc = list(results['metrics.valacc'])
        trainacc = list(results['metrics.trainacc'])
        trainlosses = list(results['metrics.trainloss'])

        vallosses.reverse()
        valacc.reverse()
        trainacc.reverse()
        trainlosses.reverse()

        # plot
        x = [i+1 for i in range(len(vallosses))]
        plt.rcParams["font.family"] = "cmr10"
        if config.args.fmow_overfit_type == 'acc':
            plt.plot(x, trainacc, label='Training', color=colors[0], linestyle='solid', marker='^' , linewidth=2)
            plt.plot(x, valacc, label='Validation', color=colors[1], linestyle='solid', marker='o' , linewidth=2)
            plt.ylabel('Accuracy', fontsize=16)
        else:
            plt.plot(x, trainlosses, label='Training', color=colors[0], linestyle='solid', marker='^' , linewidth=2)
            plt.plot(x, vallosses, label='Validation', color=colors[1], linestyle='solid', marker='o' , linewidth=2)
            plt.ylabel('Loss', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Epochs', fontsize=16)
        #plt.legend(fontsize=16, loc=8, ncol=3,
         #          bbox_to_anchor=(0.5, -1)
          #         )
        plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
        plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + 'fMoW_Overfit_{}{}.pdf'.format(config.args.fmow_overfit_type, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')


    # ablation study of smaller ODO share on static system
    elif config.args.ablation_study == 'smaller_OOD_share_v2':

        if config.args.results_repro:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs_REPRO{}.log'.format(config.args.ablation_study)
        else:
            mlrunsID_log_file = '../ai-in-the-loop/results/' + case_domain_pipe + '/overview/' + case_domain_pipe + '_mlrunsIDs{}.log'.format(config.args.ablation_study)

        # select ID
        if os.path.isfile(mlrunsID_log_file):
            with open(mlrunsID_log_file) as f:
                IDs = [int(x) for x in f]
            # if no mlruns_ID provided via args, use first ID as default (default results)
            if config.args.mlruns_ID == 0:
                IDs = [IDs[0]]
            # if ID specified via args -> use this ID
            else:
                IDs = [config.args.mlruns_ID]
        else:
            print('ERROR: no runs available. Please make sure, all runs are done and saved to the correct directory.')
            raise NotImplementedError

        print('Results are averaged over ' + str(len(IDs)) + ' seeds!')
        print('')

        assert len(IDs) == 1
        id = IDs[0]

        # get data
        results = mlflow.search_runs(experiment_ids=str(id))
        accuracy = list(results['metrics.pipe_acc_wH'])
        human_effort = list(results['metrics.human_effort'])
        hitl_accuracy = list(results['metrics.hitl_UB'])
        hitl_human_effort = list(results['metrics.hitl_human_effort'])
        svhn_rate = list(results['metrics.false_svhn'])
        mnist_rate = list(results['metrics.false_mnist'])
        fmnist_rate = list(results['metrics.false_fashionmnist'])
        svhn_rate.reverse()
        mnist_rate.reverse()
        fmnist_rate.reverse()

        odin_rate = []
        for i in range(len(fmnist_rate)):
            odin_rate.append((1-svhn_rate[i] + 1-mnist_rate[i] + 1-fmnist_rate[i])/3)

        accuracy.reverse()
        accuracy = np.array(accuracy)
        human_effort.reverse()
        human_effort = np.array(human_effort)
        hitl_accuracy.reverse()
        hitl_accuracy = np.array(hitl_accuracy)
        hitl_human_effort.reverse()
        hitl_human_effort = np.array(hitl_human_effort)

        aiitl_utility = config.args.utility_a * accuracy - config.args.utility_b * human_effort
        hitl_utility = config.args.utility_a * hitl_accuracy - config.args.utility_b * hitl_human_effort

        # plot
        x = [i for i in range(101) if i % 5 == 0]
        plt.rcParams["font.family"] = "cmr10"
        if config.args.smaller_OOD_plot == 'system':
            plt.plot(x, aiitl_utility, label='AIITL-system: Maha', color=colors[0], linestyle='solid', marker='o', linewidth=2)
            plt.plot(x, hitl_utility, label='HITL-system', color='black', linestyle='solid', linewidth=2)
            if config.args.utility_b == 99:
                plt.plot(x, hitl_utility, label='ODIN rejection rate', color=colors[1], linestyle='solid', marker='^', linewidth=2)
            plt.ylabel(r'Utility ($\alpha$={}, $\beta$={})'.format(config.args.utility_a, config.args.utility_b), fontsize=16)
        else:
            plt.plot(x, odin_rate, color=colors[1], linestyle='solid', marker='^', linewidth=2)
            #plt.ylim(0.5, 1)
            plt.ylabel('ODIN rejection rate', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Share of unknown data [%]', fontsize=16)
        if config.args.utility_b == 99:
            plt.legend(fontsize=16, loc=8, ncol=3,
                       bbox_to_anchor=(0.5, -1)
                       )
        plt.gcf().set_size_inches(config.args.fig_size[0], config.args.fig_size[1])
        plt.savefig(config.args.pipe_root + 'results/' + case_domain_pipe + '/logs/' + case_domain_pipe + '_{}_{}{}.pdf'.format(config.args.ablation_study, config.args.smaller_OOD_plot, ('REPRO' if config.args.results_repro else '')), bbox_inches='tight')

    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
