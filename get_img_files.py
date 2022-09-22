import src.utils.pipe as pipe
import src.utils.model_building as mb
import src.config
import src.utils.utils as utils
import src.utils.thresholds as th
import src.utils.mahalanobis as maha
import src.utils.odin as odin
import time
import src.data.make_data as md
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
import mlflow

"""
This script is intended to provide the image files of the correctly predicted images in the satellite case for batch 31 (static batch)
When the dynamic models are needed, set pipe_type to "dynamic"
"""

def main():

    # get config
    config = src.config.cfg

    # enforce reproducibility
    main_seed = config.args.main_seed
    print('seed in script (for control): ' + str(main_seed))
    utils.set_seed(main_seed)

    """# Build, train, save and test models (if already saved, models are loaded)
    gen_AI = mb.get_general_model()
    experts = []
    if config.args.pipe_type == 'dynamic':

        config.args.dyn_pipe = True
        print('exp1 is loaded')
        exp_AI_1 = utils.load_gpu_pickle(config.args.pipe_root + 'results/sat_multi_dynamic/models/exp_AI_1_dyn_mahamaha_lastRun.pickle')
        print('exp2 is loaded')
        exp_AI_2 = utils.load_gpu_pickle(config.args.pipe_root + 'results/sat_multi_dynamic/models/exp_AI_2_dyn_mahamaha_lastRun.pickle')
        print('exp3 is loaded')
        exp_AI_3 = utils.load_gpu_pickle(config.args.pipe_root + 'results/sat_multi_dynamic/models/exp_AI_3_dyn_mahamaha_lastRun.pickle')

    else:
        exp_AI_1, exp_AI_2, exp_AI_3 = mb.get_expert_models()
    experts = [exp_AI_1, exp_AI_2, exp_AI_3]


    ##### RUN PIPELINE #####
    if config.args.domain == 'multi':

        # set size and noise to default, if not specified. The default size and noise refer to the data batch design of the Master's Thesis
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
            elif config.args.dyn_batch_design == 'independent_recurring_genAI':
                no_batches = config.args.independent_recurring_genAI_dyn_no_batches
                if config.args.pipe_case == 'benchmark':
                    config.args.pipe_size = int(sum([int(i/no_batches) for i in [78257, 65000, 65000]])/config.args.smaller_OOD_share)
                    assert config.args.pipe_size <= 5000
                    config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [78257, 65000, 65000]]
                elif config.args.pipe_case == 'sat':
                    config.args.pipe_size = int(sum([int(i/no_batches) for i in [48565, 7164, 7560]])/config.args.smaller_OOD_share)
                    assert config.args.pipe_size <= 1900
                    config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [48565, 7164, 7560]]

        # borders are the last index borders of the general ML model and the artificial experts
        borders = [config.args.pipe_size]
        for noise in config.args.pipe_noise:
            borders.append(int(borders[-1] + noise*config.args.pipe_size))

        # run system
        print('Static pipeline (multi-domain) started:')
        print('Case: ' + config.args.pipe_case)
        print('Tuning dataset: ' + config.args.pipe_tune_dataset)
        print('Selec-mech: ' + config.args.selec_mech)
        print('Comb-mech: ' + config.args.comb_mech)
        print('Unc-score: ' + config.args.unc_score)
        start = time.time()


        step1_perf, _, _, pipe_perf_wH, pipe_perf_woH, matrix, human_effort, HITL_base_acc, _, exp_claimed_acc, _, true_inds, predloader, forwardloader = pipe.run_pipeline(gen_AI, experts)


        end = time.time()
        dur = end - start

        # results
        print('Performance including/without human labeled instances:')
        print('Gen AI acc. on claimed instances: ' + str(step1_perf) + '/..')
        for i in range(len(experts)):
            print('Exp_AI_' + str(i+1) + ' claimed acc: ' + str(exp_claimed_acc['exp_AI_' + str(i+1) + '_claimed_acc']) + '/..')
        print('')
        print('pipe_acc: ' + str(pipe_perf_wH) + '/' + str(pipe_perf_woH))
        print('')
        print('Human effort (pct. of total instances in pipeline): ' + str(human_effort['#_inst']/(borders[-1])))
        print('')
        print('Static pipeline finished.')

        print('')
        print('')
        print('no of true inds:' + str(len(true_inds)))
        print('')

        pipeloader_after_pipe = DataLoader(ConcatDataset([predloader.dataset, forwardloader.dataset]), batch_size=1, shuffle=False)"""


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


    # print/write all image files, which are predicted (and thus, allocated) correctly
    myfile = open('../ai-in-the-loop/results/sat_multi_dynamic/logs/correct_files.txt', 'w')
    for i in range(len(pipeloader)):
        img = utils.get_original_idx(pipeloader.dataset, i)[3]
        print(img)
        myfile.write(img + '\n')
    myfile.close()

if __name__ == "__main__":
    main()