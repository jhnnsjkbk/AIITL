# this code is intended to run the entire results of the dynamic systems

# all libraries
import src.utils.pipe as pipe
import src.config
import src.utils.utils as utils
import src.utils.odin as odin
from datetime import datetime
import time
import mlflow
import src.utils.model_building as mb


def main():

    ##### INITIALIZE #####
    print('Initialize models..')

    # get config
    config = src.config.cfg
    config.args.dyn_pipe = False

    # some checks
    assert config.args.pipe_type == 'dynamic'

    # enforce reproducibility
    main_seed = config.args.main_seed
    
    print('seed in script (for control): ' + str(main_seed))
    utils.set_seed(main_seed)

    # Build, train, save and test models (if already saved, models are loaded)
    gen_AI = mb.get_general_model()

    print('Initialization done.')

    ##### HYPERPARAMETER TUNING #####

    if config.args.domain == 'multi':

        print('Tune hyperparameters..')

        # tune ODIN iSUN/UCM
        tune_cache = config.args.pipe_tune_dataset
        if config.args.pipe_case == 'sat':
            config.args.pipe_tune_dataset = 'UCM'
        else:
            config.args.pipe_tune_dataset = 'iSUN'
        odin.tune_all_odin([gen_AI])
        config.args.pipe_tune_dataset = tune_cache

        print('Tuning done.')


    ##### RUN ALL PIPELINE VARIATIONS #####
    if config.args.exp_name == '':
        exp_name = 'run_' + config.args.domain + '_dynamic_pipeline_' + config.args.pipe_case + '_' + str(datetime.now().strftime("%d_%m_%Y_%H_%M"))
    else:
        exp_name = config.args.exp_name

    if config.args.domain == 'multi':

        print('All dynamic, multi-domain pipeline variations are computed and saved to mlflow')
        print('Case: ' + config.args.pipe_case)

        # set batch design to be consistent with static pipeline
        if config.args.dyn_batch_design == 'independent':
            raise NotImplementedError
        elif config.args.dyn_batch_design == 'dependent':
            if config.args.dyn_MVP_test:
                no_batches = 1000
            else:
                no_batches = config.args.dyn_no_batches
        if config.args.pipe_case == 'benchmark':
            if config.args.dyn_batch_design == 'dependent':
                config.args.pipe_size = int(5000/(no_batches+1))
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [78257, 65000, 65000]]
            elif config.args.dyn_batch_design == 'independent_recurring_genAI':
                no_batches = config.args.independent_recurring_genAI_dyn_no_batches
                config.args.pipe_size = int(sum([int(i/no_batches) for i in [78257, 65000, 65000]])/config.args.smaller_OOD_share)
                assert config.args.pipe_size <= 5000
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [78257, 65000, 65000]]
        elif config.args.pipe_case == 'sat':
            if config.args.dyn_batch_design == 'dependent':
                config.args.pipe_size = int(1900/(no_batches+1))
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [48565, 7164, 7560]]
            elif config.args.dyn_batch_design == 'independent_recurring_genAI':
                no_batches = config.args.independent_recurring_genAI_dyn_no_batches
                config.args.pipe_size = int(sum([int(i/no_batches) for i in [48565, 7164, 7560]])/config.args.smaller_OOD_share)
                assert config.args.pipe_size <= 1900
                config.args.pipe_noise = [i/no_batches/config.args.pipe_size for i in [48565, 7164, 7560]]

        # combo structure:
        # (early stop, train_patience, train_epochs, selec_mech, comb_mech, tune_dataset, tune_point_of_time,
        # trust_thresh, trust_patience, split_type)
        combos = [
            (True, 20, 300, 'gating', 'gating', 'iSUN' if config.args.pipe_case == 'benchmark' else 'UCM', 'when_trusted', 0.95, 3, 'train_val'),
            (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.95, 3, 'train_val'),
            (True, 20, 300, 'odin', 'odin', 'iSUN' if config.args.pipe_case == 'benchmark' else 'UCM', 'when_trusted', 0.95, 3, 'train_val')
        ]

        # Ablation study only for Maha

        if config.args.ablation_study == 'smaller_OOD_share':
            combos = [
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.95, int(config.args.independent_recurring_genAI_dyn_no_batches / 10), 'train_val')
            ]

        elif config.args.ablation_study == 'no_trust_early_stop':
            combos = [
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.95, 3, 'train_val')
            ]

        elif config.args.ablation_study == 'resnet18_backbone':
            combos = [
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.95, 3, 'train_val')
            ]

        elif config.args.ablation_study == 'trust_thresholds':
            combos = [
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.7, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.75, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.8, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.85, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.9, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.91, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.92, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.93, 3, 'train_val'),
                (True, 20, 300, 'maha', 'maha', 'pipe', 'when_trusted', 0.94, 3, 'train_val'),
            ]


        # run through all combos
        for idx, combo in enumerate(combos):

            # enforce reproducibility
            main_seed = config.args.main_seed
            
            print('seed in script (for control): ' + str(main_seed))
            utils.set_seed(main_seed)

            # set parameters
            config.args.train_early_stop = combo[0]
            config.args.dyn_valloss_patience = combo[1]
            config.args.dyn_multi_train_epochs = combo[2]
            config.args.dyn_multi_selec_mech = combo[3]
            config.args.dyn_multi_comb_mech = combo[4]
            config.args.dyn_multi_tune_dataset = combo[5]
            config.args.dyn_multi_tune_pot = combo[6]
            config.args.dyn_trust_thresh = combo[7]
            config.args.dyn_trust_patience = combo[8]
            config.args.dyn_multi_split = combo[9]

            # mlflow
            mlflow.set_experiment(experiment_name=exp_name)
            if idx == 0:
                exp = mlflow.get_experiment_by_name(exp_name)
                utils.adjust_mlflow_yaml(exp.experiment_id)
                utils.save_mlflow_exp_id(exp.experiment_id)
                print('Exp. ID: ' + str(exp.experiment_id))
            run_name = str(combo[0])
            i = 0
            while i < (len(combo)-1):
                run_name = run_name + '/' + str(combo[i+1])
                i = i+1
            mlflow.start_run(run_name='PARAMS_' + run_name)

            # collect params
            if config.args.pipe_case == 'benchmark':
                unc_thresh = config.args.conf_thresh_benchmark_wideresnet_pipe_wFMNIST
            elif config.args.pipe_case == 'sat':
                unc_thresh = config.args.conf_thresh_sat
            params = {
                'pipe_case': config.args.pipe_case,
                'pipe_size': config.args.pipe_size,
                'pipe_noise_1': config.args.pipe_noise[0],
                'pipe_noise_2': config.args.pipe_noise[1],
                'pipe_noise_3': config.args.pipe_noise[2],
                'Unc_mech': config.args.unc_mech,
                'Selec_mech': config.args.selec_mech,
                'bottom_x': config.args.bottom_x,
                'conf_thresh': unc_thresh,
                'Unc_score': config.args.unc_score,
                'early_stop': config.args.train_early_stop,
                'valloss patience': config.args.dyn_valloss_patience,
                'Pipe selec mech': config.args.dyn_multi_selec_mech,
                'Pipe comb mech': config.args.dyn_multi_comb_mech,
                'Pipe tune dataset': config.args.dyn_multi_tune_dataset,
                'trust_thresh': config.args.dyn_trust_thresh,
                'trust patience': config.args.dyn_trust_patience,
                'tune point of time': config.args.dyn_multi_tune_pot,
                'train_epochs': config.args.dyn_multi_train_epochs if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_train_epochs_sat,
                'train_batch_size': config.args.dyn_multi_train_batch if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_train_batch_sat,
                'split_type': config.args.dyn_multi_split,
            }
            mlflow.log_params(params)
            mlflow.end_run()


            print('Try ' + str(idx+1) + '/' + str(len(combos)) + ':')
            print('Trust thresh: ' + str(config.args.dyn_trust_thresh))
            print('Standard model parameters:')
            print('SGD/ADAM lr: ' + str(config.args.dyn_single_sgd_lr if config.args.domain == 'single' else (config.args.dyn_multi_sgd_lr if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_adam_lr_sat)))
            print('SGD/ADAM momentum: ' + str(config.args.dyn_single_sgd_mom if config.args.domain == 'single' else (config.args.dyn_multi_sgd_mom if config.args.pipe_case == 'benchmark' else 0)))
            print('SGDA/ADAM weight decay: ' + str(config.args.dyn_single_sgd_wd if config.args.domain == 'single' else (config.args.dyn_multi_sgd_wd if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_adam_wd_sat)))
            print(params)
            print('')

            # run dynamic system
            start = time.time()
            result = pipe.run_dyn_pipeline(gen_AI=gen_AI, run_name=run_name, exp_name=exp_name)
            end = time.time()
            dur = end - start

            # print results
            print('RESULTS: ')
            utils.print_dict(result)

            print('')
            print('Duration [s]: ' + str(dur))
            print('')
            print('')

            print('Try ' + str(idx+1) + '/' + str(len(combos)) + ' done.')
            print('')



    if config.args.domain == 'single':

        print('All static, single-domain pipeline variations are computed and saved to mlflow')
        print('Case: ' + config.args.pipe_case)

        # always conf thresh as selec mech, as there was no difference between metrics in static pipeline
        config.args.selec_mech = 'unc_thresh'
        config.args.unc_mech = 'S'
        config.args.unc_score = 'conf'
        config.args.dyn_multi_split = 'random'

        # combos
        if config.args.pipe_case == 'benchmark':
            trust_threshes = [0.95]  # chosen in relation to gen AI strength
            unc_threshes = [0.999]  # chosen accordingly to literature regarding Softmax scores (overconfident)
        else:
            trust_threshes = [0.9]  # chosen in relation to gen AI strength
            unc_threshes = [0.999]  # chosen accordingly to literature regarding Softmax scores (overconfident)

        # iterate through combos
        for idx, thresh in enumerate(trust_threshes):
            for idx_2, unc_thresh in enumerate(unc_threshes):

                # enforce reproducibility
                main_seed = config.args.main_seed
                
                print('seed in script (for control): ' + str(main_seed))
                utils.set_seed(main_seed)

                # set parameters
                config.args.dyn_trust_thresh = thresh
                config.args.conf_thresh_singledom = unc_thresh
                config.args.conf_thresh_singledom_sat = unc_thresh

                # mlflow
                mlflow.set_experiment(experiment_name=exp_name)
                if idx+idx_2 == 0:
                    exp = mlflow.get_experiment_by_name(exp_name)
                    utils.adjust_mlflow_yaml(exp.experiment_id)
                    utils.save_mlflow_exp_id(exp.experiment_id)
                    print('Exp. ID: ' + str(exp.experiment_id))
                run_name = str(thresh) + '_' + str(unc_thresh)
                mlflow.start_run(run_name='PARAMS_' + run_name)

                # log params
                if config.args.pipe_case == 'benchmark':
                    unc_thresh = config.args.conf_thresh_singledom
                elif config.args.pipe_case == 'sat':
                    unc_thresh = config.args.conf_thresh_singledom_sat
                params = {
                    'pipe_case': config.args.pipe_case,
                    'Unc_mech': config.args.unc_mech,
                    'Selec_mech': config.args.selec_mech,
                    'conf_thresh': unc_thresh,
                    'Unc_score': config.args.unc_score,
                    'trust_score': config.args.dyn_trust_score,
                    'trust_thresh': config.args.dyn_trust_thresh,
                    'train_epochs': config.args.c10_epochs if config.args.pipe_case == 'benchmark' else config.args.eurosat_epochs,
                    'train_batch_size': config.args.c10_wideresnet_train_batch if config.args.pipe_case == 'benchmark' else config.args.eurosat_train_batch,
                }
                mlflow.log_params(params)
                mlflow.end_run()

                print('Try ' + str(idx * len(unc_threshes) + idx_2 +1) + '/' + str(len(unc_threshes)  * len(trust_threshes)) + ':')
                print('unc_thresh: ' + str(unc_thresh))

                print('Trust score: ' + config.args.dyn_trust_score)
                print('Trust thresh: ' + str(config.args.dyn_trust_thresh))
                print('Standard model parameters:')
                print('Train batch size: ' + str(config.args.c10_wideresnet_train_batch if config.args.pipe_case == 'benchmark' else config.args.eurosat_train_batch))
                print('Train epochs: ' + str(config.args.c10_epochs if config.args.pipe_case == 'benchmark' else config.args.eurosat_epochs))
                print('SGD lr: ' + str(config.args.c10_sgd__lr if config.args.pipe_case == 'benchmark' else config.args.eurosat_sgd__lr))
                print('SGD momentum: ' + str(config.args.c10_sgd__mom if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_sgd_mom))
                print('SGD weight decay: ' + str(config.args.c10_sgd__wd if config.args.pipe_case == 'benchmark' else config.args.eurosat_sgd__wd))
                print('')

                # run system
                start = time.time()
                result = pipe.run_dyn_pipeline(gen_AI=gen_AI, run_name=run_name, exp_name=exp_name)
                end = time.time()
                dur = end - start


                # print results
                print('RESULTS: ')
                utils.print_dict(result)

                print('')
                print('Duration [s]: ' + str(dur))
                print('')
                print('')

                print('Try ' + str(idx * len(unc_threshes) + idx_2 +1) + '/' + str(len(unc_threshes)  * len(trust_threshes)) + ' done.')
                print('')

                mlflow.end_run()

                print('')
                print("Done. Download new mlflow folder and use 'mlflow ui' in terminal to observe all results.")


if __name__ == "__main__":
    main()
