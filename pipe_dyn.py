# this code is intended to run individual results of the dynamic AIITL-system

# all libraries
import src.utils.pipe as pipe
import src.config
import src.utils.utils as utils
import src.utils.thresholds as th
import time
import src.utils.model_building as mb


def main():

        ##### INITIALIZE #####
        print('Initialize models..')

        # get config
        config = src.config.cfg

        # some checks
        assert config.args.pipe_type == 'dynamic'

        # enforce reproducibility
        main_seed = config.args.main_seed
        print('seed in script (for control): ' + str(main_seed))
        utils.set_seed(main_seed)

        # Build, train, save and test models (if already saved, models are loaded)
        gen_AI = mb.get_general_model()
        experts = []
        if config.args.domain == 'multi':
                exp_AI_1, exp_AI_2, exp_AI_3 = mb.get_expert_models()
                experts = [exp_AI_1, exp_AI_2, exp_AI_3]

        print('Initialization done.')

        ##### HYPERPARAMETER TUNING #####

        if config.args.domain == 'multi':
                print('Tune hyperparameters..')

                # tune all threshold
                th.tune_all_gen_AI_thresholds(gen_AI=gen_AI, experts=experts)

                print('Tuning done.')



        ##### RUN PIPELINE #####
        print('Dynamic pipeline (' + config.args.domain + '-domain) started:')
        print('Case: ' + config.args.pipe_case)
        if config.args.domain == 'multi':
                print('No of experts: ' + str(config.args.dyn_num_experts))
        print('Trust thresh: ' + str(config.args.dyn_trust_thresh))
        print('Standard model parameters:')
        print('Train batch size: ' + str(config.args.dyn_single_train_batch if config.args.domain == 'single' else (config.args.dyn_multi_train_batch if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_train_batch_sat)))
        print('Train epochs: ' + str(config.args.dyn_single_train_epochs if config.args.domain == 'single' else (config.args.dyn_multi_train_epochs if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_train_epochs_sat)))
        print('SGD/ADAM lr: ' + str(config.args.dyn_single_sgd_lr if config.args.domain == 'single' else (config.args.dyn_multi_sgd_lr if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_adam_lr_sat)))
        print('SGD/ADAM momentum: ' + str(config.args.dyn_single_sgd_mom if config.args.domain == 'single' else (config.args.dyn_multi_sgd_mom if config.args.pipe_case == 'benchmark' else 0)))
        print('SGDA/ADAM weight decay: ' + str(config.args.dyn_single_sgd_wd if config.args.domain == 'single' else (config.args.dyn_multi_sgd_wd if config.args.pipe_case == 'benchmark' else config.args.dyn_multi_adam_wd_sat)))
        print('')

        if config.args.domain == 'single':
                config.args.selec_mech = 'unc_thresh'
                config.args.unc_mech = 'S'
                config.args.unc_score = 'conf'
                config.args.dyn_multi_split = 'random'

        start = time.time()
        result = pipe.run_dyn_pipeline(gen_AI=gen_AI)
        end = time.time()
        dur = end - start

        # print results
        print('RESULTS: ')
        utils.print_dict(result)
        print('')
        print('Duration [s]: ' + str(dur))
        print('')
        print('')
        print('Dynamic pipeline finished.')


if __name__ == "__main__":
        main()

