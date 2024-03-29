# Improving the Efficiency of Human-in-the-Loop Systems: Adding Artificial to Human Experts

This is a [PyTorch](https://pytorch.org) implementation of hybrid _AI-in-the-Loop_ (AIITL) systems.
The system creates artificial experts that incrementally learn to classify the unknown data instances which previously had to be reviewed by a human expert. 
Our approach then assesses which of the artificial experts is suitable for classifying an unknown instance and allocates the instance accordingly. 
Like this, human effort is gradually reduced. We demonstrate that our approach outperforms HITL systems by a large margin on a range of benchmark for image classification data.

Find our pre-print on [arXiv](https://arxiv.org/pdf/2307.03003.pdf). <br>
🚀: Our paper got accepted at [WI'23](https://wi2023.de/en/startseite-english/), nominated for the [best paper award](https://wi2023.de/en/best-paper-nominations/), and is part of four selected thought-provoking papers.

by Johannes Jakubik\*, Daniel Weber, Patrick Hemmer, Michael Vössing, and Gerhard Satzger

## :bulb: Approach
<img src="/results/AIITL.png"/>

We present the evaluation below.

## :checkered_flag: Results
Results on benchmark datasets for image classification

<img src="/results/benchmark_multi_dynamic_utility_scores_1.0_0.5.png" width="500" height="238"/>
<img src="/results/benchmark_multi_dynamic_utility_scores_1.0_0.5_table.png" width="500" height="128"/>


This ReadMe is structured as follows:
1. Requirements and preparation
2. Reproduction of results
4. Plots and Tables

## :speech_balloon: Requirements and preparation

### Preliminaries

* Python 3.8
* PyTorch 1.9.1
* CUDA 11.5
* Miniconda3
* Training was executed on a NVIDIA A100 GPU.

To install the required packages in your conda environment, please run in the root directory
```
pip3 install -r requirements.txt
```

## :speech_balloon: Reproduction of results

### Overview of arguments


`-pipe_case`
<br> (Experiment considered)

| Option      | Description                       |
| :-----------           | :-----------                       |
| benchmark             | Experiment on benchmark data      |

`-pipe_type`
<br> (System type considered)

| Option        | Description                                                                  |
| :----------             | :-----------                                                                      |
| dynamic                 | Dynamic AIITL-system (dynamic creation of AIITL-system starting from HITL-system)  |
| static (not focus of this repo)                | Static AIITL-system (not focus of this repo)                                    |

`--domain`
<br> (Data domain considered)

| Option        |  Description               |
| :----------            | :-----------                |
| single                | Single-domain AIITL-system |
| multi                 | Multi-domain AIITL-system  |

`-ablation_study`
<br> (Ablation study considered)

| Option        | Description                |
| :----------            | :-----------                |
| ''                | Default results |
| trust_thresholds | Impact of varying trust thresholds |
| no_trust_early_stop | Impact of excluding the early stopping mechanism for trusting artificial experts  |
| resnet18_backbone | Impact of changing the backbone from Wide-ResNet-28-10 to ResNet-18  |
| smaller_OOD_share_v2 | Impact of changing the share of unknown data considered per batch |


**Examples:**
```
python pipe_dyn.py -pipe_case=benchmark -pipe_type=dynamic --domain=multi
```
```
python all_pipe_dyn.py -pipe_case=benchmark -pipe_type=dynamic --domain=multi
```

* Please make sure all models are removed or renamed
* Training of all ML models is done automatically
* Use the flag `-pipe_tune_thresholds` in order to initiate allocation mechanism tuning. The assignment of hyperparamters is done automatically within a single run.

## :speech_balloon: Plots and Tables

* Plots and tables can be generated by running [`plots.py`](https://github.com/jhnnsjkbk/AIITL/blob/main/plots.py)
* Besides the required arguments of `-pipe_case`, `-pipe_type`, `--domain` and `-ablation_study`, choose optional arguments of the following:

**Type of plot/table**:

| Option  | Description |
| :----------            | :-----------                |
| `-softmax_thresholds`              | Impact of Softmax thresholds on static, single-domain AIITL-system |
| `-utility_graph`               | Development of utility |
| `-acc_cov_table`               | Overview of coverages and system accuracy |
| `-exp_learning_graph`               | Development of trust scores |
| `-allocation_matrix`               | Overview of allocated samples for a specific data batch |
| `-selec_comb_mechs`               | Overview of all combinations of allocation mechanisms in the static, multi-domain AIITL-system |
| `-selec_comb_mechs_baselines`              | Baselines of the static, multi-domain AIITL-system |

**Further parameters of plots:**

| Option  | Description | Required for Plot/table |
| :----------            | :-----------                | :-----------                |
| `-sub_case`               | Combination of considered allocation mechanism in the following format (string): <ul><li>dynamic AIITL-system: 'allocation_mechanism' <br> (e.g., 'gating')</li><li>static AIITL-system: 'allocation_mechanism1/allocation_mechanism2' <br> (e.g., 'gating/maha')</li></ul>| allocation_matrix & exp_learning_graph |
| `-utility_a`              | alpha of utility score (float) | All plots/tables which include utility |
| `-utility_b`               | beta of utility score (float) | All plots/tables which include utility |
| `-no_finalbatch`               | Exclude final batch (flag) | utility_graph & exp_learning_graph |
| `-batch_no`               | Number of batch considered in dynamic system (int) | allocation_matrix |
| `-fmow_overfit_type`               | 'acc' OR 'loss' (string) | Ablation Study: fmow_overfit|
| `-smaller_OOD_plot`               | 'odin' OR 'system' (string) | Ablation Study: smaller_OOD_share_v2|

**Examples:**
```
python plots.py -pipe_case=benchmark -pipe_type=dynamic --domain=multi -utility_graph -utility_a=1 -utility_b=0.5
```
```
python plots.py -pipe_case=benchmark -pipe_type=static --domain=multi -allocation_matrix -utility_a=1 -utility_b=0.5 -sub_case=gating/maha
```
```
python plots.py -pipe_case=benchmark -pipe_type=dynamic --domain=multi -allocation_matrix -utility_a=1 -utility_b=0.5 -sub_case=gating -batch_no=31
```

## :speech_balloon: License
Code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.

## :speech_balloon: References
Appreciate the work from the following repositories:
- https://github.com/facebookresearch/odin
- https://github.com/pokaxpoka/deep_Mahalanobis_detector
- https://github.com/ildoonet/pytorch-randaugment
- https://github.com/kuangliu/pytorch-cifar 

