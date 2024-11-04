# MIA-HPO: Hyperparameters in Score-Based Membership Inference Attacks.

### DEPENDENCIES

The following modules are required to run our code:
 * Python 3.8 or greater
 * PyTorch 1.11 or greater (https://pytorch.org/)
 * Opacus 1.3 or greater (https://opacus.ai/#quickstart)
 * prv_accountant 0.2.0 or greater (https://github.com/microsoft/prv_accountant)
 * Optuna 3.0 or greater (https://optuna.org/#installation)

Additionally, we make use of the following open-source libraries in our experiments:

* TIMM (for the PyTorch VIT-B implementation): Copyright 2020 Ross Wightman https://github.com/rwightman/pytorch-image-models
* Big Transfer (for the R-50 implementation): Copyright 2020 Google LLC https://github.com/google-research/big_transfer
  * Switch to the ```src``` directory and download the BiT pre-trained model for ResNet50: ```wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz```

### GENERAL SETUP

We adapted the code provided at:
- https://github.com/cambridge-mlg/dp-few-shot
- https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021

for our experiments.

**General experiment options for training models include:**

```
--dataset <cifar10,cifar100>
--feature_extractor <BiT-M-R50x1,vit-b-16> 
--examples_per_class <number of examples per class, `-1` means the whole training set, `None` enables the VTAB split>
--seed <for reproducibility, e.g., 0>
--optimizer <adam,sgd>
--private --target_epsilon <1,8>
--num_shadow_models <64>
--max_physical_batch_size  <for running under constrained memory> 
--test_batch_size <for running under constrained memory> 
```
For the FiLM experiments, we used NVIDIA V100 GPUs with 32 GB of memory. While training models on GPU, ```--max_physical_batch_size``` can be used to limit the size of the large logical batches such that they could be accommodated in the GPU Memory.

The arguments ```--private --target_epsilon <1,8> ``` is to implement differential privacy. For the non-DP setting, set ```--target_epsilon -1``` and do not use the ```--private``` flag.

We fixed ```--target_delta 1e-5```  for our experiments. 

**Other setup options:**

```
--download_path_for_tensorflow_datasets <path to dataset>
--checkpoint_dir <path to checkpoint directory>
--run_id <sub-directory depending on examples_per_class>
--exp_id <sub-directory for each experiment in a given run depending on the level of privacy>
```
The arguments ```--run_id``` and ```--exp_id``` are used to customize the path to store the results. The choice of the former depends on the examples per class whereas the latter depends on the level of privacy used in the experiments (for example, ```--rrun_id 1 --exp_id 1``` implies ```examples_per_class 100 --target_epsilon -1``` in our experiments).

**For hyperparameter optimization (HPO) use the following options:**

```
--number_of_trials 20
--train_batch_size_lb 10 --train_batch_size_ub 10000
--max_grad_norm_lb 0.2 --max_grad_norm_ub 10.0
--learning_rate_lb 1e-07 --learning_rate_ub 0.01
```
We fixed ```--epochs 40``` for our experiments.

### ATTACKS (Section III & IV)
In this section, we detail the implementation of code to study the effect of hyperparameter selection to train shadow models on the performance of score-based membership inference attacks (MIAs).

To create the MIA Grid follow the given steps:

* Use ```python3 src/build_mia_grid_head_td.py``` or  ```python3 src/build_mia_grid_film_td.py``` with the flag  ```--tune``` to sample the $\ D_0,D_1,...,D_M\$ data sets from the training data set $\ D_T\$ and the collect the corresponding optimal hyperparameters.

* Use ```python3 src/build_mia_grid_head_td.py``` or ```python3 src/build_mia_grid_film_td.py``` without the flag  ```--tune``` to train the models for the MIA grid such that $\ \mathcal{M}_{D_i,\eta_j} \leftarrow \texttt{TRAIN} (D_i, \eta_j)\$.

**NOTE:** Following additional arguments in the code to build the MIA Grid will allow the users to collect optimal hyperparameters/data set and train models for the MIA Grid in parts when limited compute is available to run the code:

```
--start_data_split
--stop_data_split
--start_hypers
--stop_hypers 
```
For example, ```python3 src/build_mia_grid_film_td.py -- start_data_split 0 --stop_data_split 5 --start_hypers 0 --stop_hypers 5 --tune``` will sample and collect optimal hyperparameters for  $\ D_0,D_1,D_2,D_3,D_4\$ of the 65 (```--num_shadow_models``` + 1) data sets sampled from $\ D_T\$. 

Once we have the logits for samples in $\ D_T\$ for the models in the MIA Grid:
* Calculate the LiRA scores of samples using ```python3 src/run_lira.py``` when the target architecture is known and can be used to train the shadow models.
* To simulate LiRA in the Black-Box setting, use ```python3 src/run_lira_bb.py``` where the arguments ```--target_stats_dir``` and ```--shadow_stats_dir``` should point to the logits collected from models trained with the target architecture and the shadow architecture respectively on same data splits sampled from $\ D_T\$.

### EMPIRICAL PRIVACY LEAKAGE DUE TO HPO (Section VI & VII)

* Use ```python3 src/train_head_ed_target_models.py``` or  ```python3 src/train_film_ed_target_models.py``` to train the target models in the ED-HPO setting. 

* To run LiRA on the target models in ED-HPO use ```python3 arc/run_lira_ed.py```.
