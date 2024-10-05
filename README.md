# MIA-HPO: Hyperparameters in Score-Based Membership Inference Attacks.
We make use of the following open-source libraries in our experiments:

* TIMM (for the PyTorch VIT-B implementation): Copyright 2020 Ross Wightman https://github.com/rwightman/pytorch-image-models
* Big Transfer (for the R-50 implementation): Copyright 2020 Google LLC https://github.com/google-research/big_transfer
  * Switch to the ```src``` directory in this repo and download the BiT pre-trained model for ResNet50: ```wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz```

General experiment options include:

```
--feature_extractor <BiT-M-R50x1,vit-b-16> 
--examples_per_class <number of examples per class, `-1` means the whole training set, `None` enables the VTAB split>
--seed <for reproducibility, e.g., 0>
--optimizer <adam,sgd>
--private --target_epsilon <1,8>
```
```--private --target_epsilon <1,8>``` is to implement differential privacy. For the non-DP setting, set ```--target_epsilon -1``` and do not use the ```--private``` flag.

### ATTACKS (Section III & IV)
In this section, we detail the implementation of code to study the effect of hyperparameter selection to train shadow models on the performance of score-based membership inference attacks (MIAs).

To create the MIA Grid follow the given steps:

* Use ```python3 src/build_mia_grid_head_td.py``` or  ```python3 src/build_mia_grid_film_td.py``` with the flag  ```--tune``` to sample the $\ D_0,D_1,...,D_M \$ data sets from the training data set $\ D_T \$ and the collect the corresponding optimal hyperparameters.

* Use ```python3 src/build_mia_grid_head_td.py``` or ```python3 src/build_mia_grid_film_td.py``` without the flag  ```--tune``` to train the models for the MIA grid such that $\ \mathcal{M}_{D_i,\eta_j} \leftarrow \texttt{TRAIN} (D_i, \eta_j)\$.



### EMPIRICAL PRIVACY LEAKAGE DUE TO HPO (Section VI & VII)

TO BE EDITED
