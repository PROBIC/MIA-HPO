# Portions of this code are excerpted from:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py

import pickle
import numpy as np
import os
from lira import compute_score_lira
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to in_indices, stat, and output score files.")
    parser.add_argument("--seed", type = int, default=None, help="Seeding used for experiments.")
    parser.add_argument("--exp_id", type=int, default=None, help="Experiment ID.")
    parser.add_argument("--run_id", type=int, default=None, help="Run ID for rerunning the whole experiment.")
    parser.add_argument("--results", default = ".", help="Path to store the processed lira scores.")
    parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                            help="Which feature extractor parameters to learn.")
    parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
    parser.add_argument("--target_epsilon", type=int,
                            default=10, help="Maximum value of epsilon allowed.")
    parser.add_argument("--num_shadow_models", type=int,
                            default = 256, help="Number of shadow models used for LIRA")
    args = parser.parse_args()

    config = args.learnable_params
    shot = args.examples_per_class
    NUM_TARGET_MODELS = args.num_shadow_models + 1
    if args.target_epsilon == -1:
        epsilon = 'inf'
    else:
        epsilon = args.target_epsilon

    with open(os.path.join(args.data_path, "Seed={}".format(args.seed), 'Run_{}'.format(args.run_id), 'experiment_{}'.format(args.exp_id), 'in_indices_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
        in_indices = pickle.load(f)
    with open(os.path.join(args.data_path, "Seed={}".format(args.seed), 'Run_{}'.format(args.run_id), 'experiment_{}'.format(args.exp_id), 'stat_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
        stat = pickle.load(f)
    if args.run_id % 2 == 0:
        # only load the TD-HPO indices 
        with open(os.path.join(args.data_path, "Seed={}".format(args.seed), 'Run_{}'.format(args.run_id-1), 'experiment_{}'.format(args.exp_id), 'tune_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
            tune_indices = pickle.load(f)
    else:
        with open(os.path.join(args.data_path, "Seed={}".format(args.seed), 'Run_{}'.format(args.run_id), 'experiment_{}'.format(args.exp_id), 'tune_{}_{}_{}.pkl'.format(config, shot, epsilon)), "rb") as f:
            tune_indices = pickle.load(f)   

    # select only the indices + stats for samples used for tuning
    curated_indices,curated_stat = [],[]
    for idx in range(NUM_TARGET_MODELS):
        curr_indices = in_indices[idx]
        curated_indices.append(np.array([curr_indices[i] for i in range(len(curr_indices)) if tune_indices[i] == True]))
        curr_stat = stat[idx]
        curated_stat.append(np.expand_dims(np.concatenate([curr_stat[i] for i in range(len(curr_stat)) if tune_indices[i] == True]),1))
    
    n = np.sum(tune_indices)
    # Now we do MIA for each model
    all_scores = []
    all_y_true = []
    for idx in range(NUM_TARGET_MODELS):
        print(f'Target model is #{idx}')
        stat_target = curated_stat[idx]  # statistics of target model, shape (n, k)
        in_indices_target = curated_indices[idx]  # ground-truth membership, shape (n,)
        # `stat_shadow` contains statistics of the shadow models, with shape
        # (num_shadows, n, k). `in_indices_shadow` contains membership of the shadow
        # models, with shape (num_shadows, n). We will use them to get a list
        # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
        # (m, k) array, for m being the number of shadow models trained with
        # (resp. without) the j-th example, and k being the number of augmentations
        # (1 in our case).
        stat_shadow = np.array(curated_stat[:idx] + curated_stat[idx + 1:])
        in_indices_shadow = np.array(curated_indices[:idx] + curated_indices[idx + 1:])
        stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(n)]
        stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(n)]

        # Compute the scores and use them for MIA
        scores = compute_score_lira(stat_target, stat_in, stat_out, fix_variance=True)

        # y_score = np.concatenate((scores[in_indices_target], scores[~in_indices_target]))
        # y_true = np.concatenate((np.zeros(len(scores[in_indices_target])),
        #                                      np.ones(len(scores[~in_indices_target]))))

        # preserve the order of samples
        y_true = [0 if mask else 1 for mask in in_indices_target]

        all_scores.append(scores)
        all_y_true.append(y_true)
    

    all_y_true = np.hstack(all_y_true)
    all_scores  = np.hstack(all_scores)
    result = {
            'y_true': all_y_true,
            'scores': all_scores
            }

    with open(os.path.join(args.data_path, "Seed={}".format(args.seed), 'Run_{}'.format(args.run_id), 'experiment_{}'.format(args.exp_id), 
                           'scores_{}_{}_{}.pkl'.format(config, shot, epsilon)), "wb") as f:
        pickle.dump(result,f)

if __name__ == '__main__':
    main()
