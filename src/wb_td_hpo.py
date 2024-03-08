# Portions of this code are excerpted from:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py

import numpy as np
import os.path
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from dataset import dataset_map
from utils import limit_tensorflow_memory_usage,\
    compute_accuracy_from_predictions, predict_by_max_logit, cross_entropy_loss, shuffle, set_seeds
from cached_data_loader import CachedFeatureLoader
from datetime import datetime
import csv
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from hpo import optimize_hyperparameters
import gc
import sys
import warnings
from lira import convert_logit_to_prob, calculate_statistic, log_loss
import pickle

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        # self.logger = Logger(self.args.results, 'log.txt')
        # self.start_time = datetime.now()
        # self.logger.print_and_log("Options: %s\n" % self.args)
        # self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.results)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss = cross_entropy_loss
        self.eps = None
        self.delta = None
        self.tune_images = None
        self.tune_labels = None
        self.num_classes = None
        self.exp_dir = None
        self.run_dir = None
        self.models = {}
        # for recording the best trials hypers
        self.optimal_args = {"seed":[],
                             "learning_rate":[],
                             "max_grad_norm":[],
                             "batch_size":[]}

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', help='Dataset to use.', choices=dataset_map.keys(), default="small_set")
        parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='BiT-M-R50x1', help="Feature extractor to use.")
        parser.add_argument("--classifier", choices=['linear'], default='linear',
                            help="Which classifier to use.")
        parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                            help="Which feature extractor parameters to learn.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--results", help="Directory to load results from.")
        # parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints',
        #                     help="Directory to save checkpoint to.")
        parser.add_argument("--train_batch_size", "-b", type=int, default=200, help="Batch size.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.003, help="Learning rate.")
        parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of fine-tune epochs.")

        parser.add_argument("--test_batch_size", "-tb", type=int, default=600, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--exp_id", type=int, default=None,
                            help="Experiment ID.")
        parser.add_argument("--run_id", type=int, default=None,
                            help="Run ID for rerunning the whole experiment.")
        # HPO
        parser.add_argument("--sampler", type=str, default="BO", help="Type of sample to be used for HPO.")
        # parser.add_argument("--type_of_tuning", type=int, default=0, 
                            # help="For TD-HPO set the variable to 0, for ED-HPO set it to 1.")
        # parser.add_argument("--ed_hpo_repeats", type=int, default=1, help="The number of trials for optuna for ED-HPO")

        # parser.add_argument("--epochs_lb", type=int, default=1, help="LB of fine-tune epochs.")
        # parser.add_argument("--epochs_ub", type=int, default=200, help="UB of fine-tune epochs.")
        parser.add_argument("--train_batch_size_lb", type=int, default=10, help="LB of Batch size.")
        parser.add_argument("--train_batch_size_ub", type=int, default=1000, help="UB of Batch size.")
        parser.add_argument("--max_grad_norm_lb", type=float, default=0.2, help="LB of maximum gradient norm.")
        parser.add_argument("--max_grad_norm_ub", type=float, default=10.0, help="UB of maximum gradient norm.")
        parser.add_argument("--learning_rate_lb", type=float, default=1e-7, help="LB of learning rate")
        parser.add_argument("--learning_rate_ub", type=float,  default=1e-2, help="UB of learning rate")
        parser.add_argument("--number_of_trials", type=int, default=20, help="The number of trials for optuna")
        # DP options
        parser.add_argument("--private", dest="private", default=False, action="store_true",
                            help="If true, use differential privacy.")
        parser.add_argument("--noise_multiplier", type=float, default=1.0, help="Noise multiplier.")
        parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm.")
        parser.add_argument("--target_epsilon", type=float, default=10.0, help="Maximum value of epsilon allowed.")
        parser.add_argument("--target_delta", type = float, default = 1e-5, help="The delta for DP training.")
        parser.add_argument("--max_physical_batch_size", type=int, default=400, help="Maximum physical batch size")
        parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam')
        parser.add_argument("--secure_rng", dest="secure_rng", default=False, action="store_true",
                            help="If true, use secure RNG for DP-SGD.")
        parser.add_argument("--accountant", type=str, default = "rdp",
                            help="The nature of the accountant used for privacy engine.")
        
        # LiRA options
        parser.add_argument("--num_shadow_models", type=int, default=256,
                            help="Number of shadow models to train tfor the LiRA attack.")

        args = parser.parse_args()
        return args


    def run(self):
        # seeding
        set_seeds(self.args.seed)
        limit_tensorflow_memory_usage(2048)

        self.accuracies = {"in":np.zeros((self.args.num_shadow_models + 1,)),
                           "out":np.zeros((self.args.num_shadow_models + 1,)),
                           "test":np.zeros((self.args.num_shadow_models + 1,))}
        
        # ensure the directory to hold results exists
        self.exp_dir = f"experiment_{self.args.exp_id}"
        self.run_dir = f"Run_{self.args.run_id}"
        directory = os.path.join(self.args.results, "Seed={}".format(self.args.seed),self.run_dir, self.exp_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        datasets = dataset_map[self.args.dataset]       
        for dataset in datasets:
            # self.logger.print_and_log("{}".format(dataset['name']))
            if dataset['enabled'] is False:
                continue

            self.num_classes = dataset['num_classes']

            self.dataset_reader = CachedFeatureLoader(path_to_cache_dir=self.args.download_path_for_tensorflow_datasets,
                                                      dataset=dataset["name"],
                                                      feature_extractor = self.args.feature_extractor,
                                                      optuna_trials = self.args.number_of_trials,
                                                      random_seed=self.args.seed
                                                      )
            
            self.feature_dim = self.dataset_reader.obtain_feature_dim()

            train_features, train_labels, training_indices, self.class_mapping = self.dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                                                                        n_classes=self.num_classes,
                                                                                                                        task="train")
            with open(os.path.join(self.args.results, "Seed={}".format(self.args.seed), self.run_dir, self.exp_dir, 'train_{}_{}_{}.pkl'.format(
                    self.args.learnable_params,
                    self.args.examples_per_class,
                    int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                    pickle.dump(training_indices, f)


            n = train_features.shape[0]
            in_indices = []  # a list of in-training indices for all models
            for m in range(self.args.num_shadow_models + 1):
                    # Generate a binary array indicating which example to include for training
                    np.random.seed(self.args.seed + m + 1) # set the seed for drawing in-samples to the model index + parent seed
                    in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))
                    x, y = train_features[in_indices[-1]], train_labels[in_indices[-1]]
                    curr_opt_hypers,_ = optimize_hyperparameters(m+1, self.args, 
                                                                 x,
                                                                 y, 
                                                                 self.feature_dim, 
                                                                 self.num_classes, 
                                                                 self.args.seed + m + 1)
                    self.optimal_args["seed"].append(self.args.seed + m + 1)
                    self.optimal_args["learning_rate"].append(curr_opt_hypers.learning_rate)
                    self.optimal_args["batch_size"].append(curr_opt_hypers.train_batch_size)
                    if self.args.private:
                        self.optimal_args["max_grad_norm"].append(curr_opt_hypers.max_grad_norm)
            
            with open(os.path.join(self.args.results, "Seed={}".format(self.args.seed), self.run_dir, self.exp_dir, 'opt_args_{}_{}_{}.pkl'.format(
                    self.args.learnable_params,
                    self.args.examples_per_class,
                    int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                    pickle.dump(self.optimal_args, f)

            with open(os.path.join(self.args.results, "Seed={}".format(self.args.seed), self.run_dir, self.exp_dir, 'in_indices_{}_{}_{}.pkl'.format(
                    self.args.learnable_params,
                    self.args.examples_per_class,
                    int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                    pickle.dump(in_indices, f)

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
