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
from gpradhan.dp_hp_tuning.src.cached_data_loader import CachedFeatureLoader
from datetime import datetime
# import csv
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss = cross_entropy_loss
        self.eps = None
        self.delta = None
        self.tune_images = None
        self.tune_labels = None
        self.num_classes = None
        self.exp_dir = None
        self.run_dir = None
        # for recording the best trials hypers
        self.hypers = {"learning_rate":[],
                             "max_grad_norm":[],
                             "batch_size":[]}
    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', help='Dataset to use.', choices=dataset_map.keys(), default="small_set")
        parser.add_argument("--target_feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='BiT-M-R50x1', help="Feature extractor for target model.")
        parser.add_argument("--shadow_feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='BiT-M-R50x1', help="Feature extractor for shadow models.")
        parser.add_argument("--classifier", choices=['linear'], default='linear',
                            help="Which classifier to use.")
        parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                            help="Which feature extractor parameters to learn.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--results", help="Directory to load results from.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--exp_id", type=int, default=None,
                            help="Experiment ID.")
        parser.add_argument("--run_id", type=int, default=None,
                            help="Run ID for rerunning the whole experiment.")
        # HPO    
        parser.add_argument("--start_data_split", type=int, default=0,
                            help="Starting index of data split to train tfor the LiRA attack.")
        parser.add_argument("--stop_data_split", type=int, default=129,
                            help="Stopping number of data split to train tfor the LiRA attack.")
        parser.add_argument("--start_hypers", type=int, default=0,
                            help="Starting index of hypers to train tfor the LiRA attack.")
        parser.add_argument("--stop_hypers", type=int, default=129,
                            help="Stopping index of hypers to train tfor the LiRA attack.")
        parser.add_argument("--num_shadow_models", type=int, default=128,
                            help="Total number of shadow models.")
        
        args = parser.parse_args()
        return args

    def create_head(self,feature_dim: int, num_classes: int):
        head = nn.Linear(feature_dim, num_classes)
        head.weight.data.fill_(0.0)
        head.bias.data.fill_(0.0)
        head.to(DEVICE)
        return head

    def run(self):
        # seeding
        set_seeds(self.args.seed)
        limit_tensorflow_memory_usage(2048)

        self.accuracies = {"in": np.zeros((self.args.stop_data_split - self.args.start_data_split, 
                                             self.args.stop_hypers - self.args.start_hypers)),
                           "out": np.zeros((self.args.stop_data_split - self.args.start_data_split, 
                                             self.args.stop_hypers - self.args.start_hypers)),
                           "test": np.zeros((self.args.stop_data_split - self.args.start_data_split, 
                                             self.args.stop_hypers - self.args.start_hypers))}
        
        # ensure the directory to hold results exists
        self.exp_dir = f"experiment_{self.args.exp_id}"
        self.run_dir = f"Run_{self.args.run_id}"
        self.directory = os.path.join(self.args.results, "Seed={}".format(self.args.seed),self.run_dir, self.exp_dir)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        datasets = dataset_map[self.args.dataset]       
        for dataset in datasets:
            if dataset['enabled'] is False:
                continue

            self.num_classes = dataset['num_classes']

            self.target_dataset_reader = CachedFeatureLoader(path_to_cache_dir=self.args.download_path_for_tensorflow_datasets,
                                                      dataset=dataset["name"],
                                                      feature_extractor = self.args.target_feature_extractor,
                                                      random_seed=self.args.seed
                                                      )
            
            self.target_feature_dim = self.target_dataset_reader.obtain_feature_dim()
            target = self.target_dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                            n_classes=self.num_classes,
                                                                            task="train")
            self.shadow_dataset_reader = CachedFeatureLoader(path_to_cache_dir=self.args.download_path_for_tensorflow_datasets,
                                                      dataset=dataset["name"],
                                                      feature_extractor = self.args.shadow_feature_extractor,
                                                      random_seed=self.args.seed
                                                      )
            
            self.shadow_feature_dim = self.shadow_dataset_reader.obtain_feature_dim()
            shadow = self.shadow_dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                                n_classes=self.num_classes,
                                                                                task="train")
        
            print("ViT",target[0].shape)
            print("R50",shadow[0].shape)

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
