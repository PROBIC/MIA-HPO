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
from cached_data_loader_v2 import CachedFeatureLoader
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
        parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='BiT-M-R50x1', help="Feature extractor to use.")
        parser.add_argument("--classifier", choices=['linear'], default='linear',
                            help="Which classifier to use.")
        parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                            help="Which feature extractor parameters to learn.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--results", help="Directory to load results from.")
        # parser.add_argument("--checkpoint_dir", "-c",
        #                     help="Directory to load data and optimal hyperparameters from.")
        parser.add_argument("--train_batch_size", "-b", type=int, default=200, help="Batch size.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.003, help="Learning rate.")
        parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of fine-tune epochs.")
        parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm.")
        parser.add_argument("--test_batch_size", "-tb", type=int, default=600, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--exp_id", type=int, default=None,
                            help="Experiment ID.")
        parser.add_argument("--run_id", type=int, default=None,
                            help="Run ID for rerunning the whole experiment.")
        # HPO
        parser.add_argument("--number_of_trials", type=int, default=20, help="The number of trials for optuna")
        parser.add_argument("--sampler", type=str, default="BO", help="Type of sample to be used for HPO.")
        parser.add_argument("--train_batch_size_lb", type=int, default=10, help="LB of Batch size.")
        parser.add_argument("--train_batch_size_ub", type=int, default=1000, help="UB of Batch size.")
        parser.add_argument("--max_grad_norm_lb", type=float, default=0.2, help="LB of maximum gradient norm.")
        parser.add_argument("--max_grad_norm_ub", type=float, default=10.0, help="UB of maximum gradient norm.")
        parser.add_argument("--learning_rate_lb", type=float, default=1e-7, help="LB of learning rate")
        parser.add_argument("--learning_rate_ub", type=float,  default=1e-2, help="UB of learning rate")
        # DP options
        parser.add_argument("--private", dest="private", default=False, action="store_true",
                            help="If true, use differential privacy.")
        parser.add_argument("--noise_multiplier", type=float, default=1.0, help="Noise multiplier.")
        parser.add_argument("--target_epsilon", type=float, default=10.0, help="Maximum value of epsilon allowed.")
        parser.add_argument("--target_delta", type = float, default = 1e-5, help="The delta for DP training.")
        parser.add_argument("--max_physical_batch_size", type=int, default=400, help="Maximum physical batch size")
        parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam')
        parser.add_argument("--secure_rng", dest="secure_rng", default=False, action="store_true",
                            help="If true, use secure RNG for DP-SGD.")
        parser.add_argument("--accountant", type=str, default = "rdp",
                            help="The nature of the accountant used for privacy engine.")
        
        # LiRA options
        parser.add_argument("--save_models", type=bool, default=True,
                                    help="If True, save the models trained for LiRA.")
        parser.add_argument("--start_data_split", type=int, default=0,
                            help="Starting index of data split to train tfor the LiRA attack.")
        parser.add_argument("--stop_data_split", type=int, default=257,
                            help="Stopping number of data split to train tfor the LiRA attack.")
        parser.add_argument("--start_hypers", type=int, default=0,
                            help="Starting index of hypers to train tfor the LiRA attack.")
        parser.add_argument("--stop_hypers", type=int, default=257,
                            help="Stopping index of hypers to train tfor the LiRA attack.")
        parser.add_argument("--num_shadow_models", type=int, default=256,
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

            self.dataset_reader = CachedFeatureLoader(path_to_cache_dir=self.args.download_path_for_tensorflow_datasets,
                                                      dataset=dataset["name"],
                                                      feature_extractor = self.args.feature_extractor,
                                                      optuna_trials = self.args.number_of_trials,
                                                      random_seed=self.args.seed
                                                      )
            
            self.feature_dim = self.dataset_reader.obtain_feature_dim()
            train_features, train_labels,_, self.class_mapping = self.dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                                                                        n_classes=self.num_classes,
                                                                                                                        task="train")

            # load the hypers and training data splits
            hypers_file_path = os.path.join(self.directory, 'opt_args_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf'))
            
            if not os.path.isfile(hypers_file_path):
                tune_features, tune_labels,tune_indices,_ = self.dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                                    n_classes=self.num_classes,
                                                                                    task="tune")
                
                tune_data_splits = [] # record of tune splits
                n = tune_features.shape[0]
                for idx in range(0,self.args.num_shadow_models+1):
                    np.random.seed(idx + 1 + self.args.seed)
                    D_i = np.random.binomial(1, 0.5, n).astype(bool)
                    x_i, y_i = tune_features[D_i], tune_labels[D_i]
                    tune_data_splits.append(D_i)
                    opt_args_i,_ = optimize_hyperparameters(idx, self.args, x_i, y_i, self.feature_dim, self.num_classes, self.args.seed) 
                    self.hypers["learning_rate"].append(opt_args_i.learning_rate)
                    self.hypers["batch_size"].append(opt_args_i.train_batch_size)
                    if opt_args_i.private:
                        self.hypers["max_grad_norm"].append(opt_args_i.max_grad_norm)

                with open(hypers_file_path, 'wb') as f:
                    pickle.dump(self.hypers,f)  

                with open(os.path.join(self.directory, 'tune_indices_{}_{}_{}.pkl'.format(
                        self.args.learnable_params,
                        self.args.examples_per_class,
                        int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                    pickle.dump(tune_indices,f) 

                with open(os.path.join(self.directory, 'tune_splits_{}_{}_{}.pkl'.format(
                        self.args.learnable_params,
                        self.args.examples_per_class,
                        int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                    pickle.dump(tune_data_splits,f)        

            else:
                with open(hypers_file_path, 'rb') as f:
                    self.hypers = pickle.load(f)

            data_file_path = os.path.join(self.directory,'in_indices_{}_{}_{}.pkl'.format(
                    self.args.learnable_params,
                    self.args.examples_per_class,
                    int(self.args.target_epsilon) if self.args.private else 'inf'))
            
            if os.path.isfile(data_file_path):
                with open(data_file_path, 'rb') as f:
                    self.data_splits = pickle.load(f)
            
            else:
                self.data_splits = []               
                for idx in range(0,self.args.num_shadow_models + 1):
                    np.random.seed(idx + 1 + self.args.seed)
                    self.data_splits.append(np.random.binomial(1, 0.5, n).astype(bool))
                
                with open(data_file_path,"wb") as f:
                    pickle.dump(self.data_splits,f)

            n = 2 * self.num_classes * self.args.examples_per_class
            self.model_stats = np.zeros(shape=(self.args.stop_data_split - self.args.start_data_split,
                                               self.args.stop_hypers - self.args.start_hypers,
                                               n, 
                                               1))
            
            print("Shape of stats array =",self.model_stats.shape)
            self.run_lira(
                x=train_features,
                y=train_labels,
                test_dataset_reader=self.dataset_reader
            )

    def train_test(
            self,
            x,y,
            num_classes,
            array_coords =(0,0),
            test_set_reader=None,
            sample_weight=None):

        self.start_time_final_run = datetime.now()
        i,j = array_coords
        in_train_features, in_train_labels = x[self.data_splits[i]].to(self.device), y[self.data_splits[i]].to(self.device)
        out_train_features, out_train_labels = x[~self.data_splits[i]].to(self.device), y[~self.data_splits[i]].to(self.device)

        self.args.learning_rate = self.hypers["learning_rate"][j]
        self.args.train_batch_size = self.hypers["batch_size"][j]
        if self.args.private:
            self.args.max_grad_norm = self.hypers["max_grad_norm"][j]

        train_loader = DataLoader(
            TensorDataset(in_train_features, in_train_labels),
            batch_size= self.args.train_batch_size if self.args.private else min(self.args.train_batch_size, 
                                                                                 self.args.max_physical_batch_size),
            shuffle=True) 

        model = self.create_head(feature_dim=self.feature_dim, num_classes=num_classes)

        if self.args.classifier == 'linear':
            self.eps, self.delta = self.fine_tune_batch(model=model, train_loader=train_loader)
            in_accuracy = self.validate_linear(model, train_loader)
            self.accuracies["in"][i][j - self.args.start_hypers] = in_accuracy 
            accuracy = (self.test_linear(model=model, dataset_reader=test_set_reader)).cpu()
            self.accuracies["test"][i][j - self.args.start_hypers] = accuracy
        else:
            print("Invalid classifier option.")
            sys.exit()

        out_dataloader = DataLoader(
                        TensorDataset(out_train_features,out_train_labels),
                        batch_size= self.args.train_batch_size if self.args.private else min(self.args.train_batch_size, self.args.max_physical_batch_size),
                        shuffle=True) 
        
        out_accuracy = self.validate_linear(model, out_dataloader)
        self.accuracies["out"][i][j - self.args.start_hypers] = out_accuracy
        
        print(f'Trained model #{i,j} with {self.data_splits[i].sum()} examples. Test Accuracy = {accuracy}. Epsilon = {self.eps}')
        stats,_ = self.get_stat_and_loss_aug(model, x, y.numpy(), sample_weight)

        self.model_stats[i][j - self.args.start_hypers] = stats # store the stats associated with model[i][j] 
        # free up memory
        if self.args.save_models:
            model_dir = os.path.join(self.directory,"lira_models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            filename = os.path.join(model_dir, 'model_{}_{}.pkl'.format(i+1, (j - self.args.start_hypers) + 1))       
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def fine_tune_batch(self, model, train_loader):
        model.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        delta = None
        if self.args.private:
            delta = self.args.target_delta
            privacy_engine = PrivacyEngine(accountant=self.args.accountant, secure_mode=self.args.secure_rng)

            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=self.args.target_epsilon,
                epochs=self.args.epochs,
                target_delta=delta,
                max_grad_norm=self.args.max_grad_norm)

        if self.args.private:
            for _ in range(self.args.epochs):
                with BatchMemoryManager(
                        data_loader=train_loader,
                        max_physical_batch_size=self.args.max_physical_batch_size,
                        optimizer=optimizer
                ) as new_train_loader:
                    for batch_images, batch_labels in new_train_loader:
                        batch_images = batch_images.to(self.device)
                        batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                        optimizer.zero_grad()
                        torch.set_grad_enabled(True)
                        logits = model(batch_images)
                        loss = self.loss(logits, batch_labels)
                        loss.backward()
                        del logits
                        optimizer.step()
                        torch.cuda.empty_cache()

        else:
            for _ in range(self.args.epochs):
                for batch_images, batch_labels in train_loader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                    optimizer.zero_grad()
                    torch.set_grad_enabled(True)
                    logits = model(batch_images)
                    loss = self.loss(logits, batch_labels)
                    loss.backward()     
                    del logits
                    optimizer.step()
                    torch.cuda.empty_cache()

        eps = None
        if self.args.private:
            eps = privacy_engine.get_epsilon(delta=delta)

        return eps, delta

    def validate_linear(self, model, val_loader):
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.type(torch.LongTensor).to(DEVICE)
                logits = model(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
            predictions = torch.hstack(predictions)
            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy

    def test_linear(self, model, dataset_reader):
        model.eval()

        with torch.no_grad():
            labels = []
            predictions = []
            test_features,test_labels = dataset_reader.load_test_data(class_mapping=self.class_mapping)
            test_loader = DataLoader(
                    TensorDataset(test_features, test_labels),
                    batch_size= self.args.test_batch_size,
                    shuffle=True)                
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                logits = model(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
                torch.cuda.empty_cache()
            predictions = torch.hstack(predictions)
            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy

    def run_lira(self, x, y, test_dataset_reader):
        # Sample weights are set to `None` by default, but can be changed here.
        sample_weight = None
        for i in range(self.args.start_data_split, self.args.stop_data_split): # dataset loop
            for j in range(self.args.start_hypers,self.args.stop_hypers): # hyperparameters loop
                self.train_test(x,y,
                                self.num_classes,
                                array_coords=(i,j),
                                test_set_reader=test_dataset_reader,
                                sample_weight=sample_weight)

        # save stat, and train/test accuracies
        filename = os.path.join(self.directory, 'stat_{}_{}_{}_r_{}_to_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf',
                self.args.start_hypers,
                self.args.stop_hypers))
        
        with open(filename, 'wb') as f:
            pickle.dump(self.model_stats, f)

        filename = os.path.join(self.directory, 'accs_{}_{}_{}_r_{}_to_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf',
                self.args.start_hypers,
                self.args.stop_hypers))
        
        with open(filename, 'wb') as f:
            pickle.dump(self.accuracies, f)

    def get_stat_and_loss_aug(self,
            model,
            x,
            y,
            sample_weight=None):
        """A helper function to get the statistics and losses.

        Here we get the statistics and losses for the images.

        Args:
            model: model to make prediction
            x: samples
            y: true labels of samples (integer valued)
            sample_weight: a vector of weights of shape (n_samples, ) that are
                assigned to individual samples. If not provided, then each sample is
                given unit weight. Only the LogisticRegressionAttacker and the
                RandomForestAttacker support sample weights.
            batch_size: the batch size for model.predict

        Returns:
            the statistics and cross-entropy losses
        """
        losses, stat= [], []
        data = x.to(DEVICE)
        with torch.no_grad():
            logits = model(data).cpu().numpy()
        prob = convert_logit_to_prob(logits)
        losses.append(log_loss(y, prob, sample_weight=sample_weight))
        stat.append(calculate_statistic(prob, y, sample_weight=sample_weight, is_logits=False))
        return np.expand_dims(np.concatenate(stat), axis=1), np.expand_dims(np.concatenate(losses), axis=1)

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()