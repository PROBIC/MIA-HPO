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
        self.optimal_args = {"learning_rate":[],
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
        parser.add_argument("--type_of_tuning", type=int, default=0, 
                            help="For TD-HPO set the variable to 0, for ED-HPO set it to 1.")
        parser.add_argument("--ed_hpo_repeats", type=int, default=1, help="The number of trials for optuna for ED-HPO")

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

        self.accuracies = {"in":np.zeros((self.args.num_shadow_models + 1,)),
                           "out":np.zeros((self.args.num_shadow_models + 1,)),
                           "test":np.zeros((self.args.num_shadow_models + 1,))}
        
        # ensure the directory to hold results exists
        self.exp_dir = f"experiment_{self.args.exp_id}"
        self.run_dir = f"Run_{self.args.run_id}"
        self.directory = os.path.join(self.args.results, "Seed={}".format(self.args.seed),self.run_dir, self.exp_dir)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

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
            with open(os.path.join(self.directory, 'train_{}_{}_{}.pkl'.format(
                    self.args.learnable_params,
                    self.args.examples_per_class,
                    int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                    pickle.dump(training_indices, f)

            # create the training and tuning datasets
            if self.args.type_of_tuning == 0:
                print("Total samples used for MIA and HPO training ={}".format(len(train_features)))
            else:
                tune_features, tune_labels, tune_indices,_ = self.dataset_reader.load_train_data(shots=self.args.examples_per_class, 
                                                                                                 n_classes=self.num_classes,
                                                                                                 task="tune")
                print("Total samples used for MIA training ={}".format(len(train_features)))  
                print("Total samples used for HPO tuning ={}".format(len(tune_features)))  

                with open(os.path.join(self.directory, 'tune_{}_{}_{}.pkl'.format(
                        self.args.learnable_params,
                        self.args.examples_per_class,
                        int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
                        pickle.dump(tune_indices, f) 

                for t in range(self.args.ed_hpo_repeats):
                    self.args,_ = optimize_hyperparameters(t+1, self.args, tune_features, tune_labels, self.feature_dim, self.num_classes, self.args.seed)
                    self.optimal_args["learning_rate"].append(self.args.learning_rate)
                    self.optimal_args["batch_size"].append(self.args.train_batch_size)
                    if self.args.private:
                        self.optimal_args["max_grad_norm"].append(self.args.max_grad_norm)

            self.run_lira(
                x=train_features,
                y=train_labels,
                test_dataset_reader=self.dataset_reader
            )

    def train_test(
            self,
            train_features,
            train_labels,
            num_classes,
            test_set_reader=None,
            model_index=None):

        self.start_time_final_run = datetime.now()
        # tuning each shadow model ONLY for TD setting
        if self.args.type_of_tuning == 0:
            self.args,_ = optimize_hyperparameters(model_index, self.args, train_features, train_labels, self.feature_dim, num_classes, self.args.seed) 
            self.optimal_args["learning_rate"].append(self.args.learning_rate)
            self.optimal_args["batch_size"].append(self.args.train_batch_size)
            if self.args.private:
                self.optimal_args["max_grad_norm"].append(self.args.max_grad_norm)
        else:
            # tune a given shadow model from hypers randomly chosen from a pre-compiled list
            choose_hypers = np.random.choice(np.arange(0,self.args.ed_hpo_repeats),1)[0]
            self.args.learning_rate = self.optimal_args["learning_rate"][choose_hypers]
            self.args.train_batch_size = self.optimal_args["batch_size"][choose_hypers]
            if self.args.private:
                self.args.max_grad_norm = self.optimal_args["max_grad_norm"][choose_hypers]

        train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size= self.args.train_batch_size if self.args.private else min(self.args.train_batch_size, self.args.max_physical_batch_size),
            shuffle=True) 

        model = self.create_head(feature_dim=self.feature_dim, num_classes=num_classes)

        if self.args.classifier == 'linear':
            self.eps, self.delta = self.fine_tune_batch(model=model, train_loader=train_loader)
            accuracy = self.validate_linear(model,train_loader)
            self.accuracies["in"][model_index] = accuracy 
            
            if test_set_reader is not None:  # use test set for testing
                accuracy = (self.test_linear(model=model, dataset_reader=test_set_reader)).cpu()
                self.accuracies["test"][model_index] = accuracy
            else:
                accuracy = 0.0  # don't test
        else:
            print("Invalid classifier option.")
            sys.exit()

        if model_index is not None:
            self.models[model_index] = model

        # free up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return accuracy, self.eps

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
        n = x.shape[0]
        # delta = self.args.target_delta

        # Train the target and shadow models. We will use one of the model in `models`
        # as target and the rest as shadow.
        # Here we use the same architecture and optimizer. In practice, they might
        # differ between the target and shadow models.
        in_indices = []  # a list of in-training indices for all models
        stat = []  # a list of statistics for all models
        losses = []  # a list of losses for all models
        for idx in range(self.args.num_shadow_models + 1):
            # Generate a binary array indicating which example to include for training
            np.random.seed(idx + 1 + self.args.seed) # set the seeding for in-samples to model index + 1 + parent seed
            in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))

            model_train_images = x[in_indices[-1]]
            model_train_labels = y[in_indices[-1]]
            model_train_images = model_train_images.to(self.device)
            model_train_labels = model_train_labels.to(self.device)

            accuracy, eps = self.train_test(
                train_features=model_train_images,
                train_labels=model_train_labels,
                num_classes=self.num_classes,
                test_set_reader=test_dataset_reader,
                model_index = idx  # save the model, so we can load it and get challenge example losses
            )
            curr_model = self.models[idx]

            out_dataloader = DataLoader(
                            TensorDataset(x[~in_indices[-1]], y[~in_indices[-1]]),
                            batch_size= self.args.train_batch_size if self.args.private else min(self.args.train_batch_size, self.args.max_physical_batch_size),
                            shuffle=True) 
            
            out_accuracy = self.validate_linear(curr_model, out_dataloader)
            self.accuracies["out"][idx] = out_accuracy
            
            print(f'Trained model #{idx} with {in_indices[-1].sum()} examples. Test Accuracy = {accuracy}. Epsilon = {eps}')
            s, l = self.get_stat_and_loss_aug(curr_model, x, y.numpy(), sample_weight)
            if np.isnan(s).any():
                print(f"NaNs found for model #{idx}")
            stat.append(s)
            losses.append(l)

            # Avoid OOM
            gc.collect()

        # save stat, losses, in_indices, optimal hypers, train/test accuracies
        with open(os.path.join(self.directory, 'stat_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(stat, f)
        with open(os.path.join(self.directory, 'losses_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(losses, f)
        with open(os.path.join(self.directory, 'in_indices_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(in_indices, f)
        with open(os.path.join(self.directory, 'opt_args_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(self.optimal_args, f) 

        with open(os.path.join(self.directory, 'accs_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
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
