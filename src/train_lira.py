# Portions of this code are excerpted from:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py

import numpy as np
import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from dataset import dataset_map
from utils import Logger, limit_tensorflow_memory_usage,\
    compute_accuracy_from_predictions, predict_by_max_logit, cross_entropy_loss, shuffle, set_seeds
from tf_dataset_reader import TfDatasetReader
from datetime import datetime
from model import DpFslLinear
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import gc
import sys
import warnings
from lira import convert_logit_to_prob, calculate_statistic, log_loss
import pickle


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, 'log.txt')
        self.start_time = datetime.now()
        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss = cross_entropy_loss
        self.print_parameter_count = True
        self.eps = None
        self.delta = None
        self.tune_images = None
        self.tune_labels = None
        self.num_classes = None
        self.exp_dir = None
        self.run_dir = None
        # self.models = {}
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
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints',
                            help="Directory to save checkpoint to.")
        parser.add_argument("--test_batch_size", "-tb", type=int, default=600, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--optuna_trials", type=int, default=1, help="Number of trials used for HP tuning.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--exp_id", type=int, default=None,
                            help="Experiment ID.")
        parser.add_argument("--run_id", type=int, default=None,
                            help="Run ID for rerunning the whole experiment.")

        # differential privacy options
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

    def init_model(self, num_classes):
        if self.args.classifier == 'linear':
            model = DpFslLinear(
                feature_extractor_name=self.args.feature_extractor,
                num_classes=num_classes,
                learnable_params=self.args.learnable_params
            )
        else:
            print("Invalid classifier option.")
            sys.exit()

        # print parameters, but only once
        if self.print_parameter_count:
            self.get_parameter_count(model)
            self.print_parameter_count = False

        model = model.to(self.device)

        return model

    def run(self):
        # seeding
        set_seeds(self.args.seed)

        limit_tensorflow_memory_usage(2048)

        self.logger.print_and_log("")  # add a blank line

        datasets = dataset_map[self.args.dataset]

        with open(os.path.join(self.args.results, "Seed={}".format(self.args.seed),
                                        "Run_{}".format(self.args.run_id),
                                        "experiment_{}".format(self.args.exp_id), 
                                        'best_hyperparameters.pkl'), "rb") as f:
            best_hyperparameters = pickle.load(f)
        
        if self.args.private:
            self.args.max_grad_norm = best_hyperparameters['max_grad_norm']
        self.args.train_batch_size = best_hyperparameters['train_batch_size']
        self.args.learning_rate = best_hyperparameters['learning_rate']
        self.args.epochs = best_hyperparameters['epochs']

        for dataset in datasets:
            if dataset['enabled'] is False:
                continue

            self.num_classes = dataset['num_classes']

            if self.args.examples_per_class == -1:
                context_set_size = -1  # this is the use the entire training set case
            elif (self.args.examples_per_class is not None) and (dataset['name'] != 'oxford_iiit_pet'):  # bug in pets
                context_set_size = self.args.examples_per_class * self.num_classes  # few-shot case
            else:
                context_set_size = 1000  # VTAB1000
                
            self.dataset_reader = TfDatasetReader(
                dataset=dataset['name'],
                task=dataset['task'],
                context_batch_size=context_set_size,
                target_batch_size=self.args.test_batch_size,
                path_to_datasets=self.args.download_path_for_tensorflow_datasets,
                num_classes=dataset['num_classes'],
                image_size=224 if 'vit' in self.args.feature_extractor else dataset['image_size'],
                examples_per_class=self.args.examples_per_class if self.args.examples_per_class != -1 else None,
                examples_per_class_seed=self.args.seed,
                tfds_seed=self.args.seed,
                device=self.device,   
                optuna_trials= self.args.optuna_trials,     
                osr=False
            )

            # create the training dataset
            train_images, train_labels = self.dataset_reader.get_mia_context_batch()
            print("Total samples used for shadow model training={}".format(len(train_images)))
            
            self.logger.print_and_log("{}".format(dataset['name']))
            self.exp_dir = f"experiment_{self.args.exp_id}"
            self.run_dir = f"Run_{self.args.run_id}"

            self.run_lira(
                x=train_images,
                y=train_labels,
                dataset_reader=self.dataset_reader
            )

    def train_test(
            self,
            train_images,
            train_labels,
            num_classes,
            data,
            sample_weight,
            test_set_reader=None,
            save_model_name=None):

        self.start_time_final_run = datetime.now()
        train_loader = DataLoader(
            TensorDataset(train_images, train_labels),
            batch_size= self.args.train_batch_size if self.args.private else min(self.args.train_batch_size, self.args.max_physical_batch_size),
            shuffle=True) 

        model = self.init_model(num_classes=num_classes)

        if self.args.classifier == 'linear':
            self.eps, self.delta = self.fine_tune_batch(model=model, train_loader=train_loader)
            if test_set_reader is not None:  # use test set for testing
                accuracy = (self.test_linear(model=model, dataset_reader=test_set_reader)).cpu()
            else:
                accuracy = 0.0  # don't test
        else:
            print("Invalid classifier option.")
            sys.exit()

        if save_model_name is not None:
            self.save_model(model=model, file_name=save_model_name)
        else:
            # Get the statistics of the current model.
            stats, losses = self.get_stat_and_loss_aug(model, data["x"], data["y"].numpy(), sample_weight)

        # free up memory
        del model
        del data
        gc.collect()
        torch.cuda.empty_cache()

        return accuracy, self.eps, (stats,losses)

    def fine_tune_batch(self, model, train_loader):
        r = torch.cuda.memory_reserved()/(1024*1024*1024)
        print("Total amount of memory reserved for fine-tuning:{}".format(r))
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
                        del loss
                        optimizer.step()

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
                    del loss
                    optimizer.step()

        torch.cuda.empty_cache()        
        a = torch.cuda.memory_allocated()/(1024*1024*1024)
        print("Total amount of memory allocated for fine-tuning:{}".format(a))
        print("Remaining free CUDA memory = {}".format(r-a))
        
        eps = None
        if self.args.private:
            eps = privacy_engine.get_epsilon(delta=delta)

        return eps, delta

    def test_linear(self, model, dataset_reader):
        model.eval()
        with torch.no_grad():
            labels = []
            predictions = []
            test_set_size = dataset_reader.get_target_dataset_length()
            num_batches = int(np.ceil(float(test_set_size) / float(self.args.test_batch_size)))
            for _ in range(num_batches):
                batch_images, batch_labels = dataset_reader.get_target_batch()
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                logits = model(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
            predictions = torch.hstack(predictions)
            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy

    def run_lira(self, x, y, dataset_reader):
            
        # Sample weights are set to `None` by default, but can be changed here.
        sample_weight = None
        n = x.shape[0]
        data = {"x":x, "y":y}
        # Train the target and shadow models. We will use one of the model in `models`
        # as target and the rest as shadow.
        # Here we use the same architecture and optimizer. In practice, they might
        # differ between the target and shadow models.
        in_indices = []  # a list of in-training indices for all models
        stat = []  # a list of statistics for all models
        loss = []
        for idx in range(self.args.num_shadow_models + 1):
            print('Training model #{}'.format(idx+1))
            # model_save_name = os.path.join(self.args.checkpoint_dir, self.run_dir, self.exp_dir,f'model{idx+1}.pt')
            # Generate a binary array indicating which example to include for training
            in_indices.append(np.random.binomial(1, 0.5, n).astype(bool))

            model_train_images = x[in_indices[-1]]
            model_train_labels = y[in_indices[-1]]
            model_train_images = model_train_images.to(self.device)
            model_train_labels = model_train_labels.to(self.device)

            accuracy, eps,curr_model_stats = self.train_test(
                train_images=model_train_images,
                train_labels=model_train_labels,
                num_classes=self.num_classes,
                data=data,
                sample_weight=sample_weight,
                test_set_reader=dataset_reader if idx == 0 else None,
                save_model_name=None  #save the model, so we can load it and get challenge example losses
            )
            
            print(f'Trained model #{idx} with {in_indices[-1].sum()} examples. Accuracy = {accuracy}. Epsilon = {eps}')
            stat.append(curr_model_stats[0]) # Get the statistics of the current model.
            loss.append(curr_model_stats[1])

            # Avoid OOM
            del model_train_images
            del model_train_labels
            
            gc.collect()
            torch.cuda.empty_cache()
            
        # save stat, in_indices, and losses
        directory = os.path.join(self.args.checkpoint_dir, "Seed={}".format(self.args.seed),self.run_dir, self.exp_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(self.args.checkpoint_dir, "Seed={}".format(self.args.seed), self.run_dir, self.exp_dir, 'stat_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(stat, f)
        with open(os.path.join(self.args.checkpoint_dir, "Seed={}".format(self.args.seed), self.run_dir, self.exp_dir, 'losses_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(loss, f)
        with open(os.path.join(self.args.checkpoint_dir, "Seed={}".format(self.args.seed), self.run_dir, self.exp_dir, 'in_indices_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                int(self.args.target_epsilon) if self.args.private else 'inf')), 'wb') as f:
            pickle.dump(in_indices, f)

    def get_parameter_count(self, model):
        model_param_count = sum(p.numel() for p in model.parameters())
        self.logger.print_and_log("Model Parameter Count = {}".format(model_param_count))
        trainable_model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.print_and_log("Model Trainable Parameter Count = {}".format(trainable_model_param_count))

        feature_extractor_param_count = sum(p.numel() for p in model.feature_extractor.parameters())
        self.logger.print_and_log("Feature Extractor Parameter Count = {}".format(feature_extractor_param_count))
        trainable_feature_extractor_param_count = sum(p.numel() for p in model.feature_extractor.parameters() if p.requires_grad)
        self.logger.print_and_log("Feature Extractor Trainable Parameter Count = {}".format(trainable_feature_extractor_param_count))

        if self.args.classifier == 'linear':
            head_param_count = sum(p.numel() for p in model.head.parameters())
        else:
            head_param_count = 0
        self.logger.print_and_log("Head Parameter Count = {}".format(head_param_count))
        if self.args.classifier == 'linear':
            trainable_head_param_count = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        else:
            trainable_head_param_count = 0
        self.logger.print_and_log("Head Trainable Parameter Count = {}".format(trainable_head_param_count))

    def save_model(self, model, file_name):
        # torch.save(model.state_dict(), os.path.join(self.args.checkpoint_dir, file_name))
        directory = os.path.join(self.args.checkpoint_dir, self.run_dir, self.exp_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), os.path.join(directory, file_name))

    def load_model(self, model, file_name):
        # model.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, file_name)))
        model.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, self.run_dir, self.exp_dir, file_name)))

    def _get_model_path(self, file_name):
        # return os.path.join(self.args.checkpoint_dir, file_name)
        return os.path.join(self.args.checkpoint_dir, self.run_dir, self.exp_dir, file_name)

    def get_stat_and_loss_aug(self,
                              model,
                              x,
                              y,
                              sample_weight=None,
                              batch_size=4096):
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
        losses, stat = [], []
        data = x.to(self.device)
        data_size = len(data)
        num_sub_batches = self._get_number_of_sub_batches(data_size, self.args.test_batch_size)
        for batch in range(num_sub_batches):
            batch_start_index, batch_end_index = self._get_sub_batch_indices(batch, data_size, self.args.test_batch_size)
            with torch.no_grad():
                logits = model(data[batch_start_index: batch_end_index]).cpu().numpy()
            prob = convert_logit_to_prob(logits)
            losses.append(log_loss(y[batch_start_index: batch_end_index], prob, sample_weight=sample_weight))
            stat.append(calculate_statistic(prob, y[batch_start_index: batch_end_index], sample_weight=sample_weight, is_logits=False))
        return np.expand_dims(np.concatenate(stat), axis=1), np.expand_dims(np.concatenate(losses), axis=1)

    def _get_number_of_sub_batches(self, task_size, sub_batch_size):
        num_batches = int(np.ceil(float(task_size) / float(sub_batch_size)))
        if num_batches > 1 and (task_size % sub_batch_size == 1):
            num_batches -= 1
        return num_batches

    def _get_sub_batch_indices(self, index, task_size, sub_batch_size):
        batch_start_index = index * sub_batch_size
        batch_end_index = batch_start_index + sub_batch_size
        if batch_end_index == (task_size - 1):  # avoid batch size of 1
            batch_end_index = task_size
        if batch_end_index > task_size:
            batch_end_index = task_size
        return batch_start_index, batch_end_index


if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
