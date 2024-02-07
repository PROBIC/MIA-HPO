import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#import scikitplot as skplt
import numpy as np
import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from dataset import dataset_map
from utils import Logger, get_git_revision_short_hash, get_slurm_job_id, limit_tensorflow_memory_usage, CsvWriter, get_mean_percent,\
    compute_accuracy_from_predictions, predict_by_max_logit, cross_entropy_loss, shuffle, set_seeds
from tf_dataset_reader import TfDatasetReader
from datetime import datetime
from model import DpFslLinear
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import optuna
import gc
import joblib
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import sys
import warnings
import torch.distributed as dist
import tensorflow as tf
import random
from dataset import dataset_map


def main():

    args = parse_command_line()
    
    datasets = dataset_map[args.dataset]
    for dataset in datasets:
        if dataset['enabled'] is False:
            continue

        if args.examples_per_class == -1:
            context_set_size = -1  # this is the use the entire training set case
            # bug in pets
        elif (args.examples_per_class is not None) and (dataset['name'] != 'oxford_iiit_pet'):
            context_set_size = args.examples_per_class * dataset['num_classes']  # few-shot case
        else:
            context_set_size = 1000  # VTAB1000

        num_classes = dataset['num_classes']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        dataset_reader = TfDatasetReader(
                dataset=dataset['name'],
                task=dataset['task'],
                context_batch_size=context_set_size,
                target_batch_size=args.test_batch_size,
                path_to_datasets=args.download_path_for_tensorflow_datasets,
                num_classes=dataset['num_classes'],
                image_size=224 if 'vit' in args.feature_extractor else dataset['image_size'],
                examples_per_class=args.examples_per_class if args.examples_per_class != -1 else None,
                examples_per_class_seed=args.seed,
                tfds_seed=args.seed,
                device=device,
                osr=False
        )

        model_name = 'R-50_seed=0_S=100_C=100'
        model = init_model(args, num_classes, device)
        load_model(args, model=model, file_name=model_name) # load the fine-tuned model
        
        # loop over the complete cifar100 dataset (both train and test) and compare the predictions and the true label to create the confusion matrix
        #produce y_pred, y_true
        predictions, labels = predict(args, model, dataset_reader, device)
    
        #plot full confusion matrix
        #fig_conf = plot_confusion_matrix(labels, predictions, range(100), range(100))
        #fig_conf.savefig(os.path.join(args.path_to_save_fig, "confusion_matrix"))
        
        #calculate acc per class: correctly classified samples / samples in the class 
        acc_per_class = confusion_matrix(labels, predictions, normalize="true").diagonal()
        print("Prediction acc per class: ", acc_per_class)
        #save acc_per_class
        with open(os.path.join(args.checkpoint_dir, 'acc_per_class.txt'), 'w') as file:
            file.write('\n'.join(str(acc) for acc in acc_per_class))

        #check if the confusion matrix is correct, can be deleted later
        print("Mean of acc per class: ", sum(acc_per_class)/len(acc_per_class))

        with open(args.path_to_label_names) as f:
            label_names = [line.rstrip() for line in f]
        
        print("Class name and corresponding label: ", dict(zip(label_names, range(100))))
        
        #rank the class based on pred_acc per class
        ranking, classes = sort_classes(acc_per_class, label_names)
        print("Ranking: ", ranking)
        print("Class rank: ", ranking.keys())
        fig_acc = plot_accuracy_per_class(ranking)
        fig_acc.savefig(os.path.join(args.path_to_save_fig, "accuracy_per_class"))
        
        classes.reverse()
        #plot
        for i in [8]:
            fig = plot_confusion_matrix(labels, predictions, classes[0:i], [label_names[j] for j in classes[0:i]])
            fig.savefig(os.path.join(args.path_to_save_fig, f"{i}_most_confused_classes"))

        misclassifications = find_misclassifications (labels, predictions, label_names)
        print(misclassifications)   
            
def parse_command_line():
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', help='Dataset to use.',
                            choices=dataset_map.keys(), default="small_set")
        parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='BiT-M-R50x1', help="Feature extractor to use.")
        parser.add_argument("--classifier", choices=['linear'], default='linear',
                            help="Which classifier to use.")
        parser.add_argument("--learnable_params", choices=['none', 'all', 'film'], default='film',
                            help="Which feature extractor parameters to learn.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--learning_rate", "-lr",
                            type=float, default=0.003, help="Learning rate.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints',
                            help="Directory to save checkpoint to.")
        parser.add_argument("--epochs", "-e", type=int,
                            default=400, help="Number of fine-tune epochs.")
        parser.add_argument("--train_batch_size", "-b",
                            type=int, default=200, help="Batch size.")
        parser.add_argument("--test_batch_size", "-tb",
                            type=int, default=600, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--seed", type=int, default=0,
                            help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--save_model", dest="save_model", default=False,
                            action="store_true", help="If true, save the fine tuned model.")
        parser.add_argument("--allowed_classes", nargs='+', type=int, default=None,
                           help="A list of classes we are interested.")
        parser.add_argument("--number_of_classes", type=int, default=None,
                            help="Number of classes we are to look at.")
        parser.add_argument("--random_select", dest="random_select", default=False,
                            action="store_true", help="If true, Randomly select C classes.")
        parser.add_argument("--path_to_label_names", default=None,
                            help="Path to access the label names.")
        parser.add_argument("--path_to_save_fig", default=None,
                            help="Path to save the figures.")

        # differential privacy options
        parser.add_argument("--private", dest="private", default=False, action="store_true",
                            help="If true, use differential privacy.")
        parser.add_argument("--noise_multiplier", type=float,
                            default=1.0, help="Noise multiplier.")
        parser.add_argument("--max_grad_norm", type=float,
                            default=1.0, help="Maximum gradient norm.")
        parser.add_argument("--target_epsilon", type=float,
                            default=10.0, help="Maximum value of epsilon allowed.")
        parser.add_argument("--max_physical_batch_size", type=int,
                            default=400, help="Maximum physical batch size")
        parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam')
        parser.add_argument("--secure_rng", dest="secure_rng", default=False, action="store_true",
                            help="If true, use secure RNG for DP-SGD.")

        # tuning params
        parser.add_argument("--tune_params", dest="tune_params", default=False, action="store_true",
                            help="If true, tune hyper-parameters.")
        parser.add_argument("--epochs_lb", type=int,
                            default=20, help="LB of fine-tune epochs.")
        parser.add_argument("--epochs_ub", type=int,
                            default=200, help="UB of fine-tune epochs.")
        parser.add_argument("--train_batch_size_lb", type=int,
                            default=10, help="LB of Batch size.")
        parser.add_argument("--train_batch_size_ub", type=int,
                            default=1000, help="UB of Batch size.")
        parser.add_argument("--max_grad_norm_lb", type=float,
                            default=0.2, help="LB of maximum gradient norm.")
        parser.add_argument("--max_grad_norm_ub", type=float,
                            default=10.0, help="UB of maximum gradient norm.")
        parser.add_argument("--learning_rate_lb", type=float,
                            default=1e-7, help="LB of learning rate")
        parser.add_argument("--learning_rate_ub", type=float,
                            default=1e-2, help="UB of learning rate")
        parser.add_argument("--save_optuna_study", dest="save_optuna_study", default=True, action="store_true",
                            help="If true, save optuna studies.")
        parser.add_argument("--number_of_trials", type=int,
                            default=20, help="The number of trials for optuna")
        parser.add_argument("--optuna_starting_checkpoint", default=None,
                            help="Path of a optuna checkpoint from which to start the study again. (Updates the number of trials)")
        args = parser.parse_args()
        return args

def init_model(args, num_classes, device):
    if args.classifier == 'linear':
        model = DpFslLinear(
            feature_extractor_name=args.feature_extractor,
            num_classes=num_classes,
            learnable_params=args.learnable_params
        )
    else:
        print("Invalid classifier option.")
        sys.exit()

    # print parameters, but only once
    #if self.print_parameter_count:
     #   self.get_parameter_count(model)
      #  self.print_parameter_count = False
    model = model.to(device)

    return model

def load_model(args, model, file_name):
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, file_name)))
    
def predict(args, model, dataset_reader, device):
    model.eval()
    with torch.no_grad():
        labels = []
        predictions = []
        #test data
        test_set_size = dataset_reader.get_target_dataset_length()
        num_batches = int(np.ceil(float(test_set_size) / float(args.test_batch_size)))
        for batch in range(num_batches):
            batch_images, batch_labels = dataset_reader.get_target_batch()        
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.type(torch.LongTensor).to(device)  
            logits = model(batch_images)
            predictions.append(predict_by_max_logit(logits))
            labels.append(batch_labels)
            del logits
        

        #train data
        train_images, train_labels = dataset_reader.get_context_batch()
        train_loader = DataLoader(
                TensorDataset(train_images, train_labels),
                batch_size=args.train_batch_size if args.private else min(args.train_batch_size,
                                                                    args.max_physical_batch_size),
                shuffle=True
            )
        
        for data in train_loader:
            batch_images, batch_labels = data
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.type(torch.LongTensor).to(device)  
            logits = model(batch_images)
            predictions.append(predict_by_max_logit(logits))
            labels.append(batch_labels)
            del logits
        
                 
        # on both CIFAR100 train and test data
        predictions = torch.hstack(predictions).cpu() #y_pred
        labels = torch.hstack(labels).cpu() #y_true

        return predictions, labels

def plot_confusion_matrix(y_true, y_pred, labels, display_labels):

    #plt.rcParams.update({'font.size': 45})
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=display_labels, xticks_rotation=45, values_format='d', cmap='Blues', ax=ax)
    plt.show()
    return fig
    
def plot_accuracy_per_class(ranking):
    
    sorted_labels = list(ranking.keys())
    sorted_acc = list(ranking.values())

    fig = plt.figure(figsize=(50, 25))
    bars = plt.bar(range(100), sorted_acc, width=0.4)
    
    hexadecimal_alphabets = '0123456789ABCDEF'
    colors = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in
                range(6)]) for i in range(100)]
    for i in range(len(colors)):
        bars[i].set_color(colors[i])
    
    plt.xticks(range(100), sorted_labels, rotation=45)
    plt.ylabel("prediction accuracy")
    plt.title("Prediction accuracy for each class")
    for i in range(len(sorted_labels)):
        plt.text(i, sorted_acc[i]+0.001, round(sorted_acc[i], 2), ha = 'center')
    plt.show()
    
    return fig

def sort_classes(acc_per_class, label_names):
    pred_accuracy = dict(zip(range(100), acc_per_class))
    ranking = sorted(pred_accuracy.items(), key=lambda x:x[1], reverse=True)
    
    classes = []
    accuracies = []
    for cl, acc in ranking:
        classes.append(cl)
        accuracies.append(acc)

    ranking = dict(zip([label_names[j] for j in classes], accuracies))

    return ranking, classes

def find_misclassifications (labels, predictions, label_names):
    cm_full = confusion_matrix(labels, predictions)
    misclassifications = dict()
    for i in range(100):
        l = list(cm_full[i])
        d = dict(zip(label_names, l))
        #print(f"The number of samples in class {i} is {sum(l)}")
        d = sorted(d.items(), key=lambda x:x[1], reverse=True)
        incorrects = d[1:4]
        print(f"{label_names[i]} is easy to be confused with: ", incorrects)
        misclassifications[label_names[i]] = incorrects
    return misclassifications


if __name__ == '__main__':
    main()
    