import torch
import sys
import gc
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from datetime import datetime
from model import DpFslLinear
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import (compute_accuracy_from_predictions,
                   cross_entropy_loss, 
                   predict_by_max_logit, 
                   set_seeds,
                   shuffle)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DISTRIBUTED_RANK = None
# custom alphas: alpha has to be > 1 and we add a bit more alphas than the default (max. 255 instead of 63)
CUSTOM_ALPHAS = [1.01, 1.05] + [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))

def create_head(feature_dim: int, num_classes: int):
    head = nn.Linear(feature_dim, num_classes)
    head.weight.data.fill_(0.0)
    head.bias.data.fill_(0.0)
    head.to(DEVICE)
    return head

def _get_sub_batch_indices(index,total_batch_size,max_physical_batch_size):
    batch_start_index = index * max_physical_batch_size
    batch_end_index = batch_start_index + max_physical_batch_size
    if batch_end_index == (total_batch_size - 1):  # avoid batch size of 1
        batch_end_index = total_batch_size
    if batch_end_index > total_batch_size:
        batch_end_index = total_batch_size
    return batch_start_index, batch_end_index

def _get_number_of_sub_batches(task_size,max_physical_batch_size):
    num_batches = int(np.ceil(float(task_size) / float(max_physical_batch_size)))
    if num_batches > 1 and (task_size % max_physical_batch_size == 1):
        num_batches -= 1
    return num_batches

def train_test(train_images, train_labels, args, feature_dim, num_classes, test_set_reader=None, validate=False):
        # seed when not doing hyperparameter tuning
    if not validate:
        set_seeds(args.seed)

    batch_size = args.train_batch_size

    if validate:  # tune hyper-parameters
        if args.examples_per_class is not None:
            train_images, train_labels = shuffle(train_images, train_labels)

        train_partition_size = int(0.7 * len(train_labels))
        train_loader = DataLoader(
            TensorDataset(train_images[:train_partition_size], train_labels[:train_partition_size]),
            batch_size=min(batch_size, train_partition_size),
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(train_images[train_partition_size:], train_labels[train_partition_size:]),
            batch_size=args.test_batch_size,
            shuffle=False
        )
    else:  # testing
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(args.seed)
        train_loader = DataLoader(
            TensorDataset(train_images, train_labels),
            batch_size=batch_size if args.private else min(args.train_batch_size,
                                                                args.max_physical_batch_size),
            shuffle=True,
            generator=train_loader_generator
        )

    if args.classifier == 'linear':
        if args.learnable_params == "none":
            model = create_head(feature_dim, num_classes)
        else:
            model = DpFslLinear(
                feature_extractor_name=args.feature_extractor,
                num_classes=num_classes,
                learnable_params=args.learnable_params
            )
    else:
        print("Invalid classifier option.")
        sys.exit()

    if DISTRIBUTED_RANK is not None:
        model = DPDDP(model)

    model = model.to(DEVICE)

    if args.classifier == 'linear':
        eps, privacy_engine = fine_tune_batch(model=model, train_loader=train_loader, args=args)
        if validate:
            accuracy = (validate_linear(model=model, val_loader=val_loader)).cpu()
        else:
            accuracy = (test_linear(model=model, dataset_reader=test_set_reader,args=args)).cpu()
    else:
        print("Invalid classifier option.")
        sys.exit()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy, eps, privacy_engine

def fine_tune_batch(model, train_loader, args):
    # r = torch.cuda.memory_reserved()/(1024*1024*1024)
    # print("Total amount of memory reserved for fine-tuning:{}".format(r))
    model.train()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.private:

        privacy_engine = PrivacyEngine(accountant=args.accountant, 
                                        secure_mode=args.secure_rng)

        seeded_noise_generator = torch.Generator(device=DEVICE)
        seeded_noise_generator.manual_seed(args.seed)

        if args.accountant == "rdp": 
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=args.target_epsilon,
                    epochs=args.epochs,
                    target_delta=args.target_delta,
                    max_grad_norm=args.max_grad_norm,
                    noise_generator=seeded_noise_generator if not args.secure_rng else None,
                    alphas=CUSTOM_ALPHAS)
        else: # No alpha term needed if no RDP accountant is being used
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=args.target_epsilon,
                    epochs=args.epochs,
                    target_delta=args.target_delta,
                    max_grad_norm=args.max_grad_norm,
                    noise_generator=seeded_noise_generator if not args.secure_rng else None)
    if args.private:
        for _ in range(args.epochs):
            with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=args.max_physical_batch_size,
                    optimizer=optimizer
            ) as new_train_loader:
                for batch_images, batch_labels in new_train_loader:
                    batch_images = batch_images.to(DEVICE)
                    batch_labels = batch_labels.type(torch.LongTensor).to(DEVICE)
                    optimizer.zero_grad()
                    torch.set_grad_enabled(True)
                    logits = model(batch_images)
                    loss = cross_entropy_loss(logits, batch_labels)
                    loss.backward()
                    del logits
                    optimizer.step()

    else:
        for _ in range(args.epochs):
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.type(torch.LongTensor).to(DEVICE)
                optimizer.zero_grad()
                torch.set_grad_enabled(True)
                batch_size = len(batch_labels)
                num_sub_batches = _get_number_of_sub_batches(batch_size,args.max_physical_batch_size)
                for batch in range(num_sub_batches):
                    batch_start_index, batch_end_index = _get_sub_batch_indices(batch, batch_size, args.max_physical_batch_size)
                    logits = model(batch_images[batch_start_index: batch_end_index])
                    loss = cross_entropy_loss(logits, batch_labels[batch_start_index: batch_end_index])
                    loss.backward()
                    del logits
                optimizer.step()
                
    torch.cuda.empty_cache()
    # a = torch.cuda.memory_allocated()/(1024*1024*1024)
    # print("Total amount of memory allocated for fine-tuning:{}".format(a))
    # print("Remaining free CUDA memory = {}".format(r-a))

    eps = None
    if args.private:
        # we need to call the accountant directly because we want to pass custom alphas
        if args.accountant == "rdp":
            eps = privacy_engine.accountant.get_epsilon(delta=args.target_delta, alphas=CUSTOM_ALPHAS)
        else:
            eps = privacy_engine.accountant.get_epsilon(delta=args.target_delta)
        return eps, privacy_engine
    else:
        return eps, None

def test_linear(model, dataset_reader,args):
    model.eval()
    with torch.no_grad():
        labels = []
        predictions = []
        test_set_size = dataset_reader.get_target_dataset_length()
        num_batches = int(np.ceil(float(test_set_size) / float(args.test_batch_size)))
        for _ in range(num_batches):
            batch_images, batch_labels = dataset_reader.get_target_batch()
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

def validate_linear(model, val_loader):
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


def objective_func(trial, train_features, train_labels, args, feature_dim:int, num_classes: int):

    if args.private:
        args.max_grad_norm = trial.suggest_float(
            'max_grad_norm',
            args.max_grad_norm_lb,
            args.max_grad_norm_ub
        )

    args.train_batch_size = trial.suggest_int(
        'batch_size',
        args.train_batch_size_lb,
        args.train_batch_size_ub
    )

    args.learning_rate = trial.suggest_float(
        'learning_rate',
        args.learning_rate_lb,
        args.learning_rate_ub
    )
    args.epochs = trial.suggest_int(
        'epochs',
        args.epochs_lb,
        args.epochs_ub
    )

    val_accuracy,_,_ = train_test(train_features,train_labels,args,feature_dim,num_classes,validate=True)
    return val_accuracy


def optimize_hyperparameters(idx, args, train_images, train_labels, feature_dim, num_classes, seed):
    # hyperparameter optimization
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(study_name=f"dp_mia_{idx}", direction="maximize", sampler=sampler)

    if args.type_of_tuning == 1:
            context_size = args.examples_per_class * num_classes
            tuning_batches = []
            for i in range(0,context_size*args.number_of_trials,context_size):
                curr_batch = (train_images[i:i+context_size,],train_labels[i:i+context_size])
                tuning_batches.append(curr_batch)
                
            assert len(tuning_batches) == args.number_of_trials

            for t in range(args.number_of_trials): 
                curr_tuning_features, curr_tuning_labels = tuning_batches[t]
                study.optimize(lambda trial: objective_func(trial, curr_tuning_features, curr_tuning_labels,
                   args, feature_dim, num_classes), n_trials=1)
    else:    
        study.optimize(lambda trial: objective_func(trial, train_images, train_labels,
                    args, feature_dim, num_classes), n_trials=args.number_of_trials)

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

    if args.private:
        args.max_grad_norm = trial.params['max_grad_norm']
    args.train_batch_size = trial.params['batch_size']
    args.learning_rate = trial.params['learning_rate']
    args.epochs = trial.params['epochs']

    return args, trial
