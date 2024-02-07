import torch
import os
import numpy as np


class CachedFeatureLoader:
    def __init__(self,
                 path_to_cache_dir: str, 
                dataset: str = "cifar10", 
                feature_extractor: str = "BiT-M-R50x1",
                optuna_trials:int=1,
                random_seed: int = 0):
        self.dataset = dataset
        self.path_to_cache_dir = path_to_cache_dir
        self.feature_extractor = feature_extractor
        self.optuna_trials = optuna_trials
        self.random_seed = random_seed

    def _load_complete_data_from_disk(self, train: True):
        split = "train" if train else "test"
        amount_of_data = "32768" if self.dataset == "patch_camelyon" and train else "-1"
        features = torch.load(os.path.join(self.path_to_cache_dir, self.dataset, f"features_{split}_{amount_of_data}_{self.feature_extractor}.pt"))
        labels = torch.load(os.path.join(self.path_to_cache_dir, self.dataset, f"labels_{split}_{amount_of_data}_{self.feature_extractor}.pt"))
        
        # load validation data for dtd and flowers and add to training data
        if train and self.dataset in ["dtd", "oxford_flowers102", "diabetic_retinopathy_detection"]:
            split = "validation"
            features_val = torch.load(os.path.join(self.path_to_cache_dir, self.dataset, f"features_{split}_{amount_of_data}_{self.feature_extractor}.pt"))
            labels_val = torch.load(os.path.join(self.path_to_cache_dir, self.dataset, f"labels_{split}_{amount_of_data}_{self.feature_extractor}.pt"))
            features = torch.cat([features, features_val], dim=0)
            labels = torch.cat([labels, labels_val], dim=0)
        return features, labels

    def _subsample_classes(self, labels: torch.Tensor, n_of_classes_to_select: int, sampling_method: str):
        n_available_classes = len(torch.unique(labels))
        np.random.seed(self.random_seed)
        # input check
        if n_available_classes < n_of_classes_to_select:
            raise ValueError("There are not enough classes available to select.")
        if n_of_classes_to_select == 0 or n_of_classes_to_select < -1:
            raise ValueError("No classes selected")

        # just return all classes
        if n_of_classes_to_select == n_available_classes or n_of_classes_to_select == -1:
            return np.array(range(0, n_available_classes))

        if sampling_method == "random":
            selected_classes = np.random.choice(list(range(0, n_available_classes)), size=n_of_classes_to_select, replace=False)

        assert n_of_classes_to_select == len(selected_classes)
        return np.array(selected_classes)

    def _subsample_shots(self, features: torch.Tensor, labels: torch.Tensor, classes: list, shots: int,task:str="train"):
        if not (shots != -1 or shots < 1 or shots * len(classes) > len(labels)):
            raise ValueError("The number of shots is not appropriate for the dataset.")

        selected_elements_list = list()
        if shots != -1:
            _, class_counts = np.unique(labels, return_counts=True)
            mia_training_indices = []
            for c in classes:
                idx_selected = np.random.choice(np.where(labels == c)[0],
                                            min(2*shots, class_counts[c]),
                                            replace=False
                                            )
                if shots < class_counts[c]:
                    class_counts[c] -= 2*shots
                mia_training_indices.extend(idx_selected)
            selected_elements_list.extend(mia_training_indices) # add selected training indices to selected elements list

            # update the indices map --> remove training indices
            indices_map = {} 
            for c in classes:
                indices_map[c] = []
                for idx in np.where(labels == c)[0]:  
                    if idx not in mia_training_indices:
                        indices_map[c].append(idx)

            tuning_indices = []
            for _ in range(self.optuna_trials):
                for c in classes:
                    idx_selected = np.random.choice(indices_map[c],
                                                    shots,
                                                    replace=True)
                    tuning_indices.extend(idx_selected) 
            selected_elements_list.extend(tuning_indices) 

            all_selected_features = features[selected_elements_list,:]
            all_selected_labels = labels[selected_elements_list]
            # map labels to interval without gaps (e.g., 1, 2 instead of 20, 50)
            class_mapping = dict()
            for c_i, c in enumerate(sorted(torch.unique(all_selected_labels))):
                all_selected_labels[all_selected_labels == c] = c_i
                original_class_name = str(c.item())
                class_mapping[original_class_name] = c_i

            print("Common indices between Training and Tuning data:={}".format(set(mia_training_indices).intersection(set(tuning_indices))))  

            print("Train set size = {}".format(len(mia_training_indices)))
            print("Tuning set size = {}".format(len(tuning_indices)))

            if task == "train":
                return all_selected_features[:len(mia_training_indices),:], all_selected_labels[:len(mia_training_indices)], mia_training_indices, class_mapping
            elif task == "tune":
                return all_selected_features[len(mia_training_indices):,:], all_selected_labels[len(mia_training_indices):], tuning_indices, class_mapping
            else:
                raise ValueError("Not a legitimate task!")

    def obtain_feature_dim(self):
        """Returns the feature dimension of the cached data."""
        all_test_features, _ = self._load_complete_data_from_disk(train=True)
        return all_test_features.shape[1]

    def load_train_data(self, shots: int, n_classes: int,task:str = "train"):
        """
        Loads training data based on shots and n_classes.
        The sampling is done at random.
        """
        np.random.seed(self.random_seed)
        all_train_features, all_train_labels = self._load_complete_data_from_disk(train=True)

        selected_classes = self._subsample_classes(all_train_labels, n_classes, sampling_method="random")
        train_features, train_labels, data_indices, class_mapping = self._subsample_shots(
            all_train_features, all_train_labels, selected_classes, shots, task=task)

        return train_features, train_labels, data_indices, class_mapping
    
    def load_test_data(self, class_mapping=None):
        """
        Loads test data based. 
        It is possible to pass an array of classes to select datafrom.
        """
        all_test_features, all_test_labels = self._load_complete_data_from_disk(train=False)
        if class_mapping is not None:
            selected_element_list = list()
            for c in class_mapping.keys():
                # get all elements of class
                selected_indicies = np.array(all_test_labels == float(c))
                selected_element_list.append(selected_indicies)

            selected_elements = np.array(selected_element_list)
            selected_elements = np.sum(selected_elements, axis=0).astype(dtype=bool)

            selected_test_features = all_test_features[selected_elements, :]
            selected_test_labels = all_test_labels[selected_elements]

            # use the correct label (the same mapping as in the selection of the training)
            for c in sorted(torch.unique(selected_test_labels)):
                selected_test_labels[selected_test_labels == c] = class_mapping.get(str(c.item()))

            assert len(torch.unique(selected_test_labels)) == len(list(class_mapping.keys()))
            assert sorted(torch.unique(selected_test_labels)) == sorted(class_mapping.values())
            return selected_test_features, selected_test_labels
        else:
            return all_test_features, all_test_labels