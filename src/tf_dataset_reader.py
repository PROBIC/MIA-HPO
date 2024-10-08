import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
MAX_IN_MEMORY = 50000

class TfDatasetReader:
    def __init__(self,
                 dataset,
                 task,
                 context_batch_size:int,
                 target_batch_size:int,
                 path_to_datasets:str,
                 num_classes,
                 image_size,
                 device="cuda", 
                 examples_per_class=None,
                 examples_per_class_seed=0,
                 tfds_seed=0,
                 optuna_trials:int=1,
                 osr=False):
        self.dataset = dataset
        self.task = task
        self.device = device
        self.image_size = image_size
        self.context_batch_size = context_batch_size
        self.target_batch_size = target_batch_size
        self.osr = osr
        self.ood_labels = None
        self.examples_per_class = examples_per_class
        self.examples_per_class_seed = examples_per_class_seed
        self.optuna_trials = optuna_trials
        tf.compat.v1.enable_eager_execution()

        if (self.examples_per_class is not None) and (dataset == "oxford_iiit_pet"):
            # special case as pets does not respect skip decoding, so we'll decode everything and trim later
            train_split = 'train'
            decoders = None
        elif self.examples_per_class is not None:
            train_split = 'train[:98%]'
            decoders = {'image': tfds.decode.SkipDecoding()}
        elif context_batch_size == -1:
            train_split = 'train'
            decoders = None
        else:
            train_split = 'train[:{}]'.format(context_batch_size)
            decoders = None
        ds_context, ds_context_info = tfds.load(
            dataset,
            split=train_split,
            shuffle_files=True,
            data_dir=path_to_datasets,
            with_info=True,
            decoders=decoders,
            read_config=tfds.ReadConfig(shuffle_seed=tfds_seed)
        )
        if (self.examples_per_class is not None) and (dataset != "oxford_iiit_pet"):
            ds_train_context, ds_tuning_context = self.sample_subset(
                ds_context,
                ds_context_info.splits["train"].num_examples,
                num_classes,
                self.examples_per_class,
                self.examples_per_class_seed
            )  
            self.decoder = ds_context_info.features['image'].decode_example
    
        self.training_iterator = ds_train_context.as_numpy_iterator()
        self.tuning_iterator = ds_tuning_context.as_numpy_iterator()
            
        self.training_dataset_length = ds_context_info.splits["train"].num_examples
        if self.context_batch_size == -1:  # all
            self.context_batch_size = self.context_dataset_length

        test_split = 'test'
        if self.dataset == 'clevr':
            test_split = 'validation'
        if 'test' in ds_context_info.splits:
            # we use the entire test set
            ds_target, ds_target_info = tfds.load(
                dataset,
                split=test_split,
                shuffle_files=False,
                data_dir=path_to_datasets,
                with_info=True)
            self.target_dataset_length = ds_target_info.splits["test"].num_examples
        else:  # there is no test split
            # get a second iterator to the training set and skip the training examples
            if self.examples_per_class is not None:
                test_split = 'train'  # this is a potential problem as the training examples will be in the test set - but we are not doing low shot on these datasets
            else:
                # TODO when test all, what do we do here -  - but we are not doing 'all' on these datasets
                test_split = 'train[{}:]'.format(context_batch_size)
            ds_target = tfds.load(
                dataset, 
                split=test_split,
                shuffle_files=False,
                data_dir=path_to_datasets
            )
            if self.examples_per_class is not None:
                self.target_dataset_length = self.context_dataset_length
            else:
                self.target_dataset_length = self.context_dataset_length - context_batch_size

        self.target_iterator = ds_target.as_numpy_iterator()
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def get_training_batch(self):
        return self._get_batch(self.training_iterator, is_target=False)

    def get_tuning_batch(self):
        return self._get_batch(self.tuning_iterator, is_target=False)

    def get_target_batch(self):
        return self._get_batch(self.target_iterator, is_target=True)

    def get_context_dataset_length(self):
        return self.context_dataset_length

    def get_target_dataset_length(self):
        return self.target_dataset_length

    def _get_batch(self, iterator, is_target=False):
        
        if is_target:
            batch_size = self.target_batch_size 
        else:
            batch_size = 2 * self.context_batch_size
        images = []
        labels = []
        for i in range(batch_size):
            try:
                item = iterator.next()
            except StopIteration:  # the last batch may be less than batch_size
                break

            # images
            if (self.examples_per_class is not None) and (not is_target) and (self.dataset != "oxford_iiit_pet"):
                images.append(self._prepare_image(self.decoder(item['image']).numpy()))
            else:
                images.append(self._prepare_image(item['image']))

            # labels
            if self.dataset == "clevr":
                labels.append(self._get_clevr_label(item, self.task))
            elif self.dataset == 'kitti':
                labels.append(self._get_kitti_label(item))
            elif self.dataset == 'smallnorb':
                if self.task == 'azimuth':
                    labels.append(item['label_azimuth'])
                elif self.task == 'elevation':
                    labels.append(item['label_elevation'])
                else:
                    raise ValueError("Unsupported smallnorb task.")
            elif self.dataset == "dsprites":
                labels.append(self._get_dsprites_label(item, self.task))
            elif self.dataset == "i_naturalist2018":
                labels.append(item['supercategory'])
            else:
                labels.append(item['label'])

        labels = np.array(labels)
        images = torch.stack(images)

        if not is_target and self.osr:
            unique_labels = np.unique(labels)
            num_labels = len(unique_labels)
            self.ood_labels = unique_labels[int(np.ceil(num_labels/2)):]
            images = images[labels < self.ood_labels[0]]
            labels = labels[labels < self.ood_labels[0]]
            print(f'number of in-distribution context images: {len(labels)}')
        elif is_target and self.osr:
            labels[labels > self.ood_labels[0]] = self.ood_labels[0]

        labels = torch.from_numpy(labels)
        labels = labels.type(torch.LongTensor)

        if (self.examples_per_class is not None) and (not is_target) and (self.dataset == "oxford_iiit_pet"):
            # set fixed examples per class for pets
            classes, class_counts = torch.unique(labels, return_counts=True)

            np.random.seed(self.examples_per_class_seed)
            indices = [idx
                       for c in range(len(classes))
                       for idx in np.random.choice(np.where(labels.numpy() == c)[0],
                                                   min(self.examples_per_class, class_counts[c]),
                                                   replace=False)
                       ]
            indices = torch.Tensor(indices).type(torch.LongTensor)
            images = torch.index_select(images, 0, indices)
            labels = torch.index_select(labels, 0, indices)

        return images, labels

    def _get_kitti_label(self, x):
        """Predict the distance to the closest vehicle."""
        # Location feature contains (x, y, z) in meters w.r.t. the camera.
        vehicles = np.where(x["objects"]["type"] < 3)  # Car, Van, Truck.
        vehicle_z = np.take(x["objects"]["location"][:, 2], vehicles)
        if len(vehicle_z.shape) > 1:
            vehicle_z = np.squeeze(vehicle_z, axis=0)
        if vehicle_z.size == 0:
            vehicle_z = np.array([1000.0])
        else:
            vehicle_z = np.append(vehicle_z, [1000.0], axis=0)
        dist = np.amin(vehicle_z)
        # Results in a uniform distribution over three distances, plus one class for "no vehicle".
        thrs = np.array([-100.0, 8.0, 20.0, 999.0])
        label = np.amax(np.where((thrs - dist) < 0))
        return label

    def _get_dsprites_label(self, item, task):
        num_classes = 16
        if task == "location":
            predicted_attribute = 'label_x_position'
            num_original_classes = 32
        elif task == "orientation":
            predicted_attribute = 'label_orientation'
            num_original_classes = 40
        else:
            raise ValueError("Bad dsprites task.")

        # at the desired number of classes. This is useful for example for grouping
        # together different spatial positions.
        class_division_factor = float(num_original_classes) / float(num_classes)

        return np.floor(float(item[predicted_attribute]) / class_division_factor)

    def _get_clevr_label(self, item, task):
        if task == "count":
            label = len(item["objects"]["size"]) - 3
        elif task == "distance":
            dist = np.amin(item["objects"]["pixel_coords"][:, 2])
            # These thresholds are uniformly spaced and result in more or less balanced
            # distribution of classes, see the resulting histogram:
            thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
            label = np.amax(np.where((thrs - dist) < 0))
        else:
            raise ValueError("Bad clevr task.")

        return label

    def _prepare_image(self, image):
        if self.dataset == "smallnorb" or self.dataset == "dsprites":
            # grayscale images where the channel needs to be squeezed to keep PIL happy
            image = np.squeeze(image)

        if self.dataset == "dsprites":  # scale images to be in 0 - 255 range to keep PIL happy
            image = image * 255.0

        im = Image.fromarray(image)
        im = im.resize((self.image_size, self.image_size), Image.LANCZOS)
        im = im.convert("RGB")
        return self.transforms(im)

    def sample_subset(self, data, num_examples, num_classes, examples_per_class, examples_per_class_seed):
        
        data = data.batch(min(num_examples, MAX_IN_MEMORY))
        data = data.as_numpy_iterator().next()

        np.random.seed(examples_per_class_seed)
            
        _, class_counts = np.unique(data["label"], return_counts=True)

        mia_training_indices, ed_indices = [],[]

        for c in range(num_classes):
            idx_selected = np.random.choice(np.where(data["label"] == c)[0],
                                        min(2*examples_per_class, class_counts[c]),
                                        replace=False)
            if 2 * examples_per_class < class_counts[c]: # update the number of remainin samples
                class_counts[c] -= examples_per_class
            mia_training_indices.extend(idx_selected)
        
        # update the indices map --> remove training indices
        indices_map = {} 
        for c in range(num_classes):
            indices_map[c] = []
            for idx in np.where(data["label"] == c)[0]:  
                if idx not in mia_training_indices:
                    indices_map[c].append(idx)
            
        for c in range(num_classes):
            idx_selected = np.random.choice(indices_map[c],
                                        min(2 * examples_per_class, class_counts[c]),
                                        replace=False)
            ed_indices.extend(idx_selected)            
        

        training_data = {'image': data['image'][mia_training_indices],
                'label': data['label'][mia_training_indices]}
        training_data = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(training_data['image']),
            tf.data.Dataset.from_tensor_slices(training_data['label'])))

        tuning_data = {'image': data['image'][ed_indices],
                'label': data['label'][ed_indices]}
        tuning_data = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(tuning_data['image']),
            tf.data.Dataset.from_tensor_slices(tuning_data['label'])))
        
        print(len(mia_training_indices))
        print(len(ed_indices))
        print("Common Indices in Training and Tuning Data = ".format(set(mia_training_indices) & set(ed_indices)))

        return  training_data.map(lambda x, y: {'image': x, 'label': y}, tf.data.experimental.AUTOTUNE),\
                tuning_data.map(lambda x, y: {'image': x, 'label': y}, tf.data.experimental.AUTOTUNE)
        

