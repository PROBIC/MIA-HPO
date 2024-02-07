import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from feature_space_cache.bit_resnet import KNOWN_MODELS
from feature_space_cache.tf_dataset_reader import TfDatasetReader
from timm.models.vision_transformer import vit_base_patch16_224_in21k
from utils import limit_tensorflow_memory_usage
from feature_space_cache.vtab_datasets import dataset_map


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        print("Options: %s\n" % self.args)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_size = 224 if 'vit' in self.args.feature_extractor else 384

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                            default='vit-b-16', help="Feature extractor to use.")
        parser.add_argument("--dataset_path", default=None, help="Path to the ImageNet dataset.")
        parser.add_argument("--feature_dim_path", help="Directory to save feature dim images to.")
        parser.add_argument("--batch_size", help="Batch size.", type=int, default=10)
        parser.add_argument('--dataset', help='Dataset to use.', choices=list(dataset_map.keys()) + ["ImageNet-1k"], default="ImageNet-1k")
        parser.add_argument('--test_set', help='Use test partition.', action="store_true", default=False)
        parser.add_argument("--seed", type=int, default=0, help="Seed for random routines.")
        parser.add_argument("--image_size", type=int, default=224, help="Image height and width.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        args = parser.parse_args()
        return args

    def run(self):
        if self.args.dataset == "ImageNet-1k":
            dataset = torchvision.datasets.ImageFolder(
                self.args.dataset_path,
                self.normalize_images)
        else:
            limit_tensorflow_memory_usage(2048)
            dataset = dataset_map[self.args.dataset]
            if len(dataset) > 1:
                raise ValueError("Can only transform one dataset at the time.")
            else:
                dataset = dataset[0]

                if self.args.examples_per_class == -1:
                    context_set_size = -1  # this is the use the entire training set case
                elif (self.args.examples_per_class is not None) and (dataset['name'] != 'oxford_iiit_pet'):  # bug in pets
                    context_set_size = self.args.examples_per_class * dataset['num_classes']  # few-shot case
                else:
                    context_set_size = 1000  # VTAB1000

                target_batch_size = 400
                dataset_reader = TfDatasetReader(
                    dataset=dataset['name'],
                    task=dataset['task'],
                    context_batch_size=context_set_size,
                    target_batch_size=target_batch_size,
                    path_to_datasets=self.args.download_path_for_tensorflow_datasets,
                    num_classes=dataset['num_classes'],
                    image_size=self.args.image_size,
                    examples_per_class=self.args.examples_per_class if self.args.examples_per_class != -1 else None,
                    examples_per_class_seed=self.args.seed,
                    tfds_seed=self.args.seed,
                    device=self.device
                )
            if not self.args.test_set:
                # train part
                train_images, train_labels = dataset_reader.get_context_batch()
                dataset = torch.utils.data.TensorDataset(train_images, train_labels)
            else:
                # test part
                test_set_size = dataset_reader.get_target_dataset_length()
                num_batches = int(np.ceil(float(test_set_size) /
                                          float(target_batch_size)))

                list_of_images = []
                list_of_labels = []
                for _ in range(num_batches):
                    batch_images, batch_labels = dataset_reader.get_target_batch()
                    list_of_images.append(batch_images)
                    list_of_labels.append(batch_labels)

                test_images = torch.concat(list_of_images)
                test_labels = torch.concat(list_of_labels)
                dataset = torch.utils.data.TensorDataset(test_images, test_labels)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, pin_memory=self.args.dataset == "ImageNet-1k")

        # compute features
        self.compute_features(data_loader)

    def compute_features(self, train_loader):
        model = DpFslLinear(feature_extractor_name=self.args.feature_extractor)

        model = model.to(self.device)

        feature_list = list()
        label_list = list()
        with torch.no_grad():
            for b_indx, (batch_images, batch_labels) in enumerate(train_loader):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                features = model(batch_images)
                feature_list.append(features.detach().cpu())
                label_list.append(batch_labels.detach().cpu())

        partition = "test" if self.args.test_set else "train"
        torch.save(torch.concat(feature_list), f=os.path.join(self.args.feature_dim_path,
                   f"features_{partition}_{self.args.examples_per_class}_{self.args.feature_extractor}.pt"))
        torch.save(torch.concat(label_list), f=os.path.join(self.args.feature_dim_path,
                   f"labels_{partition}_{self.args.examples_per_class}_{self.args.feature_extractor}.pt"))

    def normalize_images(self, image):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        ])

        im = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        im = im.convert("RGB")

        return transforms(im)


def create_feature_extractor(feature_extractor_name):
    num_classes = 0

    if feature_extractor_name == 'vit-b-16':
        feature_extractor = vit_base_patch16_224_in21k(pretrained=True, num_classes=num_classes)
    elif 'BiT' in feature_extractor_name:
        feature_extractor = KNOWN_MODELS[feature_extractor_name](head_size=num_classes, zero_head=True)
        feature_extractor.load_from(np.load(f"{feature_extractor_name}.npz"))
    else:
        print("Invalid feature extractor option.")
        sys.exit()

    return feature_extractor


class DpFslLinear(nn.Module):
    def __init__(self, feature_extractor_name):
        super(DpFslLinear, self).__init__()

        self.feature_extractor = create_feature_extractor(
            feature_extractor_name=feature_extractor_name
        )

    def forward(self, x):
        return self.feature_extractor(x)


if __name__ == "__main__":
    main()
