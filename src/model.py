import torch
import torch.nn as nn
import numpy as np
import sys
from timm.models.efficientnet import tf_efficientnetv2_s_in21k, tf_efficientnetv2_m_in21k, tf_efficientnetv2_l_in21k, \
    tf_efficientnetv2_xl_in21k, efficientnet_b0
from timm.models.vision_transformer import vit_tiny_patch16_224_in21k, vit_small_patch16_224_in21k, \
    vit_small_patch32_224_in21k, vit_base_patch8_224_in21k, vit_base_patch16_224_in21k, vit_base_patch32_224_in21k, \
    vit_large_patch16_224_in21k, vit_large_patch32_224_in21k, vit_huge_patch14_224_in21k
from timm.models.nfnet import nfnet_f0
from bit_resnet import KNOWN_MODELS
from film import enable_film, get_film_parameter_names


def create_feature_extractor(feature_extractor_name, learnable_params):
    num_classes = 0

    if feature_extractor_name == 'efficientnet-b0':
        feature_extractor = efficientnet_b0(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'efficientnet-v2-s':
        feature_extractor = tf_efficientnetv2_s_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'efficientnet-v2-m':
        feature_extractor = tf_efficientnetv2_m_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'efficientnet-v2-l':
        feature_extractor = tf_efficientnetv2_l_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'efficientnet-v2-xl':
        feature_extractor = tf_efficientnetv2_xl_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-ti-16':
        feature_extractor = vit_tiny_patch16_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-s-16':
        feature_extractor = vit_small_patch16_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-s-32':
        feature_extractor = vit_small_patch32_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-b-8':
        feature_extractor = vit_base_patch8_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-b-16':
        feature_extractor = vit_base_patch16_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-b-32':
        feature_extractor = vit_base_patch32_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-l-16':
        feature_extractor = vit_large_patch16_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-l-32':
        feature_extractor = vit_large_patch32_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'vit-h-14':
        feature_extractor = vit_huge_patch14_224_in21k(pretrained=True, num_classes=num_classes)
    elif feature_extractor_name == 'nfnet-f0':
        feature_extractor = nfnet_f0(pretrained=True, num_classes=num_classes)
    elif 'BiT' in feature_extractor_name:
        feature_extractor = KNOWN_MODELS[feature_extractor_name](head_size=num_classes, zero_head=True)
        feature_extractor.load_from(np.load(f"{feature_extractor_name}.npz"))
    else:
        print("Invalid feature extractor option.")
        sys.exit()

    #feature_extractor = torch.nn.DataParallel(feature_extractor)

    film_param_names = None
    if learnable_params == 'film':
        # freeze all the model parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False

        film_param_names = get_film_parameter_names(
            feature_extractor_name=feature_extractor_name,
            feature_extractor=feature_extractor
        )

        enable_film(film_param_names, feature_extractor)
    elif learnable_params == 'none':
        # freeze all the model parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False

    return feature_extractor, film_param_names


class DpFslLinear(nn.Module):
    def __init__(self, feature_extractor_name, num_classes, learnable_params):
        super(DpFslLinear, self).__init__()

        self.feature_extractor, _ = create_feature_extractor(
            feature_extractor_name=feature_extractor_name,
            learnable_params=learnable_params
        )

        # send a test signal through the feature extractor to get its feature dimension
        feature_extractor_dim = self.feature_extractor(torch.Tensor(1, 3, 224, 224)).size(1)

        self.head = nn.Linear(feature_extractor_dim, num_classes)
        self.head.weight.data.fill_(0.0)
        self.head.bias.data.fill_(0.0)

    def forward(self, x):
        return self.head(self.feature_extractor(x))
