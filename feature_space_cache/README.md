## Cache feature representations

It is more computationally efficient to cache feature representations and load them for training models in the HEAD setting where only the final layer has to be trained.

Use `feature_space_cache/map_to_feature_space.py` to save representations obtained from pre-trained models (ViT-B/R-50) from datasets in the feature dimension. This has to be only done once for each dataset.
