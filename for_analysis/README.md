### Plotting LiRA results:

Use ```python3 plot_roc_curve.py``` to visualize the performance of different attacks using the use the ROC curve (comparing TPR against TPR) on the log-log scale to visualize the performance of different attacks.
Requisite inputs to the file include:

```
--score_dir <path to the composite score file with likelihood scores under different attacks>
--examples_per_class <number of shots>
--learnable_params <trainable layers of the model>
--target_epsilon <level of DP>
```
