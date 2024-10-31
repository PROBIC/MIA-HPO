import numpy as np
import sys
import os.path
import argparse
import pickle
from sklearn.metrics import roc_curve
import warnings 
import matplotlib.pyplot as plt

def plot_roc_curve(scores, fpr_points): 
    y_score = -scores["y_score"]
    y_true = np.logical_not(scores["y_true"]).astype(int)
    assert len(y_true) == len(y_score)
    fpr, tpr, _ = roc_curve(y_true = y_true, y_score = y_score)
    tpr_at_fpr = []
    for fpr_point in fpr_points:
        tpr_at_fpr.append(np.interp(x=fpr_point, xp=fpr, fp=tpr))
    return (fpr,tpr, tpr_at_fpr)

def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--scores_dir", help="Directory to load scores from.")
        parser.add_argument("--examples_per_class", type=int, default=-1,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--target_epsilon", type=int, default=-1,
                            help="Level of DP used to train the models")
        parser.add_argument("--learnable_params", choices=['none', 'film'], default='none',
                            help="Which feature extractor parameters to learn.") 
        args = parser.parse_args()
        return args
        
    def run(self):

        self.args.target_epsilon = self.args.target_epsilon if self.args.target_epsilon != -1 else "inf"
        filename = os.path.join(self.args.scores_dir, 'scores_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                self.args.target_epsilon))
        
        with open(filename, 'rb') as f:
            scores = pickle.load(f)

        palette = {"ACC-LiRA":'lightcoral',"KL-LiRA":'mediumseagreen',"WB-LiRA":'blue'}
        FPRS = [1e-3,1e-2,1e-1]
        fig,ax = plt.subplots(1,1,figsize=(4,4))
        for mia in scores.keys():
            fpr, tpr, _ = plot_roc_curve(scores[mia],FPRS)
            ax.plot(fpr,tpr,label=mia,color=palette[mia])
        
        
        ax.plot([0,1],[0,1],color="gray",linestyle="--",label="Random Classifier")
        ax.set(xscale="log",yscale="log",xlim=[1e-3,1e+0],ylim=[1e-3,1e+0])
        ax.set_box_aspect(1)
        ax.legend()
        plt.savefig(os.path.join(self.args.scores_dir,f"mia_plot_{self.args.learnable_params}_{self.args.examples_per_class}_{self.args.target_epsilon}.pdf"),bbox_inches="tight")

        
if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
