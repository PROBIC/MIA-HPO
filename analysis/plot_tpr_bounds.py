import pandas as pd 
import numpy as np 
import math
import argparse
import os
import pickle
from prv_accountant.dpsgd import DPSGDAccountant
from opacus.accountants.utils import get_noise_multiplier_patched
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

def plot_roc_curve(y_true, y_score, fpr_points):
    assert len(y_true) == len(y_score)
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    tpr_at_fpr = []
    for fpr_point in fpr_points:
        tpr_at_fpr.append(np.interp(x=fpr_point, xp=fpr, fp=tpr))
    return (fpr,tpr, tpr_at_fpr)

def compute_bound(fpr, epsilon, delta):
    return min(math.exp(epsilon) * fpr + delta, 1 - math.exp(- epsilon)*(1 - delta - fpr))

def compute_tight_prv_bound(fprs, total_steps, sample_rate, noise_multiplier):
    print(f"Computing PRV tight bounds with total_steps {total_steps}, sample_rate {sample_rate} and noise_multiplier {noise_multiplier}.")
    prv_accountant = DPSGDAccountant(
        noise_multiplier=noise_multiplier,
        sampling_probability=sample_rate,
        eps_error=1e-5,
        delta_error=1e-11,
        max_steps=total_steps
    )
    all_bounds = []
    for delta in [1/500, 1/1000, 1/2000, 1/5000, 1/10000, 1e-5]:
        _, _, eps_upper = prv_accountant.compute_epsilon(num_steps=total_steps, delta=delta)
        print(f"δ={delta} results in ϵ={eps_upper}")
        bound = []
        for fpr in fprs:
            bound.append(compute_bound(fpr, eps_upper, delta))
        bound = np.hstack(bound)
        all_bounds.append(bound)

    # get tighest bound from all computed PRV bounds
    tighest_bound = np.amin(all_bounds, axis=0)

    return tighest_bound

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_file_path', help="Path to mia scores", default=".")
    parser.add_argument('--hp_file_path', help='Path to hyperparameter records', default=".")
    parser.add_argument('--plot_save_path', help='Path to save the plot', default=".")
    parser.add_argument('--config', help='type of linear layer used for fine-tuning', default="none")
    parser.add_argument('--epsilon', help='numeric value of target privacy budget', type = int, default=-1)
    parser.add_argument('--dataset', help='dataset used for fine-tuning downstream model', type = str, default="cifar10")
    parser.add_argument('--shots', help='examples_per_class', type = int, default=50)
    parser.add_argument("--feature_extractor", choices=['vit-b-16', 'BiT-M-R50x1'],
                    default='BiT-M-R50x1', help="Feature extractor to use.")
    parser.add_argument("--seeds", nargs='+', default=[], help="List of seeds for different runs.")
    parser.add_argument("--runs", nargs='+', default=[], help="List of runs (depends on the type of tuning + number of shots).")
    args = parser.parse_args()

    if args.epsilon == -1:
        args.epsilon = "inf"
        # line_color = "blue"     
    
    args.seeds = [int(s) for s in args.seeds]
    args.runs = [int(s) for s in args.runs]

    fpr_points = [1e-3, 1e-2, 1e-1]
    styles = ["solid","dotted"]
    legend = ["ITS-T","OoTS-T"]
    hp_df = pd.read_csv(os.path.join(args.hp_file_path,"hypers_config_{}_{}.csv".format(args.config,args.dataset)),header=0)

    for seed in args.seeds:
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        if args.epsilon == 1:
            exp = 5
            line_color = "purple"
        elif args.epsilon == 2:
            exp = 4
            line_color = "red"
        elif args.epsilon == 4:
            exp = 3
            line_color = "green"
        elif args.epsilon == 8:
            exp = 2
            line_color = "orange"
        
        fprs_list, tprs_list = [],[]
        for run in args.runs:
            filepath = os.path.join(args.scores_file_path,"Seed={}".format(seed),"scores_run_{}_exp_{}_config_{}_shots_{}_eps_{}.pkl".format(run,
                                                                                                                                             exp,
                                                                                                                                             "none" if args.config == "head" else args.config,
                                                                                                                                             args.shots,
                                                                                                                                             args.epsilon))
            with open(filepath, "rb") as f:
                result = pickle.load(f)
            
            y_score = -result["scores"] # negate the obtained scores calculated with flipped labels
            y_true = np.logical_not(result["y_true"]).astype(float) # flip the labels --> IN as 1 and OUT as 0.
            fprs,tprs,_ = plot_roc_curve(y_true, y_score, fpr_points)
            fprs_list.append(fprs)
            tprs_list.append(tprs)
        
        # for i in range(2):
        #     ax.plot(fprs_list[i], tprs_list[i],color = line_color ,alpha = 0.66,linestyle = styles[i], label = legend[i])

        # choose the most comprehensive of the ITS, OOTS fprs list to plot the upper bound
        if len(fprs_list[0]) < len(fprs_list[1]):
            ub_fprs = fprs_list[1]
        else:
            ub_fprs = fprs_list[0]
            
        all_tpr_upper_bounds = []
        # plotting the bounds
        for i in range(len(args.runs)):  
            curr_run_record = hp_df[(hp_df["seed"]==seed) 
                                    & (hp_df["exp_id"] == exp) 
                                    & (hp_df["model"] == args.feature_extractor) 
                                    & (hp_df["run_id"] == args.runs[i])]          
            target_epsilon, epochs, batch_size  = curr_run_record["eps"].values[0], curr_run_record["epochs"].values[0], curr_run_record["BS"].values[0]
            # calculated per the current opacus' version
            len_dataset = None
            if args.runs[i] == 1 or args.runs[i] == 2:
                if args.dataset == "cifar10":
                    len_dataset = 500
                elif args.dataset == "cifar100":
                    len_dataset = 5000
            elif args.runs[i] == 3 or args.runs[i] == 4:
                if args.dataset == "cifar10":
                    len_dataset = 250
                elif args.dataset == "cifar100":
                    len_dataset = 2500            
            expected_len_dataloader =  len_dataset // batch_size
            sample_rate = 1. / expected_len_dataloader
            total_steps = int(epochs / sample_rate)
            noise_multiplier = get_noise_multiplier_patched(target_epsilon= target_epsilon,
                                                            target_delta = 1e-5, 
                                                            sample_rate = sample_rate,
                                                            epochs = epochs,
                                                            accountant = "prv",
                                                            epsilon_tolerance= 0.01)  

            tpr_bounds = compute_tight_prv_bound(ub_fprs, total_steps, sample_rate, noise_multiplier)
            all_tpr_upper_bounds.append(tpr_bounds)
        
        min_tpr_upper_bound = np.array(all_tpr_upper_bounds).min(axis=0)
        fprs_list.append(ub_fprs)
        tprs_list.append(min_tpr_upper_bound)

        save_dict = {"tprs":tprs_list, 
                     "fprs": fprs_list}
            
        ax.plot(ub_fprs, min_tpr_upper_bound,color = "black",linestyle = "solid", label = "UB") # plot only the minimum of the 2 UBs
        
        ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
        ax.set(xscale="log",yscale="log")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), fancybox=True, shadow=True, ncol=5,fontsize=8)
        ax.legend(loc='lower right',fontsize=9)

        # save the plots
        name = args.scores_file_path.split("/")
        ax.set_title("{}, S={}, {}, {}, eps={}, seed={}".format(name[-1], args.shots, args.config, name[-3], args.epsilon,seed),fontsize=8)
        save_path = os.path.join(args.plot_save_path,"plots/{}_eps_{}_S_{}_config_{}_{}_run_{}_with_bounds.pdf".format(name[-3],
                                                                                                                       args.epsilon,
                                                                                                                       args.shots, 
                                                                                                                       args.config, 
                                                                                                                       name[-1],
                                                                                                                       seed))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        

if __name__ == "__main__":
    main()