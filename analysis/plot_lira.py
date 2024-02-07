from prv_accountant.dpsgd import DPSGDAccountant
from opacus.accountants.rdp import RDPAccountant
import math
import argparse
import csv
import pickle
import os
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_curve(x, y, xlabel, ylabel, ax, label, color, style, title=None):
    ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
    ax.plot(x, y, lw=2, label=label, color=color, linestyle=style)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set(aspect=1, xscale='log', yscale='log')
    if title is not None:
        ax.title.set_text(title)


def compute_attack_advantage(fpr, tpr):
    return max(tpr - fpr)


def compute_bound(fpr, epsilon, delta):
    return min(math.exp(2 * epsilon) * fpr + (1 + math.exp(epsilon)) * delta,
               1-math.exp(-2*epsilon)*(1-(1+math.exp(epsilon))*delta-fpr))


def compute_tight_prv_bound(fpr, cached = False):
    # params for eps = 1, S=10 and Head with R-50
    total_steps = 398
    sample_rate = 0.5
    noise_multiplier = 29.0625

    print(f"Computing PRV tight bounds with total_steps {total_steps}, sample_rate {sample_rate} and noise_multiplier {noise_multiplier}.")
    # PRV
    if not cached:
        prv_accountant = DPSGDAccountant(
            noise_multiplier=noise_multiplier,
            sampling_probability=sample_rate,
            eps_error=1e-5,
            delta_error=1e-11,
            max_steps=total_steps
        )
    all_bounds = []
    for i, delta in enumerate([1/1000, 1/2000, 1/5000, 1e-4, 1e-5, 1e-6]):
        if cached:
            eps_upper = [0.864171699946296, 0.9426598672691959, 1.0394010035538905,
                     1.1081376646228707, 1.3151737688955718, 1.4977559753135155][i]
        else:
            eps_low, eps_estimate, eps_upper = prv_accountant.compute_epsilon(num_steps=total_steps, delta=delta)

        print(f"δ={delta} results in ϵ={eps_upper}")

        bound = []
        for i in fpr:
            bound.append(compute_bound(i, eps_upper, delta))
        bound = np.hstack(bound)
        all_bounds.append(bound)

    # get tighest bound from all computed PRV bounds
    tighest_bound = np.amin(all_bounds, axis=0)

    return tighest_bound


def compute_rdp_bound(fpr):
    epsilon = 1
    delta = 1/1000
    bound = []
    for i in fpr:
        bound.append(compute_bound(i, epsilon, delta))
    bound = np.hstack(bound)
    return bound


def compute_rd_bound_multiple_delta(fpr):
    # params for eps = 1, S=10 and Head with R-50
    total_steps = 398
    sample_rate = 0.5
    noise_multiplier = 29.0625

    print(
        f"Computing RDP bounds with multiple delta and total_steps {total_steps}, sample_rate {sample_rate} and noise_multiplier {noise_multiplier}.")

    rdp_accountant = RDPAccountant()
    rdp_accountant.history.append((noise_multiplier, sample_rate, total_steps))
    all_bounds = []
    for delta in [1/1000, 1/2000, 1/5000, 1e-4, 1e-5, 1e-6]:
        eps, alpha = rdp_accountant.get_privacy_spent(delta=delta)
        print(f"δ={delta} results in ϵ={eps}")
        bound = []
        for i in fpr:
            bound.append(compute_bound(i, eps, delta))
        bound = np.hstack(bound)
        all_bounds.append(bound)

    # get tighest bound from all computed PRV bounds
    tighest_bound = np.amin(all_bounds, axis=0)

    return tighest_bound

def plot_roc_curve(y_true_list, y_score_list, legend_list, colors, styles, fpr_points, save_path, title,
                   shot_list, epsilon_list, flip_legend=False, plot_bound=False, plot_rdp_bound=False):
    assert len(y_true_list) == len(y_score_list)
    assert len(legend_list) == len(y_true_list)
    tpr_at_fpr_results = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for y_true, y_score, legend, color, style, shot, epsilon in zip(y_true_list, y_score_list, legend_list,
                                                                    colors, styles, shot_list, epsilon_list):
        # get the AUC
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        attack_advantage = compute_attack_advantage(fpr, tpr)
        tpr_at_fpr = {
            'legend': legend,
            'values': [],
            'auc': auc,
            'shot': shot,
            'epsilon': epsilon,
            'adv': attack_advantage
        }
        for fpr_point in fpr_points:
            tpr_at_fpr['values'].append(np.interp(x=fpr_point, xp=fpr, fp=tpr))
        tpr_at_fpr_results.append(tpr_at_fpr)

        if plot_bound and shot == '10' and legend == "Head: S=10":  # should only happen in the case of epsilon == '1' and shot == '10' and Head
            if plot_rdp_bound:
                rdp_bound = compute_rdp_bound(fpr)
                plot_curve(x=fpr, y=rdp_bound, xlabel='FPR', ylabel='TPR', ax=ax, color='C4', style=':',
                           label=r"UB (RDP $δ=$1e-3): $S=10$", title=None)
                rdp_tighest_bound = compute_rd_bound_multiple_delta(fpr)
                plot_curve(x=fpr, y=rdp_tighest_bound, xlabel='FPR', ylabel='TPR', ax=ax, color='blue', style=':',
                           label=r"UB (RDP multiple $δ$): $S=10$", title=None)

            tighest_bound = compute_tight_prv_bound(fpr, cached=True)
            plot_curve(x=fpr, y=tighest_bound, xlabel='FPR', ylabel='TPR', ax=ax, color='red', style=':',
                       label=r"UB (PRV multiple $δ$): $S=10$" if plot_rdp_bound else r"Upper Bound: $S=10$", title=None)

        if shot == "10" or not plot_rdp_bound:
            # plot the roc curve
            plot_curve(x=fpr, y=tpr, xlabel='FPR', ylabel='TPR', ax=ax, label='{0:}, TPR={1:1.3f}'.format(
                legend, tpr_at_fpr['values'][0]), color=color, style=style, title=title)

    if flip_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[::-1], labels=labels[::-1], loc='lower right', fontsize=5.0)
    else:
        plt.legend(loc='lower right', fontsize=5.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return tpr_at_fpr_results


class CsvWriter:
    def __init__(self, file_path, header):
        self.file = open(file_path, 'w', encoding='UTF8', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)

    def __del__(self):
        self.file.close()

    def write_row(self, row):
        self.writer.writerow(row)


fpr_points = [1e-3, 1e-2, 1e-1]

# colors1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# colors2 = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# line_styles = [
#     '-', '--'
# ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to in_indices, stat, and output score files.")
    parser.add_argument("--shots", type=int, default=100,help="List of shots for the plots")
    parser.add_argument("--configs", nargs='+',default=[],help="List of configs for the plots")
    parser.add_argument("--epsilons", nargs='+',default=[],help="List of epsilons for the plots")
    parser.add_argument("--run_ids", nargs='+', default=[], help="# of Runs for the various experiments.")
    parser.add_argument("--exp_ids", nargs='+',default=[], help="# of Experiment. NOTE: They are a 1-1 mapping for each epsilon input")
    # parser.add_argument("--accountant", default = "rdp",type = str, help="Type of Accountant/ Privacy Mechanism used for exps.")
    parser.add_argument("--seeds", nargs='+',default=[], help="Seeds setting of the experiment")
    args = parser.parse_args()

    color_list = {"inf":"darkgreen",
                  1:"red",
                  2:"orange",
                  4:"deepskyblue",
                  8:"blue"}
    
    # convert epsilons/ shots to their numeric equivalent
    num_epsilons = []
    for i in range(len(args.epsilons)):
        if args.epsilons[i] == '-1':
            num_epsilons.append("inf")
        else:
            num_epsilons.append(int(args.epsilons[i]))
    tpr_at_fpr_dict = {"eps":[],
                       "tuning":[],
                       "seed":[],
                       1e-3:[],
                       1e-2:[],
                       1e-1:[]
                       }
    for seed in args.seeds:
        legend_list, colors, styles, shot_list, epsilon_list,y_true_list,y_score_list = [],[],[],[],[],[],[]
        for run in args.run_ids:
            curr_id = int(run)
            for i in range(len(num_epsilons)):
                file_path = os.path.join(args.data_path, 'Seed={}'.format(seed),'scores_run_{}_exp_{}_config_{}_shots_{}_eps_{}.pkl'.format(
                        curr_id, args.exp_ids[i], args.configs[i], args.shots, num_epsilons[i]))                  
                with open(file_path, "rb") as f:
                    result = pickle.load(f)
                    y_true_list.append(result['y_true'])
                    y_score_list.append(result['scores'])
                    colors.append(color_list[num_epsilons[i]])
                    if curr_id% 2 == 1:
                        styles.append("solid")
                    elif curr_id == 2:
                        styles.append("dotted")
                    else:
                        styles.append("dashdot")
                    shot_list.append(args.shots)
                    epsilon_list.append(num_epsilons[i])
                    if curr_id == 1 or curr_id == 3:
                        run_type = "ITS-T"
                    elif curr_id% 2 == 0:
                        run_type = "OOTS-T"
                    legend_list.append('{}|config={}|eps={}'.format(run_type,'H' if args.configs[i] == 'none' else 'FiLM',num_epsilons[i]))
        
        tpr_at_fpr_results = plot_roc_curve(y_true_list, y_score_list, legend_list, colors, styles,fpr_points, 
                                            os.path.join(args.data_path, 'plots/roc_plot_tpr_vs_fpr_S_{}_seed_{}.pdf'.format(args.shots,seed)), 
                                            "S={}|seed={}".format(args.shots,seed), 
                                            shot_list, epsilon_list, flip_legend=False, plot_bound=True)
        for i, legend in enumerate(legend_list):
            print('fpr, tpr @ {} for seed = {}'.format(legend,seed))
            print('auc = {0:1.3f}'.format(tpr_at_fpr_results[i]['auc']))
            print('adv = {0:1.3f}'.format(tpr_at_fpr_results[i]['adv']))
            run_type,_, eps = legend.split("|")
            tpr_at_fpr_dict["eps"].append(eps.split("=")[-1])
            tpr_at_fpr_dict["tuning"].append(run_type)
            tpr_at_fpr_dict["seed"].append(seed)
            for fpr, tpr in zip(fpr_points, tpr_at_fpr_results[i]['values']):
                print('{0:1.3f}, {1:1.3f}'.format(fpr, tpr))
                tpr_at_fpr_dict[round(fpr,3)].append(tpr)

    # with open(os.path.join(args.data_path,'tpr_at_fpr_dict.pkl'), 'wb') as handle:
    #     pickle.dump(tpr_at_fpr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                

if __name__ == '__main__':
    main()
