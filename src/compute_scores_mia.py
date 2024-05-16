import numpy as np
import sys
import os.path
import argparse
import pickle
from lira import compute_score_lira
from sklearn.metrics import roc_curve
import warnings 

def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.scores = {"CMIA":None,
                       "WBMIA": None,
                       "MIA-KL":None,
                       "MIA-H":None,
                       "MIA-J":None}
                    #    "MIA-MD":None,
                    #    "MIA-NMD":None}
        self.opt_args = {
                       "MIA-KL":None,
                       "MIA-H":None,
                       "MIA-J":None}
                    #    "MIA-MD":None,
                    #    "MIA-NMD":None}

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--stats_dir", help="Directory to load stats from.")
        parser.add_argument("--indices_dir", "-c",
                            help="Directory to load in_indices from.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--exp_id", type=int, default=None,
                            help="Experiment ID.")
        parser.add_argument("--run_id", type=int, default=None,
                            help="Run ID for rerunning the whole experiment.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--learnable_params", choices=['none', 'film'], default='none',
                            help="Which feature extractor parameters to learn.")
        parser.add_argument("--num_models", type=int, default=257,
                            help="Total number of shadow models.")
        
        args = parser.parse_args()
        return args
        
    def run(self):
        
        # ensure the directory to hold results exists
        self.exp_dir = f"experiment_{self.args.exp_id}"
        self.run_dir = f"Run_{self.args.run_id}"
        self.stats_dir = os.path.join(self.args.stats_dir, "Seed={}".format(self.args.seed),self.run_dir, self.exp_dir)
        self.indices_dir = os.path.join(self.args.indices_dir, "Seed={}".format(self.args.seed),self.run_dir, self.exp_dir)

        if self.args.exp_id == 1:
            self.target_epsilon = "inf"
        elif self.args.exp_id == 2:
            self.target_epsilon = 8
        elif self.args.exp_id == 3:
            self.target_epsilon = 2
        elif self.args.exp_id == 4:
            self.target_epsilon = 1
        else:
            print("Invalid experiment option.")
            sys.exit()


        # load the training data splits and stats
        filename = os.path.join(self.stats_dir, 'stat_{}_{}_{}_r_0_to_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                self.target_epsilon,
                self.args.num_models))
        
        with open(filename, 'rb') as f:
            stats = pickle.load(f)

        with open(os.path.join(self.indices_dir, 'in_indices_{}_{}_{}.pkl'.format(
                self.args.learnable_params,
                self.args.examples_per_class,
                self.target_epsilon)), 'rb') as f:
            in_indices = pickle.load(f)

        # Complete TD
        self.scores["CMIA"] = run_complete_mia(stats, in_indices,use_global_variance = False)
        # WB-MIA
        self.scores["WBMIA"] = run_white_box_mia(stats,in_indices,use_global_variance=False)
        # MIA-KL
        opt_hypers_per_model_min = find_optimal_hypers(stats,in_indices,metric="KL")   
        self.opt_args["MIA-KL"] = opt_hypers_per_model_min
        self.scores["MIA-KL"] = run_white_box_mia_sim(stats,in_indices,opt_hypers_per_model_min)
        # MIA-J
        opt_hypers_per_model_min = find_optimal_hypers(stats,in_indices,metric="jeffreys")   
        self.opt_args["MIA-J"] = opt_hypers_per_model_min
        self.scores["MIA-J"] = run_white_box_mia_sim(stats,in_indices,opt_hypers_per_model_min)
        # MIA-H
        opt_hypers_per_model_min = find_optimal_hypers(stats,in_indices,metric="hellinger")  
        self.opt_args["MIA-H"] = opt_hypers_per_model_min
        self.scores["MIA-H"] = run_white_box_mia_sim(stats,in_indices,opt_hypers_per_model_min)
        # # MIA-NMD
        # opt_hypers_per_model_min = find_optimal_hypers(stats,in_indices,metric="carlini")   
        # self.opt_args["MIA-NMD"] = opt_hypers_per_model_min
        # self.scores["MIA-NMD"] = run_white_box_mia_sim(stats,in_indices,opt_hypers_per_model_min)
        # # MIA-MD
        # opt_hypers_per_model_min = find_optimal_hypers(stats,in_indices,metric="mean") 
        # self.opt_args["MIA-MD"] = opt_hypers_per_model_min  
        # self.scores["MIA-MD"] = run_white_box_mia_sim(stats,in_indices,opt_hypers_per_model_min)

        filename = os.path.join(self.stats_dir, 'scores_{}_{}_{}.pkl'.format(
        self.args.learnable_params,
        self.args.examples_per_class,
        self.target_epsilon))
            
        with open(filename, 'wb') as f:
            pickle.dump(self.scores, f)   

        filename = os.path.join(self.stats_dir, 'opt_shadow_hypers_{}_{}_{}.pkl'.format(
        self.args.learnable_params,
        self.args.examples_per_class,
        self.target_epsilon))
            
        with open(filename, 'wb') as f:
            pickle.dump(self.opt_args, f)       
        
        
def run_complete_mia(stat, in_indices, use_global_variance=False):
    N = stat.shape[0]
    cmia_stat = []
    for i in range(N):
        cmia_stat.append(stat[i,i,:,:])

    n = len(cmia_stat[0])
    # Now we do MIA for each model
    all_scores = []
    all_y_true = []
    for idx in range(N):
        print(f'Target model is #{idx}')
        stat_target = cmia_stat[idx]  # statistics of target model, shape (n, k)
        in_indices_target = in_indices[idx]  # ground-truth membership, shape (n,)

        stat_shadow = np.array(cmia_stat[:idx] + cmia_stat[idx + 1:])
        in_indices_shadow = np.array(in_indices[:idx] + in_indices[idx + 1:])
        stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(n)]
        stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(n)]

        # Compute the scores and use them for MIA
        scores = compute_score_lira(stat_target, stat_in, stat_out,fix_variance=use_global_variance)
        # preserve the order of samples
        y_true = [0 if mask else 1 for mask in in_indices_target]

        all_scores.append(scores)
        all_y_true.append(y_true)
    
    return {"y_true": np.hstack(all_y_true),
           "y_score": np.hstack(all_scores)}

def compute_score(target_stats, shadow_stats, target_in_indices, shadow_in_indices,use_global_variance=False):
    n = len(target_stats)
    stat_in = [shadow_stats[:, j][shadow_in_indices[:, j]] for j in range(n)]
    stat_out =  [shadow_stats[:, j][~shadow_in_indices[:, j]] for j in range(n)]
    # Compute the scores and use them for MIA
    scores = compute_score_lira(target_stats, stat_in, stat_out,fix_variance=use_global_variance)
    # preserve the order of samples
    y_true = [0 if mask else 1 for mask in target_in_indices]
    return y_true, scores


def run_white_box_mia(stats,indices,use_global_variance=False):
    in_indices = np.array(indices)
    all_y_true, all_y_score = [],[]
    N_MODELS = stats.shape[0]
    for i in range(N_MODELS): 
        print(f"Target model M[{i}][{i}]")
        curr_stats = stats[:,i,:,:]
        target = curr_stats[i,:,:]
        shadow = np.vstack([curr_stats[:i,:,:], curr_stats[i+1:,:,:]])
        target_in = in_indices[i,:]
        shadow_in = np.vstack([in_indices[:i,:], in_indices[i+1:,:]])
        curr_y_true, curr_y_score = compute_score(target, shadow, target_in, 
                                                  shadow_in,use_global_variance=use_global_variance)
        all_y_true.append(curr_y_true)
        all_y_score.append(curr_y_score)
    return {"y_true": np.hstack(all_y_true), "y_score": np.hstack(all_y_score)}

def hellinger_normal(P,Q):
    mu_p, mu_q, s_p, s_q = np.mean(P), np.mean(Q), np.std(P), np.std(Q)
    exp = np.exp(-(mu_p - mu_q)**2/(4*(s_p**2 + s_q**2)))
    base = np.sqrt((2*s_p*s_q)/(s_p**2 + s_q**2))
    return np.sqrt(1 - base**exp)

def carlini_version(P,Q):
    return np.abs(np.mean(P) - np.mean(Q))/ (np.std(P) + np.std(Q))

def mean_difference(P,Q):
    return np.abs(np.mean(P) - np.mean(Q))

def kl_divergence(P,Q, direction="forward"):
    mu_p, mu_q, s_p, s_q = np.mean(P), np.mean(Q), np.std(P), np.std(Q)
    if direction == "forward":
        return 0.5 * ((((mu_p-mu_q)**2 + s_p**2)/s_q**2) - np.log(s_p**2/s_q**2) - 1)
    else:
        return 0.5 * ((((mu_p-mu_q)**2 + s_q**2)/s_p**2) - np.log(s_q**2/s_p**2) - 1)

def jeffrey_divergence(P,Q):
    D_pq = kl_divergence(P,Q, direction="forward")
    D_qp = kl_divergence(P,Q, direction="backward")
    return D_pq + D_qp

def find_optimal_hypers(stats, in_indices,metric = "KL"):
    in_indices = np.array(in_indices)
    N_MODELS = stats.shape[0]
    opt_hypers_per_model = np.zeros((N_MODELS,))
    for i in range(N_MODELS):
        print(f"Currently targetting model #{i+1}")
        stats_target = stats[i,i,:,:].flatten()
        stats_shadow = np.hstack([stats[:,:i,:,:],stats[:,i+1:,:,:]]) # select all columns but the target hypers
        shadow_indices = np.vstack([in_indices[:i,:],in_indices[i+1:,:]])
        per_column_overlap = np.zeros((N_MODELS-1,))
        for j in range(stats_shadow.shape[1]): # for the remaining columns compute overlap with target distribution/model
            overlaps = np.zeros((N_MODELS-1,))
            # select all entries - models trained on target dataset
            curr_shadow_column = np.vstack([stats_shadow[:i,j,:,:],stats_shadow[i+1:,j,:,:]])
            for k in range(curr_shadow_column.shape[0]):
                if metric == "hellinger":
                    overlaps[k] = hellinger_normal(stats_target[shadow_indices[k]],curr_shadow_column[k,shadow_indices[k],:])
                elif metric == "carlini":
                    overlaps[k] = carlini_version(stats_target[shadow_indices[k]],curr_shadow_column[k,shadow_indices[k],:])
                elif metric == "KL":
                    overlaps[k] = kl_divergence(stats_target[shadow_indices[k]],curr_shadow_column[k,shadow_indices[k],:],direction="forward")
                elif metric == "jeffreys":
                    overlaps[k] = jeffrey_divergence(stats_target[shadow_indices[k]],curr_shadow_column[k,shadow_indices[k],:])
                else:
                    overlaps[k] = mean_difference(stats_target[shadow_indices[k]],curr_shadow_column[k,shadow_indices[k],:])
            per_column_overlap[j] = np.mean(overlaps)
        # for the target index, impute np.inf as the similarity measure. 
        per_column_overlap = np.insert(per_column_overlap,i,np.inf)
        opt_hypers_per_model[i] = np.argmin(per_column_overlap) 

    return opt_hypers_per_model        

def run_white_box_mia_sim(stats,indices,opt_hypers_per_model,use_global_variance=False):
    in_indices = np.array(indices)
    all_y_true, all_y_score = [],[]
    N_MODELS = stats.shape[0]
    for i in range(N_MODELS): 
        print(f"Target model M[{i}][{i}]")
        target_stats = stats[i,i,:,:]
        target_indices = in_indices[i,:]
        shadow_column = int(opt_hypers_per_model[i])
        shadow_stats = np.vstack([stats[:i,shadow_column,:,:],stats[i+1:,shadow_column,:,:]])
        shadow_indices = np.vstack([in_indices[:i,:], in_indices[i+1:,:]])
        curr_y_true, curr_y_score = compute_score(target_stats, shadow_stats, target_indices, 
                                                  shadow_indices,use_global_variance=use_global_variance)
        all_y_true.append(curr_y_true)
        all_y_score.append(curr_y_score)
    return {"y_true": np.hstack(all_y_true), "y_score": np.hstack(all_y_score)}

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )
        main()
