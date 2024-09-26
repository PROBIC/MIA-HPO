import pandas as pd 
import numpy as np 
import math
import argparse
import os
from prv_accountant.dpsgd import DPSGDAccountant
from opacus.accountants.utils import get_noise_multiplier_patched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file_path', help='Path to hyperparameter records', default=".")
    parser.add_argument('--epsilon', type=int, default=-1)
    parser.add_argument('--save_file_path', default=".")

    args = parser.parse_args()

    hp_df = pd.read_csv(args.hp_file_path,header=0)
    hp_df = hp_df[hp_df["DP"] == args.epsilon]
    total_steps_df, noise_multiplier_df, sample_rate_df = [],[],[]

    for i in range(hp_df.shape[0]):  
        current_record = hp_df.iloc[i,:]         
        target_epsilon, epochs, batch_size  = current_record["DP"], 40, current_record["BS"]
        print(target_epsilon,batch_size)
        # calculated per the current opacus' version
        len_dataset = None
        if current_record["Shots"] == 50:
            if current_record["Dataset"] == "CIFAR10":
                len_dataset = 500
            elif current_record["Dataset"] == "CIFAR100":
                len_dataset = 5000
        elif current_record["Shots"] == 100:
            if current_record["Dataset"] == "CIFAR10":
                len_dataset = 1000
            elif current_record["Dataset"] == "CIFAR100":
                len_dataset = 10000    

        expected_len_dataloader =  len_dataset // batch_size
        sample_rate = 1. / expected_len_dataloader
        total_steps = int(epochs / sample_rate)
        print(target_epsilon, batch_size, sample_rate, epochs, total_steps)
        noise_multiplier = get_noise_multiplier_patched(target_epsilon = target_epsilon,
                                                        target_delta = 1e-5, 
                                                        sample_rate = sample_rate,
                                                        epochs = epochs,
                                                        accountant = "prv",
                                                        epsilon_tolerance= 0.01)  
        total_steps_df.append(total_steps)
        noise_multiplier_df.append(noise_multiplier)
        sample_rate_df.append(sample_rate)

    hp_df["total_steps"] = total_steps_df
    hp_df["sample_rate"] = sample_rate_df
    hp_df["noise_multiplier"] = noise_multiplier_df

    hp_df.to_csv(args.save_file_path + f"hyper_with_dp_{args.epsilon}.csv",index=None)
        
if __name__ == "__main__":
    main()