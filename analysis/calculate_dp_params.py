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
    args = parser.parse_args()

    hp_df = pd.read_csv(args.hp_file_path,header=0)
    total_steps_df, noise_multiplier_df, sample_rate_df = [],[],[]

    for i in range(hp_df.shape[0]):  
        current_record = hp_df.iloc[i,:]         
        target_epsilon, epochs, batch_size  = current_record["eps"], current_record["epochs"], current_record["BS"]
        print(target_epsilon,batch_size)
        if target_epsilon == -1.0:
            total_steps_df.append(-1.0)
            noise_multiplier_df.append(-1.0)
            sample_rate_df.append(-1.0)        
        else:    
            # calculated per the current opacus' version
            len_dataset = None
            if current_record["S"] == 50:
                if current_record["dataset"] == "cifar10":
                    len_dataset = 500
                elif current_record["dataset"] == "cifar100":
                    len_dataset = 5000
            elif current_record["S"] == 25:
                if current_record["dataset"] == "cifar10":
                    len_dataset = 250
                elif current_record["dataset"] == "cifar100":
                    len_dataset = 2500    

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

    hp_df["total steps"] = total_steps_df
    hp_df["sample rate"] = sample_rate_df
    hp_df["noise multiplier"] = noise_multiplier_df

    hp_df.to_csv(args.hp_file_path,index=None)
        
if __name__ == "__main__":
    main()