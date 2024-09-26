import numpy as np
import pandas as pd
from load_hpo_data import process_hyperparams
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm
import argparse

NEW_DELTAS = [10**-x for x in [1, 2, 3, 4, 5, 6, 7, 8]]


def compute_subsampling_ratio(shots, n_classes, batch_size):
    dataset_size = shots * n_classes
    n_batches = np.ceil(dataset_size / batch_size)
    return 1 / n_batches


def get_new_eps_for_newdeltas(orig_noise_multiplier : float, subsampling_ratio: float, steps: int):
    prv_accountant = create_accountant("prv")
    prv_accountant.history = [(float(orig_noise_multiplier), float(subsampling_ratio), int(steps))]
    
    new_epsilons = list()
    for delta in NEW_DELTAS:
        new_epsilons.append(prv_accountant.get_epsilon(delta))
    return new_epsilons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_file_path', help='Path to hyperparameter records', default=".")
    args = parser.parse_args()

    df_hyper = process_hyperparams(pd.read_csv("data/hypers_dp_hp_tuning.csv"))
    columns = [f"eps for delta {delta}" for delta in NEW_DELTAS]

    rows = []

    for c in columns:
        df_hyper[c] = ""

    for i, row in tqdm(df_hyper.iterrows()):
        row[columns] = (
            get_new_eps_for_newdeltas(row["noise multiplier"], row["sample rate"], row["total steps"])
            if row["eps"] > 0
            else [-1 for d in NEW_DELTAS]
        )
        rows.append(row)

    df_hyper_new = pd.concat(rows, axis=1).T

    df_hyper_new.to_csv("data/hypers_dp_hp_tuning_more_deltas.csv", index=False)