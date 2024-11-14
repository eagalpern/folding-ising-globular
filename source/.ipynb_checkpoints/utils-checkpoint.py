import pandas as pd
import numpy as np
import os
import pickle
from fit_ff import *
from free_energy_profile import *
from fit_ff import *



def create_folder(path):
    """Create a folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def setup_paths(folder, prot_name, run_str="", multi = False):
    """
    Set up directory paths for a given protein name and run identifier.
    
    Parameters:
    - folder: Base directory for simulations.
    - prot_name: Protein name used for directory naming.
    - multi: For multiple runs, group into folder
    - run_str: For multiple runs, optional string to differentiate runs.
    
    Returns:
    - folder_: Path to the created protein-specific directory.
    """
    if multi:
        group_folder = os.path.join(folder, f"multi_seq{run_str}")
        create_folder(group_folder)
    else:
        group_folder = folder
    folder_ = os.path.join(group_folder, prot_name)
    create_folder(folder_)
    return folder_ + '/'


def extract_features(ff_file, states_file, q_hist_file, nwin, k, num_cores, out_dir):
    """
    Perform feature extraction analysis for folding simulation data.
    """
    t_, st = multi_ff_fit_i(num_cores, ff_file, states_file)
    FE, obs, barr, eq_steps = FE_analysis(ff_file, q_hist_file, nwin, k, num_cores, out_dir)
    coop_score = (eq_steps[1:-1] == 0).sum() / len(eq_steps[1:-1])
    tf, width, std_tf, std_width = sigmoid_ff_fit_i(out_dir, num_cores)
    
    return {
        "tf": tf,
        "width": width,
        "std_tf": std_tf,
        "std_width": std_width,
        "coop_score": coop_score,
        "t_": t_,
    }


def save_features(folder, features):
    """
    Save extracted features to a pickle file.
    
    Parameters:
    - folder: Directory where features should be saved.
    - features: Dictionary of extracted features to save.
    """
    with open(os.path.join(folder, 'features.pkl'), "wb") as f:
        pickle.dump(features, f)

def load_features(folder):
    """
    Load features from a pickle file.
    
    Parameters:
    - folder: Directory where features are loaded from.
    """
    with open(os.path.join(folder, 'features.pkl'), "rb") as f:  
        features = pickle.load(f)
    return features