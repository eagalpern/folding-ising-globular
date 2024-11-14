import pandas as pd
import numpy as np
import os
import sys
import pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from monte_carlo import *
from free_energy_profile import *
from fit_ff import *
from utils import *


# =============================================================================
# Simulation Functions
# =============================================================================

def run_simulation(seq, folder_, potts, breaks, params):
    """
    Run a folding simulation for a single sequence using parallel processing.
    """
    args1 = (seq, folder_, potts['J'], potts['h'], params['AAdict'], None, None, 0, params['m'], 
             params['gaps_out'], params['si0'], params['k'], params['nsteps'], params['transient'], 
             params['save_each'], params['ninst'], params['ntsteps'], params['DT'], 
             params['ff_file'], params['states_file'], params['q_hist_file'], params['ulf_file'], 
             params['DH_file'], params['DS_file'], params['tini_'], params['tfin_'], None, 
             False, params['ts_auto'], breaks, params['interactions_off'])
    args2 = (params['DH_file'], params['DS_file'], params['ulf_file'], 0, params['k'], params['nsteps'], 
             params['transient'], params['save_each'], params['ninst'], params['ntsteps'], 
             params['order'], params['extrap'], params['cp_factor'], params['ff_file'], 
             params['states_file'], params['q_hist_file'], False, None, params['num_cores'])
    
    Parallel(n_jobs=params['num_cores'])(delayed(main_fold_1seq_first_round_i)(i, *args1) for i in range(params['num_cores']))
    Parallel(n_jobs=params['num_cores'])(delayed(main_fold_1seq_second_round_i)(i, *args2) for i in range(params['num_cores']))

# =============================================================================
# Main Function
# =============================================================================

def ising_simulation(potts, breaks, folder, **kwargs):
    """
    Main function for running an Ising simulation for protein folding.
    
    Parameters:
    - potts: Dictionary of Potts model parameters.
    - breaks: Foldon breakpoints or controls.
    - folder: Base directory for simulation files.
    - **kwargs: Additional simulation parameters that override defaults.
    
    Returns:
    - Extracted features of the folding simulation, such as the folding temperature and the cooperativity score.
    """

# =============================================================================
# Parameter Definitions
# =============================================================================

    # Simulation Control Parameters
    sim_params = {
        # Unique identifier for the simulation run, useful for output filenames or logging
        "run_str": kwargs.get("run_str", ''),

        # Flag to allow for multiple proteins in the same simulation; False for single protein
        "multi_prot": kwargs.get("multi_prot", False),

        # Number of proteins to simulate if `multi_prot` is True, random choice from input MSA
        "Nprot": kwargs.get("Nprot", 100),

        # Name of the protein or reference sequence
        "prot_name": kwargs.get("prot_name", 'reference_seq'),

        # Sequence of amino acids for the protein(s) being simulated
        "seq": kwargs.get("seq", None),

        # Toggle interactions off; useful for testing purposes
        "interactions_off": kwargs.get("interactions_off", False),
    }

    # Parallelization Parameters
    parallel_params = {
        # Number of CPU cores to use for parallel processing
        "num_cores": kwargs.get("num_cores", cpu_count()),
    }

    # Temperature and Folding Dynamics Parameters
    temp_fold_params = {
        # Entropy per residue 
        "si0": kwargs.get("si0", 1),

        # Boltzmann constant 
        "k": kwargs.get("k", 1),

        # Selection Temperature 
        "Tsel": kwargs.get("Tsel", 1),

        # Initial and final temperature points for folding dynamics
        "tini_": kwargs.get("tini_", 1),
        "tfin_": kwargs.get("tfin_", 12),

        # Whether to use an automated temperature for transitions
        "ts_auto": kwargs.get("ts_auto", True),

        # Temperature step size for scanning out of the given temperature range
        "DT": kwargs.get("DT", 0.9),

        # Critical temperature factor for folding simulation, affects convergence
        "cp_factor": kwargs.get("cp_factor", 20),

        # Interval at which the simulation state is saved (e.g., for monitoring or restarting)
        "save_each": kwargs.get("save_each", 20),

        # Initial steps to ignore as transient behavior
        "transient": kwargs.get("transient", 50),

        # Number of simulation steps in total
        "nsteps": kwargs.get("nsteps", 10000),

        # Number of instances to simulate (used if running multiple trajectories)
        "ninst": kwargs.get("ninst", 200),

        # Number of constant temperature steps at which the simulation will be made
        "ntsteps": kwargs.get("ntsteps", 40),
    }

    # Amino Acid and Sequence Parameters
    aa_seq_params = {
        # Name of the FASTA file containing the reference sequence or sequences
        "fastaname": kwargs.get("fastaname", 'MSA_nogap.fasta'),
        
        # Do not consider sequence gaps
        "gaps_out": kwargs.get("gaps_out", True),

        # Dictionary encoding amino acid symbols to numeric codes, useful for mapping in simulations
        "AAdict": kwargs.get("AAdict", {
            'Z': 4, 'X': 0, '-': 0, 'B': 3, 'J': 8, 'A': 1, 'C': 2, 'D': 3, 
            'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 
            'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 
            'W': 19, 'Y': 20
        }),
    }

    # File and Directory Parameters
    file_dir_params = {
        # Path to the fraction folded files 
        "ff_file": kwargs.get("ff_file", "ff"),

        # Path to the states file 
        "states_file": kwargs.get("states_file", "st"),

        # Path to the q histogram file
        "q_hist_file": kwargs.get("q_hist_file", "q_hist"),

        # Path to the folding units lenght file
        "ulf_file": kwargs.get("ulf_file", "ulf"),

        # Path to the enthalpy and entropy matrix files
        "DH_file": kwargs.get("DH_file", "DH"),
        "DS_file": kwargs.get("DS_file", "DS"),
    }
    

    # Feature Extraction Parameters
    feature_params = {
        # Window size for any sliding window feature extraction, such as in analysis steps
        "nwin": kwargs.get("nwin", 10),
    }

    # Combine all parameter groups into one dictionary for use throughout the script
    params = {**sim_params, **parallel_params, **temp_fold_params, 
              **aa_seq_params, **file_dir_params, **feature_params}

    # Calculate derived parameters based on user inputs and constants
    params["m"] = -1 / (params["k"] * params["Tsel"])
    params["order"] = int(params["ntsteps"]/40)
    params["extrap"] = int(params["ntsteps"]/20)
    

    # Load MSA if sequence is not provided
    if params["seq"] is None:
        MSA, weights, names = load_fasta(os.path.join(folder, params["fastaname"]))
    

    
    # Run multi-protein simulation or single protein simulation
    if params["multi_prot"]:
        sequences = np.random.choice(MSA, params["Nprot"], p=weights / sum(weights), replace=False)
        features_list = []
        for i, seq in enumerate(sequences):
            folder_ = setup_paths(folder, names[i], kwargs.get("run_str", ''), True)
            file_dir_params = {key: folder_ + value for key,value in file_dir_params.items()}
            params.update(file_dir_params)
            run_simulation(seq, folder_, potts, breaks, params)
            features = extract_features(params['ff_file'], params['states_file'], 
                                        params['q_hist_file'], params['nwin'], 
                                        params['k'], params['num_cores'], folder_)
            features_list.append(features)
        return pd.DataFrame(features_list)

    else:
        seq = params["seq"] if params["seq"] is not None else MSA[0]
        folder_ = setup_paths(folder, params["prot_name"], kwargs.get("run_str", ''))
        file_dir_params = {key: folder_ + value for key,value in file_dir_params.items()}
        params.update(file_dir_params)
        run_simulation(seq, folder_, potts, breaks, params)
        features = extract_features(params['ff_file'], params['states_file'], 
                                    params['q_hist_file'], params['nwin'], 
                                    params['k'], params['num_cores'], folder_)
        save_features(folder_, features)
        return features

