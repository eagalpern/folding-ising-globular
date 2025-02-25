import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
import seaborn as sns

def predict_y_multidimensional(coef_df, x_values):
    """
    Predicts the target variable y and its error given the values of the input features and the coefficient DataFrame.
    
    Parameters:
    - coef_df (pd.DataFrame): DataFrame containing the labels, coefficients, and standard errors of the polynomial.
    - x_values (list or np.array): Array or list of input feature values.
    
    Returns:
    - y_pred (float): Predicted value of the target variable.
    - y_error (float): Prediction error (uncertainty) of the target variable.
    """
    # Initialize the predicted value with the intercept
    intercept = coef_df.loc[coef_df['label'] == '1', 'coef'].values[0]
    intercept_error = coef_df.loc[coef_df['label'] == '1', 'error'].values[0]
    y_pred = intercept
    y_error_squared = intercept_error ** 2
    
    # Iterate over the remaining coefficients
    for index, row in coef_df.iterrows():
        if row['label'] != '1':
            # Extract the feature index from the label
            feature_label = row['label']
            feature_indices = [int(i) for i in feature_label[1:].split(" ") if i.isdigit()]
            feature_product = np.prod([x_values[i] for i in feature_indices])
            # Add the contribution of the term to the predicted value
            y_pred += row['coef'] * feature_product
            y_error_squared += (row['error'] * feature_product) ** 2
    
    y_error = np.sqrt(y_error_squared)
    
    return y_pred, y_error


def plot_coop_prediction_DMS_2(ax,
                               bs,# ali_seq_num_pdb,
                               predicted_coop,
                               predicted_coop_wt=0,
                               center_wt=False,
                               xtick_sample=2,
                               label='Predicted Cooperativity',
                               cmap='seismic_r',
                               vmin=0,
                               vmax=1,
                               label_fontsize=14,
                               ticklabel_fontsize=10,  # Set size for tick labels
                               num_fontsize=12,        # Set size for numbers on the color bar
                               show_cbar=True):        # New parameter to show/hide color bar
                               
    AAdict_clean = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 
                    'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 
                    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 
                    'W': 19, 'Y': 20}
 
    # Proportional tick size settings
    tick_length = label_fontsize * 0.2
    tick_width = label_fontsize * 0.1
    cbar_labelsize = label_fontsize 

    # Create the heatmap with or without the color bar based on show_cbar
    if center_wt:
        heatmap = sns.heatmap(predicted_coop.T, cmap=cmap,
                               center=predicted_coop_wt, ax=ax,
                               cbar=show_cbar, cbar_kws={'label': label, 'shrink': 1})
    else:
        heatmap = sns.heatmap(predicted_coop.T, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                               cbar=show_cbar, cbar_kws={'label': label, 'shrink': 1})

    # Set font sizes for axis labels
    ax.set_xlabel("Position", fontsize=label_fontsize)
    ax.set_ylabel("Residue", fontsize=label_fontsize)

    # Set tick labels and their size
    ax.set_yticks(np.arange(0.5, predicted_coop.shape[1] + 0.5))
    ax.set_yticklabels([x for x in AAdict_clean.keys()], fontsize=ticklabel_fontsize)

    
        
    ax.set_xticks(bs-bs[0]+0.5)
    ax.set_xticklabels(bs,
                       fontsize=ticklabel_fontsize)
    
    #ax.set_xticks(np.arange(0.5, predicted_coop.shape[0] + 0.5, xtick_sample))
    #ax.set_xticklabels(np.arange(ali_seq_num_pdb[0], ali_seq_num_pdb[-1] + 1, xtick_sample),
    #                   fontsize=ticklabel_fontsize)

    # Customize tick size (length and width)
    ax.tick_params(axis='both', which='both', length=tick_length, width=tick_width)

    # Set a border for the main plot
    for spine in ax.spines.values():
        spine.set_visible(True)  # Ensure the spine is visible
        spine.set_linewidth(0.5)  # Set the width of the border
        spine.set_color('black')   # Set the color of the border

    # If color bar is shown, customize tick number size and label size
    if show_cbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=num_fontsize * 1.2)  # Set color bar tick number font size
        cbar.set_label(label, fontsize=cbar_labelsize)      # Set color bar label font size

        # Set a border for the color bar
        cbar.ax.spines['top'].set_visible(True)
        cbar.ax.spines['bottom'].set_visible(True)
        cbar.ax.spines['left'].set_visible(True)
        cbar.ax.spines['right'].set_visible(True)
        
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.5)  # Set the width of the color bar border
            spine.set_color('black')   # Set the color of the color bar border

            
def make_prediction_plots(ax_0,
                          coef,
                          dif_ei_mean_wt,
                          es_mean_wt,
                          decoy_dif_ei_mean,
                          decoy_es_mean,
                          seq_ix,
                          xtick_sample,
                          fontsize,
                          tick_size,
                          num_size,
                          show_cbar = True):
    

    predicted_coop_wt, error_coop = predict_y_multidimensional(coef,[dif_ei_mean_wt,es_mean_wt])

    predicted_coop = np.zeros_like(decoy_dif_ei_mean)
    for pos in range(predicted_coop.shape[0]):
        for aa in range(predicted_coop.shape[1]):
            predicted_coop[pos,aa], error_coop = predict_y_multidimensional(coef, 
                                                                [decoy_dif_ei_mean[pos,aa],
                                                                 decoy_es_mean[pos,aa]])

         
        
    max_ = max( abs( predicted_coop.flatten() -predicted_coop_wt))
    
    plot_coop_prediction_DMS_2(ax_0, 
                                     seq_ix,#ali_seq_num_pdb_,
                                     predicted_coop-predicted_coop_wt,
                                     predicted_coop_wt*0,
                                     center_wt=False,
                                     label=r'$\Delta \rho $',
                                     vmin = -max_,
                                     vmax = max_,
                                     xtick_sample = xtick_sample,
                                     label_fontsize = fontsize,
                                     ticklabel_fontsize = tick_size,
                                     num_fontsize = num_size,
                                     show_cbar = show_cbar)      
    
    return predicted_coop,predicted_coop_wt,error_coop


def create_mutational_table(matrix,
                            wt_aa_list,
                            value_name_str ='',
                            initial_pos = 1,
                            _AA = '-ACDEFGHIKLMNPQRSTVWY'):
    """
    Create a mutational table from a matrix of mutational decoy energies.

    Args:
        matrix (2D array-like): The decoy energy matrix (rows=positions, cols=amino acids).
        wt_aa_list (list): List of wild-type amino acids, one per position.
        _AA (str, optional): Amino acid alphabet. Defaults to '-ACDEFGHIKLMNPQRSTVWY'.
    Returns:
        pd.DataFrame: A DataFrame with columns ['Position', 'wt_aa', 'mut_aa', 'value', 'difference'].
    """
    aa_list = list(_AA)  # Convert to list of amino acids
    n_positions, n_mutations = matrix.shape

    # Initialize lists to store table data
    positions, wt_aas, mut_aas, decoy_energies, ddes = [], [], [], [], []

    # Iterate over positions and amino acids
    for pos in range(n_positions):
        wt_aa = wt_aa_list[pos]  # Get the wild-type amino acid for this position
        wt_energy = matrix[pos, aa_list.index(wt_aa)]  # Get the wild-type decoy energy
        #print(wt_energy,pos, wt_aa)
        for aa_idx, mut_aa in enumerate(aa_list):
            decoy_energy = matrix[pos, aa_idx]
            dde = decoy_energy - wt_energy

            # Append to the respective lists
            positions.append(pos + initial_pos)  #
            wt_aas.append(wt_aa)
            mut_aas.append(mut_aa)
            decoy_energies.append(decoy_energy)
            ddes.append(dde)
    

    
    # Create a DataFrame
    mutational_table = pd.DataFrame({
        'Position': positions,
        'wt_aa': wt_aas,
        'mut_aa': mut_aas,
        value_name_str+'_mutant': decoy_energies,
        value_name_str+'_difference': ddes
    })
    mutational_table['mutant'] = mutational_table['wt_aa']+mutational_table['Position'].astype(str)+mutational_table['mut_aa']

    return mutational_table
