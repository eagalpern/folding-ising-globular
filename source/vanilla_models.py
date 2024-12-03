import numpy as np
from evolutionary_energy import *
from pdb_utils import *
from fasta_utils import *

def make_vanilla_pots(h_average,
                       J_average,
                       h_params,
                       pdb_file,
                       pdb_ali_beg,
                       pdb_ali_end,
                       d = 8, # 3d distance threshold 
                       k = 0): # sequence distance

    if h_params['vanilla_h'] == 'vanilla':
        new_h = make_vanilla_h(h_average,h_params['dssp_data'])
    
    elif h_params['vanilla_h'] == 'info':
        new_h = make_info_h(h_average,
                            h_params['MSA_num'],
                            h_params['weights'],
                            h_params['npos'])
        
    elif h_params['vanilla_h'] == 'uniform':
        new_h = make_uniform_h(h_average,
                             h_params['npos'])
    else:
        
        new_h = make_vanilla_h_rsa(h_average,h_params['dssp_data'],
                                   h_params['RSA_df'])
                                   

    
    new_J = make_vanilla_J(J_average,
                           pdb_file,
                           pdb_ali_beg,
                           pdb_ali_end,
                           d = d,  
                           k = k)
                        
    new_potts =  {'h': new_h, 'J': new_J}
    
    return new_potts

def make_info_h(h_average,
                MSA_num,
                weights,
                npos,
                naa = 21):
  
    fi = freq(MSA_num,npos,naa,weights)
    info=np.log(naa)-np.nansum(-fi*np.log(fi),axis=1)
    new_h = np.tile(info*h_average/info.sum(),(naa,1)).T
    return new_h

def make_vanilla_h(h_average,
                   dssp_data,
                   naa = 21):
    positions =~np.isnan(dssp_data.exon_freq)
    sec_structure = np.array([int(x) for x in dssp_data.bin_ss[positions].values])
    new_h = np.tile(h_average*sec_structure/sec_structure.sum(),(naa,1)).T
    return new_h

def make_vanilla_h_rsa(h_average,
                       dssp_data,
                       df_relative_sasa,
                       naa = 21,
                       epsilon = 1e-15):
    positions =~np.isnan(dssp_data.exon_freq)
    RSA = df_relative_sasa.Relative_ASA[positions].values
    signal = -np.log(RSA+epsilon)
    new_h = np.tile(h_average*signal/signal.sum(),(naa,1)).T
    return new_h


def make_uniform_h(h_average,
                   npos,
                   naa = 21):
    new_h = np.tile(h_average*np.ones(npos)/npos,(naa,1)).T
    return new_h


def make_vanilla_J(J_average,
                   pdb_file,
                   pdb_ali_beg,
                   pdb_ali_end,
                   naa = 21,
                   d = 8, # 3d distance threshold 
                   k = 0): # sequence distance  

    contact_map = get_distance_matrix(pdb_file, chain= 'A')< d
    contact_map = contact_map[:,pdb_ali_beg:pdb_ali_end][pdb_ali_beg:pdb_ali_end,:]
    filtered_cmap = eliminate_diagonals(contact_map, k=k)
    J_av_cmap = filtered_cmap*J_average*2/filtered_cmap.sum()
    new_J = np.repeat(J_av_cmap[:, :, np.newaxis, np.newaxis], naa, axis=2)  # Repeat along the 3rd axis
    new_J = np.repeat(new_J, naa, axis=3)  # Repeat along the 4th axis
    return new_J


