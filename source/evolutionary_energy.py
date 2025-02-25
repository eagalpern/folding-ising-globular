import numpy as np
import pandas as pd
import numba
from numba import njit, jit, prange


@njit(inline="always")
def energy_eval(h_i: numba.types.Array(np.float64,ndim=2,layout='F'),
                J_ij :numba.types.Array(np.float64,ndim=4,layout='F'),
                A: np.array):
    L=A.shape[0]
        
    E=np.zeros((L,L))
    for i in range(L):
        E[i,i] = h_i[i, A[i]]
        for j in range(i + 1, L):
            E[i,j] = J_ij[i, j, A[i], A[j]]
    return -E

@jit(nopython=True)
def hj_tot(A,Hi,Jij):
    hi_sum = 0.0
    Jij_sum = 0.0
    L=len(A)
    for i in range(L):
        hi_sum += Hi[i, A[i]]
        for j in range(i + 1, L):
            Jij_sum += Jij[i, j, A[i], A[j]]
    return Jij_sum, hi_sum


@jit(nopython=True, parallel=True)
def compute_weighted_averages(A, Hi, Jij, weights):
    n_sequences = A.shape[0]
    L = A.shape[1]
    J_weighted_sum = 0.0
    h_weighted_sum = 0.0
    weight_sum = 0.0  # To normalize weights
    
    for k in prange(n_sequences):
        hi_sum = 0.0
        Jij_sum = 0.0
        seq = A[k]
        weight = weights[k]
        
        for i in range(L):
            hi_sum += Hi[i, seq[i]]
            for j in range(i + 1, L):
                Jij_sum += Jij[i, j, seq[i], seq[j]]
        
        J_weighted_sum += Jij_sum * weight
        h_weighted_sum += hi_sum * weight
        weight_sum += weight
    
    # Normalize weighted sums to compute averages
    J_average = J_weighted_sum / weight_sum
    h_average = h_weighted_sum / weight_sum
    
    return J_average, h_average


# REDUCE EVALUATED ENERGY MATRIX ACCORDING TO CUSTOM BREAKS
def energy_submatrix(evo_energy_full,breaks):
    evo_energy_s=pd.DataFrame(index=range(len(breaks)),columns=range(len(breaks)),data=0,dtype=float)
    for n in range(len(breaks)):
        if n==len(breaks)-1:
            pos_n=range(breaks[n],len(evo_energy_full))
        else:
            pos_n=range(breaks[n],breaks[(n+1)])
        for m in range(n,len(breaks)):

            if m==len(breaks)-1:
                pos_m=range(breaks[m],len(evo_energy_full))
                
            else:
                pos_m=range(breaks[m],breaks[(m+1)])
            
            evo_energy_s.loc[n,m]=evo_energy_full.loc[pos_n,pos_m].values.sum()
    return evo_energy_s

def energy_submatrix_np(evo_energy_full, breaks):
    """
    Reduce the evaluated energy matrix according to custom breaks.

    Parameters:
    -----------
    evo_energy_full : np.ndarray
        The full energy matrix of shape (L, L), where L is the length of the sequence.
    breaks : list or np.ndarray
        A list of indices defining the breaks for submatrix aggregation.

    Returns:
    --------
    np.ndarray
        A reduced energy matrix of shape (len(breaks), len(breaks)).
    """
    num_breaks = len(breaks)
    evo_energy_s = np.zeros((num_breaks, num_breaks), dtype=float)

    for n in range(num_breaks):
        # Define the row range for the current break
        if n == num_breaks - 1:
            pos_n = slice(breaks[n], evo_energy_full.shape[0])
        else:
            pos_n = slice(breaks[n], breaks[n + 1])

        for m in range(n, num_breaks):
            # Define the column range for the current break
            if m == num_breaks - 1:
                pos_m = slice(breaks[m], evo_energy_full.shape[1])
            else:
                pos_m = slice(breaks[m], breaks[m + 1])

            # Sum the submatrix and assign it to the reduced matrix
            evo_energy_s[n, m] = np.sum(evo_energy_full[pos_n, pos_m])

    # Mirror the upper triangle to the lower triangle to make the matrix symmetric
  #  for n in range(num_breaks):
  #      for m in range(n + 1, num_breaks):
  #          evo_energy_s[m, n] = evo_energy_s[n, m]

    return evo_energy_s


# MRA -> Ising energy
def seq_to_ising_DCA(seq,Jij,Hi,AAdict,breaks,m,gaps_out):

    evo_energy_full=pd.DataFrame(energy_eval(Hi,Jij,np.array([AAdict[a] for a in seq])))

   
    if gaps_out:
        gap_index=gap_idx(seq,repeat_field=False)

        if len(gap_index)>0:
            evo_energy_full.loc[gap_index]=0
            evo_energy_full[gap_index]=0

    
    evo_energy=energy_submatrix(evo_energy_full,breaks)
      
    DH=evo_energy/m
        
    return DH,breaks


def gap_idx(seq,repeat_field=True):
    if repeat_field:
        gap_ix=np.where(seq.values=='-')[0]*replen+np.where(seq.values=='-')[1]
    else:
        gap_ix=np.where(seq=='-')[0]
    return gap_ix


def si0_to_DS_units_len_DCA(si0,seq,breaks,gaps_out):    
#    units_len=np.array(rep_frag_len[j]*nrep)
    units_len=np.concatenate([breaks[1:]-breaks[:-1],np.array([len(seq)-breaks[-1]])])

    
    # los gaps no suman entropía
    DS_all=pd.DataFrame(index=range(len(seq)),columns=range(len(seq)),data=0.0)
    np.fill_diagonal(DS_all.values,si0)

    if gaps_out:
        gap_index=gap_idx(seq,repeat_field=False)

        if len(gap_index)>0:
            DS_all.loc[gap_index]=0
            DS_all[gap_index]=0
            
            
            gap_units=np.array([np.argmax(breaks[g>=breaks]) for g in gap_index])
            values, counts = np.unique(gap_units, axis=0, return_counts=True)
            units_len[values]=units_len[values]-counts   
    DS=energy_submatrix(DS_all,breaks=breaks) 
    return DS,units_len


def compute_energy_averages(seq,
                            potts,
                            breaks,
                            m,
                            AAdict  = {
            'Z': 4, 'X': 0, '-': 0, 'B': 3, 'J': 8, 'A': 1, 'C': 2, 'D': 3, 
            'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 
            'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 
            'W': 19, 'Y': 20
        }):
    DH_,breaks = seq_to_ising_DCA(seq,potts['J'],potts['h'],AAdict,breaks,m,gaps_out=True)
    DS,ul = si0_to_DS_units_len_DCA(si0=0,seq=seq,breaks=breaks,gaps_out=True) # este podria no repetirlo
    DH = DH_.values
    N=DH.shape[0]
    norm_evo=np.zeros((N,N))
    dif_e=np.zeros((N,N))

    for p in range(N):
        norm_evo[p,p]=0
        if p<(N-1):        
            for q in range(p+1,N):
                norm_evo[p,q]=DH[p,q]/ul[q]/ul[p]
                dif_e[p,q]=abs((DH[q,q]/ul[q])-(DH[p,p]/ul[p]))

    es_mean = norm_evo.flatten()[norm_evo.flatten()!=0].mean()
    dif_ei_mean = dif_e.flatten()[dif_e.flatten()!=0].mean()
    
    return es_mean, dif_ei_mean


# Define the vectorized function
def compute_decoy_energy_position_wise_vectorized(seq, potts_model, _AA = '-ACDEFGHIKLMNPQRSTVWY'):
    """
    Compute the change in the energy matrix for all possible point mutations (vectorized).

    Parameters:
    -----------
    seq : str
        The input sequence.
    potts_model : dict
        The Potts model containing couplings J and local fields h.

    Returns:
    --------
    delta_E : np.ndarray
        A 4D array of shape (seq_len, 21, seq_len, seq_len) representing the change in the energy matrix
        for all possible mutations.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, aa1 = np.meshgrid(np.arange(seq_len), np.arange(21), indexing='ij', sparse=True)

    # Initialize the change in energy matrix
    delta_E = np.zeros((seq_len, 21, seq_len, seq_len))

    # Compute changes in the diagonal elements (local fields h)
    # Shape: (seq_len, 21)
    delta_h = - potts_model['h'][pos1, aa1] + potts_model['h'][pos1, seq_index[pos1]]

    # Apply delta_h to the diagonal elements of delta_E
    for pos in range(seq_len):
        delta_E[pos, :, pos, pos] = delta_h[pos, :]



    # Compute changes in the off-diagonal elements (couplings J)
    # Shape: (seq_len, seq_len, 21)
    reduced_j = potts_model['J'][range(seq_len), :, seq_index, :].astype(np.float32)

    # Compute the coupling corrections
    # Shape: (seq_len, seq_len, 21)
    j_correction = np.zeros((seq_len, seq_len, 21))

    # Compute the change in couplings: J_ij(a_i, :) - J_ij(a_i', :)
    j_correction += reduced_j[:, pos1, seq_index[pos1]]  # J_ij(a_i, :)
    j_correction -= reduced_j[:, pos1, aa1]             # J_ij(a_i', :)

    # Reshape j_correction to match the shape of delta_E[np.arange(seq_len), :, np.arange(seq_len), :]
    j_correction = j_correction.transpose(1, 2, 0)  # Shape: (seq_len, 21, seq_len)

    # Apply j_correction to the off-diagonal elements of delta_E
    delta_E[np.arange(seq_len), :, np.arange(seq_len), :] += j_correction
    delta_E[np.arange(seq_len), :, :, np.arange(seq_len)] += j_correction


    return delta_E

def compute_decoy_energy_averages(seq, potts_model, breaks, m, AAdict  = {
            'Z': 4, 'X': 0, '-': 0, 'B': 3, 'J': 8, 'A': 1, 'C': 2, 'D': 3, 
            'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 
            'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 
            'W': 19, 'Y': 20
        }):
    # Compute DS and ul for the wildtype sequence (only once)
    _, ul = si0_to_DS_units_len_DCA(si0=0, seq=seq, breaks=breaks, gaps_out=True)
    DH_,breaks = seq_to_ising_DCA(seq,potts_model['J'],potts_model['h'],AAdict,breaks,m,gaps_out=True)

    # Compute delta_E for all decoys
    delta_E = compute_decoy_energy_position_wise_vectorized(seq, potts_model)

    # Initialize arrays to store es_mean and dif_ei_mean for each decoy
    seq_len = len(seq)
    es_mean_decoy = np.zeros((seq_len, 21))
    dif_ei_mean_decoy = np.zeros((seq_len, 21))

    # Iterate over all positions and amino acids
    for pos in range(seq_len):
        for aa in range(21):
            # Compute DH for the decoy
            DH_decoy = DH_.values + energy_submatrix_np(delta_E[pos, aa], breaks)/m  # Shape: (seq_len, seq_len)

            # Compute es_mean and dif_ei_mean for the decoy
            N = DH_decoy.shape[0]
            norm_evo = np.zeros((N, N))
            dif_e = np.zeros((N, N))

            for p in range(N):
                norm_evo[p, p] = 0
                if p < (N - 1):        
                    for q in range(p + 1, N):
                        # Normalize by ul[p] and ul[q] (position-specific)
                        norm_evo[p, q] = DH_decoy[p, q] / ul[q] / ul[p]
                        dif_e[p, q] = abs((DH_decoy[q, q] / ul[q]) - (DH_decoy[p, p] / ul[p]))

            # Compute es_mean and dif_ei_mean for this decoy
            es_mean_decoy[pos, aa] = norm_evo.flatten()[norm_evo.flatten() != 0].mean()
            dif_ei_mean_decoy[pos, aa] = dif_e.flatten()[dif_e.flatten() != 0].mean()

    return es_mean_decoy, dif_ei_mean_decoy