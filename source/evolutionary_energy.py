import numpy as np
import pandas as pd
import numba
from numba import njit


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