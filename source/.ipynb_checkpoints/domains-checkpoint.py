import numpy as np
import pandas as pd
import os
import random
import itertools


def domain_partition(t_,lim,nantozero=True,max_combinations=1000000):
    
    if nantozero:
        t_[np.isnan(t_)]=0
    else:
        t_[np.isnan(t_)]=np.min(t_[~np.isnan(t_)])

    ix=np.argsort(np.array(t_))
    ix_part=[]
    #ok_part=[]
    difs=[]

    overlap=True

    # first check the trivial partitions 
    forced_sep=((np.where((t_[ix][1:]-t_[ix][:-1])>lim)[0]) +1)
    if len(forced_sep)>0: 
        # non-overlap condition
        if all(np.array([(max(x)- min(x)) for x in np.split(t_[ix],forced_sep) if len(x)>0])<lim):
            final_part=np.split(ix,forced_sep)
            overlap=False
            
        else:
            # remaining separators are positions of the rejected partitions
            partition_ok=np.array([(max(x)- min(x)) for x in np.split(t_[ix],forced_sep) if len(x)>0])<lim
            aux=np.array(np.split(np.arange(len(t_)),forced_sep),dtype=object)[~partition_ok]
            remaining_separators=np.concatenate([x[1:] for x in aux])
    else:
        remaining_separators=np.arange(1,len(t_))
    # if there are overlapping domains, we need to add separators
    if overlap:
        L=0
        
        
        while True:

            for add_sep in itertools.combinations((remaining_separators), L):
                sep=sorted(list(add_sep)+forced_sep.tolist())
                # domain condition: maximum temperature difference within = lim
                if all(np.array([(max(x)- min(x)) for x in np.split(t_[ix],sep) if len(x)>0])<lim):
                    #ok_part.append(np.split(ts_,sep))
                    partition=np.split(ix,sep)
                    sum_dif=0
                    if L>1:
                        for x in range(len(partition)-1): # temperature difference between domain extrema
                            sum_dif=+t_[partition[x+1][0]]-t_[partition[x][-1]]
                    ix_part.append(partition)
                    difs.append(sum_dif)

            if (len(ix_part)>0) or (L==(len(remaining_separators))):
                break

            L=L+1  # split the elements into L+1+len(forced_sep) domains
            
            # check if we can handle the next separator list
            it_comb=sum(1 for ignore in itertools.combinations((remaining_separators), L))
            if it_comb>max_combinations:
                raise ValueError('Overlap too long, can not handle '+str(it_comb)+' combinatios')
                
        #if more than one L+1 domain partition is possible we choose the one that maximizes temp diff between domains
        final_part=ix_part[np.argmax(difs)] 

    return final_part,overlap

def domain_temperature(t_,partition):
    t_dom=np.zeros(len(t_))
    for x,p in enumerate(partition):
        t_dom[p]=np.mean(t_[p])
    return t_dom

def domain_matrix(t_dom):
    mat=np.zeros((len(t_dom),len(t_dom)))
    mat[:]= np.nan
    for x in range(len(t_dom)):
        for y in range(len(t_dom)):
            if t_dom[x]==t_dom[y]:
                mat[x,y]=t_dom[x]
    return mat
