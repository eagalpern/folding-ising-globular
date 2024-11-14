import numpy as np
import pandas as pd
import os
import pickle
from scipy.optimize import curve_fit


# def multi_ff_fit(out_dir_,folder,rep_frag_len,ff_file_,states_file_,L,j=1):
    
#     n_units=len(rep_frag_len[j])*L
#     ff_file=out_dir_+ff_file_
#     states_file=out_dir_+states_file_

#     ff=np.loadtxt(ff_file)
#     allstates=np.load(states_file+'.npy')
#     Tfs,eTfs=[],[]
#     for p in range(n_units):
#         #RMSD,popt,pcov=sig_fit_v3(ff[:,0],allstates[2,p,:])
#         RMSD,popt,pcov=sig_fit_v4(ff[:,0],allstates[2,p,:])
#         std=np.sqrt(np.diag(pcov))
#         Tfs.append(popt[1])
#         eTfs.append(std[1])
    
#     t_=np.array(Tfs)
#     return t_


def multi_ff_fit_i(N,ff_file,states_file):

    ff=np.loadtxt(ff_file+'_0')
    for fi in range(N):
        if fi==0:
            sti=np.load(states_file+'_'+str(fi)+'.npy')
        else:
            sti+=np.load(states_file+'_'+str(fi)+'.npy')
    st=sti/N
    n_units=st.shape[1]
    Tfs,eTfs=[],[]
    for p in range(n_units):
        #RMSD,popt,pcov=sig_fit_v3(ff[:,0],st[:,p])
        RMSD,popt,pcov=sig_fit_v4(ff[:,0],st[:,p])
        std=np.sqrt(np.diag(pcov))
        Tfs.append(popt[1])
        eTfs.append(std[1])
    
    t_=np.array(Tfs)
    return t_,st

def sigmoid_ff_fit_i(out_dir, num_cores):
    for i in range(num_cores):
        ff=np.loadtxt(out_dir+'ff_'+str(i))
        if i==0:
            ff_=np.zeros((ff.shape[0],num_cores))
        ff_[:,i]=ff[:,1]    
    RMSD,popt,pcov=sig_fit_v4(ff[:,0],ff_.mean(axis=1))
    tf = popt[1]
    std_tf = np.sqrt(pcov[1,1])
    
    width = popt[0]
    std_width = np.sqrt(pcov[0,0])
    return tf, width, std_tf, std_width

    
def sig_fit_v4(X,Y):


    def fsigmoid(x, a, b,c):
        return c * np.exp(-(x-b)/a) / (1.0 + np.exp(-(x-b)/a))
    
    try:
        
        
        p0 = [(X[1]-X[0])*2, np.mean(X), 1]
        bounds = ([(X[1]-X[0])/10, np.min(X), 0], [np.max(X)-np.min(X), np.max(X),1])
        
        popt, pcov = curve_fit(fsigmoid, X, Y, method='trf', p0=p0, bounds=bounds) 
        RMSD= np.sqrt(sum((Y-fsigmoid(X, *popt))**2)/len(Y))

    except RuntimeError:
        print("Error: curve_fit failed")
        RMSD=np.nan
        popt=[np.nan,np.nan,np.nan]
        pcov=np.array([[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])

    except ValueError:
        print("Error: wrong input")
        RMSD=np.nan
        popt=[np.nan,np.nan,np.nan]
        pcov=np.array([[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])
    return RMSD,popt,pcov

def fsigmoid(x, a, b,c):
    return c * np.exp(-(x-b)/a) / (1.0 + np.exp(-(x-b)/a))