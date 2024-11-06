import numpy as np
import pandas as pd
import random
import numba
from numba import jit, njit
from joblib import Parallel, delayed


##### OUTPUT

def reshape_jij_gremlin(Jij_gremlin_dir,npos,naa):
    Jij_gremlin=np.zeros((npos,npos,naa,naa))

    ncol=npos

    for i in range(npos):
        for j in range(i+1,npos):

            w_idx = np.arange(npos)[np.stack(np.triu_indices(ncol,1),-1)]
            n = int(np.where((w_idx[:,0] == i)&(w_idx[:,1] == j))[0])

            Jij_gremlin[i,j,:,:]=Jij_gremlin_dir[n]
    return Jij_gremlin



def zero_sum_gauge(Hi,Jij,m=2.0):
  #  print(abs(Hi.sum(axis=1)).max(),abs(Jij.sum(axis=2)).max(),abs(Jij.sum(axis=3)).max())

    # m=2 (j!=i)
    # m=1 (j>i)
    npos,q=Hi.shape

    ha=Hi.sum(axis=1)
    Ja=Jij.sum(axis=2) 
    Jb=Jij.sum(axis=3)
    Jab=Jij.sum(axis=(2,3))

    Ja_=np.einsum('lijk->ijlk',np.tile(Ja,(q,1,1,1)))
    Jb_=np.einsum('lijk->ijkl',np.tile(Jb,(q,1,1,1)))
    Jab_=np.einsum('lkij->ijkl', np.tile(Jab,(q,q,1,1)))   

    Hi_=np.zeros(Hi.shape)
    for i in range(npos):
        for a in range(q):
            Hi_[i,a]=Hi[i,a]-ha[i]/q+(Jij[i,:,a,:].sum()/q-Jij[i,:,:,:].sum()/(q**2)+Jij[:,i,:,a].sum()/q-Jij[:,i,:,:].sum()/(q**2))/m
        
    Jij_=Jij-Ja_/q-Jb_/q+Jab_/(q**2)
 
  #  print('deberian ser cero')
  #  print(abs(Hi_.sum(axis=1)).max(),abs(Jij_.sum(axis=2)).max(),abs(Jij_.sum(axis=3)).max())
    return Hi_,Jij_


@jit(nopython=True)
def E_tot(A,Hi,Jij):
    hi_sum = 0.0
    Jij_sum = 0.0
    L=len(A)
    for i in range(L):
        hi_sum += Hi[i, A[i]]
        for j in range(i + 1, L):
            Jij_sum += Jij[i, j, A[i], A[j]]
    return -Jij_sum - hi_sum


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



@njit(inline="always")
def MCseq(nsteps: numba.int64,npos: numba.int64 ,Naa: numba.int64,temp: numba.float64,
          Hi: numba.types.Array(np.float64,ndim=2,layout='F'),
          Jij: numba.types.Array(np.float64,ndim=4,layout='F'),
          save_each:numba.int64 ,transient:numba.int64):

    seq=np.array([np.random.randint(Naa) for i in range(npos)]) # generate random sequence
    e0=E_tot(seq,Hi,Jij)
    energies=np.zeros(0)
    seq_to_save=np.zeros((int((nsteps-transient)/save_each),npos),dtype=numba.int64)
    for i in range(nsteps):
        residues=list(range(0,Naa))
        x=np.random.randint(npos) # choice random position in sequence 
        old_res=seq[x]
        residues.remove(old_res)
        seq[x] = np.random.choice(np.array(residues)) # mutation
        ef=E_tot(seq,Hi,Jij) # energy after mutation
        de=ef-e0 # change in energy
        # metropolis criterium
        if de<=0: 
            e0=ef
        else:
            if random.uniform(0,1)<np.exp(-de/(temp)):
                e0=ef
            else:
                seq[x] = old_res # don't accept    
        if i%save_each==0 and i>=transient:

            seq_to_save[int((i-transient)/save_each),:]=seq
            ## tendria que cada tanto guardar un estado, puede ser escribiendo un file
            energies=np.append(energies,e0) 
    return list(energies),[list(i) for i in seq_to_save]    
    


def generate_seq_ensemble(path,num_cores,v_file,w_file,NSeq,temp=1.0,
                          Naa=21,transient=40000,save_each=5000,gremlin=True): 
    Hi=np.load(path+v_file)
    npos,naa=Hi.shape
    if gremlin:
        Jij_gremlin=np.load(path+w_file)
        Jij=reshape_jij_gremlin(Jij_gremlin,npos,naa)
    else:
        Jij=np.load(path+w_file)
    nseq=int(NSeq/num_cores)
    nsteps=transient+5000*nseq
    args=nsteps,npos,Naa,temp,Hi,Jij,save_each,transient
    r=Parallel(n_jobs=num_cores,verbose=10)(delayed(MCseq)(*j) for (i,j) in [(i_,args) for i_ in np.arange(num_cores)])
    energies_, seqs_= zip(*r)
    energies=np.concatenate(energies_)
    ali=np.concatenate(seqs_)
    np.save(path+'simulated_energies_full_gremlin',energies)
    np.save(path+'simulated_ali_full_gremlin',ali)

# joint & marginal frequencies (not excluding i=j)
def freq(ali,npos,Naa,w):
    fij=np.zeros((npos,npos,Naa,Naa))
    fi=np.zeros((npos,Naa))
    for i in range(npos):
        fi[i,:]=np.histogram(ali[:,i],bins=np.arange(-0.5,Naa+0.5),weights=w)[0]
        for j in range(npos):
            #if j!=i:
             fij[i,j,:,:]=np.histogram2d(ali[:,i],ali[:,j],bins=np.arange(-0.5,Naa+0.5),weights=w)[0]
    fi=fi/sum(w)
    fij=fij/sum(w)

    return fij,fi

def plot_freq(ax,x,y,xlabel,ylabel,xlim=[10e-5,1.2],ylim=[10e-5,1.2],a=0.1):
    #xy = np.vstack([x,y])
    #z = gaussian_kde(xy)(xy)
    #idx = z.argsort()
    #x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x,y,marker='.',alpha=a)
    ax.plot([0,1.2],[0,1.2],'grey',alpha=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

# rbm
import sys
sys.path.append('/home/ezequiel/libraries/PGM/source/')
sys.path.append('/home/ezequiel/libraries/PGM/utilities/')
import rbm,utilities
import Proteins_utils, RBM_utils, utilities,sequence_logo,plots_utils,bm

def convert_GaussianRBM_to_BM(RBM,max_size=600000):
    assert RBM.hidden == 'Gaussian'
    N = RBM.n_v
    nature = RBM.visible
    n_c = RBM.n_cv
    n_h=RBM.n_h
    BM = bm.BM(N=N,nature=nature,n_c=n_c)
    
    if not nature in ['Bernoulli','Spin','Potts']:
        print('Boltzmann Machine %s not supported'%nature)
        return
    
    if nature == 'Bernoulli':
        couplings_BM = np.dot(RBM.weights.T, 1.0/RBM.hlayer.gamma[:,np.newaxis] *  RBM.weights)        
        fields_BM = RBM.vlayer.fields - np.dot(RBM.hlayer.theta/RBM.hlayer.gamma,RBM.weights)    + 0.5 * couplings_BM[np.arange(N),np.arange(N)]
        couplings_BM[np.arange(N),np.arange(N)] *= 0 # Important: Must have zero diagonal.
    elif nature == 'Spin':
        couplings_BM = np.dot(RBM.eights.T,1.0/RBM.hlayer.gamma[:,np.newaxis] * RBM.weights)        
        couplings_BM[np.arange(N),np.arange(N)] *= 0 # Important: Must have zero diagonal.        
        fields_BM = RBM.vlayer.fields - np.dot(RBM.hlayer.theta/RBM.hlayer.gamma,RBM.weights)
    elif nature == 'Potts':
        
        if N*n_c*n_h>max_size:
            couplings_BM=np.zeros((N,N,n_c,n_c))
            for i in range(n_h):
                couplings_BM+=RBM.weights[i,np.newaxis,:,np.newaxis,:] * RBM.weights[i,:,np.newaxis,:,np.newaxis]/RBM.hlayer.gamma[i]
            
        else:
            couplings_BM = (1.0/RBM.hlayer.gamma[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] *  RBM.weights[:,np.newaxis,:,np.newaxis,:] * RBM.weights[:,:,np.newaxis,:,np.newaxis]).sum(0)
        fields_BM = RBM.vlayer.fields - np.tensordot(RBM.hlayer.theta/RBM.hlayer.gamma,RBM.weights,axes=[(0),(0)]) + 0.5 * couplings_BM[np.arange(N),np.arange(N)][:,np.arange(n_c),np.arange(n_c)]
        couplings_BM[np.arange(N),np.arange(N)] *= 0 # Important: Must have zero diagonal.
    
    
    BM.layer.fields = fields_BM
    BM.layer.fields0 = RBM.vlayer.fields0
    BM.layer.couplings = couplings_BM
    return BM