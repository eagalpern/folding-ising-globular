import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt, colors
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib

import os
import random

import seaborn as sns
from importlib import reload  

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from scipy.signal import argrelextrema
from scipy.optimize import curve_fit


import itertools
import py3Dmol

import MC_functions as mcf

mcf=reload(mcf)

# =============================================================================
#  DOMAINS
# =============================================================================

def multi_ff_fit(out_dir_,folder,rep_frag_len,ff_file_,states_file_,L,j=1):
    
    n_units=len(rep_frag_len[j])*L
    ff_file=out_dir_+ff_file_
    states_file=out_dir_+states_file_

    ff=np.loadtxt(ff_file)
    allstates=np.load(states_file+'.npy')
    Tfs,eTfs=[],[]
    for p in range(n_units):
        #RMSD,popt,pcov=sig_fit_v3(ff[:,0],allstates[2,p,:])
        RMSD,popt,pcov=sig_fit_v4(ff[:,0],allstates[2,p,:])
        std=np.sqrt(np.diag(pcov))
        Tfs.append(popt[1])
        eTfs.append(std[1])
    
    t_=np.array(Tfs)
    return t_

# NCORES VERSION (and new st file shape)

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


# =============================================================================
#  FREE ENERGY
# =============================================================================

def free_energy(T,k,N_q): 
    return -k*T*np.log(N_q/sum(N_q))

def vect(obs,Nq):
    # cantidad de temperaturas (donde calcule FE) para las cuales el minimo de FE esta en q.
    # esos intervalos de T tienen que ser constantes sino esto no tiene sentido
    # el primer y ultimo valor deben ser>0 y no tienen relevancia, depende de como hice la simulacion
    aux=obs.groupby('abs_min').Temp.count()
    eq_steps=np.zeros(Nq,dtype=int)
    eq_steps[aux.index]=aux
    
    # el alto de las barreras cada q, si es que la barrrera maxima esta en q para alguna 
    # si dos barreras estan entre los mismos (o pegados) minimos, elijo entre ellas la más alta
    aux2=obs.groupby('wh_barr').h_barr.max()
    aux2=aux2.loc[aux2.index>0]
    aux3=pd.DataFrame(columns=['wh_barr','h_barr'])
    for ai in range(len(aux)-1):
        if (aux.index[ai+1]-aux.index[ai])>1:
            candidatos=aux2[(aux2.index > aux.index[ai]) & (aux2.index < aux.index[ai+1])]
            if len(candidatos)==1:
                aux3.loc[len(aux3),'wh_barr']=candidatos.index[0]
                aux3.loc[len(aux3)-1,'h_barr']=candidatos[candidatos.index[0]]

            elif len(candidatos)>1:
                print(candidatos)
                ca=candidatos[candidatos==(np.sort(candidatos)[-1])]
                aux3.loc[len(aux3),'wh_barr']=ca.index[0]
                aux3.loc[len(aux3)-1,'h_barr']=ca[ca.index[0]]
    barr=np.zeros(Nq,dtype=float)
    barr[aux3.wh_barr.tolist()]=aux3.h_barr
    
    
    
    return barr, eq_steps

def obs_(FE,Temps):
    
    Nframes=FE.shape[0]
    
    if Nframes!=len(Temps):
        print('Wrong dimensions')
        return 1
        
    Nq=FE.shape[1]

    obs=pd.DataFrame(np.zeros((Nframes,6)))
    obs.columns=['nbarr','wh_barr','h_barr','dif_mins','abs_min','abs_min_2']

    for i in range(Nframes):
        x=FE[i,:]

        qmax=argrelextrema(x, np.greater)[0]
        qmin=argrelextrema(x, np.less,mode='wrap')[0]

        obs.nbarr[i]=len(qmax)

        if len(qmin)==1:
            obs.abs_min[i]=qmin
        elif len(qmax)>0:
            glob_min1_ix=np.argmin(x[qmin])
            glob_min1=qmin[glob_min1_ix]
            qmin_= np.delete(qmin, glob_min1_ix)
            glob_min2=qmin_[np.argmin(x[qmin_])]

            qmins=[glob_min1,glob_min2]
            qmins.sort()
            qmax_=[]

            # seleccionar barreras relevantes
            for qm in qmax: 
                if qm>qmins[0] and qm<qmins[1]:
                    qmax_.append(qm)
            
            if len(qmax_)>0:
                qbarr=qmax_[np.argmax(x[qmax_])]

                obs.abs_min[i]=glob_min1
                obs.abs_min_2[i]=glob_min2
                obs.dif_mins[i]=abs(x[qmins][0]-x[qmins][1])
                obs.wh_barr[i]=qbarr 
                obs.h_barr[i]=abs(x[qbarr]-max([abs(y) for y in  x[qmins]]))
            else:
                obs.abs_min[i]=glob_min1
        else:
            obs.abs_min[i]=np.argmin(x)

    obs=obs.astype({'nbarr':int,'wh_barr':int,'h_barr':float,'dif_mins':float,'abs_min':int,'abs_min_2':int})
    obs['Temp']=Temps
    return obs



def FE_analysis(ff_file,q_hist_file,nwin,k,num_cores,save_dir):

    
    if num_cores>1:
        ff=np.loadtxt(ff_file+'_1')   

        for fi in range(num_cores):
            if fi==0:
                q_hist=np.load(q_hist_file+'_'+str(fi)+'.npy')
            else:
                q_hist+=np.load(q_hist_file+'_'+str(fi)+'.npy')
    else:
        ff=np.loadtxt(ff_file)   
        q_hist=np.load(q_hist_file+'.npy')
        
    ts=ff[:,0]    
    
    FE=np.zeros((nwin,np.shape(q_hist)[1]))
    FE[:] = np.inf

    lims=np.linspace(ts[0],ts[-1],nwin+1)

    Temps=[]
    for it in range(nwin):
        if it==(nwin-1):
            inwin=np.where((ts>=lims[it]) & (ts<=lims[it+1]))[0] # last point in last partition
        else:
            inwin=np.where((ts>=lims[it]) & (ts<lims[it+1]))[0]

        if len(inwin)==0:
            print('Warning: empty temperature window ['+str(lims[it])+','+str(lims[it+1])+')')
        t_=np.mean(ts[inwin]) 
        Temps.append(t_)
        FE[it,:]=free_energy(t_,k,q_hist[inwin,:].sum(axis=0))
    
    '''
    
    # old: only regular windows
    fpw=int(np.floor(len(ts)/nwin)) #files per window # temps per window
    nrows=fpw*nwin #rows to use #total # len(ts) corregido si algo queda afuera. si es multiplo de nwin es al pedo

    FE=np.zeros((nwin,np.shape(q_hist)[1]))
    FE[:] = np.inf
    Temps=[]
    for it in range(nwin):
        fini=fpw*it
        ffin=fpw*(it+1)
        t_=(ts[fini]+ts[ffin-1])/2 # window temp
        Temps.append(t_)
        FE[it,:]=free_energy(t_,k,q_hist[fini:ffin,:].sum(axis=0))
    '''
    np.savetxt(save_dir+'FE_matrix.csv',FE)
    np.savetxt(save_dir+'FE_temps.csv',Temps)

    obs=obs_(FE,Temps)
    obs.to_csv(save_dir+'FE_obs.csv')

    barr, eq_steps=vect(obs,FE.shape[1])

    np.savetxt(save_dir+'barr.csv',barr)
    np.savetxt(save_dir+'eq_steps.csv',eq_steps)

    return FE,obs,barr, eq_steps

# =============================================================================
#  DATASET STATS
# =============================================================================

def array_energy(array_map,MRA_num,Jij,Hi,outdir):
    rep_len=len(MRA_num[0])
    for i in range(len(array_map)):
        nseq=MRA_num.iloc[array_map['abs.init.row'][i]:(array_map['abs.fin.row'][i]+1)]
        E,I=mcf.energy_eval_sum(Jij,Hi,nseq)
        array_map.loc[i,'energy_intra']=E
        array_map.loc[i,'energy_inter']=I
    array_map['nrep']=array_map.apply(lambda x: (x['abs.fin.row']-x['abs.init.row']+1),axis=1)
    array_map['E_per_rep']=array_map.apply(lambda x: x.energy_intra/x.nrep,axis=1)
    array_map['I_per_pair']=array_map.apply(lambda x: x.energy_inter/(x.nrep-1) if x.nrep>1 else 0,axis=1)
    array_map['E_norm']=array_map['E_per_rep']+array_map['I_per_pair']
    array_map.to_csv(outdir+'map_energy')
    return array_map

def ff_fit(t_map,out_dir_):
    with open(out_dir_+'sig_fit.txt', "w+") as file_:
        file_.write('RMSD'+'\t'+'mT'+'\t'+'Tf'+'\t'+'maxf'+'\t'+'emT'+'\t'+'eTf'+'\t'+'emaxf''\n')
 
    
    for i in range(len(t_map)):
        prot_name=t_map.prot_name[i]
        out_dir=out_dir_+prot_name+'/'
        ff_file=out_dir+'ff_file'
        ff=np.loadtxt(ff_file)
        RMSD,popt,pcov=sig_fit_v4(ff[:,0],ff[:,1])
        #RMSD,popt,pcov=sig_fit_v3(ff[:,0],ff[:,1])

        std=np.sqrt(np.diag(pcov))
        with open(out_dir_+'sig_fit.txt', "a") as file_:
            file_.write(str_to_save([RMSD,*popt,*std])+'\n')


def array_energy(array_map,MRA_num,Jij,Hi,outdir):
    rep_len=len(MRA_num[0])
    for i in range(len(array_map)):
        nseq=MRA_num.iloc[array_map['abs.init.row'][i]:(array_map['abs.fin.row'][i]+1)]
        E,I=mcf.energy_eval_sum(Jij,Hi,nseq)
        array_map.loc[i,'energy_intra']=E
        array_map.loc[i,'energy_inter']=I
    array_map['nrep']=array_map.apply(lambda x: (x['abs.fin.row']-x['abs.init.row']+1),axis=1)
    array_map['E_per_rep']=array_map.apply(lambda x: x.energy_intra/x.nrep,axis=1)
    array_map['I_per_pair']=array_map.apply(lambda x: x.energy_inter/(x.nrep-1) if x.nrep>1 else 0,axis=1)
    array_map['E_norm']=array_map['E_per_rep']+array_map['I_per_pair']
    array_map.to_csv(outdir+'map_energy')
    return array_map
            
def dom_stats(t_map,out_dir2_,lim,save_dir):
    
    # all domains
    all_dom_len=[]
    all_dom_temp=[]
    all_dom_prot_name=[]
    all_dom_nrep=[]
    overlap=[]
    dom_order=[]
    n_dom_array=[]
    partitions=[]
    
    for h_,L in enumerate(t_map.Nrep.unique()):
        print(L)
        n_units=2*L

        f_tf=out_dir2_+'tf_fits/'+'Tf_fit_'+str(L)+'.txt'
        tfs=pd.read_csv(f_tf,sep='\t',index_col=None)

        nucl=[]
   
        for j in range(len(tfs)):
            
            # array
            name=t_map.loc[t_map.Nrep==L].reset_index().prot_name[j]
            
            # domain calculation
            t_=tfs.values[j,:]
            
            try:
                partition,ovl=domain_partition(t_,lim)
                
                t_dom=domain_temperature(t_,partition)

                # all domains
                all_dom_len+=[len(x) for x in partition]
                all_dom_temp+=[np.mean(t_[x]) for x in partition]
                all_dom_prot_name+=np.repeat(name,len(partition)).tolist()
                all_dom_nrep+=np.repeat(L,len(partition)).tolist()
                overlap+=np.repeat(ovl,len(partition)).tolist()
                dom_order+=(np.arange(len(partition))[::-1]).tolist()
                n_dom_array+=np.repeat(len(partition),len(partition)).tolist()

                partitions+=partition
                
                
            except:
                print('error',name,L)
                
    # all domain table
    df_dom=pd.DataFrame({'dom_len':all_dom_len,'temp':all_dom_temp,'array_len':all_dom_nrep,
                     'prot_name':all_dom_prot_name,'array_overlap':overlap,'dom_order':dom_order,
                        'domain':partitions,'n_dom_array':n_dom_array})
    
        
    df_dom.to_csv(save_dir+'df_dom_'+str(lim))
    
    return 0



def load_energy(i,t_map,out_dir_,m):
    
    #prot_name=t_map['array.name'][i]+'_'+str(t_map['array.p.ini'][i])+'_'+str(t_map['array.p.fin'][i])
    
    
    out_dir=out_dir_+t_map.prot_name[i]+'/'
    DH_file=out_dir+'DH'
    ulf_file=out_dir+'ulf'
    ul=np.loadtxt(ulf_file)
    for n, i in enumerate(ul):
        if i == 0:
            ul[n] = 1
        #asi no divergen si la energia es cero
    evo_e=np.loadtxt(DH_file)*m
    dif_e=np.zeros(evo_e.shape)
    norm_evo=np.zeros(evo_e.shape)
    int_e=np.zeros(evo_e.shape[0])
    surf_e=np.zeros(evo_e.shape[0]-1)
    
    int_e_norm=np.zeros(evo_e.shape[0])
    surf_e_norm=np.zeros(evo_e.shape[0]-1)
    surf_e_norm_rel=np.zeros(evo_e.shape[0]-1)
    
    # agrego esto
    vec_e=np.zeros(evo_e.shape[0])
    
    
    for p in range(evo_e.shape[0]):
        norm_evo[p,p]=0
        int_e[p]=evo_e[p,p]
        int_e_norm[p]=evo_e[p,p]/ul[p]
        if p<(evo_e.shape[0]-1):        
            for q in range(p+1,evo_e.shape[0]):
                surf_e[p]=sum(evo_e[p,(p+1):])
                norm_evo[p,q]=evo_e[p,q]/ul[q]/ul[p]
                surf_e_norm[p]+=evo_e[p,q]/ul[q]/ul[p]
                surf_e_norm_rel[p]+=evo_e[p,q]/ul[q]/ul[p]/(evo_e[q,q]/ul[q]+evo_e[p,p]/ul[p])

                dif_e[p,q]=abs((evo_e[q,q]/ul[q])-(evo_e[p,p]/ul[p]))
                
        vec_e[p]=(np.sum(evo_e[p,:])+np.sum(evo_e[:,p])-evo_e[p,p])/ul[p]
    return vec_e,norm_evo,dif_e,int_e,surf_e,int_e_norm,surf_e_norm,surf_e_norm_rel

def calc_eq_steps_(eq_steps,Temps):
    fedt=Temps[1]-Temps[0]
    eq_steps_=eq_steps*fedt
    if eq_steps[0]>0:
        eq_steps_[0]=fedt
    if eq_steps[-1]>0:
        eq_steps_[-1]=fedt
    return eq_steps_


def jumps_(eq_steps):
    aux=np.where(eq_steps!=0)[0]
    jumps=np.zeros(len(aux)-1,dtype=int)
    for a in range(len(aux)-1):
        jumps[a]=aux[a+1]-aux[a]
    return jumps


# =============================================================================
#  Energies

def map_array_energy(t_map,out_dir2_,m,rep_frag_len,j):


    evf='evo_energy/'
    os.system('mkdir '+out_dir2_+evf)

    # cargar la distribución de largos
    
    
    
    largos=t_map.groupby('Nrep').Nrep.count()
   
    # armar matrices 4 np.array vacias, una x largo para c/cosa
    i=-1

    for L in largos.index:
        n_units=len(rep_frag_len[j])*L

        norm_evo_array=np.zeros((largos[L],n_units**2))
        f_norm_evo_array=out_dir2_+evf+str(L)+'_norm_evo'
        dif_array=np.zeros((largos[L],n_units**2))
        f_dif=out_dir2_+evf+str(L)+'_dif'
        norm_evo_array_int=np.zeros((largos[L],(n_units-2)**2))
        f_norm_evo_array_int=out_dir2_+evf+str(L)+'_norm_evo_int'
        dif_array_int=np.zeros((largos[L],(n_units-2)**2))
        f_dif_int=out_dir2_+evf+str(L)+'_dif_int'
        Ei=np.zeros((largos[L],n_units))
        f_Ei=out_dir2_+evf+str(L)+'_Ei'
        Es=np.zeros((largos[L],n_units-1))
        f_Es=out_dir2_+evf+str(L)+'_Es'
        Ein=np.zeros((largos[L],n_units))
        f_Ein=out_dir2_+evf+str(L)+'_Ein'
        Esn=np.zeros((largos[L],n_units-1))
        f_Esn=out_dir2_+evf+str(L)+'_Esn'
        Esn_rel=np.zeros((largos[L],n_units-1))
        f_Esn_rel=out_dir2_+evf+str(L)+'_Esn_rel'

        vec_e_=np.zeros((largos[L],n_units))
        f_vec_e=out_dir2_+evf+str(L)+'_vec_e'
        
        for i_ in range(largos[L]):
            i+=1
            try:
                vec_e,norm_evo,dif_e,int_e,surf_e,int_e_norm,surf_e_norm,surf_e_norm_rel=load_energy(i,t_map,out_dir2_,m)
               
                N=norm_evo.shape[0]
                norm_evo_int_=norm_evo[2:,:][:,:-2]
                dif_int_=dif_e[2:,:][:,:-2]
                dif_array_int[i_,:]=np.reshape(dif_int_,dif_int_.shape[0]*dif_int_.shape[1])
                norm_evo_array_int[i_,:]=np.reshape(norm_evo_int_,dif_int_.shape[0]*dif_int_.shape[1])

                dif_array[i_,:]=np.reshape(dif_e,N**2)
                norm_evo_array[i_,:]=np.reshape(norm_evo,N**2)

                Ei[i_,:]=int_e
                Es[i_,:]=surf_e
                Ein[i_,:]=int_e_norm
                Esn[i_,:]=surf_e_norm
                Esn_rel[i_,:]=surf_e_norm_rel
                
                
                vec_e_[i_,:]=vec_e


            except:
                print('Error ',i)
                norm_evo_int_=np.nan
                dif_int_=np.nan
                dif_array_int[i_,:]=np.nan 
               # norm_evo_array_int[i_,:]=np.nan esto no lo voy a usar lo dejo asi para que dif_array_int se calcule bien

                dif_array[i_,:]=np.nan
                #norm_evo_array[i_,:]=np.nan esto no lo voy a usar lo dejo asi para que dif_array_int se calcule bien

                
                Ei[i_,:]=np.nan
                Es[i_,:]=np.nan
                Ein[i_,:]=np.nan
                Esn[i_,:]=np.nan
                Esn_rel[i_,:]=np.nan
                
                vec_e_[i_,:]=np.nan


        # save stuff
        dif_array_int=dif_array_int[:,norm_evo_array_int.sum(axis=0)!=0]
        norm_evo_array_int=norm_evo_array_int[:,norm_evo_array_int.sum(axis=0)!=0]


        dif_array=dif_array[:,norm_evo_array.sum(axis=0)!=0]
        norm_evo_array=norm_evo_array[:,norm_evo_array.sum(axis=0)!=0]

        fs=f_norm_evo_array,f_dif,f_norm_evo_array_int,f_dif_int,f_Ei,f_Es,f_Ein,f_Esn,f_Esn_rel,f_vec_e
        arr=norm_evo_array,dif_array,norm_evo_array_int,dif_array_int,Ei,Es,Ein,Esn,Esn_rel,vec_e_
        for f,a in zip(fs,arr):
            np.save(f,a)
            
            
# =============================================================================
#  Features

def map_array_features(t_map,out_dir2_,rep_frag_len,j):


    vf='features/'
    os.system('mkdir '+out_dir2_+vf)
    
# cargar la distribución de largos

    largos=t_map.groupby('Nrep').Nrep.count()
    i=0

    max_n_units=len(rep_frag_len[j])*max(largos.index)

    all_eq_steps_=np.zeros((len(t_map),max_n_units+1))
    all_eq_steps_int=np.zeros((len(t_map),max_n_units-4))

    f_all_eq_steps_=out_dir2_+vf+'all_eq_steps_'
    f_all_eq_steps_int=out_dir2_+vf+'all_eq_steps_int'

    for L in largos.index:
        n_units=len(rep_frag_len[j])*L

        barreras=np.zeros((largos[L],n_units+1))
        eq_steps=np.zeros((largos[L],n_units+1))
        eq_steps_=np.zeros((largos[L],n_units+1))


        f_barreras=out_dir2_+vf+str(L)+'_barrs'
        f_eq_steps=out_dir2_+vf+str(L)+'_eq_steps'
        f_eq_steps_=out_dir2_+vf+str(L)+'_eq_steps_'

        for i_ in range(largos[L]):
            #prot_name=t_map['array.name'][i]+'_'+str(t_map['array.p.ini'][i])+'_'+str(t_map['array.p.fin'][i])
            out_dir=out_dir2_+t_map.prot_name[i]+'/'
            Temps_i=np.loadtxt(out_dir+'FE_temps.csv')
            eq_steps_i=np.loadtxt(out_dir+'eq_steps.csv')

            barreras[i_,:]=np.loadtxt(out_dir+'barr.csv')
            eq_steps[i_,:]=eq_steps_i
            eq_steps_[i_,:]=calc_eq_steps_(eq_steps_i,Temps_i)

            all_eq_steps_[i,:(n_units+1)]=calc_eq_steps_(eq_steps_i,Temps_i)
            all_eq_steps_int[i,:(n_units-4)]=calc_eq_steps_(eq_steps_i,Temps_i)[:-5]

            i+=1
        # save stuff

        fs=f_barreras,f_eq_steps,f_eq_steps_
        arr=barreras,eq_steps,eq_steps_
        for f,a in zip(fs,arr):
            np.save(f,a)
    np.save(f_all_eq_steps_,all_eq_steps_)
    np.save(f_all_eq_steps_int,all_eq_steps_int)
    
    
### add columns to array_map 

def add_col_map(t_map,out_dir2_,t_map_name,rep_frag_len,j,ali_dir):

    
    evf='evo_energy/'
    os.system('mkdir '+out_dir2_+evf)
    
    vf='features/'
    os.system('mkdir '+out_dir2_+vf)
    
    
    nucleations=[]
    nucleations_units_len=[]

    t_map['max_jump']=0
    t_map['n_jump']=0
    t_map['n_barr']=0
    t_map['first_barr']=0
    t_map['highest_barr']=0


    # energy mean and std
    t_map['Ei_norm_std']=0
    t_map['Ei_std']=0
    t_map['Es_mean']=0
    t_map['Es_norm_mean']=0
    t_map['Ei_norm_mean']=0

    t_map['Es_norm_rel_mean']=0
    t_map['Es_all_mean']=0
    t_map['Ei_dif_mean']=0

    t_map['Es_all_mean_int']=0
    t_map['Ei_dif_mean_int']=0


# cargar la distribución de largos
    largos=t_map.groupby('Nrep').Nrep.count()
  
    for iL,L in enumerate(largos.index):

    #L=5
        n_units=len(rep_frag_len[j])*L
        f_norm_evo_array=out_dir2_+evf+str(L)+'_norm_evo.npy'
        f_dif=out_dir2_+evf+str(L)+'_dif.npy'
        f_norm_evo_array_int=out_dir2_+evf+str(L)+'_norm_evo_int.npy'
        f_dif_int=out_dir2_+evf+str(L)+'_dif_int.npy'
        f_Ei=out_dir2_+evf+str(L)+'_Ei.npy'
        f_Es=out_dir2_+evf+str(L)+'_Ei.npy'
        f_Ein=out_dir2_+evf+str(L)+'_Ein.npy'
        f_Esn=out_dir2_+evf+str(L)+'_Esn.npy'
        f_Esn_rel=out_dir2_+evf+str(L)+'_Esn_rel.npy'

        f_eq_steps=out_dir2_+vf+str(L)+'_eq_steps.npy'
        f_barreras=out_dir2_+vf+str(L)+'_barrs.npy'

        dif_array_int=np.load(f_dif_int)
        norm_evo_array_int=np.load(f_norm_evo_array_int)
        dif_array=np.load(f_dif)
        norm_evo_array=np.load(f_norm_evo_array)
        Ei=np.load(f_Ei)
        Es=np.load(f_Es)
        Ein=np.load(f_Ein)
        Esn=np.load(f_Esn)
        Esn_rel=np.load(f_Esn_rel)

        eq_steps=np.load(f_eq_steps)
        barreras=np.load(f_barreras)


        t_map.Ei_std.loc[t_map.Nrep==L]=Ei.std(axis=1)
        t_map.Ei_norm_std.loc[t_map.Nrep==L]=Ein.std(axis=1)
        t_map.Ei_norm_std.loc[t_map.Nrep==L]=Ein.std(axis=1)
        t_map.Ei_norm_mean.loc[t_map.Nrep==L]=Ein.mean(axis=1)

        t_map.Es_mean.loc[t_map.Nrep==L]=Esn.mean(axis=1)
        t_map.Es_norm_mean.loc[t_map.Nrep==L]=Esn.mean(axis=1)
        t_map.Es_norm_rel_mean.loc[t_map.Nrep==L]=Esn_rel.mean(axis=1)    
        t_map.Es_all_mean.loc[t_map.Nrep==L]=norm_evo_array.mean(axis=1)
        t_map.Ei_dif_mean.loc[t_map.Nrep==L]=dif_array.mean(axis=1)

        t_map.Es_all_mean_int.loc[t_map.Nrep==L]=norm_evo_array_int.mean(axis=1)
        t_map.Ei_dif_mean_int.loc[t_map.Nrep==L]=dif_array_int.mean(axis=1)


        for i in range(len(eq_steps)):
            for x in jumps_(eq_steps[i])[jumps_(eq_steps[i])>1]:
                nucleations.append(x)
                nucleations_units_len.append(n_units)

        t_map.max_jump.loc[t_map.Nrep==L]=[max(jumps_(eq_steps[i])) for i in range(len(eq_steps))]
        t_map.n_jump.loc[t_map.Nrep==L]=[sum(jumps_(eq_steps[i])>1) for i in range(len(eq_steps))]
        t_map.n_barr.loc[t_map.Nrep==L]=[sum(barreras[i]>0) for i in range(len(barreras))]

        aux=np.zeros(len(barreras))
        for i in range(len(barreras)):
            bar=np.where(barreras[i]>0)[0]
            if len(bar)>0:
                aux[i]=min(bar) 

        t_map.first_barr.loc[t_map.Nrep==L]=aux

        t_map.highest_barr.loc[t_map.Nrep==L]=[max(barreras[i]) for i in range(len(barreras))]


    # mas features
    f_all_eq_steps_=out_dir2_+vf+'all_eq_steps_.npy'
    all_eq_steps_=np.load(f_all_eq_steps_)
    ocupados=all_eq_steps_!=0
    t_map['rho']=((t_map.Nrep*2+1)-ocupados.sum(axis=1))/(t_map.Nrep*2-1)
    t_map['DT_total']=all_eq_steps_.sum(axis=1)
    t_map['max_int_DT']=all_eq_steps_.max(axis=1)
    t_map['wh_max_int_DT']=np.argmax(all_eq_steps_,axis=1)

    f_all_eq_steps_int=out_dir2_+vf+'all_eq_steps_int.npy'
    all_eq_steps_int=np.load(f_all_eq_steps_int)
    ocupados_int=all_eq_steps_int!=0
    t_map['rho_int']=((t_map.Nrep*2-4)-ocupados_int.sum(axis=1))/(t_map.Nrep*2-5)
    t_map['DT_total_int']=all_eq_steps_int.sum(axis=1)
    t_map['max_int_DT_int']=all_eq_steps_int.max(axis=1)
    t_map['wh_max_int_DT_int']=np.argmax(all_eq_steps_int,axis=1)

    t_map.to_csv(ali_dir+t_map_name+'.csv',index=False)
    
    

def add_only_energy_col_map(t_map,out_dir2_,t_map_name,rep_frag_len,j,ali_dir):

    
    evf='evo_energy/'
    os.system('mkdir '+out_dir2_+evf)
    
    
    # energy mean and std
    t_map['Ei_norm_std']=0
    t_map['Ei_std']=0
    t_map['Es_mean']=0
    t_map['Es_norm_mean']=0
    t_map['Ei_norm_mean']=0

    t_map['Es_norm_rel_mean']=0
    t_map['Es_all_mean']=0
    t_map['Ei_dif_mean']=0

    t_map['Es_all_mean_int']=0
    t_map['Ei_dif_mean_int']=0


# cargar la distribución de largos
    largos=t_map.groupby('Nrep').Nrep.count()
  
    for iL,L in enumerate(largos.index):

    #L=5
        n_units=len(rep_frag_len[j])*L
        f_norm_evo_array=out_dir2_+evf+str(L)+'_norm_evo.npy'
        f_dif=out_dir2_+evf+str(L)+'_dif.npy'
        f_norm_evo_array_int=out_dir2_+evf+str(L)+'_norm_evo_int.npy'
        f_dif_int=out_dir2_+evf+str(L)+'_dif_int.npy'
        f_Ei=out_dir2_+evf+str(L)+'_Ei.npy'
        f_Es=out_dir2_+evf+str(L)+'_Ei.npy'
        f_Ein=out_dir2_+evf+str(L)+'_Ein.npy'
        f_Esn=out_dir2_+evf+str(L)+'_Esn.npy'
        f_Esn_rel=out_dir2_+evf+str(L)+'_Esn_rel.npy'

        
        Ei=np.load(f_Ei)
        Es=np.load(f_Es)
        Ein=np.load(f_Ein)
        Esn=np.load(f_Esn)
        Esn_rel=np.load(f_Esn_rel)
        
        dif_array_int=np.load(f_dif_int)
        norm_evo_array_int=np.load(f_norm_evo_array_int)
        dif_array=np.load(f_dif)
        norm_evo_array=np.load(f_norm_evo_array)
        

        t_map.Ei_std.loc[t_map.Nrep==L]=Ei.std(axis=1)
        t_map.Ei_norm_std.loc[t_map.Nrep==L]=Ein.std(axis=1)
        t_map.Ei_norm_std.loc[t_map.Nrep==L]=Ein.std(axis=1)
        t_map.Ei_norm_mean.loc[t_map.Nrep==L]=Ein.mean(axis=1)

        t_map.Es_mean.loc[t_map.Nrep==L]=Esn.mean(axis=1)
        t_map.Es_norm_mean.loc[t_map.Nrep==L]=Esn.mean(axis=1)
        t_map.Es_norm_rel_mean.loc[t_map.Nrep==L]=Esn_rel.mean(axis=1)    
        t_map.Es_all_mean.loc[t_map.Nrep==L]=norm_evo_array.mean(axis=1)
        t_map.Ei_dif_mean.loc[t_map.Nrep==L]=dif_array.mean(axis=1)

        t_map.Es_all_mean_int.loc[t_map.Nrep==L]=norm_evo_array_int.mean(axis=1)
        t_map.Ei_dif_mean_int.loc[t_map.Nrep==L]=dif_array_int.mean(axis=1)


        

    t_map.to_csv(ali_dir+t_map_name+'.csv',index=False)


# =============================================================================
#  ONE PROTEIN PLOT
# =============================================================================

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def tick_function(X):
    s_='$\sigma_{'
    return [s_+str(i+1)+'}$' for i in range(len(X))]

def combined_heatmap_3(ax,prot_name,evo_energy_full,evo_energy,DH,breaks,
                                                  AAdict,replen,rep_frag_len,j,m):
 
    evo_energy_s,evo_energy_av=mcf.energy_average_matrix(evo_energy_full,breaks)
    evo_energy_av=evo_energy_av.where(np.triu(np.ones(evo_energy_s.shape)).astype(bool),0)

    evo_energy_s[evo_energy_s==0]=np.nan
    evo_energy_full[evo_energy_full==0]=np.nan

    vmin_s=min([evo_energy_full.min().min(),-evo_energy_full.max().max()])
    vmax_s=-vmin_s
    print(vmax_s,vmin_s)
    ha=sns.heatmap(evo_energy_full,cmap='seismic',ax=ax,center=0,vmin=vmin_s,vmax=vmax_s)

    evo_energy_s=evo_energy_s.where(np.triu(np.ones(evo_energy_s.shape)).astype(bool),0)

    sns.heatmap(evo_energy_s.transpose(),mask=np.triu(evo_energy_s),cmap='seismic',
                ax=ax, center=0,cbar=False)


    ax.hlines(breaks[1:], *ax.get_xlim(),'grey',linewidths=0.1)

    ax.vlines(breaks[1:],*ax.get_ylim(),'grey',linewidths=0.1)
    

    
    ax.axhline(y=0, color='k',linewidth=1)
    #ax.axhline(y=len(evo_energy_s)-0.2, color='k',linewidth=1)
    ax.axvline(x=0, color='k',linewidth=1)
    #ax.axvline(x=len(evo_energy_s)-0.3, color='k',linewidth=1)

    
    new_tick_locations = []
    for ib,b in enumerate(breaks):
        if b==breaks[-1]:
            aux=(b+len(evo_energy_s)+1)/2 
        else:
            aux=(b+breaks[ib+1])/2
        
        new_tick_locations.append(aux)
        

    
    
    ax.set_xticks(new_tick_locations)
    ax.set_xticklabels(tick_function(range(len(breaks)+1)),rotation = 0,fontsize=8)
  
    ax.set_yticks(new_tick_locations)
    ax.set_yticklabels(tick_function(range(len(breaks)+1)),rotation = 0,fontsize=8)
    
    
    ax.set_xlabel('Folding unit',fontsize=7)
    ax.set_ylabel('Folding unit',fontsize=7)

    
    ax2 = ax.twinx().twiny()
    ax2.xaxis.set_label_position('top') 
    ax2.set_xlabel('Amino-acid sequence position',fontsize=7,labelpad=10)

    ax2.set_xlim(ax.get_xlim())
    ax3=ax.twiny().twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_ylabel('Amino-acid sequence position',rotation=-90,fontsize=7,labelpad=10)

    
    ax2.set_xticks(np.append(breaks,len(evo_energy_s)))
    ax2.set_xticklabels(np.append(breaks+1,len(evo_energy_s)+1),rotation = 0,fontsize=7)
    ax3.set_xticks([])
    ax2.set_yticks([])


    ax3.set_yticks(np.append(breaks,len(evo_energy_s)))
    ax3.set_yticklabels(np.append(breaks+1,len(evo_energy_s)+1),rotation = 0,fontsize=7)
 #   ax2.tick_params(axis='y', which='major', labelsize=8)
 #   ax2.tick_params(axis='x', which='major', labelsize=8)

   # ax2.set_ylabel('Amino-acid sequence position')
  
    
    cbar = ha.collections[0].colorbar

    cbar.ax.set_aspect('auto')
    cbar.ax.set_ylim([vmin_s,-vmin_s])

    pos = cbar.ax.get_position()
    cbar2=cbar.ax.twinx()
    cbar2.set_ylim([-evo_energy_s.min().min(),evo_energy_s.min().min()])
 #   cbar2.set_ylim([-100,100])
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_ylabel('Evolutionary Energy',fontsize=7,labelpad=3)
    cbar.ax.tick_params(axis='y', which='major', labelsize=6)
    cbar2.set_ylabel('Ising Energy',rotation=-90,fontsize=7)
    cbar2.tick_params(axis='y', which='major', labelsize=6)

    pos.x0 += 0.2
    pos.x1+=0.15
    cbar.ax.set_position(pos)
    cbar2.set_position(pos)
    return 


    
def plot_ff_and_prob(fig,ax,out_dir_,ff_file_,states_file_,plot_exp_data=False,exp_data=None,st=True,DT=0,save=False):

    ff_file=out_dir_+ff_file_
    ff=np.loadtxt(ff_file)
    
    if st:
        ax[0,1].remove()  # remove unused upper right axes

        ax_ff=ax[0,0]
        ax_st=ax[1,0]
        ax_bar=ax[1,1]
        
        states_file=out_dir_+states_file_+'.npy'
        st=np.load(states_file)[2,:,:]

        st_=pd.DataFrame(st)
        st_.columns=[round(x) for x in ff[:,0]]
        sns.heatmap(st_,ax=ax_st,cbar_ax=ax_bar,xticklabels=100,cmap='RdBu')
        ax_st.set_title('')
        ax_st.set_xlabel('T')
        ax_st.set_ylabel('element')
        ax_st.set_yticklabels(np.arange(1,9,1))
        ax_st.tick_params(axis='x', rotation=0)

        ax_bar.set_ylabel('Prob folding')
        
    else:
        ax_ff=ax
    


   # ax_st.scatter(x=ff[:,0],y=ff[:,1],label='sim',linewidth=2,color='white')

    
    ax_ff.plot(ff[:,0],ff[:,1],label='simulation',linewidth=2,color='k',zorder=3)

    if plot_exp_data:
        init=np.argmin(abs(exp_data.temp[0]-ff[:,0]))
        fin=np.argmin(abs(exp_data.temp[len(exp_data)-1]-ff[:,0]))
        ax_ff.scatter(x=exp_data.temp,y=exp_data.ff*ff[init,1],color='red',label='experimental data',
                      s=10,zorder=2)
        ax_ff.legend()
        ax_ff.axvline(ff[init,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)
        ax_ff.axvline(ff[fin,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)

    if len(DT)==1: 
        ax_ff.set_xlim(min(ff[:,0]),max(ff[:,0]))
    else:
        ax_ff.set_xlim(DT[0],DT[1])
    ax_ff.set_xlabel('Temperature')
    ax_ff.set_ylabel('Folded fraction')
    
   # ax_ff.axvline(ff[426,0],color='grey',linewidth=2,linestyle='--',alpha=0.7)
   # ax_ff.axvline(ff[476,0],color='grey',linewidth=2,linestyle='--',alpha=0.7)
    if save:
        fig.savefig(out_dir_+'ff.pdf')


def plot_ff_mutants(ax,out_dir_,prot_names,exp_datas,labels,DT=[0],save=False):

   
    ff=np.loadtxt(out_dir_+prot_names[0]+'/ff')   
    ax_ff=ax

    init=np.argmin(abs(exp_datas[0].temp[0]-ff[:,0]))
    fin=np.argmin(abs(exp_datas[0].temp[len(exp_datas[0])-1]-ff[:,0]))
    ax_ff.axvline(ff[init,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)
    ax_ff.axvline(ff[fin,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)

    for i,exp_data in enumerate(exp_datas):
        ax_ff.scatter(x=exp_data.temp,y=exp_data.ff*ff[init,1],label=labels[i]+' exp',
                        s=10,zorder=2)
        ff=np.loadtxt(out_dir_+prot_names[i]+'/ff')
        ax_ff.plot(ff[:,0],ff[:,1],linewidth=2,zorder=3,label=labels[i])

    
    if len(DT)==1: 
        ax_ff.set_xlim(min(ff[:,0]),max(ff[:,0]))
    else:
        ax_ff.set_xlim(DT[0],DT[1])
        ff=np.loadtxt(out_dir_+prot_names[0]+'/ff')   
        ax_ff.set_ylim([0,ff[init,1]*1.05])
    ax_ff.set_xlabel('Temperature')
    ax_ff.set_ylabel('Folded fraction')
    ax_ff.legend()

    if save:
        fig.savefig(out_dir_+'ff.pdf')
    return

# NCORES VERSION

def plot_ff_i(fig,ax_ff,out_dir_,ff_file,DT=[0],num_cores=1,
              save=False,errorbar=False,plot_exp_data=False,exp_data=None):    
    
    if num_cores>1:

        ff=np.loadtxt(ff_file+'_1')   

        ffs=np.zeros((len(ff),num_cores))

        for fi in range(num_cores):
            ff=np.loadtxt(ff_file+'_'+str(fi))   
            ffs[:,fi]=ff[:,1]
        if errorbar:
            ax_ff.errorbar(x=ff[:,0],y=ffs.mean(axis=1),yerr=ffs.std(axis=1)/np.sqrt(num_cores),fmt='.')
        else:
            ax_ff.plot(ff[:,0],ffs.mean(axis=1),label='simulation',linewidth=2,color='k',zorder=3)
        ff[:,1]=ffs.mean(axis=1)

        
        
    else:
        ff=np.loadtxt(ff_file)
        ax_ff.plot(ff[:,0],ff[:,1],label='simulation',linewidth=2,color='k',zorder=3)
   
    if plot_exp_data:
        init=np.argmin(abs(exp_data.temp[0]-ff[:,0]))
        fin=np.argmin(abs(exp_data.temp[len(exp_data)-1]-ff[:,0]))
        ax_ff.scatter(x=exp_data.temp,y=exp_data.ff*ff[init,1],color='red',label='experimental data',
                      s=10,zorder=2)
        ax_ff.legend()
        ax_ff.axvline(ff[init,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)
        ax_ff.axvline(ff[fin,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)


    if len(DT)==1: 
        ax_ff.set_xlim(min(ff[:,0]),max(ff[:,0]))
    else:
        ax_ff.set_xlim(DT[0],DT[1])
    ax_ff.set_xlabel('Temperature')
    ax_ff.set_ylabel('Folded fraction')

    if save:
        fig.savefig(out_dir_+'ff.pdf')

def domains_and_fe(fig,ax,out_dir_,t_,nrep,DT=0,inter_t=2,cbar_ax=False,save=False,lw=.1,all_ticks=True,
                   ftick=1,ls=10,cbar_label=True,nwin=50,lim=5):
    ax_fq=ax[0]
    FQT_file=out_dir_+'FE_matrix.csv'
    temps_file=out_dir_+'FE_temps.csv'
    FQT=pd.read_csv(FQT_file,sep=' ',header=None)
    temps=pd.read_csv(temps_file,sep=' ',header=None)
    
    if len(DT)==1:
        itemps=np.arange(0,nwin-1,inter_t)
        
    else:
        itemps=np.arange(np.argmin(abs(DT[0]-temps)),np.argmin(abs(DT[1]-temps)),inter_t)

    nf=len(itemps)



    viridis = plt.cm.get_cmap('viridis', nf)
    colors=viridis(np.linspace(0,1,nf))


    for ci,it in enumerate(itemps):

        temp_=temps.loc[it]
        FQ=FQT.loc[it]
        ax_fq.plot(FQ,label='T ='+str(round(temp_[0])),c=colors[ci],linewidth=1,alpha=0.7)
    #ax_fq.legend()
   # ax_fq.set_title('Free energy')
    ax_fq.set_xlabel('Folded elements (Q)')
    #ax_fq.set_ylabel('Free energy')
    ax_fq.set_ylabel('$\Delta f$')
    
    if all_ticks:
        ax_fq.set_xticks(range(nrep*2+1))
    #ax_fq.set_xlim([0,nrep*2+4])
    
    if cbar_ax:
        colors=apparent_domains([ax[2],ax[1]],t_,vmin=temps.loc[min(itemps)],vmax=temps.loc[max(itemps)],lim=lim,
                                cbar_ax=cbar_ax,lw=lw,ftick=ftick,ls=ls)
    else:
        colors=apparent_domains(ax[1],t_,vmin=temps.loc[min(itemps)],vmax=temps.loc[max(itemps)],lim=5,cbar_ax=cbar_ax,
                                lw=lw,ftick=ftick,ls=ls,cbar_label=cbar_label)
    if save:
        fig.savefig(out_dir_+'domains_and_fe.pdf') 
    
    return colors


def apparent_domains(ax_,t_,lim=5,vmin=0,vmax=500,cbar_ax=False,lw=.1,ftick=1,ls=10,cbar_label=True):
    cmap=cm.get_cmap('viridis')
    
    partition,overlap=domain_partition(t_,lim)
    t_dom=domain_temperature(t_,partition)
    mat=domain_matrix(t_dom)

    data=pd.DataFrame(mat)
    
    if cbar_ax:
        ax=ax_[0]
        ha=sns.heatmap(data,ax=ax,cmap=cmap,linewidths=lw,vmin=vmin,vmax=vmax,cbar_ax=ax_[1])
        ax_[1].yaxis.tick_left()
        ax_[1].yaxis.set_label_position("left")
        ax_[1].set_ylabel('Temperature',fontsize=ls)
        ax_[1].tick_params(axis='both', which='major', labelsize=ls-1)

    else:
        ax=ax_
        ha=sns.heatmap(data,ax=ax,cmap=cmap,linewidths=lw,vmin=vmin,vmax=vmax)
        cbar = ha.collections[0].colorbar
        if cbar_label:
            cbar.ax.set_ylabel('Temperature',fontsize=ls)
        cbar.ax.tick_params(axis='both', which='major', labelsize=ls-1)
  #  ax.set_title('Apparent domains')

    
    ax.set_xlabel('element')
    #
    ax.set_yticks(ticks=np.arange(0.5,len(t_)+0.5,ftick))
    ax.set_xticks(ticks=np.arange(0.5,len(t_)+0.5,ftick))
    ax.set_xticklabels(np.arange(1,len(t_)+1,ftick),rotation=0)
    ax.set_yticklabels(np.arange(1,len(t_)+1,ftick),rotation=0)
    ax.set_ylabel('element')
   
   
    for x in np.arange(1,len(t_)+1,1):


        ax.axvline(x, color='grey',linewidth=lw)
        ax.axhline(x, color='grey',linewidth=lw)
    for x in [0,len(t_)+2]:
        ax.axvline(x, color='black',alpha=1,linewidth=lw)
        ax.axhline(x, color='black',alpha=1,linewidth=lw)
        
    norm = Normalize(vmin,vmax)
    rgba_values = cmap(norm(t_))
    colors=[]
    for rgba in rgba_values:
        colors.append(matplotlib.colors.rgb2hex(rgba))   
    
    return colors

def view_3d_domains(alinum,t_,colors,pdb,AF=False):
    t_[np.isnan(t_)]=0
    # colores segun Tf de cada elemento directamente




    if AF:
        pdb_filename='/home/ezequiel/SynologyDrive/folding_ank/pdbs/'+pdb+'.pdb'

        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
        view.addModel(open(pdb_filename,'r').read(),'pdb')
        
    else:
        view = py3Dmol.view(query='pdb:'+pdb,width=800, height=600)
        
    view.setBackgroundColor('white')

    view.setStyle({'cartoon':{'color':'white'}})
    view.setStyle({'chain':'B'},{'opacity':0 })
    view.setStyle({'chain':'C'},{'opacity':0 })


    #change residue color
    for e_,rfin in enumerate(alinum.fin):

        view.addStyle({'chain':'A','resi':[str(alinum.ini[e_])+'-'+str(rfin)]},{'cartoon':{'color':colors[e_]}})
        view.addStyle({'chain':'A','resi':[str(alinum.no[e_])]},{'cartoon':{'color':'white'}})
    view.zoomTo(viewer=(0,0))

    return view

def show_tfs(t_,prot_name,pdb,colors,png,AF=False,dir_='/home/ezequiel/SynologyDrive/folding_ank/msa/ali_num/'):    
    alinum=pd.read_csv(dir_+prot_name+'.csv')
    # el pdb de notch esta corrido
    if pdb=='1OT8':
        alinum.ini+=-1901
        alinum.fin+=-1901
    
    view=view_3d_domains(alinum,t_,colors,pdb,AF)
    

    
    print(t_)
    if png:
        view.show()
        time.sleep(20)
        png=view.png()

    return view


# =============================================================================
#  DATASET PLOT
# =============================================================================



def fit_levels(ax,t_map,x_,y_,degree,ticks,colors,alpha,xlims,ylims):
    y=t_map.frac_saltada
    X=np.zeros((len(t_map),2))
    for i_,i in enumerate(t_map.index):
        X[i_,:]=t_map[x_][i],-t_map[y_][i]


    poly = PolynomialFeatures(degree=degree)
    poly_variables = poly.fit_transform(X)

    regression = linear_model.LinearRegression(fit_intercept=False) 


    model = regression.fit(poly_variables, y)
    score = model.score(poly_variables, y)
    coef = pd.DataFrame({'label':poly.get_feature_names(),'coef':regression.coef_})


    poly = PolynomialFeatures(degree=degree)

    N=100
    predict_x0, predict_x1 = np.meshgrid(np.linspace(*xlims, N), 
                                         np.linspace(*ylims, N))
    predict_x = np.concatenate((predict_x0.reshape(-1, 1), 
                                predict_x1.reshape(-1, 1)), 
                               axis=1)
    predict_x_ = poly.fit_transform(predict_x)
    predict_y = regression.predict(predict_x_)

    ax.contour(predict_x0, predict_x1, predict_y.reshape(predict_x0.shape),ticks,colors=colors,alpha=alpha)
    
def plot_correlation_err(x,y,xlabel,ylabel,title,ax,yerr,alpha=0.1,fontsize=12,numsize=10):
    ax.errorbar(x=x,y=y,yerr=yerr,alpha=alpha,ls=' ')
    linreg = LinearRegression()
    linreg.fit(x[:,None], y,sample_weight=1/yerr)
    rsq=linreg.score(x[:,None], y,sample_weight=1/yerr)
    slp=linreg.coef_[0]
    xp = np.linspace(x.min(), x.max(), 3)
    yp = linreg.predict(xp[:,None])
    ax.plot(xp, yp, "r", lw=3,label='$R^2 =$ '+str(round(rsq,2)),linewidth=1 )
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.set_title(title,fontsize=fontsize)

    ax.legend(fontsize=fontsize-2)
    return linreg

def Tf_boxplot(fit_pars_p,t_map_pt_e,ax):
    ia=~np.isnan(fit_pars_p).any(axis=1)
    reps=t_map_pt_e.loc[ia].Nrep.unique()
    d=dict(zip(reps, [fit_pars_p.loc[ia].Tf[t_map_pt_e.Nrep==L]for L in reps]))
    ax.boxplot(d.values(),patch_artist=True)
    ax.set_xticklabels(d.keys())
    ax.set_xlabel('Length [repeats]')
    ax.set_ylabel('Folding temperature')
    

# =============================================================================
#  EXTRA FUNCTIONS
# =============================================================================


def str_to_save(x):
    xstr=''
    for i in range(len(x)):
        xstr=xstr+str(x[i])+ '\t'
    return xstr

    
def sig_fit_v3(X,Y):
    from scipy.optimize import curve_fit

    def fsigmoid_old(x, a, b,c):
        return c * np.exp(-a*(x-b)) / (1.0 + np.exp(-a*(x-b)))
    try:
        popt, pcov = curve_fit(fsigmoid, X, Y, method='lm', p0=[1,np.mean(X),1]) 
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

def fsigmoid_old(x, a, b,c):
        return c * np.exp(-a*(x-b)) / (1.0 + np.exp(-a*(x-b)))
    
    
    
def sig_fit_v4(X,Y):
    from scipy.optimize import curve_fit


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


