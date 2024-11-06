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

import MC_functions_plm as mcf

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
    # Create a mask where N_q is zero
    zero_mask = (N_q == 0)
    # Compute the log values, treating the zero values separately
    log_values = -k * T * np.log(np.where(zero_mask, np.nan, N_q) / np.sum(N_q))
    # Replace the computed log values with np.inf where N_q was zero
    log_values[zero_mask] = np.inf
    return log_values
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

        #obs.nbarr[i]=len(qmax)
        obs.loc[i,'obs'] =len(qmax)
        if len(qmin)==1:
            #obs.abs_min[i]=qmin
            obs.loc[i,'abs_min']=qmin
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

                #obs.abs_min[i]=glob_min1
                #obs.abs_min_2[i]=glob_min2
                #obs.dif_mins[i]=abs(x[qmins][0]-x[qmins][1])
                #obs.wh_barr[i]=qbarr 
                #obs.h_barr[i]=abs(x[qbarr]-max([abs(y) for y in  x[qmins]]))
                obs.loc[i,'abs_min']=glob_min1
                obs.loc[i,'abs_min_2']=glob_min2
                obs.loc[i,'dif_mins']=abs(x[qmins][0]-x[qmins][1])
                obs.loc[i,'wh_barr']=qbarr 
                obs.loc[i,'h_barr']=abs(x[qbarr]-max([abs(y) for y in  x[qmins]]))
            else:
                #obs.abs_min[i]=glob_min1
                obs.loc[i,'abs_min']=glob_min1
        else:
            #obs.abs_min[i]=np.argmin(x)
            obs.loc[i,'abs_min']=np.argmin(x)

    obs=obs.astype({'nbarr':int,'wh_barr':int,'h_barr':float,'dif_mins':float,'abs_min':int,'abs_min_2':int})
    obs['Temp']=Temps
    return obs



def FE_analysis(ff_file,q_hist_file,nwin,k,num_cores,save_dir,save=True):

    
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
    obs=obs_(FE,Temps)
    barr, eq_steps=vect(obs,FE.shape[1])

    
    if save:
        np.savetxt(save_dir+'FE_matrix.csv',FE)
        np.savetxt(save_dir+'FE_temps.csv',Temps)

        obs.to_csv(save_dir+'FE_obs.csv')


        np.savetxt(save_dir+'barr.csv',barr)
        np.savetxt(save_dir+'eq_steps.csv',eq_steps)

    return FE,obs,barr, eq_steps




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
    s_=r'$\sigma_{'
    return [s_+str(i+1)+'}$' for i in range(len(X))]

def combined_heatmap_3(ax,prot_name,evo_energy_full,evo_energy,DH,breaks,
                                                  AAdict,replen,rep_frag_len,j,m):
 
    evo_energy_s,evo_energy_av=mcf.energy_average_matrix(evo_energy_full,breaks)
    evo_energy_av=evo_energy_av.where(np.triu(np.ones(evo_energy_s.shape)).astype(bool),0)

    evo_energy_s[evo_energy_s==0]=np.nan
    evo_energy_full[evo_energy_full==0]=np.nan

    vmin_s=min([evo_energy_full.min().min(),-evo_energy_full.max().max()])
    vmax_s=-vmin_s
  #  print(vmax_s,vmin_s)
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


    #viridis = plt.colormaps['viridis'](nf)
    viridis = plt.colormaps['viridis']
    colors = viridis(np.linspace(0, 1, nf))
    #viridis = plt.cm.get_cmap('viridis', nf)
    #colors=viridis(np.linspace(0,1,nf))


    for ci,it in enumerate(itemps):

        temp_=temps.loc[it]
        FQ=FQT.loc[it]
        ax_fq.plot(FQ,label='T ='+str(round(temp_[0])),c=colors[ci],linewidth=1,alpha=0.7)
    #ax_fq.legend()
   # ax_fq.set_title('Free energy')
    ax_fq.set_xlabel('Folded elements (Q)')
    #ax_fq.set_ylabel('Free energy')
    ax_fq.set_ylabel(r'$\Delta f$')
    
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
    #cmap=cm.get_cmap('viridis')
    cmap = matplotlib.colormaps['viridis']
    
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






# =============================================================================
#  EXTRA FUNCTIONS
# =============================================================================


def str_to_save(x):
    xstr=''
    for i in range(len(x)):
        xstr=xstr+str(x[i])+ '\t'
    return xstr
    
    
    
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


