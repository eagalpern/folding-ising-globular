import numpy as np
import pandas as pd
import os
import pickle
from matplotlib import pyplot as plt, colors
from matplotlib import cm
from matplotlib.colors import Normalize,rgb2hex
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib
import seaborn as sns

from domains import *
from free_energy_profile import *
from fit_ff import *



# =============================================================================
#  SINGLE PROTEIN PLOT
# =============================================================================

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def load_and_plot_ff(ax_ff,
                     folder,
                     cmap_ = 'viridis',
                     num_cores = 8,
                     vmin = 250,
                     vmax = 450,
                     prot_name = 'reference_seq',
                     plot_ff_protein = False,
                     plot_ff_foldon = True,
                     scatter_foldon = True,
                     t_ = None,
                     temp_color = True):
    
    out_dir_= folder + prot_name+'/'
    ff_file=out_dir_+'ff'
    states_file=out_dir_+'st'
    q_hist_file=out_dir_+'q_hist'

   #N=num_cores


    ff=np.loadtxt(ff_file+'_0')   
    ffs=np.zeros((len(ff),num_cores))
    for fi in range(num_cores):
        ff=np.loadtxt(ff_file+'_'+str(fi))   
        ffs[:,fi]=ff[:,1]

    ff=np.loadtxt(ff_file+'_0')
    for fi in range(num_cores):
        if fi==0:
            sti=np.load(states_file+'_'+str(fi)+'.npy')
        else:
            sti+=np.load(states_file+'_'+str(fi)+'.npy')
    st=sti/num_cores
    n_units=st.shape[1]


    # get colors from tf fit
    if t_ is None:
        Tfs,eTfs=[],[]
        for p in range(n_units):
            #RMSD,popt,pcov=sig_fit_v3(ff[:,0],st[:,p])
            RMSD,popt,pcov=sig_fit_v4(ff[:,0],st[:,p])
            std=np.sqrt(np.diag(pcov))
            Tfs.append(popt[1])
            eTfs.append(std[1])
        t_=np.array(Tfs)
     #   print(t_)

    if temp_color:
        cmap = colormaps.get_cmap(cmap_)
        norm = Normalize(vmin, vmax)
        color_foldon = cmap(norm(t_))

    else:
        new_cmap,color_foldon = rand_cmap(len(t_), 
                                         type='bright',
                                         first_color_black=False, 
                                         last_color_black=False,
                                         verbose=False)

    if plot_ff_foldon:
        for i in range(n_units):
            if scatter_foldon:
                ax_ff.scatter(ff[:,0],st[:,i],c=color_foldon[i])
            ax_ff.plot(ff[:,0],st[:,i],c=color_foldon[i])
    if plot_ff_protein:
        ax_ff.plot(ff[:,0],ffs.mean(axis=1),label='simulation',linewidth=1.5,color='k',zorder=3, ls='--')
    #ff[:,1]=ffs.mean(axis=1)
    ax_ff.set_xlabel('Temperature')
    ax_ff.set_ylabel('Folded fraction')

def apparent_domains_2(ax_,t_,ul,
                       lim=5,vmin=0,vmax=500,cbar_ax=False,lw=.1,ftick=1,ls=10,cbar_label=True):
    cmap = matplotlib.colormaps['viridis']
    
    partition,overlap=domain_partition(t_,lim)
    t_dom=domain_temperature(t_,partition)
    mat=domain_matrix(t_dom)

    
    data=pd.DataFrame(mat)
    # Step 1: Expand the DataFrame based on the sizes in `ul`
    N = len(ul)
    expanded_data = []
    for i in range(N):
        row_expansion = []
        for j in range(N):
            # Repeat the value in the cell according to ul[i] uniformly for rows and columns
            row_expansion.extend([data.iloc[i, j]] * ul[j])
        expanded_data.extend([row_expansion] * ul[i])

    expanded_df = pd.DataFrame(expanded_data)

    if cbar_ax:
        ax=ax_[0]
        ha=sns.heatmap(expanded_df,ax=ax,cmap=cmap,
                       linewidths=0, linecolor='white',
                       vmin=vmin,vmax=vmax,cbar_ax=ax_[1])
        ax_[1].yaxis.tick_left()
        ax_[1].yaxis.set_label_position("left")
        ax_[1].set_ylabel('Temperature',fontsize=ls)
        ax_[1].tick_params(axis='both', which='major', labelsize=ls-1)

    else:
        ax=ax_
        ha=sns.heatmap(expanded_df,ax=ax,cmap=cmap,linewidths=0, linecolor='white',
                       vmin=vmin,vmax=vmax,)
        cbar = ha.collections[0].colorbar
        if cbar_label:
            cbar.ax.set_ylabel('Temperature',fontsize=ls)
        cbar.ax.tick_params(axis='both', which='major', labelsize=ls-1)
    #  ax.set_title('Apparent domains')

    ax.set_ylabel('element')
    ax.set_xlabel('element')
    col_positions = np.append(0,np.cumsum(ul))# - 0.5
    
    #
    ax.set_yticks(ticks=col_positions[:-1]+ul/2)
    ax.set_xticks(ticks=col_positions[:-1]+ul/2)
    ax.set_xticklabels(np.arange(1,len(t_)+1,ftick),rotation=0)
    ax.set_yticklabels(np.arange(1,len(t_)+1,ftick),rotation=0)

    for pos in col_positions:
        ax.axvline(pos, color='grey', linewidth=lw)
        ax.axhline(pos, color='grey', linewidth=lw)

    norm = Normalize(vmin,vmax)
    rgba_values = cmap(norm(t_))
    colors=[]
    for rgba in rgba_values:
        colors.append(matplotlib.colors.rgb2hex(rgba))   

    return colors

def domains_and_fe_2(ax,out_dir_,t_,DT=0,inter_t=2,cbar_ax=False,save=False,lw=.1,all_ticks=True,
                   ftick=1,ls=10,cbar_label=True,nwin=50,lim=5,lw_fq=1,alpha_fq=.7, 
                     ul= None,noninf = True, t0 = 50):
    ax_fq=ax[0]
    FQT_file=out_dir_+'FE_matrix.csv'
    temps_file=out_dir_+'FE_temps.csv'
    FQT=pd.read_csv(FQT_file,sep=' ',header=None)
    temps=pd.read_csv(temps_file,sep=' ',header=None)
    #print(temps)
    if noninf:
        FQT[FQT==np.inf]=-1
        fmax = np.max(FQT)
        FQT[FQT==-1]=fmax*2
        ax_fq.set_ylim([0,fmax])

    
    if len(DT)==1:
        itemps=np.arange(0,nwin-1,inter_t)
        vmin = 100
        vmax = 500
        
    else: 
       # itemps=np.arange(np.argmin(abs(DT[0]-temps)),np.argmin(abs(DT[1]-temps)),inter_t)
        itemps=np.arange(np.argmin(abs(DT[0]-temps-t0)),np.argmin(abs(DT[1]-temps+t0)),inter_t)
        vmin = DT[0]
        vmax = DT[1]
    nf=len(itemps)
    #print(vmin, vmax)

    viridis = plt.colormaps['viridis']
    colors = viridis(np.linspace(0, 1, nf))
   

    for ci,it in enumerate(itemps):

        temp_=temps.loc[it]
        FQ=FQT.loc[it]
        ax_fq.plot(FQ,label='T ='+str(round(temp_[0])),c=colors[ci],linewidth=lw_fq,alpha=alpha_fq)
    #ax_fq.legend()
   # ax_fq.set_title('Free energy')
    ax_fq.set_xlabel('Folded elements (Q)')
    #ax_fq.set_ylabel('Free energy')
    ax_fq.set_ylabel('$\Delta f$')
    
    if all_ticks:
        ax_fq.set_xticks(range(len(t_)+1))
    #ax_fq.set_xlim([0,nrep*2+4])
  
    
    if cbar_ax:
        colors=apparent_domains_2([ax[2],ax[1]],t_,ul,vmin=vmin,vmax=vmax,lim=lim,
                                cbar_ax=cbar_ax,lw=lw,ftick=ftick,ls=ls)
    else:
        colors=apparent_domains_2(ax[1],t_,ul,vmin=vmin,vmax=vmax,lim=5,cbar_ax=cbar_ax,
                                lw=lw,ftick=ftick,ls=ls,cbar_label=cbar_label)
        

   
    return colors

def plot_ising(folder,
               ax_ff,
               ax_domains_and_fe,
               main_path = '/home/ezequiel/Deposit/ising_rbm/',
               prot_name = 'reference_seq',
               fit_tfs = False,
               vmin = 150,
               vmax = 500,
               inter_t=1, # Plot single seq 
               nwin=10, # Free energy calc
               lim_=5, # Plot single seq
               lw=.6, # line width domains 
               lw_fq = 1.5, # line width df
               alpha_fq= 1, # alpha df
               fontsize = 10,
               noninf = False,
               t0 = 0):
    
    #family = tsel_fam.family[i]
    #folder = main_path+family+'/'+tsel_fam.loc[tsel_fam.family==family,"best_ali"].values[0]+'/'
    #print(family)
    tini_ = vmin#-50
    tfin_ = vmax#+50


    out_dir_= folder + prot_name+'/'
    ff_file=out_dir_+'ff'
    states_file=out_dir_+'st'
    q_hist_file=out_dir_+'q_hist'
    ulf_file=out_dir_+'ulf'

    ul=np.loadtxt(ulf_file,int)

    num_cores=8
    ff=np.loadtxt(ff_file+'_0')   
    ffs=np.zeros((len(ff),num_cores))
    for fi in range(num_cores):
        ff=np.loadtxt(ff_file+'_'+str(fi))   
        ffs[:,fi]=ff[:,1]

    ff=np.loadtxt(ff_file+'_0')
    for fi in range(num_cores):
        if fi==0:
            sti=np.load(states_file+'_'+str(fi)+'.npy')
        else:
            sti+=np.load(states_file+'_'+str(fi)+'.npy')
    st=sti/num_cores
    n_units=st.shape[1]


    # get colors from tf fit
    
    if fit_tfs:

        Tfs,eTfs=[],[]
        for p in range(n_units):
            #RMSD,popt,pcov=sig_fit_v3(ff[:,0],st[:,p])
            RMSD,popt,pcov=sig_fit_v4(ff[:,0],st[:,p])
            std=np.sqrt(np.diag(pcov))
            Tfs.append(popt[1])
            eTfs.append(std[1])
        t_=np.array(Tfs)
    
    else:
        with open(out_dir_+'features.pkl', "rb") as f:  
            features = pickle.load(f)
            t_ = features['t_']

    load_and_plot_ff(ax_ff,
                     folder,
                     cmap_ = 'viridis',
                     vmin = vmin,
                     vmax = vmax,
                     prot_name = prot_name,
                     t_ = t_,
                     plot_ff_protein = True,
                     plot_ff_foldon = True,
                     scatter_foldon = False,
                     temp_color = True)

    DT_=[tini_,tfin_]
    colors=domains_and_fe_2(ax_domains_and_fe,out_dir_,t_,
                            inter_t=inter_t,DT=DT_,cbar_ax=True,
                            nwin=nwin,lim=lim_,ul=ul,lw=lw,lw_fq = lw_fq, alpha_fq=alpha_fq, ls=fontsize,
                            noninf = noninf, t0 = t0)
    


#

def build_axes():
    fig = plt.figure(constrained_layout=True,figsize=cm2inch(17,12))
    widths = [6,0.2, 3.8 ]
    heights = [3, 3]
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    ax=[]
    for row in range(2):
        for col in range(3):
            ax.append(fig.add_subplot(spec[row, col]))
    ax[1].remove()        
    ax[2].remove()        
    return fig,ax



def build_axes_2(N,
                width_cm = 16,
                height_cm = 4):
    fig = plt.figure(constrained_layout=True,figsize=cm2inch(width_cm,N*height_cm))
    widths = [4, 4, 0.2, 3.8 ]
    heights = [1]*N
    spec = fig.add_gridspec(ncols=4, nrows=N, width_ratios=widths,
                              height_ratios=heights)
    ax=[]
    for row in range(N):
        for col in range(4):
            ax.append(fig.add_subplot(spec[row, col]))
    #ax[1].remove()        
    #ax[2].remove()        
    return fig, ax






# def tick_function(X):
#     s_=r'$\sigma_{'
#     return [s_+str(i+1)+'}$' for i in range(len(X))]

# def combined_heatmap_3(ax,prot_name,evo_energy_full,evo_energy,DH,breaks,
#                                                   AAdict,replen,rep_frag_len,j,m):
 
#     evo_energy_s,evo_energy_av=mcf.energy_average_matrix(evo_energy_full,breaks)
#     evo_energy_av=evo_energy_av.where(np.triu(np.ones(evo_energy_s.shape)).astype(bool),0)

#     evo_energy_s[evo_energy_s==0]=np.nan
#     evo_energy_full[evo_energy_full==0]=np.nan

#     vmin_s=min([evo_energy_full.min().min(),-evo_energy_full.max().max()])
#     vmax_s=-vmin_s
#   #  print(vmax_s,vmin_s)
#     ha=sns.heatmap(evo_energy_full,cmap='seismic',ax=ax,center=0,vmin=vmin_s,vmax=vmax_s)

#     evo_energy_s=evo_energy_s.where(np.triu(np.ones(evo_energy_s.shape)).astype(bool),0)

#     sns.heatmap(evo_energy_s.transpose(),mask=np.triu(evo_energy_s),cmap='seismic',
#                 ax=ax, center=0,cbar=False)


#     ax.hlines(breaks[1:], *ax.get_xlim(),'grey',linewidths=0.1)

#     ax.vlines(breaks[1:],*ax.get_ylim(),'grey',linewidths=0.1)
    

    
#     ax.axhline(y=0, color='k',linewidth=1)
#     #ax.axhline(y=len(evo_energy_s)-0.2, color='k',linewidth=1)
#     ax.axvline(x=0, color='k',linewidth=1)
#     #ax.axvline(x=len(evo_energy_s)-0.3, color='k',linewidth=1)

    
#     new_tick_locations = []
#     for ib,b in enumerate(breaks):
#         if b==breaks[-1]:
#             aux=(b+len(evo_energy_s)+1)/2 
#         else:
#             aux=(b+breaks[ib+1])/2
        
#         new_tick_locations.append(aux)
        

    
    
#     ax.set_xticks(new_tick_locations)
#     ax.set_xticklabels(tick_function(range(len(breaks)+1)),rotation = 0,fontsize=8)
  
#     ax.set_yticks(new_tick_locations)
#     ax.set_yticklabels(tick_function(range(len(breaks)+1)),rotation = 0,fontsize=8)
    
    
#     ax.set_xlabel('Folding unit',fontsize=7)
#     ax.set_ylabel('Folding unit',fontsize=7)

    
#     ax2 = ax.twinx().twiny()
#     ax2.xaxis.set_label_position('top') 
#     ax2.set_xlabel('Amino-acid sequence position',fontsize=7,labelpad=10)

#     ax2.set_xlim(ax.get_xlim())
#     ax3=ax.twiny().twinx()
#     ax3.set_ylim(ax.get_ylim())
#     ax3.set_ylabel('Amino-acid sequence position',rotation=-90,fontsize=7,labelpad=10)

    
#     ax2.set_xticks(np.append(breaks,len(evo_energy_s)))
#     ax2.set_xticklabels(np.append(breaks+1,len(evo_energy_s)+1),rotation = 0,fontsize=7)
#     ax3.set_xticks([])
#     ax2.set_yticks([])


#     ax3.set_yticks(np.append(breaks,len(evo_energy_s)))
#     ax3.set_yticklabels(np.append(breaks+1,len(evo_energy_s)+1),rotation = 0,fontsize=7)
#  #   ax2.tick_params(axis='y', which='major', labelsize=8)
#  #   ax2.tick_params(axis='x', which='major', labelsize=8)

#    # ax2.set_ylabel('Amino-acid sequence position')
  
    
#     cbar = ha.collections[0].colorbar

#     cbar.ax.set_aspect('auto')
#     cbar.ax.set_ylim([vmin_s,-vmin_s])

#     pos = cbar.ax.get_position()
#     cbar2=cbar.ax.twinx()
#     cbar2.set_ylim([-evo_energy_s.min().min(),evo_energy_s.min().min()])
#  #   cbar2.set_ylim([-100,100])
#     cbar.ax.yaxis.set_label_position("left")
#     cbar.ax.set_ylabel('Evolutionary Energy',fontsize=7,labelpad=3)
#     cbar.ax.tick_params(axis='y', which='major', labelsize=6)
#     cbar2.set_ylabel('Ising Energy',rotation=-90,fontsize=7)
#     cbar2.tick_params(axis='y', which='major', labelsize=6)

#     pos.x0 += 0.2
#     pos.x1+=0.15
#     cbar.ax.set_position(pos)
#     cbar2.set_position(pos)
#     return 


    
# def plot_ff_and_prob(fig,ax,out_dir_,ff_file_,states_file_,plot_exp_data=False,exp_data=None,st=True,DT=0,save=False):

#     ff_file=out_dir_+ff_file_
#     ff=np.loadtxt(ff_file)
    
#     if st:
#         ax[0,1].remove()  # remove unused upper right axes

#         ax_ff=ax[0,0]
#         ax_st=ax[1,0]
#         ax_bar=ax[1,1]
        
#         states_file=out_dir_+states_file_+'.npy'
#         st=np.load(states_file)[2,:,:]

#         st_=pd.DataFrame(st)
#         st_.columns=[round(x) for x in ff[:,0]]
#         sns.heatmap(st_,ax=ax_st,cbar_ax=ax_bar,xticklabels=100,cmap='RdBu')
#         ax_st.set_title('')
#         ax_st.set_xlabel('T')
#         ax_st.set_ylabel('element')
#         ax_st.set_yticklabels(np.arange(1,9,1))
#         ax_st.tick_params(axis='x', rotation=0)

#         ax_bar.set_ylabel('Prob folding')
        
#     else:
#         ax_ff=ax
    


#    # ax_st.scatter(x=ff[:,0],y=ff[:,1],label='sim',linewidth=2,color='white')

    
#     ax_ff.plot(ff[:,0],ff[:,1],label='simulation',linewidth=2,color='k',zorder=3)

#     if plot_exp_data:
#         init=np.argmin(abs(exp_data.temp[0]-ff[:,0]))
#         fin=np.argmin(abs(exp_data.temp[len(exp_data)-1]-ff[:,0]))
#         ax_ff.scatter(x=exp_data.temp,y=exp_data.ff*ff[init,1],color='red',label='experimental data',
#                       s=10,zorder=2)
#         ax_ff.legend()
#         ax_ff.axvline(ff[init,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)
#         ax_ff.axvline(ff[fin,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)

#     if len(DT)==1: 
#         ax_ff.set_xlim(min(ff[:,0]),max(ff[:,0]))
#     else:
#         ax_ff.set_xlim(DT[0],DT[1])
#     ax_ff.set_xlabel('Temperature')
#     ax_ff.set_ylabel('Folded fraction')
    
#    # ax_ff.axvline(ff[426,0],color='grey',linewidth=2,linestyle='--',alpha=0.7)
#    # ax_ff.axvline(ff[476,0],color='grey',linewidth=2,linestyle='--',alpha=0.7)
#     if save:
#         fig.savefig(out_dir_+'ff.pdf')


# def plot_ff_mutants(ax,out_dir_,prot_names,exp_datas,labels,DT=[0],save=False):

   
#     ff=np.loadtxt(out_dir_+prot_names[0]+'/ff')   
#     ax_ff=ax

#     init=np.argmin(abs(exp_datas[0].temp[0]-ff[:,0]))
#     fin=np.argmin(abs(exp_datas[0].temp[len(exp_datas[0])-1]-ff[:,0]))
#     ax_ff.axvline(ff[init,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)
#     ax_ff.axvline(ff[fin,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)

#     for i,exp_data in enumerate(exp_datas):
#         ax_ff.scatter(x=exp_data.temp,y=exp_data.ff*ff[init,1],label=labels[i]+' exp',
#                         s=10,zorder=2)
#         ff=np.loadtxt(out_dir_+prot_names[i]+'/ff')
#         ax_ff.plot(ff[:,0],ff[:,1],linewidth=2,zorder=3,label=labels[i])

    
#     if len(DT)==1: 
#         ax_ff.set_xlim(min(ff[:,0]),max(ff[:,0]))
#     else:
#         ax_ff.set_xlim(DT[0],DT[1])
#         ff=np.loadtxt(out_dir_+prot_names[0]+'/ff')   
#         ax_ff.set_ylim([0,ff[init,1]*1.05])
#     ax_ff.set_xlabel('Temperature')
#     ax_ff.set_ylabel('Folded fraction')
#     ax_ff.legend()

#     if save:
#         fig.savefig(out_dir_+'ff.pdf')
#     return

# # NCORES VERSION

# def plot_ff_i(fig,ax_ff,out_dir_,ff_file,DT=[0],num_cores=1,
#               save=False,errorbar=False,plot_exp_data=False,exp_data=None):    
    
#     if num_cores>1:

#         ff=np.loadtxt(ff_file+'_1')   

#         ffs=np.zeros((len(ff),num_cores))

#         for fi in range(num_cores):
#             ff=np.loadtxt(ff_file+'_'+str(fi))   
#             ffs[:,fi]=ff[:,1]
#         if errorbar:
#             ax_ff.errorbar(x=ff[:,0],y=ffs.mean(axis=1),yerr=ffs.std(axis=1)/np.sqrt(num_cores),fmt='.')
#         else:
#             ax_ff.plot(ff[:,0],ffs.mean(axis=1),label='simulation',linewidth=2,color='k',zorder=3)
#         ff[:,1]=ffs.mean(axis=1)

        
        
#     else:
#         ff=np.loadtxt(ff_file)
#         ax_ff.plot(ff[:,0],ff[:,1],label='simulation',linewidth=2,color='k',zorder=3)
   
#     if plot_exp_data:
#         init=np.argmin(abs(exp_data.temp[0]-ff[:,0]))
#         fin=np.argmin(abs(exp_data.temp[len(exp_data)-1]-ff[:,0]))
#         ax_ff.scatter(x=exp_data.temp,y=exp_data.ff*ff[init,1],color='red',label='experimental data',
#                       s=10,zorder=2)
#         ax_ff.legend()
#         ax_ff.axvline(ff[init,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)
#         ax_ff.axvline(ff[fin,0],color='grey',linewidth=0.5,linestyle='--',alpha=0.3,zorder=1)


#     if len(DT)==1: 
#         ax_ff.set_xlim(min(ff[:,0]),max(ff[:,0]))
#     else:
#         ax_ff.set_xlim(DT[0],DT[1])
#     ax_ff.set_xlabel('Temperature')
#     ax_ff.set_ylabel('Folded fraction')

#     if save:
#         fig.savefig(out_dir_+'ff.pdf')

# def domains_and_fe(fig,ax,out_dir_,t_,nrep,DT=0,inter_t=2,cbar_ax=False,save=False,lw=.1,all_ticks=True,
#                    ftick=1,ls=10,cbar_label=True,nwin=50,lim=5):
#     ax_fq=ax[0]
#     FQT_file=out_dir_+'FE_matrix.csv'
#     temps_file=out_dir_+'FE_temps.csv'
#     FQT=pd.read_csv(FQT_file,sep=' ',header=None)
#     temps=pd.read_csv(temps_file,sep=' ',header=None)
    
#     if len(DT)==1:
#         itemps=np.arange(0,nwin-1,inter_t)
        
#     else:
#         itemps=np.arange(np.argmin(abs(DT[0]-temps)),np.argmin(abs(DT[1]-temps)),inter_t)

#     nf=len(itemps)


#     #viridis = plt.colormaps['viridis'](nf)
#     viridis = plt.colormaps['viridis']
#     colors = viridis(np.linspace(0, 1, nf))
#     #viridis = plt.cm.get_cmap('viridis', nf)
#     #colors=viridis(np.linspace(0,1,nf))


#     for ci,it in enumerate(itemps):

#         temp_=temps.loc[it]
#         FQ=FQT.loc[it]
#         ax_fq.plot(FQ,label='T ='+str(round(temp_[0])),c=colors[ci],linewidth=1,alpha=0.7)
#     #ax_fq.legend()
#    # ax_fq.set_title('Free energy')
#     ax_fq.set_xlabel('Folded elements (Q)')
#     #ax_fq.set_ylabel('Free energy')
#     ax_fq.set_ylabel(r'$\Delta f$')
    
#     if all_ticks:
#         ax_fq.set_xticks(range(nrep*2+1))
#     #ax_fq.set_xlim([0,nrep*2+4])
    
#     if cbar_ax:
#         colors=apparent_domains([ax[2],ax[1]],t_,vmin=temps.loc[min(itemps)],vmax=temps.loc[max(itemps)],lim=lim,
#                                 cbar_ax=cbar_ax,lw=lw,ftick=ftick,ls=ls)
#     else:
#         colors=apparent_domains(ax[1],t_,vmin=temps.loc[min(itemps)],vmax=temps.loc[max(itemps)],lim=5,cbar_ax=cbar_ax,
#                                 lw=lw,ftick=ftick,ls=ls,cbar_label=cbar_label)
#     if save:
#         fig.savefig(out_dir_+'domains_and_fe.pdf') 
    
#     return colors


# def apparent_domains(ax_,t_,lim=5,vmin=0,vmax=500,cbar_ax=False,lw=.1,ftick=1,ls=10,cbar_label=True):
#     #cmap=cm.get_cmap('viridis')
#     cmap = matplotlib.colormaps['viridis']
    
#     partition,overlap=domain_partition(t_,lim)
#     t_dom=domain_temperature(t_,partition)
#     mat=domain_matrix(t_dom)

#     data=pd.DataFrame(mat)
    
#     if cbar_ax:
#         ax=ax_[0]
#         ha=sns.heatmap(data,ax=ax,cmap=cmap,linewidths=lw,vmin=vmin,vmax=vmax,cbar_ax=ax_[1])
#         ax_[1].yaxis.tick_left()
#         ax_[1].yaxis.set_label_position("left")
#         ax_[1].set_ylabel('Temperature',fontsize=ls)
#         ax_[1].tick_params(axis='both', which='major', labelsize=ls-1)

#     else:
#         ax=ax_
#         ha=sns.heatmap(data,ax=ax,cmap=cmap,linewidths=lw,vmin=vmin,vmax=vmax)
#         cbar = ha.collections[0].colorbar
#         if cbar_label:
#             cbar.ax.set_ylabel('Temperature',fontsize=ls)
#         cbar.ax.tick_params(axis='both', which='major', labelsize=ls-1)
#   #  ax.set_title('Apparent domains')

    
#     ax.set_xlabel('element')
#     #
#     ax.set_yticks(ticks=np.arange(0.5,len(t_)+0.5,ftick))
#     ax.set_xticks(ticks=np.arange(0.5,len(t_)+0.5,ftick))
#     ax.set_xticklabels(np.arange(1,len(t_)+1,ftick),rotation=0)
#     ax.set_yticklabels(np.arange(1,len(t_)+1,ftick),rotation=0)
#     ax.set_ylabel('element')
     
#     for x in np.arange(1,len(t_)+1,1):


#         ax.axvline(x, color='grey',linewidth=lw)
#         ax.axhline(x, color='grey',linewidth=lw)
#     for x in [0,len(t_)+2]:
#         ax.axvline(x, color='black',alpha=1,linewidth=lw)
#         ax.axhline(x, color='black',alpha=1,linewidth=lw)
        
#     norm = Normalize(vmin,vmax)
#     rgba_values = cmap(norm(t_))
#     colors=[]
#     for rgba in rgba_values:
#         colors.append(matplotlib.colors.rgb2hex(rgba))   
    
#     return colors






# =============================================================================
#  EXTRA FUNCTIONS
# =============================================================================

