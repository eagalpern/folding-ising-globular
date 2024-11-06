import pandas as pd
from matplotlib import pyplot as plt, colors
import numpy as np
import os
from joblib import Parallel, delayed
from importlib import reload  
import MC_functions_plm as mcf
import process_and_plot_functions as pf

import pickle
import sys
sys.path.append('/home/ezequiel/libraries/PGM/source/')
sys.path.append('/home/ezequiel/libraries/PGM/utilities/')
import rbm,utilities
import Proteins_utils, RBM_utils, utilities,sequence_logo,plots_utils,bm

from Bio import SeqIO
import pickle
from numba import njit
import numba
import seaborn as sns
from DCA_output_func import *
#from model import CouplingsModel
#import tools
mcf=reload(mcf)
pf=reload(pf)

# =============================================================================
#  EXTRA FUNCTIONS
# =============================================================================

def extract_sequence_from_fasta(uniprot_id, fasta_file):
        found_sequence = None
        with open(fasta_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if uniprot_id in record.description:
                    found_sequence = record.seq
                    break  # Stop searching after finding the first match (if there are multiple)

        return found_sequence


def load_fasta(fasta_file):
        fasta_sequences = list(SeqIO.parse(fasta_file,'fasta'))
        MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
        names=[]
        w_list=[]
        for j in range(len(fasta_sequences)):
            MSA[j,:]=[i for i in fasta_sequences[j].seq]
            names.append(fasta_sequences[j].id)
            w_list.append(float(fasta_sequences[j].description.split(' ')[1]))
        name_list=[names[i].split('.')[0] for i in range(len(names))]
        ini=[int(names[i].split('/')[1].split('-')[0]) for i in range(len(names))]
        fin=[int(names[i].split('/')[1].split('-')[1]) for i in range(len(names))]
        weights=np.array(w_list)
        return  MSA, weights,name_list
    

def load_simple_fasta(fasta_file):
        fasta_sequences = list(SeqIO.parse(fasta_file,'fasta'))
        MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
        names=[]
        for j in range(len(fasta_sequences)):
            MSA[j,:]=[i for i in fasta_sequences[j].seq]
            names.append(fasta_sequences[j].id)
        return  MSA, names
    

def convert_rbm_potts(family,
                      n_h = 500, # RBM number of hidden units
                      n_iter = 500, # RBM epochs
                      hidden = 'Gaussian', # RBM potential
                      l1b = 0.25, # RBM regularization
                      main_path = '/home/ezequiel/Deposit/ising_rbm/', # rbm path
                      rbm_run_name = 'rbm_complete_set'): # rbm folder
    
    rbm_name='RBM_'+str(n_h)+'_'+str(n_iter)+'_'+f"{l1b:.3f}".replace('.', 'd')+'_'+hidden
    path_rbm=main_path+family+'/'+rbm_run_name+'/'
    RBM=RBM_utils.loadRBM(path_rbm+rbm_name)
    BM = convert_GaussianRBM_to_BM(RBM)
    Hi_,Jij_=zero_sum_gauge(BM.layer.fields,BM.layer.couplings)
    potts={'h':Hi_, 'J':Jij_}
    return potts


def get_breaks(family,
               path_ = '/home/ezequiel/Deposit/foldon_data/'): # exon data path

    if family=='DHFR':     # special case DHFR mapping to 7dfr
        breaks= np.load(path_+'DHFR/ecoli_breaks.npy')
    else:
        exon_freq_all=pd.read_csv(path_+'exon_freq_allfam.csv',index_col=0)
        exon_freq_fam=exon_freq_all[exon_freq_all.family==family]
        exon_bs_pdb=exon_freq_fam.pdb_original_res_num[exon_freq_fam.exon_bs].values
        breaks=exon_bs_pdb[:-1]-exon_bs_pdb[0]
    return breaks


def load_breaks_ctrl_(family,
                    N=10,
                    str_='exon_energies.pkl',
                    str_ctrl='ctrl_energies',
                    folder='DIF_TOTAL/',
                    path_='/home/ezequiel/Deposit/foldon_data/'):

    path_f=path_+family+'/'
    path_frustra_total=path_f+folder
    
    #with open(path_frustra_total+str_, 'rb') as f:
    #    ft=pd.DataFrame(pickle.load(f))
    #ft['family']=family
    
    breaks_ctrl=[]
    for n in range(N):
        with open(path_frustra_total+str_ctrl+str(n)+'.pkl', 'rb') as f:
            aux=pd.DataFrame(pickle.load(f))
        breaks_ctrl_=aux.exon_start.values-aux.exon_start.values[0]
        breaks_ctrl.append(breaks_ctrl_)
    return breaks_ctrl

def adapt_breaks(breaks_,
                 positions_original,
                 positions_ecoli):
    breaks_ecoli_=[positions_ecoli[positions_original==b][0] for b in breaks_]
    breaks=breaks_ecoli_-breaks_ecoli_[0]
    return breaks


def load_breaks_ctrl(family,
                    N=10,
                    str_='exon_energies.pkl',
                    str_ctrl='ctrl_energies',
                    folder='DIF_TOTAL/',
                    path_='/home/ezequiel/Deposit/foldon_data/'):

    if family == 'DHFR':
        positions_original = np.load(path_+'DHFR/positions_original.npy')
        positions_ecoli = np.load(path_+'DHFR/positions_ecoli.npy')
        breaks_ctrl = [adapt_breaks(breaks_c,
             positions_original = positions_original,
             positions_ecoli = positions_ecoli) for breaks_c in load_breaks_ctrl_('DHFR',
                                                                                 N,str_,
                                                                                 str_ctrl,
                                                                                 folder,
                                                                                 path_)]
    else:
        breaks_ctrl = load_breaks_ctrl_(family,N,str_,str_ctrl,folder,path_)
    
    return breaks_ctrl
        


def sigmoid_ff_fit_i(out_dir, num_cores):
    for i in range(num_cores):
        ff=np.loadtxt(out_dir+'ff_'+str(i))
        if i==0:
            ff_=np.zeros((ff.shape[0],num_cores))
        ff_[:,i]=ff[:,1]    
    RMSD,popt,pcov=pf.sig_fit_v4(ff[:,0],ff_.mean(axis=1))
    tf = popt[1]
    std_tf = np.sqrt(pcov[1,1])
    
    width = popt[0]
    std_width = np.sqrt(pcov[0,0])
    return tf, width, std_tf, std_width


# =============================================================================
#  MAIN FUNCTION
# =============================================================================

def ising_simulation(family,
                     potts,
                     breaks,
                     folder,
                     plot = True,
                     run_str = '', # multi seq : add string to identify the folder
                     interactions_off = False, # turn off surface energy terms
                     ctrl_breaks = False, # using breaks from exon ctrl group
                     ctrl_i = None, # breaks ctrl group id
                     seq = None, # imput single seq instead of MSA
                     multi_prot = False, # run many sequences for the same family or one
                     Nprot = 100, # if multi prot, number of sequences to simulate 
                     fastaname='MSA_nogap.fasta',  
                     si0=1, # MC entropy per residue
                     k=1, # MC Boltzmann constant
                     Tsel=1, # MC Family selection temperature
                     num_cores=8, # MC Cores for computing
                     tini_=1, # MC Initial temperature
                     tfin_=12, # MC Final temperature
                     ts_auto=True, # MC if run BORDER T CHECK, move t limits tofold and unfold 
                     DT=0.9, # MC temperature step for border check
                     gaps_out=True, # MC ignore gaps 
                     ninst=200, # MC 
                     ntsteps=40, # MC
                     cp_factor=20, # MC 
                     save_each=20, # MC
                     transient=50, # MC
                     nsteps=10000, # MC
                     inter_t=1, # Plot single seq 
                     nwin=10, # Free energy calc
                     nrep=4, # Plot single seq
                     lim_=5, # Plot single seq
                     AAdict={'Z':4,'X':0,'-':0,'B':3,'J':8,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12
            ,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}):  # potts AA code
    
    t_=[0]
    
    # =============================================================================
    #  LOAD MSA
    # =============================================================================      
    
    
    if seq is None:
        MSA, weights,names=load_fasta(folder+fastaname)
        


    # =============================================================================
    #  MC parameters
    # =============================================================================
    m = - 1/(k*Tsel) 
    j=0 # does nothing
    nrep=0 # does nothing
    replen,rep_frag_len = None,None # does nothing
    order,extrap=int(ntsteps/40),int(ntsteps/20)
    custom_ts=False # if true, accept to customize temperatures (ts_)
    ts_=None # custom temperatures
    custom_cp=False # if true, accept to customize critical points (cps_)
    cps_= None  # custom critical points (indexes)


    # =============================================================================
    #  SIMULATION
    # =============================================================================
    
    
    #  MULTI PROT
    # =============================================================================
    
    if multi_prot:
        if Nprot<MSA.shape[0]:
            ix=np.random.choice(range(MSA.shape[0]),Nprot,p=weights/sum(weights),replace=False)
        else:
            ix = np.arange(MSA.shape[0])
        #ix=np.random.choice(range(MSA.shape[0]),Nprot,p=weights/sum(weights),replace=False)
        for ix_ in ix:
            uniprot_id=names[ix_]
            seq=MSA[ix_]
            #seq_=extract_sequence_from_fasta(uniprot_id,path_f+fastaname)
            #print(seq_)
            #seq=np.array([np.char.upper(x) for x in seq_])
            if interactions_off:
                prot_name = uniprot_id + 'no_interactions'
            else:
                prot_name=uniprot_id
                
            group_folder= folder + 'multi_seq'+run_str+'/'
            try:
                os.system('mkdir '+ group_folder)
            except:
                pass
            folder_= group_folder + prot_name
            try:
                os.system('mkdir '+ folder_)
            except:
                pass

            ff_file=folder_+'/ff'
            states_file=folder_+'/st'
            q_hist_file=folder_+'/q_hist'
            ulf_file=folder_+'/ulf'
            DH_file=folder_+'/DH'
            DS_file=folder_+'/DS'
            print(uniprot_id)

            args1=seq,folder_,potts['J'],potts['h'],AAdict,replen,rep_frag_len,j,m,gaps_out,si0,k,nsteps,transient,save_each,ninst,ntsteps,DT,ff_file,states_file,q_hist_file,ulf_file,DH_file,DS_file,tini_,tfin_,ts_,custom_ts,ts_auto,breaks,interactions_off
            args2=DH_file,DS_file,ulf_file,j,k,nsteps,transient,save_each,ninst,ntsteps,order,extrap,cp_factor,ff_file,states_file,q_hist_file,custom_cp,cps_,num_cores

            Parallel(n_jobs=num_cores)(delayed(mcf.main_fold_1seq_first_round_i)(i,*j) for (i,j) in [(i_,args1) for i_ in np.arange(num_cores)])
            Parallel(n_jobs=num_cores)(delayed(mcf.main_fold_1seq_second_round_i)(i,*j) for (i,j) in [(i_,args2) for i_ in np.arange(num_cores)])
    
            #  FEATURES multi PROT
            # =============================================================================    
            t_,st=pf.multi_ff_fit_i(num_cores,ff_file,states_file)
            FE,obs,barr,eq_steps=pf.FE_analysis(ff_file,q_hist_file,nwin,k,num_cores,folder_+'/')

            coop_score = (eq_steps[1:-1]==0).sum() / len(eq_steps[1:-1])
            tf, width, std_tf, std_width = sigmoid_ff_fit_i(folder_+'/', num_cores)


            features = {
                "tf": tf,
                "width": width,
                "std_tf": std_tf,
                "std_width": std_width,
                "coop_score": coop_score,
                "t_": t_,
            }
            with open(folder_+'/features.pkl', "wb") as f:  # Note the "wb" mode for writing binary data
                    pickle.dump(features, f)

    
    
    
    #  SINGLE PROT
    # =============================================================================
    else: #only the reference protein for the family (first one on MSA)
        if seq is None:
            seq=MSA[0]
        if interactions_off:
            prot_name = 'reference_seq' + 'no_interactions'
        else:
            prot_name='reference_seq'
        
        if ctrl_breaks:
            folder = folder + '_ctrl_'+ str(ctrl_i) + '_'
            
        folder_= folder + prot_name
        try:
            os.system('mkdir '+ folder_)
        except:
            pass

        ff_file=folder_+'/ff'
        states_file=folder_+'/st'
        q_hist_file=folder_+'/q_hist'
        ulf_file=folder_+'/ulf'
        DH_file=folder_+'/DH'
        DS_file=folder_+'/DS'
        args1=seq,folder_,potts['J'],potts['h'],AAdict,replen,rep_frag_len,j,m,gaps_out,si0,k,nsteps,transient,save_each,ninst,ntsteps,DT,ff_file,states_file,q_hist_file,ulf_file,DH_file,DS_file,tini_,tfin_,ts_,custom_ts,ts_auto,breaks,interactions_off
        args2=DH_file,DS_file,ulf_file,j,k,nsteps,transient,save_each,ninst,ntsteps,order,extrap,cp_factor,ff_file,states_file,q_hist_file,custom_cp,cps_,num_cores
        Parallel(n_jobs=num_cores)(delayed(mcf.main_fold_1seq_first_round_i)(i,*j) for (i,j) in [(i_,args1) for i_ in np.arange(num_cores)])
        Parallel(n_jobs=num_cores)(delayed(mcf.main_fold_1seq_second_round_i)(i,*j) for (i,j) in [(i_,args2) for i_ in np.arange(num_cores)])

        
        
        #  FEATURES SINGLE PROT
        # =============================================================================    
        out_dir_= folder_+'/'
        #ff_file=out_dir_+'ff'
        #states_file=out_dir_+'st'
        #q_hist_file=out_dir_+'q_hist'
        
        t_,st=pf.multi_ff_fit_i(num_cores,ff_file,states_file)
        FE,obs,barr,eq_steps=pf.FE_analysis(ff_file,q_hist_file,nwin,k,num_cores,out_dir_)
        
        coop_score = (eq_steps[1:-1]==0).sum() / len(eq_steps[1:-1])
        tf, width, std_tf, std_width = sigmoid_ff_fit_i(out_dir_, num_cores)

        
        features = {
            "tf": tf,
            "width": width,
            "std_tf": std_tf,
            "std_width": std_width,
            "coop_score": coop_score,
            "t_": t_,
        }
        
        with open(out_dir_+'features.pkl', "wb") as f:  # Note the "wb" mode for writing binary data
                pickle.dump(features, f)

    #  PLOT SINGLE PROT
    # =============================================================================    
        if plot:
            fig = plt.figure(constrained_layout=True,figsize=pf.cm2inch(17,12))
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


            #folder_= main_dir + prot_name

            DT_=[tini_,tfin_]
            pf.plot_ff_i(fig,ax[0],out_dir_,ff_file,DT=DT_,num_cores=num_cores,exp_data=None,plot_exp_data=False)
            #ax[0].scatter(x=exp_data.temp,y=exp_data.ff,color='red',label='experimental data',
            #                      s=10,zorder=2)
            #ax[0].set_ylim([-0.1,1.1])
            colors=pf.domains_and_fe(fig,[ax[3],ax[4],ax[5]],out_dir_,t_,nrep,inter_t=inter_t,DT=DT_,cbar_ax=True,
                                     nwin=nwin,lim=lim_)
           # print(t_)

            plt.title(family)
        
    return features