import pandas as pd
import numpy as np
import os
import sys
import pickle
from joblib import Parallel, delayed
from monte_carlo import *
from free_energy_profile import *
from fit_ff import *


def ising_simulation(potts,
                     breaks,
                     folder,
                     run_str = '', # multi seq : add string to identify the folder
                     interactions_off = False, # turn off surface energy terms
                     ctrl_breaks = False, # using breaks from exon ctrl group
                     ctrl_i = None, # breaks ctrl group id
                     seq = None, # imput single seq instead of MSA
                     multi_prot = False, # run many sequences for the same family or one
                     Nprot = 100, # if multi prot, number of sequences to simulate 
                     fastaname='MSA_nogap.fasta',  
                     prot_name = 'reference_seq', 
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
                     nwin=10, # Free energy calc
                     AAdict={'Z':4,'X':0,'-':0,'B':3,'J':8,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12
            ,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}):  # potts AA code
    
    
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
        for ix_ in ix:
            uniprot_id=names[ix_]
            seq=MSA[ix_]
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

            Parallel(n_jobs=num_cores)(delayed(main_fold_1seq_first_round_i)(i,*j) for (i,j) in [(i_,args1) for i_ in np.arange(num_cores)])
            Parallel(n_jobs=num_cores)(delayed(main_fold_1seq_second_round_i)(i,*j) for (i,j) in [(i_,args2) for i_ in np.arange(num_cores)])
    
            #  FEATURES multi PROT
            # =============================================================================    
            t_,st=multi_ff_fit_i(num_cores,ff_file,states_file)
            FE,obs,barr,eq_steps=FE_analysis(ff_file,q_hist_file,nwin,k,num_cores,folder_+'/')

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
            
            prot_name = prot_name + 'no_interactions'
        
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
        Parallel(n_jobs=num_cores)(delayed(main_fold_1seq_first_round_i)(i,*j) for (i,j) in [(i_,args1) for i_ in np.arange(num_cores)])
        Parallel(n_jobs=num_cores)(delayed(main_fold_1seq_second_round_i)(i,*j) for (i,j) in [(i_,args2) for i_ in np.arange(num_cores)])

        
        
        #  FEATURES SINGLE PROT
        # =============================================================================    
        out_dir_= folder_+'/'
       
        t_,st=multi_ff_fit_i(num_cores,ff_file,states_file)
        FE,obs,barr,eq_steps=FE_analysis(ff_file,q_hist_file,nwin,k,num_cores,out_dir_)
        
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

   
        
    return features