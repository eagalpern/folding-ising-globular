import numpy as np
import pandas as pd
import time
import numba
from numba import njit
import scipy.signal as sps
import random
from evolutionary_energy import *


@njit(inline="always")
def folding_energy_ss_speed(state: np.array,temp: np.float64, 
                            DHarr: numba.types.Array(np.float64,ndim=2,layout='F') ,
                            DS: numba.types.Array(np.float64,ndim=2,layout='F')):
    H=np.sum(DHarr[np.where(state>0)[0]][:,np.where(state>0)[0]])
    S=np.sum(DS[np.where(state==0)[0]][:,np.where(state==0)[0]])
    G=-H-temp*S 
    # Ferreiro et al (2008) : evo_energy is DH_unfolding>0; S is DS_unfolding>0
    return G


@njit(inline="always")
def montecarlo_speed_complete(evo_energy: numba.types.Array(np.float64,ndim=2,layout='F'),
                temp:np.float64 ,DS: numba.types.Array(np.float64,ndim=2,layout='F'),
                        units_len: np.array,k: np.float64 ,nsteps: np.array ,
                transient: numba.int64, save_each: numba.int64, state: np.array):
    n=evo_energy.shape[0]
    m=np.zeros(0)
    q=np.zeros(0)
   # Es=np.zeros(0)
    stat_av=np.zeros(n)
    e0=folding_energy_ss_speed(state,temp,evo_energy,DS) # initial energy

    for i in range(nsteps):
        # MC loop
        x=np.random.randint(n) # choice random spin 
        state[x] = int(not state[x]) # flip it
        ef=folding_energy_ss_speed(state,temp,evo_energy,DS) # energy

        de=ef-e0 # change in energy
    
        # metropolis criterium
        if de<=0: 
            e0=ef
        else:
            if random.uniform(0,1)<np.exp(-de/(k*temp)):
                e0=ef
            else:
                state[x] = int(not state[x]) # flip it back     
        # save fraction folded
        if i%save_each==0 and i>transient:
            m=np.append(m,np.sum(state*units_len)) # pseudomagnetization * residue
            q=np.append(q,np.sum(state)) # pseudomagnetization * residue
            #Es=np.append(Es,e0)
            stat_av+=state # aca sumar estados
        
        q_hist=np.histogram(q,bins=np.arange(-0.5,n+1.5,1))[0]

        
    return state,np.mean(m)/np.sum(units_len),np.std(m/np.sum(units_len))/np.sqrt(m.shape[0]), stat_av/m.shape[0],q_hist
    # last_state,fraction folded average, fraction folded error, state average, histogram  


def temp_denaturation_first_round(DH,DS,units_len,nrep,j,ts,k,ninst,nsteps_,transient_,save_each_):
        
    if nrep>0:  
        # repeat proteins
        nsteps=nsteps_[j,nrep]
        transient=transient_[j,nrep]
        save_each=save_each_[j,nrep]
    else:
        # goblular proteins
        nsteps=nsteps_
        transient=transient_
        save_each=save_each_
    
    nsave=int((nsteps-transient)/save_each)-1 
    n=DH.shape[0]

    state=np.array([random.randint(0, 1) for _ in range(n)])

    folded_res=np.zeros(len(ts))
    folded_res_err=np.zeros(len(ts))
    states_av=np.zeros((len(ts),n))
    q_hist=np.zeros((len(ts),n+1))
    
    for t,temp in enumerate(ts):
        state,folded_res[t],folded_res_err[t],states_av[t,:],q_hist[t,:]=montecarlo_speed_complete(DH,ts[t],DS,units_len,k,nsteps,transient,save_each,state)

    
    time.sleep(0.2)     

    return folded_res,folded_res_err,states_av,q_hist


def temp_denaturation_second_round(critical_points,folded_res,folded_res_err,states_av,q_hist,
                                   DH,DS,units_len,nrep,ts,j,k,ninst_,nsteps_,transient_,
                                   save_each_,order,extrap,cp_factor):
    
    if nrep>0:  
        # repeat proteins
        nsteps=nsteps_[j,nrep]
        transient=transient_[j,nrep]
        save_each=save_each_[j,nrep]
        ninst=ninst_[j]
    else:
        # goblular proteins
        nsteps=nsteps_
        transient=transient_
        save_each=save_each_
        ninst=ninst_
    

    save_critical=save_each*cp_factor
    nsteps_critical=save_each*cp_factor*ninst
    nsave=int((nsteps-transient)/save_each)-1 
    n=DH.shape[0]

    state=np.array([random.randint(0, 1) for _ in range(n)])

    for t in critical_points:
            #print(t)
            state,folded_res[t],folded_res_err[t],states_av[t,:],q_hist[t,:]=montecarlo_speed_complete(DH,ts[t],DS,units_len,k,nsteps_critical,save_critical,save_critical,state)
    
    time.sleep(0.2)     

    return folded_res,folded_res_err,states_av,q_hist


def main_fold_1seq_first_round_i(i_,seq,out_dir_,Jij,Hi,AAdict,replen,rep_frag_len,j,m,gaps_out,si0,
                                 k,nsteps_,transient_,save_each_,ninst,ntsteps,DT,
                                 ff_file,states_file,q_hist_file,ulf_file,DH_file,DS_file,
                                 tini_=None,tfin_=None,ts_=None,custom_ts=False,ts_auto=True,breaks=None,
                                interactions_off=False):
    nsteps=nsteps_
    transient=transient_
    save_each=save_each_
    nrep=0

    
    DH,breaks=seq_to_ising_DCA(seq,Jij,Hi,AAdict,breaks,m,gaps_out)
    DS,units_len=si0_to_DS_units_len_DCA(si0,seq,breaks,gaps_out)
    
    if interactions_off:
        for i in range(len(DH)):
            for j in range(len(DH.columns)):
                if i != j:  
                    DH.iloc[i, j] = 0  
    
    if i_==0:
        np.savetxt(ulf_file,units_len,fmt='%i')
        np.savetxt(DH_file,DH.values)
        np.savetxt(DS_file,DS.values)

    if custom_ts:
        ts=ts_
         
    else:   
        if ts_auto:
            #4) correr rapido solo ff(T) en los extremos para ver si pasamos de plegado a desplegado 
            tini,tfin=border_check(DT,DH.values,DS.values,units_len,tini_,tfin_,k,
                                          nsteps,transient,save_each)

        else:
            tini=tini_
            tfin=tfin_

        ts=np.linspace(tini,tfin,ntsteps)

    ff=np.zeros((len(ts),2))
    ff_err=np.zeros((len(ts),2))

    ff[:,0]=ts
    ff_err[:,0]=ts

    allstates=np.zeros((len(ts),len(DH)))
     
    args=DH.to_numpy(),DS.to_numpy(),units_len,nrep,j,ts,k,ninst,nsteps,transient,save_each
    ff[:,1],ff_err[:,1],allstates,q_hist=temp_denaturation_first_round(*args)

    np.savetxt(ff_file+'_err_'+str(i_), ff_err)
    np.savetxt(ff_file+'_'+str(i_), ff)
    np.save(states_file+'_'+str(i_),allstates)
    np.save(q_hist_file+'_'+str(i_), q_hist)
    
    del ff,ff_err,allstates,q_hist
    return i_


def main_fold_1seq_second_round_i(i_,DH_file,DS_file,ulf_file,j,k,nsteps,transient,save_each,ninst,ntsteps,
                                  order,extrap,cp_factor,
                                ff_file,states_file,q_hist_file,custom_cp=False,cps_=None,num_cores=None,nrep=0):
    
    DH=np.loadtxt(DH_file)
    DS=np.loadtxt(DS_file)
    ff=np.loadtxt(ff_file+'_'+str(i_)) 
    ff_err=np.loadtxt(ff_file+'_err_'+str(i_))
    allstates=np.load(states_file+'_'+str(i_)+'.npy')
    q_hist=np.load(q_hist_file+'_'+str(i_)+'.npy')

    units_len=np.loadtxt(ulf_file)
    ts=ff[:,0]
    
    if custom_cp:
        cps=cps_
    else:
        ff_errs=np.zeros((len(ff_err),num_cores))

        for fi in range(num_cores):
            ff_err_i=np.loadtxt(ff_file+'_err_'+str(fi))
            ff_errs[:,fi]=ff_err_i[:,1]
        
        ix=sps.argrelmax(ff_errs.mean(axis=1),order=order)[0]
    
        if len(ix>0):
            critical_points=np.concatenate([np.arange((ix_-extrap),(ix_+extrap+1)) for ix_ in ix])
            critical_points=critical_points[critical_points>=0]
            cps=critical_points[critical_points<len(ts)]
       
        else:
            cps=[]
        del ff_errs,ff_err_i
        
    if(len(cps)>0):
        #print(cps)
        DH.shape[0]
        args=cps,ff[:,1],ff_err[:,1],allstates,q_hist,DH,DS,units_len,nrep,ts,j,k,ninst,nsteps,transient,save_each,order,extrap,cp_factor
        ff[:,1],ff_err[:,1],allstates,q_hist=temp_denaturation_second_round(*args)

        np.savetxt(ff_file+'_err_'+str(i_), ff_err)
        np.savetxt(ff_file+'_'+str(i_), ff)
        np.save(states_file+'_'+str(i_),allstates)
        np.save(q_hist_file+'_'+str(i_), q_hist)

    del ff,ff_err,allstates,q_hist
   
    return i_


def border_check(DT,DH,DS,units_len,tini,tfin,k,nsteps,transient,save_each):
    
    n=DH.shape[0]
    state=np.array([random.randint(0, 1) for _ in range(n)])
    folded_res=np.zeros(4)
    run_=True
    while run_:
        for t,temp in enumerate([tini,tini,tfin,tfin]):
            state,folded_res[t],frer,stav,q_hist=montecarlo_speed_complete(DH,
                                                 temp,DS,units_len,k,nsteps,transient,save_each,state)

        if folded_res[0]<1:
            tini+=-DT
        if folded_res[1]>0:
            tfin+=DT
        if (folded_res==[1,1,0,0]).all():    
            run_=False
        if tini==0:
            tini=10
            if (folded_res[2:]==[0,0]).all():
                run_=False
        if tfin>=350:
            if (folded_res[:1]==[1,1]).all():
                run_=False
        
    del state,folded_res,frer,stav,q_hist
    return tini-5,tfin+5 




