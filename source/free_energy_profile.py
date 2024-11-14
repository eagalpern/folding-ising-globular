import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


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
               # print(candidatos)
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
