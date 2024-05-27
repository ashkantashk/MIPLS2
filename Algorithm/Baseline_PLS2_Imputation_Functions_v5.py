# -*- coding: utf-8 -*-
"""
Created on Fri Feb. 15, 2023
@author: Ashkan

Comprised a main function for PLS2-based imputation and several operational functions for a variety of applications

List of the Functions:
---------------------
1) PLS2Based_Imputation(XI, YI, App, Just_do_min, Opt_LV, Max_LV, cv_mode, NSplits, GM_type=None, plt_enb=None, YT=None) 
--> return YI_P1, YI_P2, predP1, predP2, MV, MV_new, MV_idx, LV_cnt, idxy, Intermediate_MV_idx, Lowest_MV_idx, Max_Value

2) nd_rmse(A, B) --> r
    
3) find_comps(rmsecvs, just_do_min=None) --> r
    
4) handle_missing(X, copy=False) --> Q

5) optlv(rmsecv, default_comp = 0, shoulder_change_lim = 5, local_min_change = 1, startat = 0, just_do_min=None) --> different output argumental options  

6) venetian_blinds_indices(data, num_splits= int, random_state=None, shuffle=None) --> train_sets, val_sets

7) rmse(predictions, targets) --> p.sqrt(((predictions - targets) ** 2).mean())

8) Conv_trend_plot(YT, MV, no_MV, n_idx, MV_idx_1, idxy,  MV_new =None, 
                    stp_xticks=None, legend=False, leg_font_size=None, 
                    Log_plt_norm = None, LV_param=None) --> plt.shows 
    
"""
#%%
# Init stuff.

from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
import progressbar 

from ikpls.numpy_ikpls import PLS
#%%

def nd_rmse(A, B):
    # Calculates RMSE for A sample mode 0 and A variable mode 1, for each A componet mode 2,
    # where B is the reference with size of mode B0 == A0 and size of mode B1 == A1.
    # Ignore missing values in A and B        
        
    r = np.empty((A.shape[2], A.shape[1]), dtype=np.float64)
    for ii in range(A.shape[2]):
        for jj in range(A.shape[1]):
            r[ii,jj] = np.sqrt(np.nanmean((A[:,jj,ii] - B[:,jj])**2))
            
    return r            

## Return for the optimal no. LVs based on the first local minimum rule
def find_comps(rmsecvs, just_do_min=None):
    r = np.empty((rmsecvs.shape[1],), dtype=np.int16)
    
    for i in range(rmsecvs.shape[1]):
        r[i] = optlv(rmsecvs[:,i], startat = 0, just_do_min=just_do_min) 
    
    return r    
## Replacing MVs with the mean of non-missing values for each variable (column-wise)
def handle_missing(X, copy=False):
    Q = X.copy() if copy else X
    for i in range(Q.shape[1]):       
        Q[np.isnan(Q)[:,i], i] = Q[ ~np.isnan(Q)[:,i], i].mean()
    return Q

## Search for the optimal no. LVs based on the first local minimum rule
def optlv(rmsecv, default_comp = 0, shoulder_change_lim = 5, local_min_change = 1, startat = 0, just_do_min=None):# Added just_do_min=None to the code

    # Find number of components based on CV rmse
    # 0. Check for global minimum
    if just_do_min:
        return np.where(rmsecv == np.nanmin(rmsecv))[0][0]


    # 1. check for first local error minimum, but excpect min_change 
    for n in range(startat+1, len(rmsecv)):
        if rmsecv[n-1] < rmsecv[n]:
            return n-1

    # 2. If no local minimum, find first shoulder (e.g. < n% rmsecv change between components)
    for n in range(startat+1, len(rmsecv)):
        if np.abs(100-100*rmsecv[n-1]/rmsecv[n]) < shoulder_change_lim:
            return n-1
    # 3. If there is niether no local minimum nor first shoulder
    return len(rmsecv)-1

def venetian_blinds_indices(data, num_splits= int, random_state=None, shuffle=None):
    if random_state != None:
        # Set the random seed for reproducibility
        np.random.seed(random_state)

    # Create a list of indices and shuffle it
    indices = np.arange(data.shape[0])
    if shuffle!=None:
        np.random.shuffle(indices)

    # Create empty lists for training and validation sets
    train_sets = [[] for _ in range(num_splits)]
    val_sets = [[] for _ in range(num_splits)]

    # Iterate over the shuffled indices
    for i in range(len(indices)):
        # Determine which split this sample belongs to
        split = i % num_splits

        # Add to the appropriate training and validation sets
        for j in range(num_splits):
            if j == split:
                val_sets[j].append(indices[i])
            else:
                train_sets[j].append(indices[i])

    for ii in range(num_splits):
        val_sets[ii] = np.array(val_sets[ii])
        train_sets[ii] = np.array(train_sets[ii])

    # Return the training and validation sets
    return train_sets, val_sets

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

## Plotting the convergence trend of MVs existing in the samples as mentioned above
def Conv_trend_plot(MV, no_MV, n_idx, MV_idx_1, idxy, YT, MV_new =None, 
                    stp_xticks=None, legend=False, leg_font_size=None, 
                    Log_plt_norm = None, LV_param=None):
     ## Inputs:
        ### Mandatory input arguments:
        # MV: all the updates for existing MVs in the dataset
        # no_Mv: number of MVs
        # n_idx: the sample index 
        # MV_idx_1: the indices of MVs in original dataset
        # idxy: the indices of the original MVs in tuple format
        # YT: The ground truth values for the missing values
        ### Optional input variables:
        # stp_xticks: the step size for the ticks belonging to x axis
        # legend: descriptions of the real and converging trends of variables
        # leg_font_size: the font size for the legend
        # Log_plt_norm: plotting in a logarithmic format
        # LV_param: comprising three data --> (1) LV_cnt: no. LVs at each step per MV
        #                                     (2) unique_pairs: a dictionary of the unique pairs of LVs per MV
        #                                     (3) Keys: the keys for the unique_pairs dictionary
    MarkerLOT = ['o','d','s','*','v','^', '<','>','8','p','P','h','H','D','.','X'] # Marker lookup table
    import math
    MV_1 = np.array(MV)
    idx_tmp = np.where(MV_idx_1[:,0]==n_idx)[0]
    MVt = MV_1[idx_tmp]
    MVs_idx = idxy[1][np.where(idxy[0]==n_idx)].tolist()
    lent = int(len(MVt)/no_MV)
    if MV_new is None:
        MV_t = np.empty((no_MV,lent))
        xx = range(1,lent+1)
        for ii in range(no_MV):
            MV_t[ii,:] = [MVt[ii+no_MV*jj] for jj in range(lent)]
            import os
            import random
            # Generate a random seed
            random_seed = int.from_bytes(os.urandom(4), byteorder="big")
            # Set the random seed
            random.seed(random_seed)
            col_map = np.random.rand(3, )
            if Log_plt_norm!=None:
                if Log_plt_norm ==1:
                    Marker_1 = MarkerLOT[np.random.randint(1,len(MarkerLOT))]
                    plt.semilogy(xx,MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]], Marker_1, markersize=5, 
                                 color=col_map,linestyle='--',
                               label=f'Pred. Results for MV#{ii+1} in sample #{n_idx}')
                    if LV_param!=None:
                        LV_cnt = LV_param[0]
                        unique_pairs = LV_param[1]
                        Keys = LV_param[2]
                        labels = [LV_cnt[unique_pairs[Keys[ii]][jj]] for jj in range(len(unique_pairs[Keys[ii]]))]
                        labels = [format(x, 'd') for x in labels]
                        MVt = math.log10(MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]])
                        for x,y, label in zip(xx, MVt, labels):
                            plt.text(x, y, label, fontsize=10)
                elif Log_plt_norm==2:
                    Marker_1 = MarkerLOT[np.random.randint(1,len(MarkerLOT))]
                    plt.loglog(xx,MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]],Marker_1, markersize=5,
                               color=col_map,linestyle='--',
                               label=f'Pred. Results for MV#{ii+1} in sample #{n_idx}')
                    if LV_param!=None:
                        LV_cnt = LV_param[0]
                        unique_pairs = LV_param[1]
                        Keys = LV_param[2]
                        labels = [LV_cnt[unique_pairs[Keys[ii]][jj]] for jj in range(len(unique_pairs[Keys[ii]]))]
                        labels = [format(x, 'd') for x in labels]
                        MVt = math.log10(MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]])
                        for x,y, label in zip(math.log10(xx), MVt, labels):
                            plt.text(x, y, label, fontsize=10)
            else:
                Marker_1 = MarkerLOT[np.random.randint(1,len(MarkerLOT))]
                plt.plot(xx,MV_t[ii,:],Marker_1, markersize=5,
                         color=col_map,linestyle='--',
                          label=f'Pred. Results for MV#{ii+1} in sample #{n_idx}')
                plt.plot(xx,np.repeat(YT[n_idx,MVs_idx[ii]],len(xx)),color=col_map,
                          linestyle = ':', label=f'Real Value for MV#{ii+1}')
                if LV_param!=None:
                    LV_cnt = LV_param[0]
                    unique_pairs = LV_param[1]
                    Keys = LV_param[2]
                    labels = [LV_cnt[unique_pairs[Keys[ii]][jj]] for jj in range(len(unique_pairs[Keys[ii]]))]
                    labels = [format(x, 'd') for x in labels]
                    MVt = MV_t[ii,:]
                    for x,y, label in zip(xx, MVt, labels):
                        plt.text(x, y, label, fontsize=10)
    else:
        lent = int(len(MVt)/no_MV) 
        lent1 = int(len(MV_new)/len(idxy[0]))
        MV_t = np.empty((no_MV,lent+lent1), dtype=np.float64)
        xx = range(1,lent1+lent+1)
        if len(idx_tmp)==no_MV:
            for kk in np.arange(lent1-1,-1,-1):
                MV_t[:,-kk-1] = np.squeeze(MV_new[np.argwhere(idxy[0]==n_idx)+len(idxy[0])*(lent1-kk-1)])
        for ii in range(no_MV):
            MV_t[ii,:lent] = [MVt[ii+no_MV*jj] for jj in range(lent)]
            if len(idx_tmp)!=no_MV:
                for kk in np.arange(lent1-1,-1,-1):
                    MV_t[:,-kk-1] = np.squeeze(MV_new[np.argwhere(idxy[0]==n_idx)+len(idxy[0])*(lent1-kk-1)])
            col_map = np.random.rand(3, )
            if not Log_plt_norm is None:
                if Log_plt_norm ==1:
                    Marker_1 = MarkerLOT[np.random.randint(1,len(MarkerLOT))]
                    plt.semilogy(xx,MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]], Marker_1,
                                 markersize=5,
                                 color=col_map,linestyle='--')
                    if LV_param!=None:
                        LV_cnt = LV_param[0]
                        unique_pairs = LV_param[1]
                        Keys = LV_param[2]
                        labels = [LV_cnt[unique_pairs[Keys[ii]][jj]] for jj in range(len(unique_pairs[Keys[ii]]))]
                        labels = [format(x, 'd') for x in labels]
                        MVt = math.log10(MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]])
                        for x,y, label in zip(xx, MVt, labels):
                            plt.text(x, y, label, fontsize=10)
                    if stp_xticks is None:
                        stp_xticks = 5
                    plt.xticks(np.arange(1,lent1+lent+1,stp_xticks))
                elif Log_plt_norm==2:
                    Marker_1 = MarkerLOT[np.random.randint(1,len(MarkerLOT))]
                    plt.loglog(xx,MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]], Marker_1, markersize = 5,
                               color=col_map,linestyle='--')
                    if LV_param!=None:
                        LV_cnt = LV_param[0]
                        unique_pairs = LV_param[1]
                        Keys = LV_param[2]
                        labels = [LV_cnt[unique_pairs[Keys[ii]][jj]] for jj in range(len(unique_pairs[Keys[ii]]))]
                        labels = [format(x, 'd') for x in labels]
                        MVt = math.log10(MV_t[ii,:]*100/YT[n_idx,MVs_idx[ii]])
                        for x,y, label in zip(math.log10(xx), MVt, labels):
                            plt.text(x, y, label, fontsize=10)
                    if stp_xticks == None:
                        stp_xticks = 5
                    plt.xticks(np.arange(1,lent1+lent+1,stp_xticks))
            else:
                Marker_1 = MarkerLOT[np.random.randint(1,len(MarkerLOT))]
                plt.plot(xx,MV_t[ii,:], Marker_1,markersize=5, 
                         color=col_map,linestyle='--',
                         label=f'Pred. Results for MV#{ii+1} in sample #{n_idx}')
                plt.plot(xx,np.repeat(YT[n_idx,MVs_idx[ii]],len(xx)), 
                         color=col_map,#np.random.shuffle(col_map),
                          linestyle = 'dotted', linewidth = 3,
                          label=f'Real Value for MV#{ii+1}')
                if LV_param!=None:
                    LV_cnt = LV_param[0]
                    unique_pairs = LV_param[1]
                    Keys = LV_param[2]
                    labels = [LV_cnt[unique_pairs[Keys[ii]][jj]] for jj in range(len(unique_pairs[Keys[ii]]))]
                    labels = [format(x, 'd') for x in labels]
                    MVt = MV_t[ii,:]
                    for x,y, label in zip(xx, MVt, labels):
                        plt.text(x, y, label, fontsize=10)
    if Log_plt_norm != None:
        if Log_plt_norm ==1:
            plt.semilogy(xx,np.repeat(100,len(xx)),color='blue', linewidth = 2,
                       linestyle = '-', label=f'Real Value for MV#{ii+1}')
        elif Log_plt_norm==2:
            plt.loglog(xx,np.repeat(100,len(xx)),color='blue', linewidth = 2,
                     linestyle = '-', label=f'Real Value for MV#{ii+1}')
    else:
        if stp_xticks == None:
            stp_xticks = 5
        if MV_new is None:
            plt.xticks(np.arange(1,lent+1,stp_xticks))
        else:
            plt.xticks(np.arange(1,lent1+lent+1,stp_xticks))
    plt.xlabel('Iterations',fontsize=12)
    plt.ylabel(f'Updates for MVs in sample#{n_idx}', fontsize=12)
    plt.title(f'Iterative Changes for imputed MVs in sample#{n_idx}', fontsize=14)
    plt.grid(); 
    if legend:
        if leg_font_size == None:
            leg_font_size = 4
        plt.legend(fontsize=leg_font_size)
    plt.show()
###############################################################################
def PLS2Based_Imputation(XI, YI1, App, Just_do_min, Opt_LV, Max_LV, cv_mode, 
                         Nsplits, rnd_stat = None, gm_type=None, YT=None, 
                         Thresh_itr = None, CNT=None, Thresh = None, tmp_val=None, 
                         tmp_val2=None,Strat_shuffle = None, verbose=None):
    # App: 
    #0) 'A0xy' --> A single stratification, solely with samples sorted in ascending order based on the number of Missing Values (MVs).
    #1) 'A1xy' --> Update per sample from the lowest no. MVs to the highest number
    #2) 'A2xy' --> update all var.s in each strat. once at a time
    #3) 'A3xy' --> update the var.s in each substrat. once at a time
    ###########################################################################
    # Select local (None) or global minimum (True)
    # Just_do_min= True # True #None ## x options in Azxy: --> None ==> x=1 & True ==> x=2 
    # Opt_LV = 'allvars' #'pervar' # 'allvars' ## y options for Azxy: --> 'pervar'==> y=1 & 'allvars'==> y=2 
    #%% ###########################################################################
    YI = np.copy(YI1)
    missingmap_yi = np.isnan(YI)
    missing_yi = missingmap_yi.sum(axis=1)
    # Stratify Samples based on the no. MVs
    uq_missing_yi, uq_missing_yi_idx, uq_missing_yi_cnts = np.unique(missing_yi, axis=0, return_counts=True, return_index=True)
    # Initial model on zero missing samples, and progressivly add missing samples    
    c_ix = np.where(missing_yi == 0)[0]
    ## New YT based on the new stratified and substratified samples with synthesized MVs
    if YT is not None:
        YT_new = np.empty((YI.shape), dtype = np.float64)
        YT_new[-len(c_ix):,:] = YT[c_ix,:]
    #%%
    if Just_do_min:
        GM_type = gm_type # 1: most frequent LVs , 2: average LVs, 3: mean of minimums
    else:
        GM_type = 0
    #%% Managing stratified samples:
    if App == 'A0xy':
        # Initial model on zero missing samples, and progressivly add missing samples    
        c_ix = np.where(missing_yi == 0)[0]
        p_ix = [np.where(missing_yi != 0)[0]]
        if Strat_shuffle is not None:
            np.random.seed(Strat_shuffle)
            np.random.shuffle(p_ix[0])
        # The best case is related to the samples with the lowest no. MVs
        Lowest_MV_idx = 1
        Max_Value = len(p_ix)
    if App == 'A1xy':
        # Initial model on zero missing samples, and progressivly add missing samples    
        c_ix = np.where(missing_yi == 0)[0]
        p_ix = []
        for n1 in uq_missing_yi[1:]:
            for n2 in np.where(missing_yi == n1)[0]:
                p_ix.append(n2)
        # The best case is related to the samples with the lowest no. MVs
        Lowest_MV_idx = 1
        Max_Value = len(p_ix)
        iter = 0
    elif App == 'A2xy' or App=='A3xy':
        St_idx = {}
        for n1 in range(1,len(uq_missing_yi_cnts[1:])+1):
            St_idx['S'+f'{n1}'] = np.where(missing_yi == uq_missing_yi[n1])[0]
            if Strat_shuffle is not None:
                np.random.seed(Strat_shuffle)
                np.random.shuffle(St_idx['S'+f'{n1}'])
        KEYS1 = list(St_idx.keys())
        Lowest_MV_idx = St_idx[KEYS1[0]][0]
        # Initial model on zero missing samples, and progressivly add missing samples    
        p_ix = []
        for n1 in uq_missing_yi[1:]:
            for n2 in np.where(missing_yi == n1)[0]:
                p_ix.append(n2)
        if App =='A3xy':
            #% For approach A3xx
            subSt_idxI = {}
            # YI_new = np.copy(YT_new)
            cnt = 0
            ## Substratification:
            for nn in KEYS1:
                sub_YI = YI[St_idx[nn],:]
                # sub_YT = YT[St_idx[nn],:]
                # sub_XT = XT[St_idx[nn],:]
                sub_mmap_yi = np.where(np.isnan(sub_YI))
                _, uq_m_xi_cnt = np.unique(sub_mmap_yi[0], axis=0, return_counts=True)
                subarrays = sub_mmap_yi[1].reshape(-1,uq_m_xi_cnt[0])
                # subarrays = sub_mmap_yi[1].reshape(-1,1)
                uq_m_yi, uq_m_yi_cnts = np.unique(subarrays, axis=0, return_counts=True)
                cnt1=1
                for ii,kk in enumerate(sorted(np.unique(uq_m_yi_cnts),reverse=True)):
                    idt=np.where(uq_m_yi_cnts==kk)[0]
                    for jj in range(len(idt)):
                        idt1 = np.where((subarrays == uq_m_yi[idt[jj]]).all(axis=1))[0]
                        subSt_idxI[nn +'_'+f'{ii+jj+cnt1}'] = St_idx[nn][idt1]
                        # if Strat_shuffle is not None:
                        #     np.random.shuffle(subSt_idxI[nn +'_'+f'{ii+jj+cnt1}'])
                        cnt += len(idt1)
                    cnt1+=(len(idt)-1)
            St_idx = subSt_idxI.copy()
            KEYS1 = list(St_idx.keys())
            # The best case is related to the samples with the lowest no. MVs
            Lowest_MV_idx = St_idx[KEYS1[0]][0]
        Max_Value = len(KEYS1)
    #%%
    # Calculating the index of a sample with the intermediate no. MVs
    # The intermediate case is according to the samples with the median no. MVs
    if len(uq_missing_yi)!=2:
        Intermediate_MV_idx = np.argmin(abs(np.median(uq_missing_yi[1:])-uq_missing_yi[1:]))
    else:
        Intermediate_MV_idx == 1
    #%%
    # Do imputation
    n_comps = Max_LV 
    if YI.shape[0]>=YI.shape[1]:
        P = PLS(algorithm=1)
    else:
        P = PLS(algorithm=2)
    #%%
    # Variables for recording the iterative updates of MVs
    MV = [] 
    MV_idx = []
    LV_cnt = []
    #%% Cross_validation mode
    if rnd_stat is None:
        rnd_stat = 42
    # Nsplits = 30
    Nsplits_old = np.copy(Nsplits).tolist()
    #%%
    len_old = len(c_ix)
    if Thresh_itr is None:
        Thresh_itr = 0#5e-2
    if tmp_val is None:
        tmp_val = Max_LV -1 
    with progressbar.ProgressBar(max_value= Max_Value) as bar:
        if App == 'A0xy':
           Nsplits = np.copy(Nsplits_old).tolist()
           # Preprocess (autoscale) calibration data
           MIXC = XI[c_ix,:].mean(axis=0)
           SIXC = XI[c_ix,:].std(axis=0)
           XIP = (XI[c_ix, :] - MIXC) / SIXC
           MIYC = YI[c_ix].mean(axis=0)
           SIYC = YI[c_ix].std(axis=0)
           YIP = (YI[c_ix, :] - MIYC) / SIYC
           pi_ix = p_ix[0]
           PXIP = (XI[pi_ix, :] - MIXC) / SIXC
           if np.ndim(PXIP) == 1: # convert a 1D array into a 2D one with, e.g. converting from MX0 or 0xM into 1xM
               PXIP = PXIP.reshape(1,-1)
           No_samples = len(XIP)
           if Nsplits>No_samples:
              Nsplits = No_samples
           if cv_mode == 'KFold':
               # Do CV fit
               pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
               cv = model_selection.KFold(n_splits=Nsplits, random_state=rnd_stat, shuffle=True)
               for cv_ix, v_ix in cv.split(X=XIP):
                   P.fit(XIP[cv_ix,:], YIP[cv_ix, :], n_comps)
                   pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
           elif cv_mode=='Venetian':
               # Do CV fit
               pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
               cv_ix_all, vv_ix_all = venetian_blinds_indices(XIP, num_splits=Nsplits, random_state=rnd_stat, shuffle=True)
               for cv_ix, v_ix in zip(cv_ix_all,vv_ix_all):
                   P.fit(XIP[cv_ix,:], YIP[cv_ix, :], n_comps)
                   pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
           pred_cv = np.moveaxis(pred_cv, 0, 2)
           rmse_cv = nd_rmse(pred_cv, YIP)
           # Do fit for imputation    
           P.fit(XIP, YIP, n_comps)
           pred_ix = np.concatenate((c_ix, pi_ix))
           pred_cal = P.predict(np.concatenate((XIP, PXIP)))
           pred_cal = np.moveaxis(pred_cal, 0, 2)
           # Pick optimal components
           opt_comps = find_comps(rmse_cv, just_do_min=Just_do_min)
           opt_comps[opt_comps> tmp_val] = tmp_val
           if tmp_val2 is not None:
               if tmp_val2>=0 and tmp_val2<=Max_LV-1:
                   opt_comps[opt_comps> tmp_val2] = tmp_val2
               opt_comps_old = np.copy(opt_comps)
           if GM_type == 1:
               a1,b1 = np.unique(opt_comps, return_counts=True)
               LV_glob = a1[np.argmax(b1)]
           elif GM_type ==3:
               # LV_glob = np.argmin(rmse_cv.mean(axis=1))   
               LV_glob = np.argmin(rmse_cv.sum(axis=1))                        
           else:
               LV_glob = int(np.round(np.mean(opt_comps)))
           if tmp_val2 is not None:
               if tmp_val2>=0 and tmp_val2<=Max_LV-1 and LV_glob > tmp_val2:
                   LV_glob = tmp_val2
               LV_glob_old = np.copy(LV_glob)
           if LV_glob > tmp_val:
               LV_glob = tmp_val
           pred = np.empty(pred_cal.shape[0:2], dtype=np.float64)
           for n in range(pred_cal.shape[0]):
               for m in range(pred_cal.shape[1]):
                   if Opt_LV == 'pervar':
                       pred[n,m] = pred_cal[n,m,opt_comps[m]]
                       if missingmap_yi[pred_ix[n], m]:
                           LV_cnt.append(opt_comps[m]+1)
                   elif Opt_LV == 'allvars':
                       pred[n,m] = pred_cal[n,m,LV_glob]
                       if missingmap_yi[pred_ix[n], m]:
                           LV_cnt.append(LV_glob+1)
           # Move imputed sample to calibration set
           c_ix = np.append(c_ix, values=pi_ix)
           p_ix = p_ix[len(pi_ix):]
           if len(p_ix) ==0:
               predP1 = np.empty(pred.shape, dtype = np.float64)
               predP1[np.where(missingmap_yi)]=pred[np.where(missingmap_yi)].copy()
           # Reverse preprocessing
           pred *= SIYC
           pred += MIYC
           # Absolute of the cases in which negative values appear in the prediction
           pred = np.abs(pred)
           if len_old == len(c_ix)-len(pi_ix):
               pred_old = pred.copy()
           else:
               pred_old = np.concatenate((pred_old,pred[pi_ix,:]))
           # Replace original missing with predicted 
           for n in range(pred.shape[0]):
               for m in range(pred.shape[1]):
                   if missingmap_yi[pred_ix[n], m]:
                       if abs(pred[n,m]-pred_old[n,m])/np.max([pred[n,m],pred_old[n,m]])<=Thresh_itr:
                           YI[pred_ix[n], m] = pred[n,m]
                           MV.append(pred[n,m])
                       else:
                           YI[pred_ix[n], m] = pred_old[n,m]
                           MV.append(pred_old[n,m])
                       MV_idx.append([pred_ix[n], m])
           if len_old != len(c_ix)-len(pi_ix):
               pred_old = pred.copy()
           if verbose is not None:
               bar.update(1)
        elif App == 'A2xy' or App == 'A3xy':
            for ii in range(len(KEYS1)):
                Nsplits = np.copy(Nsplits_old)
                Nsplits = Nsplits.tolist()
                # Preprocess (autoscale) calibration data
                MIXC = XI[c_ix,:].mean(axis=0)
                SIXC = XI[c_ix,:].std(axis=0)
                XIP = (XI[c_ix, :] - MIXC) / SIXC
                MIYC = YI[c_ix].mean(axis=0)
                SIYC = YI[c_ix].std(axis=0)
                YIP = (YI[c_ix, :] - MIYC) / SIYC
                # Preprocess (autoscale) imputation row(s)
                # This is one-strat-at-a-time:
                p_ix = St_idx[KEYS1[ii]]
                pi_ix = np.append(c_ix, values = p_ix)
                PXIP = (XI[p_ix, :] - MIXC) / SIXC
                if np.ndim(PXIP) == 1:
                    PXIP = PXIP.reshape(1,-1)
                No_samples = len(XIP)
                if Nsplits>No_samples:
                   Nsplits = No_samples
                if cv_mode == 'KFold':
                    # Do CV fit
                    pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
                    cv = model_selection.KFold(n_splits=Nsplits, random_state=rnd_stat, shuffle=True)
                    for cv_ix, v_ix in cv.split(X=XIP):
                        P.fit(XIP[cv_ix,:], YIP[cv_ix, :], n_comps)
                        pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
                elif cv_mode=='Venetian':
                    # Do CV fit
                    pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
                    cv_ix_all, vv_ix_all = venetian_blinds_indices(XIP, num_splits=Nsplits, random_state=rnd_stat, shuffle=True)
                    for cv_ix, v_ix in zip(cv_ix_all,vv_ix_all):
                        P.fit(XIP[cv_ix,:], YIP[cv_ix, :], n_comps)
                        pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
                pred_cv = np.moveaxis(pred_cv, 0, 2)
                rmse_cv = nd_rmse(pred_cv, YIP)
                # Do fit for imputation    
                P.fit(XIP, YIP, n_comps)
                pred_ix = pi_ix
                # pred_cal = P.predict(np.concatenate((XIP, PXIP)))
                pred_cal = P.predict(np.concatenate((XIP, PXIP)))
                pred_cal = np.moveaxis(pred_cal, 0, 2)
                # Pick optimal components
                opt_comps = find_comps(rmse_cv, just_do_min=Just_do_min)
                opt_comps[opt_comps> ii + tmp_val] = ii + tmp_val
                if tmp_val2 is not None:
                    if ii==0:
                        if tmp_val2>=0 and tmp_val2<=Max_LV-1:
                            opt_comps[opt_comps>tmp_val2] = tmp_val2
                        opt_comps_old = np.copy(opt_comps)
                    else:
                        tmp2 = opt_comps - opt_comps_old
                        ind_pos = np.where((tmp2 != 1) & (tmp2>0))
                        ind_neg = np.where((tmp2!=-1) & (tmp2<0))
                        opt_comps[ind_pos] = opt_comps_old[ind_pos]+1 
                        opt_comps[ind_neg] = opt_comps_old[ind_neg]-1
                        # opt_comps = np.copy(opt_comps_old)+1
                        opt_comps[opt_comps>=Max_LV] = Max_LV-1
                        opt_comps[opt_comps<0] = 0
                        opt_comps_old = np.copy(opt_comps)
                if GM_type == 1:
                    a1,b1 = np.unique(opt_comps, return_counts=True)
                    LV_glob = a1[np.argmax(b1)]
                elif GM_type ==3:
                    # LV_glob = np.argmin(rmse_cv.mean(axis=1))                        
                    LV_glob = np.argmin(rmse_cv.sum(axis=1))                        
                else:
                    LV_glob = int(np.round(np.mean(opt_comps)))
                if tmp_val2 is not None:
                    if ii==0:
                        if tmp_val2>=0 and tmp_val2<=Max_LV-1:
                            opt_comps[opt_comps>tmp_val2] = tmp_val2
                        LV_glob_old = np.copy(LV_glob)
                    else:
                        tmp2 = LV_glob - LV_glob_old
                        if tmp2 > 1:
                            LV_glob = LV_glob_old + 1
                        elif tmp2<-1:
                            LV_glob = LV_glob_old - 1     
                        if LV_glob>=Max_LV:
                            LV_glob = Max_LV -1
                        elif LV_glob<0:
                            LV_glob = 0
                if LV_glob > ii + tmp_val:
                    LV_glob = ii + tmp_val
                pred = np.empty(pred_cal.shape[0:2], dtype=np.float64)
                for n in range(pred_cal.shape[0]):
                    for m in range(pred_cal.shape[1]):
                        if Opt_LV == 'pervar':
                            pred[n,m] = pred_cal[n,m,opt_comps[m]]
                            if missingmap_yi[pred_ix[n], m]:
                                LV_cnt.append(opt_comps[m]+1)
                        elif Opt_LV == 'allvars':
                            pred[n,m] = pred_cal[n,m,LV_glob]
                            if missingmap_yi[pred_ix[n], m]:
                                LV_cnt.append(LV_glob+1)
                        
                # Move imputed sample to calibration set
                # c_ix = np.append(c_ix, values=pi_ix)
                c_ix = pi_ix
                if ii == len(KEYS1)-1:
                    # predP = pred.copy()
                    predP1 = np.empty(pred.shape, dtype = np.float64)
                    predP1[np.where(missingmap_yi)]=pred[np.where(missingmap_yi)].copy()
                # Reverse preprocessing
                pred *= SIYC
                pred += MIYC
                # pred *= SIYP
                # pred += MIYP
                pred = np.abs(pred)
                if len_old == len(c_ix)-len(p_ix):
                    pred_old = pred.copy()
                else:
                    pred_old = np.concatenate((pred_old,pred[-len(p_ix):,:]))
                # Replace original missing with predicted 
                for n in range(pred.shape[0]):
                    for m in range(pred.shape[1]):
                        if missingmap_yi[pred_ix[n], m]:
                            if abs(pred[n,m]-pred_old[n,m])/np.max([pred[n,m],pred_old[n,m]])<=Thresh_itr:
                                YI[pred_ix[n], m] = pred[n,m]
                                MV.append(pred[n,m])
                            else:
                                YI[pred_ix[n], m] = pred_old[n,m]
                                MV.append(pred_old[n,m])
                            MV_idx.append([pred_ix[n], m])
                if len_old != len(c_ix)-len(pi_ix):
                    pred_old = pred.copy()
                if verbose is not None:
                    bar.update(ii)
        elif App =='A1xy':
            while len(p_ix) > 0:
                Nsplits = np.copy(Nsplits_old)
                Nsplits = Nsplits.tolist()
                # Preprocess (autoscale) calibration data
                MIXC = XI[c_ix,:].mean(axis=0)
                SIXC = XI[c_ix,:].std(axis=0)
                XIP = (XI[c_ix, :] - MIXC) / SIXC
                MIYC = YI[c_ix].mean(axis=0)
                SIYC = YI[c_ix].std(axis=0)
                YIP = (YI[c_ix, :] - MIYC) / SIYC
                # Preprocess (autoscale) imputation row(s)
                # This is one-row-at-a-time :
                pi_ix = [p_ix[0]]
                PXIP = (XI[pi_ix, :] - MIXC) / SIXC
                if np.ndim(PXIP) == 1: # convert a 1D array into a 2D one with, e.g. converting from MX0 or 0xM into 1xM
                    PXIP = PXIP.reshape(1,-1)
                No_samples = len(XIP)
                if Nsplits>No_samples:
                   Nsplits = No_samples
                if cv_mode == 'KFold':
                    # Do CV fit
                    pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
                    cv = model_selection.KFold(n_splits=Nsplits, random_state=rnd_stat, shuffle=True)
                    for cv_ix, v_ix in cv.split(X=XIP):
                        P.fit(XIP[cv_ix,:], YIP[cv_ix, :], n_comps)
                        pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
                elif cv_mode=='Venetian':
                    # Do CV fit
                    pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
                    cv_ix_all, vv_ix_all = venetian_blinds_indices(XIP, num_splits=Nsplits, random_state=rnd_stat, shuffle=True)
                    for cv_ix, v_ix in zip(cv_ix_all,vv_ix_all):
                        P.fit(XIP[cv_ix,:], YIP[cv_ix, :], n_comps)
                        pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
                pred_cv = np.moveaxis(pred_cv, 0, 2)
                rmse_cv = nd_rmse(pred_cv, YIP)
                # Do fit for imputation    
                P.fit(XIP, YIP, n_comps)
                pred_ix = np.concatenate((c_ix, pi_ix))
                pred_cal = P.predict(np.concatenate((XIP, PXIP)))
                pred_cal = np.moveaxis(pred_cal, 0, 2)
                # Pick optimal components
                opt_comps = find_comps(rmse_cv, just_do_min=Just_do_min)
                opt_comps[opt_comps> iter + tmp_val] = iter + tmp_val
                if tmp_val2 is not None:
                    if iter==0:
                        if tmp_val2>=0 and tmp_val2<=Max_LV-1:
                            opt_comps[opt_comps>tmp_val2] = tmp_val2
                        opt_comps_old = np.copy(opt_comps)
                    else:
                        tmp2 = opt_comps - opt_comps_old
                        ind_pos = np.where((tmp2 != 1) & (tmp2>0))
                        ind_neg = np.where((tmp2!=-1) & (tmp2<0))
                        opt_comps[ind_pos] = opt_comps_old[ind_pos]+1 
                        opt_comps[ind_neg] = opt_comps_old[ind_neg]-1
                        opt_comps[opt_comps>=Max_LV] = Max_LV-1
                        opt_comps[opt_comps<0] = 0
                        opt_comps_old = np.copy(opt_comps)
                if GM_type == 1:
                    a1,b1 = np.unique(opt_comps, return_counts=True)
                    LV_glob = a1[np.argmax(b1)]
                elif GM_type ==3:
                    # LV_glob = np.argmin(rmse_cv.mean(axis=1))                        
                    LV_glob = np.argmin(rmse_cv.sum(axis=1))
                else:
                    LV_glob = int(np.round(np.mean(opt_comps)))
                if tmp_val2 is not None:
                    if iter==0:
                        if tmp_val2>=0 and tmp_val2<=Max_LV-1:
                            opt_comps[opt_comps> tmp_val2] = tmp_val2
                        LV_glob_old = np.copy(LV_glob)
                    else:
                        tmp2 = LV_glob - LV_glob_old
                        if tmp2 > 1:
                            LV_glob = LV_glob_old + 1
                        elif tmp2<-1:
                            LV_glob = LV_glob_old - 1
                            
                        if LV_glob>=Max_LV:
                            LV_glob = Max_LV -1
                        elif LV_glob<0:
                            LV_glob = 0
                if LV_glob > iter + tmp_val:
                    LV_glob = iter + tmp_val
                pred = np.empty(pred_cal.shape[0:2], dtype=np.float64)
                for n in range(pred_cal.shape[0]):
                    for m in range(pred_cal.shape[1]):
                        if Opt_LV == 'pervar':
                            pred[n,m] = pred_cal[n,m,opt_comps[m]]
                            if missingmap_yi[pred_ix[n], m]:
                                LV_cnt.append(opt_comps[m]+1)
                        elif Opt_LV == 'allvars':
                            pred[n,m] = pred_cal[n,m,LV_glob]
                            if missingmap_yi[pred_ix[n], m]:
                                LV_cnt.append(LV_glob+1)
                # Move imputed sample to calibration set
                c_ix = np.append(c_ix, values=pi_ix)
                p_ix = p_ix[len(pi_ix):]
                if len(p_ix) ==0:
                    predP1 = np.empty(pred.shape, dtype = np.float64)
                    predP1[np.where(missingmap_yi)]=pred[np.where(missingmap_yi)].copy()
                # Reverse preprocessing
                pred *= SIYC
                pred += MIYC
                pred = np.abs(pred)
                if len_old == len(c_ix)-len(pi_ix):
                    pred_old = pred.copy()
                else:
                    pred_old = np.concatenate((pred_old,pred[-1,:].reshape(1,-1)))
                # Replace original missing with predicted 
                for n in range(pred.shape[0]):
                    for m in range(pred.shape[1]):
                        if missingmap_yi[pred_ix[n], m]:
                            if abs(pred[n,m]-pred_old[n,m])/np.max([pred[n,m],pred_old[n,m]])<=Thresh_itr:
                                YI[pred_ix[n], m] = pred[n,m]
                                MV.append(pred[n,m])
                            else:
                                YI[pred_ix[n], m] = pred_old[n,m]
                                MV.append(pred_old[n,m])
                            MV_idx.append([pred_ix[n], m])
                if len_old != len(c_ix)-len(pi_ix):
                    pred_old = pred.copy()
                iter+=1
                if verbose is not None:
                    bar.update(iter)
    #%%
    YI_P1 = np.copy(YI)
    unique_pairs = {}
    for index, inner_list in enumerate(MV_idx):
        pair = tuple(inner_list)
        if pair not in unique_pairs:
            unique_pairs[pair] = [index]
        else:
            unique_pairs[pair].append(index)
    Keys = list(unique_pairs.keys())
    ## Plotting the updates of the selected MVs to observe their convergence trend 
    #%%
    idxy = np.empty((2,len(Keys)), dtype=np.int64)
    idxy[0] = [Keys[ii][0] for ii in range(len(Keys))]
    idxy[1] = [Keys[ii][1] for ii in range(len(Keys))]
    idxy = tuple(idxy)    
    #%% Updating MVs until reaching a convergence
    YI_P2 = np.copy(YI_P1)
    LV_cnt_new = []
    pred_old = np.copy(pred)
    if YT is None:
        RMSE_old = 1
        # RMSE_old1 = 1
    else:
        RMSE_old = root_mean_squared_error(YI_P1[idxy], YT[idxy])
        # RMSE_old1 = root_mean_squared_error(YI_P1, YT)
    RMSE_new = 2*RMSE_old
    # RMSE_new1 = 2*RMSE_old1
    if CNT is None:
        CNT = 401
    CNT0 = np.copy(CNT)
    if Thresh is None:
        Thresh = 5e-6
    MV_new = np.array(MV[-len(Keys):])
    while np.abs(RMSE_new-RMSE_old)>Thresh and CNT!=1: #and np.abs(RMSE_new1-RMSE_old1)>Thresh:
        Nsplits = np.copy(Nsplits_old)
        Nsplits = Nsplits.tolist()
        # Preprocess (autoscale) calibration data
        RMSE_old = RMSE_new
        MIXC = XI.mean(axis=0)
        SIXC = XI.std(axis=0)
        XIP = (XI - MIXC) / SIXC

        MIYC = YI_P2.mean(axis=0)
        SIYC = YI_P2.std(axis=0)
        YIP = (YI_P2 - MIYC) / SIYC
        
        No_samples = len(XIP)
        if Nsplits>No_samples:
           Nsplits = No_samples
        if cv_mode == 'KFold':
            # Do CV fit
            pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
            cv = model_selection.KFold(n_splits=Nsplits, random_state=rnd_stat, shuffle=True)
            for c_ix, v_ix in cv.split(X=XIP):
                P.fit(XIP[c_ix,:], YIP[c_ix, :], n_comps)
                pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
        elif cv_mode=='Venetian':
            # Do CV fit
            pred_cv = np.empty((n_comps, YIP.shape[0], YIP.shape[1]), dtype=np.float64)
            cv_ix_all, vv_ix_all = venetian_blinds_indices(XIP, num_splits=Nsplits, random_state=rnd_stat, shuffle=True)
            for c_ix, v_ix in zip(cv_ix_all,vv_ix_all):
                P.fit(XIP[c_ix,:], YIP[c_ix, :], n_comps)
                pred_cv[:, v_ix, :] = P.predict(XIP[v_ix,:])
        pred_cv = np.moveaxis(pred_cv, 0, 2)
        rmse_cv = nd_rmse(pred_cv, YIP)
        # Do fit for imputation    
        P.fit(XIP, YIP, n_comps)
        pred_cal = P.predict(XIP)
        pred_cal = np.moveaxis(pred_cal, 0, 2)
        # Pick optimal components
        opt_comps = find_comps(rmse_cv, just_do_min=Just_do_min)
        if tmp_val2 is not None:
            opt_comps = np.copy(opt_comps_old)+1
            opt_comps[opt_comps>=Max_LV] = Max_LV -1
            opt_comps_old = np.copy(opt_comps)
        if GM_type == 1:
            a1,b1 = np.unique(opt_comps, return_counts=True)
            LV_glob = a1[np.argmax(b1)]
        elif GM_type ==3:
            LV_glob = np.argmin(rmse_cv.mean(axis=1))
        else:
            LV_glob = int(np.round(np.mean(opt_comps)))
        if tmp_val2 is not None:
           LV_glob = LV_glob_old + 1
           if LV_glob>=Max_LV:
              LV_glob = Max_LV-1
        pred = np.empty(pred_cal.shape[0:2], dtype=np.float64)
        for n in range(pred_cal.shape[0]):
            for m in range(pred_cal.shape[1]):
                if Opt_LV == 'pervar':
                    pred[n,m] = pred_cal[n,m,opt_comps[m]]
                    if missingmap_yi[pred_ix[n], m]:
                        LV_cnt_new.append(opt_comps[m]+1)
                elif Opt_LV == 'allvars':
                    pred[n,m] = pred_cal[n,m,LV_glob]
                    if missingmap_yi[pred_ix[n], m]:
                        LV_cnt_new.append(LV_glob+1)
        # predP = pred.copy()
        if YT is None:
            predP2 = np.copy(YIP)
        else:
            YTP = (YT - YT.mean(axis=0)) / YT.std(axis=0)
            predP2 = np.copy(YTP)
        predP2[idxy]=pred[idxy]
        # Reverse preprocessing
        pred *= SIYC
        pred += MIYC
        pred = np.abs(pred)
        # Replace original missing with predicted 
        YI_Ptmp = np.copy(YI_P2)
        YI_P2 = np.copy(YI1)
        YI_P2[idxy] = pred[idxy]
        MV_new = np.concatenate((MV_new, YI_P2[idxy]),axis=0)
        CNT-=1
        if YT is None:
            RMSE_new = root_mean_squared_error(YI_P2[idxy], YI_Ptmp[idxy])
            #RMSE_new1 = root_mean_squared_error(YI_P2, YI_Ptmp)
        else:
            RMSE_new = root_mean_squared_error(YI_P2[idxy], YT[idxy])
            #RMSE_new1 = root_mean_squared_error(YI_P2, YT)
        if verbose is not None:
            print(f'Differential RMSE for iter. #{CNT0-CNT} equals to= {np.abs(np.round(RMSE_new-RMSE_old,6))}')
    #%%
    return YI_P1, YI_P2, predP1, predP2, MV, MV_new, MV_idx, LV_cnt, idxy, Intermediate_MV_idx, Lowest_MV_idx, Max_Value
