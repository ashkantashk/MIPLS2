# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:12:19 2023

@author: Ashkan
"""
## Loading essential libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from Algorithms.cpu_bootstrap_new import bootstrap_index_generator_new
Y_preprocess_mode = 'autoscale'  #'mean_centering' #  'none'
if Y_preprocess_mode == 'mean_centering':
    from Algorithms.numpy_improved_kernel_revised import PLS
    from multi_output_rmse_with_MC import rmse_output_variables
elif Y_preprocess_mode == 'autoscale':
    from Algorithms.numpy_improved_kernel_revised_Y_autoscaled import PLS
    from multi_output_rmse_with_AS import rmse_output_variables
else:
    from Algorithms.numpy_improved_kernel import PLS
    from multi_output_rmse import rmse_output_variables
from sklearn.model_selection import train_test_split
from scipy.io import loadmat  # , savemat
from sklearn.metrics import mean_squared_error
import pickle
from essential_Funcs_for_Presenting_PLS_Results import Intermediate_Plot_PLS1_2,\
    Par_idx_finder, Plot_Measured_Predicted_NMR_MV, Calc_Opt_LVs_PLS2_PLS1#,\
    #Plot_Measured_Predicted_NMR_PLS, Plot_Measured_Predicted_NMR_PLS, #\ 
    #q2_calc, Plot_Measured_vs_Predicted, Plot_Measured_Predicted_NMR
###############################################################################
#%###################### List of Variables Descriptions #####################%#
folder_path = r'Baseline_PLS2_Model\RealWildStuff'
# nu_list : a counter for the number of stratification
# 
#%%# Loading Full original LP NMR data
Path = r'Baseline_PLS2_Model\Original_Data'
X = loadmat(Path + r'\NMR_r9.mat')
X = X['NMR_r9']
X1_org = np.copy(X)
## Loading uc LP data
Y = loadmat(Path + r'\UC65_values.mat')
Y = Y['uc_65vars']
Y1_org = np.copy(Y)
#%%#
# Considering the ratio of the missing to non-missing values for each variable separately
import random
rand_seed = 42
missing_value_distribution = []
for ii in range(Y1_org.shape[1]):
    missing_value_distribution.append(np.sum(Y1_org[:, ii] == 0))
original_rows = Y1_org.shape[0]
original_rows_with_MVs = len([rows for rows in range(Y1_org.shape[0]) if np.any(Y1_org[rows, :] == 0)])
original_cols = Y.shape[1]
# New array dimensions
NSAMVs = 20  # No. Samples containing Artificial Missing Values
new_rows = NSAMVs
new_cols = original_cols
columns_without_missing = [col for col in range(original_cols) if np.all(Y1_org[:, col] != 0)]
# Calculate the ratio of missing values along the Y-axis for columns with missing values
y_ratio = original_rows / (original_rows + len(columns_without_missing))

# Calculate the total number of missing values in the original data
total_missing_values = sum(missing_value_distribution)  # or np.sum(Y==0)

# Calculate the number of missing values to distribute in the new array
new_missing_values = int(total_missing_values * (new_rows * new_cols) / (original_rows_with_MVs * original_cols))

# Generate a new 2D array with the same column structure as the original array
new_array = np.full((new_rows, new_cols), None)
random.seed(rand_seed)
# Distribute missing values based on the original distribution
for col, original_missing_count in enumerate(missing_value_distribution):
    new_missing_count = int(np.round(new_missing_values * (original_missing_count / total_missing_values)))
    # Get the rows with missing values in the original column
    rows_with_missing = [row for row in range(original_rows) if Y1_org[row, col] == 0]
    for _ in range(new_missing_count):
        if rows_with_missing:
            # Select a random row with a missing value in the original column
            original_row = random.choice(rows_with_missing)
            # Find a row that doesn't have a missing value in the selected column
            new_row = random.choice([r for r in range(new_rows) if new_array[r, col] is None])
            # Copy the missing value from the original row to the new row
            new_array[new_row, col] = "missing"
            rows_with_missing.remove(original_row)
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx_z_new = []
for ii in range(Y1_org.shape[0]):
    if any(Y1_org[ii, :] == 0):
        idx_z_new.append(ii)
# Y_z_new = np.zeros((len(idx_z_new),Y1_org.shape[1]))   
Y_z_new = Y1_org[idx_z_new, :]
X_z_new = X1_org[idx_z_new, :]
# new_array = np.vstack((new_array,Y_z_new))
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx_z_y = []
idx_z_x = []
for ii in range(Y.shape[1]):
    if np.sum(new_array[:, ii] == 'missing') != 0:
        tmp = np.where(np.isin(new_array[:, ii], 'missing'))[0].tolist()
        idx_z_x = np.concatenate((idx_z_x, tmp))
        idx_z_y = np.concatenate((idx_z_y, np.repeat(ii, len(tmp))))
idx_z_x = idx_z_x.astype(int)
idx_z_y = idx_z_y.astype(int)
# %% ###########################################################################
## Loading LP NMR data and Loading uc LP data
with open(folder_path + r'\uc308_imput_with_update_AvgWeights_HalfCI_Best_TrVal_04-07-2023_21-11.pkl', 'rb') as f:
    SS1 = pickle.load(f)  # SS1 = [list1,list2,rmsep,Best_LV,Best_set]
Y = np.concatenate((SS1['Y_good'], SS1['Y_eval']))
Y_org1 = np.copy(Y)
X = np.concatenate((SS1['X_good'], SS1['X_eval']))
X_org1 = np.copy(X)
# %%
from random import seed, shuffle  # , choices, sample
rand_seed = 42#50  # 40
seed(rand_seed)
arr = np.arange(len(Y))
shuffle(arr)
Len_trval = 25  # the No. tr. and val. samples with no M.V.s
Len_eval = 13  # the No. of eval samples containing 8.57% M.V.s
Y_eval = Y[arr[-Len_eval:], :]
X_eval = X[arr[-Len_eval:], :]
Y = Y[arr[0:-Len_eval], :]
Y11_org = np.copy(Y)
X = X[arr[0:-Len_eval], :]
# Selecting randomly  out of 8.7% of all good samples for being kept unchanged
# tmp = choices(range(len(Y)),k=int(0.17*len(Y)))
# Resorting samples based on the unchangable samples
YY1 = Y[0:-Len_trval, :]
# Define the number of elements to select
num_elements = int(Y.size * 0.0857)

# Get the total number of elements in the matrix
total_elements = YY1.size

# Insert random indices to the NSAMVs No. of samples including no missing values
# by replace the selected elements with -10
for ii1 in range(len(idx_z_x)):
    YY1[idx_z_x[ii1], idx_z_y[ii1]] = -10
# Reconstruct the original data
for ii in range(len(YY1)):
    Y[ii, :] = YY1[ii, :]
Y_vir_MVs = np.copy(Y)  # Y containing artificially embedded MVs
# %%
idx_nz = []  # np.arange(len(Y)-Len_trval,len(Y))
idx_z = []  # np.arange(len(Y)-Len_trval(Y))
Rep_cnt = []
for ii in range(Y.shape[0]):
    if all(Y[ii, :] > 0):
        idx_nz.append(ii)
    else:
        idx_z.append(ii)
        Rep_cnt.append(np.sum(Y[ii, :] <= 0))
Rep_cnt = np.array(Rep_cnt)
## Stratification of the input and output variables based on the No. Zeros
idx_z_n = np.array(idx_z)
pnt = 0  # The number of stratifications
for ii in range(Y.shape[1]):
    if np.sum(Rep_cnt == ii) != 0:
        pnt += 1
Y1 = np.copy(Y)  # Samples with real and artificial MVs
# Y_org = np.vstack((Y_z_new, Y11_org))
Y_org = np.copy(Y11_org)
#%% ###########################################################################
#### labels for uc variables ####
Path = r'C:\Users\tsz874\Downloads\FAB_Projects\PL_W0610 shared wish Ashkan'
uc_labels = loadmat(Path + r'\UC65_variables.mat')
uc_labels = uc_labels['matches']
uc_proc_labels = []
tmp = uc_labels.tolist()
for ii in range(0, len(uc_labels)):
    uc_proc_labels.append(tmp[ii][0].tolist()[0])
uc_proc_labels = np.array(uc_proc_labels)
###############################################################################
par_1 = ['VLDLTG', 'IDLTG', 'LDLPhoslip', 'HDL-2bPhoslip', 'LDLChol', 'HDLChol', \
         'HDL-2bApoA', 'HDL-3ApoA']
par_2 = ['VLDL${tg}$', 'IDL${tg}$', 'LDL${Phoslp}$', 'HDL2b${Phoslp}$', \
         'LDL${chol}$', 'HDL${chol}$', 'HDL2b${apoA}$', 'HDL3${apoA}$']
###############################################################################
# %% Bootstraping applied to the list of all good samples
XX = X[idx_nz, :]
YY = Y[idx_nz, :]
# Defining Hyperparameters:
Max_LV = 24 # X.shape[1]
X_good = XX
Y_good = YY
# Generating cross-validation indices
indices = np.arange(YY.shape[0])
TestSize = 0.4
Rnd_stat = 42
Len_tr = int(np.round((1 - TestSize) * Len_trval))
Len_val = Len_trval - Len_tr
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_good, Y_good, test_size=TestSize, random_state=Rnd_stat)
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))
# %% Replacing the zeros with mean of the non-zeros values for each variable
Mean_Vals = np.mean(Y_good, axis=0)
# %%
Rank_pnt = np.zeros((pnt, 2))
cnt = 0
cnt1 = 1
cnt2 = 0
Y_MV = []  # missing values replaced with zero values
Y_MVM = []  # missing values imputed with the mean values
Y_NMV = []  # MVs replaced with the original and real values
X_MV = []
MV_loc = []
while cnt < Y.shape[0] - len(idx_nz):
    tmp = np.argwhere(Rep_cnt == cnt1)
    if len(tmp) != 0:
        cnt += len(tmp)
        if cnt1 == np.min(Rep_cnt):
            Rank_pnt[cnt2, 1] = len(tmp) - 1
            Y_MV.append(np.squeeze(Y1[idx_z_n[tmp], :]))
            Y_MV = np.array(Y_MV[0])
            Y_MVM.append(np.squeeze(Y[idx_z_n[tmp], :]))
            Y_MVM = np.array(Y_MVM[0])
            Y_NMV.append(np.squeeze(Y_org[idx_z_n[tmp], :]))
            Y_NMV = np.array(Y_NMV[0])
            X_MV.append(np.squeeze(X[idx_z_n[tmp], :]))
            X_MV = np.array(X_MV[0])
        else:
            Y_MV = np.vstack((Y_MV, np.squeeze(Y1[idx_z_n[tmp], :])))
            Y_MVM = np.vstack((Y_MVM, np.squeeze(Y[idx_z_n[tmp], :])))
            Y_NMV = np.vstack((Y_NMV, np.squeeze(Y_org[idx_z_n[tmp], :])))
            X_MV = np.vstack((X_MV, np.squeeze(X[idx_z_n[tmp], :])))
            Rank_pnt[cnt2][0] = Rank_pnt[cnt2 - 1][1] + 1
            Rank_pnt[cnt2][1] = len(tmp) + Rank_pnt[cnt2 - 1][1]
        cnt2 += 1
    cnt1 += 1
Rank_pnt = Rank_pnt.astype(int)
# Extracting the coordinates of the artificially embedded MVs inside Y_vir_MVs
idxy_init = tuple(list(np.where(np.isin(Y_MV, -10))))
Y_MV[Y_MV == -10] = 0
#%% Evaluation ################################################################
ind_11 = 1
par1 = par_1[ind_11]  # 'LDLPhoslip'#'IDLTGKorr'
indices = Par_idx_finder(par1, uc_proc_labels)
indices = indices[0]
#%%
# Main Code with random stratification
rand_seed = 42
seed(rand_seed)
nu_list = np.arange(pnt).tolist()
# shuffle(nu_list)
MV = []
idxy_col = []
sold = 0
#%%
MD_tr = np.median(Y_train1,axis=0)
MD_val = np.median(Y_val,axis=0)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
estimator.fit(X_train1, Y_train1, Max_LV)
y_pred_tr = estimator.predict(X_train1)*np.std(Y_train1,axis=0)+np.mean(Y_train1, axis=0)
y_pred_val = estimator.predict(X_val)*np.std(Y_val,axis=0)+np.mean(Y_val, axis=0)
# Lab_tr = [f'LV{ii+1}' for ii in range(Max_LV)]
# y_preds_tr = {key: [] for key in Lab_tr}
N_uc_var = Y_train1.shape[1]
y_preds_tr = np.empty((Max_LV,N_uc_var))
y_preds_val = np.copy(y_preds_tr)
no_show = True
for ii in range(Max_LV):
    y_preds_tr[ii,:] = [mean_squared_error(y_pred_tr[ii,:,jj],Y_train1[:,jj],squared=False) 
                   for jj in range(N_uc_var)]
    y_preds_tr[ii,:]/=MD_tr
    y_preds_val[ii,:] = [mean_squared_error(y_pred_val[ii,:,jj],Y_val[:,jj],squared=False) 
                   for jj in range(N_uc_var)]
    y_preds_val[ii,:]/=MD_val
    if not no_show:
        plt.bar(np.arange(1,N_uc_var+1),y_preds_tr[ii,:],
                width=0.9, color=np.array([0.2, 0.8, 0]), 
                edgecolor = 'black', 
                linewidth=1, alpha=0.5)
        plt.title(f'RMSECN vs. uc Variables for LV={ii+1}');plt.xlabel('65 uc Var.')
        plt.ylabel('RMSECN (% of AVG_MD_tr)');plt.show()
        plt.bar(np.arange(1,Y_val.shape[1]+1),y_preds_val[ii,:],
                width=0.9, color=np.array([0.8, 0.2, 0]), 
                edgecolor = 'black', 
                linewidth=1, alpha=0.5)
        plt.title(f'RMSECV vs. uc Variables for LV={ii+1}');plt.xlabel('65 uc Var.');
        plt.ylabel('RMSECV (% of AVG_MD_val)');plt.show()
#%%
if not no_show:
    RMSEC = [np.mean(y_preds_tr[ii,:]) for ii in range(Max_LV)]
    RMSECV = [np.mean(y_preds_val[ii,:]) for ii in range(Max_LV)]
    LV_x = np.arange(1,Max_LV+1)
    lent = 14
    plt.plot(LV_x[:lent],RMSEC[:lent], 'ro', 
             linestyle = '-.', label='RMSEC')
    plt.plot(LV_x[:lent],RMSECV[:lent], 'g*', 
             linestyle='--', label='RMSECV')
    plt.title('RMSEs vs. LVs for all uc Var.');plt.xlabel('LVs');
    plt.ylabel('RMSEs (% of AVG_MDs)');plt.legend();plt.grid();plt.show()

#%%
if not no_show:
    for jj in range(N_uc_var):
        plt.plot(np.arange(1,Max_LV+1),y_preds_tr[:,jj], 'ro', 
                 linestyle = '-.', label=f'RMSEC for {uc_proc_labels[jj]}')
        plt.plot(np.arange(1,Max_LV+1),y_preds_val[:,jj], 'g*', 
                 linestyle='--', label=f'RMSECV for {uc_proc_labels[jj]}')
        plt.title(f'RMSEs vs. LVs for uc Var. #{jj+1} {uc_proc_labels[jj]}');plt.xlabel('LVs');
        plt.ylabel('RMSEs (% of AVG_MDs)');plt.legend();plt.grid();plt.show()
#%%
#%%
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
def Baseline_PLS2_Modeling_for_Calc_Normalized_RMSEs(X_tr_val, Y_tr_val, X_MV, 
                                                     Y_MVM, Y_MV, Y_NMV, Max_LV, 
                                                     estimator, 
                                                     Y_preprocess_mode, 
                                                     no_show=None):
    # This function calculates the normalized (percentage of RMSEs per median) for each variable
    # Inputs: 
    # X_tr_val = X_good, Y_tr_val = Y_good : both containing 25 samples 
    # X_MV , Y_MVM , Y_MV, Y_NMV : containing 20 samples with 136 synthetic missing values (MVs)
    # Max_LV : maximum no. of LVs considered for implementing PLS2 models
    # estimator : the PLS2 algorithm using setting up parameters like the type of algorithm, the no. 
    # LVs, applying an specific preprocessing to the processing data, etc.
    # No_show: optional input for whether displaying or not the output results
    ##
    # Intermediate calculated variables:
    # y_pred_trval : predicted training Y variables as the ones without no MVs
    # y_pred_te : predicted test variables including missing values
    # MD_trval : contaning the median values per variable
    # MD_te : containing the median values per variable (MVs are initially imputed with the mean of 
    # the other values)
    ##
    # outputs:
    # RMSECN: normalized RMSECs for each variable in the training set & each PLS2 model with an LV 
    # from 1 to Max_LV
    # RMSECVN: normalized RMSECs for each variable in the test set & each PLS2 model with an  LV 
    # from 1 to Max_LV
    # RMSECN_noMV: normalized RMSECs for the variables in the training set with no matching MVs 
    # in the test set & each PLS2 model with an LV from 1 to Max_LV
    # RMSECN_noMV: normalized RMSECs for the variables in the test set with no MVs & each PLS2 
    # model with an LV from 1 to Max_LV
    MD_trval = np.median(Y_tr_val,axis=0)
    MD_te = np.zeros((Y_MVM.shape[1],1))
    for ii in range(len(MD_te)):
        MD_te[ii] = np.median(Y_MV[np.where(Y_MV[:,ii]!=0),ii])
    MD_te = np.squeeze(MD_te)
    estimator.fit(X_tr_val, Y_tr_val, Max_LV)
    if Y_preprocess_mode == 'mean_centering':
        y_pred_trval = estimator.predict(X_tr_val)+np.mean(Y_tr_val, axis=0)
        y_pred_te = estimator.predict(X_MV)+np.mean(Y_MVM, axis=0)
    elif Y_preprocess_mode == 'autoscale':
        y_pred_trval = estimator.predict(X_tr_val)*np.std(Y_tr_val,axis=0)+np.mean(Y_tr_val, axis=0)
        y_pred_te = estimator.predict(X_MV)*np.std(Y_MVM,axis=0)+np.mean(Y_MVM, axis=0)
    else:
        y_pred_trval = estimator.predict(X_tr_val)
        y_pred_te = estimator.predict(X_MV)
    N_uc_var = Y_tr_val.shape[1]
    RMSECN = [100*mean_squared_error(Y_tr_val/MD_trval,\
                               y_pred_trval[ii,:,:]/MD_trval,squared=False) 
            for ii in range(Max_LV)]
    RMSECVN = [100*mean_squared_error(Y_MVM/MD_te,\
                               y_pred_te[ii,:,:]/MD_te,squared=False) 
            for ii in range(Max_LV)]
    RMSECN_noMV = np.zeros((Max_LV,N_uc_var))#np.copy(RMSECN)
    RMSECVN_noMV = np.copy(RMSECN_noMV)
    for ii in range(Max_LV):
        for kk in range(Y_MV.shape[0]):
            idxt = np.where(Y_MV[kk,:]==0)
            MD_te_n = np.delete(MD_te, idxt)
            MD_trval_n = np.delete(MD_trval, idxt)
            yrt = np.delete(y_pred_trval[ii,:,:], idxt, axis=1)
            ypt = np.delete(Y_tr_val, idxt, axis=1)
            RMSECN_noMV[ii,kk] = mean_squared_error(yrt/MD_trval_n, ypt/MD_trval_n,squared=False)*100
            yrt = np.delete(Y_MV[kk,:], idxt)
            ypt = np.delete(y_pred_te[ii,kk,:], idxt)
            RMSECVN_noMV[ii,kk] = mean_squared_error(yrt/MD_te_n, ypt/MD_te_n, squared=False)*100
        if no_show!=None and not no_show:
            plt.bar(np.arange(1,N_uc_var+1),RMSECN[ii,:],
                    width=0.9, color=np.array([0.2, 0.8, 0]), 
                    edgecolor = 'black', 
                    linewidth=1, alpha=0.5)
            plt.title(f'RMSECN vs. uc Variables for LV={ii+1}');plt.xlabel('65 uc Var.')
            plt.ylabel('RMSECN (% of AVG_MD_tr)');plt.show()
            plt.bar(np.arange(1,Y_eval.shape[1]+1),RMSECVN[ii,:],
                    width=0.9, color=np.array([0.8, 0.2, 0]), 
                    edgecolor = 'black', 
                    linewidth=1, alpha=0.5)
            plt.title(f'RMSECVN vs. uc Variables for LV={ii+1}');plt.xlabel('65 uc Var.');
            plt.ylabel('RMSECVN (% of AVG_MD_val)');plt.show()
    if no_show!=None and not no_show:
        XX = np.arange(1,Max_LV+1)
        Lt = Max_LV
        plt.plot(XX[:Lt],RMSECN[:Lt], '*', color='r',
                 linestyle='--', linewidth=2, label = 'PLS2-based RMSECN')
        plt.plot(XX[:Lt],RMSECVN[:Lt], 'o', color='b', 
                 linestyle = '-.', linewidth=2, label = 'PLS2-based RMSECVN')
        plt.title('RMSECN & RMSECVN vs. LVs');plt.xlabel('LVs');
        plt.ylabel('RMSECN & RMSECVN (% of AVG_MD_val)');
        plt.xticks(XX[:Lt]);plt.grid()
        plt.legend();plt.show()
        plt.plot(XX[:Lt],np.mean(RMSECN_noMV,axis=1)[:Lt], '*', color='r',
                 linestyle='--', linewidth=2, label = 'PLS2-based RMSECN')
        plt.plot(XX[:Lt],np.mean(RMSECVN_noMV,axis=1)[:Lt], 'o', color='b', 
                 linestyle = '-.', linewidth=2, label = 'PLS2-based RMSECVN')
        plt.title('RMSECN & RMSECVN not including indices for MVs vs. LVs ');
        plt.xlabel('LVs'); plt.ylabel('RMSECN & RMSECVN (% of AVG_MD_val)');
        plt.xticks(XX[:Lt]);plt.grid()
        plt.legend();plt.show()
    return RMSECN, RMSECVN, RMSECN_noMV, RMSECVN_noMV
###############################################################################   
#%%
# Main Code with random stratification
rand_seed = 42
seed(rand_seed)
nu_list = np.arange(pnt).tolist()
# shuffle(nu_list)
MV = []
idxy_col = []
sold = 0
max_iter = 1
for kk in range(pnt):  # for kk in range(Y_MV.shape[0]):
    if kk == 0:
        idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                                  num_iters=max_iter)
    else:
        X_train1 = X[:X.shape[0] - X_val.shape[0], :]
        Y_train1 = Y[:X.shape[0] - X_val.shape[0], :]
        idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                                  num_iters=max_iter)
    X_train_val = np.vstack((X_train1, X_val))
    Y_train_val = np.vstack((Y_train1, Y_val))
    ###########################################################################
    # Cross validate procedure
    # Define model
    algorithm = 1
    estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
    fit_params = {'A': Max_LV}
    cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                        scoring=rmse_output_variables(),
                        cv=idx_generator, n_jobs=-1, verbose=1,
                        fit_params=fit_params, return_estimator=False,
                        error_score='raise')
    ###########################################################################
    tmp = cv['test_lowest_mean_rmse']
    tmp1 = cv['test_num_components_lowest_mean_rmse']
    # clearing tmp and tmp1 from nan values
    tmp1 = tmp1[np.logical_not(np.isnan(tmp))]
    tmp = tmp[np.logical_not(np.isnan(tmp))]
    mup = np.mean(tmp)
    stdp = np.std(tmp)
    ###########################################################################
    no_show = 10
    if no_show != 1 or kk%no_show==0:
        Intermediate_Plot_PLS1_2(Max_LV,tmp,tmp1,max_iter,cv,estimator, X_train_val,
                                     Y_train_val, X_eval, Y_eval, indices, par_1, par_2,
                                     ind_11, Y_preprocess_mode)
    ###########################################################################
    tmp33 = nu_list[kk]
    ###########################################################################
    # Random selection of samlpes inside each stratification of samples with MVs
    XX_tmp = X_MV[int(Rank_pnt[tmp33][0]):int(Rank_pnt[tmp33][1])+1,:]
    Y_MVM_tmp = Y_MVM[int(Rank_pnt[tmp33][0]):int(Rank_pnt[tmp33][1])+1,:]
    Y_MV_tmp = Y_MV[int(Rank_pnt[tmp33][0]):int(Rank_pnt[tmp33][1])+1,:]
    ss1 = XX_tmp.shape[0]
    for pp1 in range(ss1):
        # idx_tmp = list(np.where(Y_MV_tmp[pp1,:]==0)[0])
        # LV_cnt = [int(cv[f'test_num_components_lowest_var{jj1+1}_rmse'][0]) for jj1 in idx_tmp]
        B_LV_all = tmp1[0]
        estimator.fit(X_train1, Y_train1, B_LV_all)
        Y_tmp = np.vstack((Y_train1,Y_MV_tmp[pp1,:],Y_val))
        X_train1 = np.vstack((X_train1,XX_tmp[pp1,:]))
        Y_train1 = np.vstack((Y_train1,Y_MVM_tmp[pp1,:]))
        X = np.vstack((X_train1,X_val))
        Y = np.vstack((Y_train1,Y_val))
        y_mv_pred = np.zeros((1,sold+np.sum(Y_MV_tmp[pp1,:]==0)))
        sold = y_mv_pred.shape[1]  # size old
        # idxy = tuple(list(np.where(np.isin(Y_tmp,0))))
        idxy = list(np.where(np.isin(Y_tmp,0)))
        if kk==0 and pp1==0:
            idxyold = np.copy(idxy)
        else:
            idxy[0] = np.concatenate((idxyold[0],idxy[0]),axis=0)
            idxy[1] = np.concatenate((idxyold[1],idxy[1]),axis=0)
            idxyold = np.copy(idxy)
        LV_cnt = [int(cv[f'test_num_components_lowest_var{jj1+1}_rmse'][0]) for jj1 in idxy[1]]
###############################################################################
        for pp2 in range(y_mv_pred.shape[1]):
            # estimator.fit(X,Y, LV_cnt[pp2])
            estimator.fit(X,np.expand_dims(Y[:,idxy[1][pp2]],axis=1), LV_cnt[pp2])
            if Y_preprocess_mode == 'autoscale':
                y_pred_tmp = estimator.predict(X, LV_cnt[pp2]) \
                            * np.std(Y[:,idxy[1][pp2]], axis=0) + \
                            np.mean(Y[:,idxy[1][pp2]], axis=0)
            elif Y_preprocess_mode == 'mean_centering':
                y_pred_tmp = estimator.predict(X, LV_cnt[pp2]) \
                            + np.mean(Y[:,idxy[1][pp2]], axis=0)
            y_mv_pred[0][pp2] = np.abs(y_pred_tmp[idxy[0][pp2]][0])
        #%#
        Y[tuple(idxy)] = y_mv_pred
        # MV.append(Y[tuple(idxy)])
        idxy_col.append(tuple(idxy)[1])
        idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                                  num_iters=1)
        fit_params = {'A': Max_LV}
        cv = cross_validate(estimator=estimator, X=X, y=Y,
                            scoring=rmse_output_variables(),
                            cv=idx_generator, n_jobs=-1, verbose=1,
                            fit_params=fit_params, return_estimator=False,
                            error_score='raise')
###############################################################################
        tmp = cv['test_lowest_mean_rmse']
        tmp1 = cv['test_num_components_lowest_mean_rmse']
        # clearing tmp and tmp1 from nan values
        tmp1 = tmp1[np.logical_not(np.isnan(tmp))]
        tmp = tmp[np.logical_not(np.isnan(tmp))]
        mup = np.mean(tmp)
        stdp = np.std(tmp)
        #################
        no_show = 10
        if no_show != 1 or 10%no_show==0:
            Intermediate_Plot_PLS1_2(Max_LV,tmp,tmp1,1,cv,estimator, X_train_val,
                                     Y_train_val, X_eval, Y_eval, indices, par_1, par_2,
                                     ind_11, Y_preprocess_mode)
        #################
        estimator.fit(X,Y,tmp1[0])
        if Y_preprocess_mode == 'autoscale':
            y_pred_tmp = estimator.predict(X, tmp1[0]) \
                        * np.std(Y, axis=0) + np.mean(Y, axis=0)
        elif Y_preprocess_mode == 'mean_centering':
            y_pred_tmp = estimator.predict(X, tmp1[0]) \
                        + np.mean(Y, axis=0)
        Y[tuple(idxy)] = y_pred_tmp[tuple(idxy)]
        MV.append(Y[tuple(idxy)])
        print(f'Running of the imputation method for sample #{pp1+1} belonging',
              f' to rank {kk+1} of stratified samples has finished')
#%% ###########################################################################
# Saving the resutls of PLS2-based imputation
import datetime
# get the current date and time
now = datetime.datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M")
file_name = 'uc58_imput_with_update_Baseline_Model_' + now + '_V0.pkl'
data = {'X_good': X_good, 'Y_good': Y_good, 'X': X, 'Y': Y, 
        'MV': MV, 'X_eval':X_eval, 'Y_eval':Y_eval, 'idxy': idxy, 
        'idxy_init':idxy_init,'idxy_col':idxy_col}
with open(file_name, 'wb') as f:
    pickle.dump(data, f)
# %% ###########################################################################
sold = 0
for jj in range(len(MV)):
    ss1 = len(MV)-jj+1
    ss2 = len(MV[jj])-sold
    mv2 = np.ones((ss1,ss2))*Y_NMV[idxy_init[0][:ss2],idxy_init[1][:ss2]]
    mv1 = np.ones((ss1, ss2)) * Mean_Vals[idxy_col[jj][sold:]].T
    for ii in range(ss1-1):
        mv1[ii+1,:] = MV[ii+jj][sold:sold+ss2]
    sold+=ss2
    for ii in range(ss2):
        col_tmp = np.random.rand(3, )
        plt.plot(range(ss1), mv1[:,ii], 'o', color=col_tmp, linestyle='--',
                 label=f'Pred. Resuts for MV_#{ii+1} of Rank #{jj+1}')
        plt.plot(range(ss1), mv2[:,ii], color=col_tmp, linestyle='-',
                 label=f'Mean Value for MV_#{ii+1} of Rank #{jj+1}')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('Predicted Y values')
    plt.legend(fontsize=6)
    if jj == 0:
        plt.title(f'Predicted Values per iterations for Stratified Samples #{jj + 1} with #{len(MV[jj])} MVs')
    else:
        plt.title(
            f'Predicted Values per iterations for Stratified Samples #{jj + 1} with #{len(MV[jj]) - len(MV[jj - 1])} MVs')
    plt.show()
#%% ###########################################################################
# Extra Code for converging the calculated missing values
with open(file_name, 'rb') as f:
    SS11 = pickle.load(f)  # [X_good,Y_good,X_eval,Y_eval,X,Y,MV,idxy]
K11 = list(SS11.keys())
X_good = SS11[K11[0]]
Y_good = SS11[K11[1]]
X = SS11[K11[2]]
Y = SS11[K11[3]]
MV = SS11[K11[4]]
X_eval = SS11[K11[5]]
Y_eval = SS11[K11[6]]
idxy = SS11[K11[7]]
idxy_init = SS11[K11[8]]
#%% 
TMP = np.ones((len(MV[-1]),)) * np.reshape(Mean_Vals[idxy[-1]], [len(MV[-1]), ])
tmp22 = np.ones((len(MV[-1]), 1)) * np.reshape(Mean_Vals[idxy[-1]], [len(MV[-1]), 1])
idxy1 = np.copy(idxy)
cnt22 = 0
X_train1 = X[:X.shape[0] - X_val.shape[0], :]
while mean_squared_error(TMP, MV[-1], squared=False) > 5e-5:  # and len(np.argwhere(np.abs(MV[-1])<1))>0.012*len(TMP):    
    Y_train1 = Y[:X.shape[0] - X_val.shape[0], :]
    idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                              num_iters=max_iter)
    X_train_val = np.vstack((X_train1, X_val))
    Y_train_val = np.vstack((Y_train1, Y_val))
    ############################################################################
    # Cross validate procedure
    # Define model
    algorithm = 1
    estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
    fit_params = {'A': Max_LV}
    cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                        scoring=rmse_output_variables(),
                        cv=idx_generator, n_jobs=-1, verbose=1,
                        fit_params=fit_params, return_estimator=False,
                        error_score='raise')
    ###########################################################################
    tmp = cv['test_lowest_mean_rmse']
    tmp1 = cv['test_num_components_lowest_mean_rmse']
    # clearing tmp and tmp1 from nan values
    tmp1 = tmp1[np.logical_not(np.isnan(tmp))]
    tmp = tmp[np.logical_not(np.isnan(tmp))]
    mup = np.mean(tmp)
    stdp = np.std(tmp)
    no_show = 10
    if no_show != 1 or 10%no_show==0:
        Intermediate_Plot_PLS1_2(Max_LV,tmp,tmp1,1,cv,estimator, X_train_val,
                                     Y_train_val, X_eval, Y_eval, indices, par_1, par_2,
                                     ind_11, Y_preprocess_mode)
    #####
    for pp1 in range(y_mv_pred.shape[1]):
        estimator.fit(X, Y, tmp1[0])
        if Y_preprocess_mode == 'autoscale':
            y_pred_tmp = estimator.predict(X, tmp1[0]) \
                         * np.std(Y, axis=0) + np.mean(Y, axis=0)
        elif Y_preprocess_mode == 'mean_centering':
            y_pred_tmp = estimator.predict(X, tmp1[0]) \
                         + np.mean(Y, axis=0)
    Y[tuple(idxy)] = np.abs(y_pred_tmp[tuple(idxy)])
    # Non-weighted average
    # Y[idxy] = np.mean(y_mv_pred,axis=0)
    # Weighted average
    # Y[tuple(idxy)] = np.average(y_mv_pred, axis=0, weights=np.squeeze(Weights).tolist())
    TMP = np.copy(MV[-1])
    tmp22 = np.concatenate((tmp22, TMP.reshape([len(MV[-1]), 1])), axis=1)
    MV.append(Y[tuple(idxy)])
    cnt22 += 1
    tmp333 = len(np.argwhere(np.abs(MV[-1]) < 1))
    tmp44 = mean_squared_error(TMP, MV[-1], squared=False)
    print(f'Running of the extra iteration #{cnt22} with ' \
          f'#{tmp333} imputations less than "One" & ' \
          f'MSRE={np.round(tmp44 * 1e5) / 1e5} has finished')
#%% ###########################################################################
# Plotting the results
magnitudes = np.linalg.norm(tmp22, axis=1)
normalized_array = tmp22 / magnitudes[:, np.newaxis]
normalized_array *= (100 / np.expand_dims(normalized_array[:, 0], axis=1))
plt.plot(normalized_array.T, '*', linestyle='--')
plt.grid()
plt.xlabel('iteration')
plt.ylabel('Predicted Missing values')
plt.title('Predicted Missing Values per extra iterations')
plt.show()
# get the current date and time
now = datetime.datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M")
file_name = 'Updated_uc58_imput_with_update_Baseline_Model_' + now + '_V0.pkl'
data = {'X': X, 'Y': Y, 'Y_org': Y_org, 'X_good': X_good, 'Y_good': Y_good,
        'X_val': X_val, 'Y_val': Y_val, 'Y_NMV': Y_NMV, 'Y_MV': Y_MV, 'Y_MVM':Y_MVM,
        'X_eval':X_eval, 'Y_eval':Y_eval, 'MV': MV, 'idxy': idxy,'idxy_init':idxy_init,
        'idxy_col':idxy_col, 'Mean_Vals':Mean_Vals}
with open(file_name, 'wb') as f:
    pickle.dump(data, f)
#%%
# file_name ='Updated_uc58_after_imputation_06-10-2023_15-35_new16_new4.pkl'
with open(file_name, 'rb') as f:
    SS11 = pickle.load(f)
X = SS11['X']
Y = SS11['Y']
Y_org = SS11['Y_org']
X_val = SS11['X_val']
Y_val = SS11['Y_val']
X_eval = SS11['X_eval']
Y_eval = SS11['Y_eval']
X_good = SS11['X_good']
Y_good = SS11['Y_good']
idxy = SS11['idxy']
idxy_init = SS11['idxy_init']
Y_NMV = SS11['Y_NMV']
Y_MVM = SS11['Y_MVM']
Y_MV = SS11['Y_MV']
MV = SS11['MV']
# %% Plotting the original values and the results of convergence
for jj in range(pnt):
    if jj == 0:
        mv1 = np.ones((len(MV) + 1 - jj, len(MV[jj]))) * np.repeat(Mean_Vals[idxy_col[jj]], len(MV) + 1 - jj, axis=1).T
        # mv1 = np.zeros((len(MV)+1-jj,len(MV[jj])))
    else:
        mv1 = np.ones((len(MV) + 1 - jj, len(MV[jj]) - len(MV[jj - 1]))) * \
              np.repeat(Mean_Vals[idxy_col[jj][len(MV[jj - 1]):]], len(MV) + 1 - jj, axis=1).T
        # mv1 = np.zeros((pnt+1-jj,len(MV[jj])-len(MV[jj-1])))
    for ii in range(len(MV) - jj):
        if jj == 0:
            mv1[ii + 1, :] = MV[ii + jj][:len(MV[jj])]
        else:
            mv1[ii + 1, :] = MV[ii + jj][len(MV[jj - 1]):len(MV[jj])]
    plt.plot(range(len(MV) + 1 - jj), mv1, 'o', linestyle='--',
             label='results of Predictions for missing values of Rank {ii+1}')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('Predicted Y values')
    if jj == 0:
        plt.title(f'Predicted Values per iterations for Stratified Samples #{jj + 1} with #{len(MV[jj])} MVs')
    else:
        plt.title(
            f'Predicted Values per iterations for Stratified Samples #{jj + 1} with #{len(MV[jj]) - len(MV[jj - 1])} MVs')
    plt.show()
# %%
sold = 0
for jj in range(len(MV)):
    ss1 = len(MV)-jj+1
    ss2 = len(MV[jj])-sold
    mv2 = np.ones((ss1,ss2))*Y_NMV[idxy_init[0][:ss2],idxy_init[1][:ss2]]
    mv1 = np.ones((ss1, ss2)) * Mean_Vals[idxy_col[jj][sold:]].T
    for ii in range(ss1-1):
        mv1[ii+1,:] = MV[ii+jj][sold:sold+ss2]
    sold+=ss2
    for ii in range(ss2):
        col_tmp = np.random.rand(3, )
        plt.plot(range(ss1), mv1[:,ii], 'o', color=col_tmp, linestyle='--',
                 label=f'Pred. Resuts for MV_#{ii+1} of Rank #{jj+1}')
        plt.plot(range(ss1), mv2[:,ii], color=col_tmp, linestyle='-',
                 label=f'Mean Value for MV_#{ii+1} of Rank #{jj+1}')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('Predicted Y values')
    plt.legend(fontsize=6)
    if jj == 0:
        plt.title(f'Predicted Values per iterations for Stratified Samples #{jj + 1} with #{len(MV[jj])} MVs')
    else:
        plt.title(
            f'Predicted Values per iterations for Stratified Samples #{jj + 1} with #{len(MV[jj]) - len(MV[jj - 1])} MVs')
    plt.show()
# %%
sold = 0
sold1 = 0
for jj in range(pnt):
    tmp33 = nu_list[jj]
    s1 = np.sum(np.isin(Y_MV[Rank_pnt[tmp33][0]:Rank_pnt[tmp33][1] + 1, :], 0))
    ind_jj = np.diff(Rank_pnt[jj])[0]+1
    if jj == 0:
        s2 = len(MV) + 1 - jj
        MV0 = np.ones((s1, s2)) * np.repeat(Mean_Vals[idxy_col[jj]], s2, axis=1)
        # MV0 = np.zeros((s1,s2))
    else:
        sold1+=ind_jj
        s2 = len(MV) + 1 - ind_jj
        MV0 = np.ones((s1, s2)) * \
              np.repeat(Mean_Vals[idxy_col[sold1][sold:]], s2, axis=1)
    for ii in range(sold1, s2):
        MV0[:, ii] = MV[ii + sold1 - 1][sold:s1 + sold]
    s_tmp = np.where(np.isin(Y_MV[Rank_pnt[tmp33][0]:Rank_pnt[tmp33][1] + 1, :], 0))
    Y_tmp = Y_NMV[Rank_pnt[tmp33][0]:Rank_pnt[tmp33][1] + 1, :]
    tmp9 = np.expand_dims(100*(MV0[0,:]-MV0[0,0])/(MV0[0,-1]-MV0[0,0]),axis=0)
    for ii in range(MV0.shape[0]-1):
        tmp9 = np.concatenate((tmp9,np.expand_dims(100*(MV0[ii,:]-MV0[ii,0])/(MV0[ii,-1]-MV0[ii,0]),
                                                   axis=0)),axis=0)
    # Create a matplotlib figure
    Markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '+', '*', 'x', 'D', 'd']
    epsilon = 1e-9
    fig, ax = plt.subplots()
    for kk in range(len(s_tmp[1])):
        if kk < len(Markers):
            # plt.plot(range(s2), (MV0[kk, :] * 100) / (Y_tmp[s_tmp][kk]+epsilon),
            #          Markers[kk], markersize=4, linestyle='--',
            #          label=f'Conv. MV{kk + 1}')
            plt.plot(range(s2), (MV0[kk, :] * 100) / (Y_tmp[s_tmp][kk]+epsilon),
                     Markers[kk], markersize=4, linestyle='--',
                     label=f'Conv. MV{kk + 1}')
        else:
            plt.plot(range(s2), (MV0[kk] * 100) / (Y_tmp[s_tmp][kk]+epsilon), 'o',
                     linestyle='--', markersize=4,
                     label=f'Conv. MV{kk + 1}')
        # plt.plot(range(s2),np.repeat(Y_tmp[s_tmp][kk],len(MV)+1-jj),linestyle='-',
        #          label=f'Real MV{kk+1}');
    plt.plot(100 * np.ones((s2, 1)), linestyle='-',
             label='Norm_RV(100)', linewidth=2);
    plt.title(f'Real vs. Converged values for Stratifiction#{tmp33 + 1}')
    plt.ylabel('Real & converged MVs')
    plt.xlabel('Convergence Steps')
    # plt.xticks(np.linspace(0,len(MV)-jj,int((len(MV)+1-jj)/2)+1))
    x_tmp = range(0, s2)
    num_ticks = 10
    step_size = s2 // (num_ticks - 1)
    plt.xticks(x_tmp[::step_size], x_tmp[::step_size])
    plt.grid()
    # Add a legend
    pos = ax.get_position()
    if s1 <= 20:
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), fontsize=7)
    plt.show()
    sold += s1
# %%
YY_tmp = Y[Len_tr:-Len_val, :]
################################################################################
rmse_zero = mean_squared_error(Y_MV[idxy_init], Y_NMV[idxy_init], squared=False)
rmse_mean = mean_squared_error(Y_MVM[idxy_init], Y_NMV[idxy_init], squared=False)
rmse_PLS2 = mean_squared_error(YY_tmp[idxy_init], Y_NMV[idxy_init], squared=False)
print(' RMSE for zero-value imputed MVs || RMSE for mean-value imputed MVs || RMSE for PLS2-value imputed MVs')
print('========================================================================================================')
print('             {:.3f}                               {:.3f}                             {:.3f}'.format(rmse_zero,
                                                                                                           rmse_mean,
                                                                                                           rmse_PLS2))
# %% Evaluation #######################################################################################
ind_11 = 1
par1 = par_1[ind_11]  # 'LDLPhoslip'#'IDLTGKorr'
indices = Par_idx_finder(par1, uc_proc_labels)
indices = indices[0]
App = 2 # Appraoch ==1: Most Frequent #LVs among all bootstraps, 
# Appraoch ==2: The #LVs matched with the minimum average RMSE values for all bootstraps
# %%
# (0) 
# Generating cross-validation indices
X3, X_val, Y3, Y_val = \
    train_test_split(X_good, Y_good, test_size=TestSize, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train1 = np.vstack((X3, X_MV))
Y_train1 = np.vstack((Y3, Y_MV))
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
#%%
# (1) 
# Generating cross-validation indices
X3, X_val, Y3, Y_val = \
    train_test_split(X_good, Y_good, test_size=TestSize, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train1 = np.vstack((X3, X_MV))
Y_train1 = np.vstack((Y3, Y_MV))
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
#
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0], 
                                          num_iters=max_iter)
algorithm = 1  
estimator = PLS(algorithm=algorithm, Signal_Type = 'NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val, 
                    scoring= rmse_output_variables(), 
                    cv=idx_generator, n_jobs=-1, verbose=1, 
                    fit_params=fit_params, return_estimator=False, 
                    error_score='raise')
# PLS2-based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
                                   bins=range(1,Max_LV+2))[0].tolist())+1
elif App==2:
    LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
    LV_no = 5
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)*np.std(Y_train_val, axis=0)\
        + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)*np.std(Y_eval, axis=0)\
        + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)\
        + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)\
        + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:,indices], y_pred_te[:,indices],squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes', 
#                                 text='PLS2 Alg. & no imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], 
#                                 text='PLS2 Alg. & no imputation')
estimator.fit(X_train1,Y_train1,LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val,LV_no)*np.std(Y_val, axis=0)\
        + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val,LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:,indices],squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & no imputation')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
                                Y_train_val[:,indices], y_pred_tr[:,indices], 
                                rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
                                text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                                idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var'+str(indices+1)+'_rmse'],
                                   bins=range(1,Max_LV+2))[0].tolist())+1
elif App ==2:
    _, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
   LV_no = 6
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:,indices],axis=1), LV_no)
if Y_preprocess_mode == 'autoscale': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)\
        *np.std(Y_train_val[:,indices], axis=0)\
            + np.mean(Y_train_val[:,indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)*np.std(Y_eval[:,indices], axis=0)\
        + np.mean(Y_eval[:,indices], axis=0)
elif Y_preprocess_mode == 'mean_centering': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)\
        + np.mean(Y_train_val[:,indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)\
        + np.mean(Y_eval[:,indices], axis=0)
rmsep = mean_squared_error(Y_eval[:,indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes', 
#                                 text='PLS1 Alg. & no imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],  
#                                 text= 'PLS1 Alg. & no imputation')
estimator.fit(X_train1,np.expand_dims(Y_train1[:,indices],axis=1),LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val,LV_no)*np.std(Y_val[:,indices], axis=0)\
        + np.mean(Y_val[:,indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val,LV_no)\
        + np.mean(Y_val[:,indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val,squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & no imputation')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:,indices], y_pred_te, 
                                Y_train_val[:,indices], y_pred_tr, 
                                rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
                                text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                                idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (2-0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# Generating cross-validation indices
X_train1 = X[:-Len_val,:]
Y_train1 = Y[:-Len_val,:]
X_val = X[-Len_val:,:]
Y_val = Y[-Len_val:,:]
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))

estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (2) Evaluating the performance of a bootstrapping cross validated PLS2 algorithm
# with Generating cross-validation indices
# X_train1, X_val, Y_train1, Y_val = \
#     train_test_split(X, Y, test_size=0.3, random_state=Rnd_stat)
X_train1 = X[:-Len_val,:]
Y_train1 = Y[:-Len_val,:]
X_val = X[-Len_val:,:]
Y_val = Y[-Len_val:,:]
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))
# max_iter = 1000
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                          num_iters=max_iter)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                    scoring=rmse_output_variables(),
                    cv=idx_generator, n_jobs=-1, verbose=1,
                    fit_params=fit_params, return_estimator=False,
                    error_score='raise')
# PLS2 based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
                                   bins=range(1, Max_LV + 2))[0].tolist()) + 1
elif App ==2:
    LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
    LV_no = 24
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te[:, indices], squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
estimator.fit(X_train1, Y_train1, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:, indices], squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & PLS2-based Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1 based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
                                   bins=range(1, Max_LV + 2))[0].tolist()) + 1
elif App==2:
    _, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
    LV_no = 7
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval[:, indices], axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val, squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te,
                               Y_train_val[:, indices], y_pred_tr,
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & PLS2-based Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (3-0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
# Generating cross-validation indices
Y_train_val = np.vstack((Y_good, Y_MVM))
X_train_val = np.vstack((X_good, X_MV))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=Rnd_stat)
estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (3) Evaluating the performance of Mean value Imputation
# with Generating cross-validation indices
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
Y_train_val = np.vstack((Y_good, Y_MVM))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=Rnd_stat)
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                          num_iters=max_iter, seed=42)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                    scoring=rmse_output_variables(),
                    cv=idx_generator, n_jobs=-1, verbose=1,
                    fit_params=fit_params, return_estimator=False,
                    error_score='raise')
# PLS2 based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
                                   bins=range(1, Max_LV + 2))[0].tolist()) + 1
elif App==2:
    LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
    LV_no = 5
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te[:, indices], squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
estimator.fit(X_train1, Y_train1, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:, indices], squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Mean Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1 based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
                                    bins=range(1, Max_LV + 2))[0].tolist()) + 1
elif App==2:
    _, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
    LV_no = 10
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval[:, indices], axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val, squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te,
                               Y_train_val[:, indices], y_pred_tr,
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Mean Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (4-0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
# Generating cross-validation indices
Y_train_val = np.vstack((Y_good, Y_NMV))
X_train_val = np.vstack((X_good, X_MV))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=Rnd_stat)
estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (4) Evaluating the performance of Output variables with Original values
# with Generating cross-validation indices
Y3 = np.vstack((Y_good, Y_NMV))
X3 = np.vstack((X_good, X_MV))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X3, Y3, test_size=0.3, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = np.vstack((Y_good, Y_MV))
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
Y_train_val = np.vstack((Y_good, Y_NMV))
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                          num_iters=max_iter, seed=42)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                    scoring=rmse_output_variables(),
                    cv=idx_generator, n_jobs=-1, verbose=1,
                    fit_params=fit_params, return_estimator=False,
                    error_score='raise')
# PLS2 based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
                                   bins=range(1, Max_LV + 2))[0].tolist()) + 1
elif App==2:
    LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
else:
    LV_no = 5
LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te[:, indices], squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
estimator.fit(X_train1, Y_train1, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:, indices], squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. for Real Values of MVs', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1 based implementation
if App ==1:
    LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
                                    bins=range(1, Max_LV + 2))[0].tolist()) + 1
elif App==2:
    _, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)        
else:
    LV_no = 10
_, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval[:, indices], axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val, squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te,
                               Y_train_val[:, indices], y_pred_tr,
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. for Real Values of MVs', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
#%% Evaluation #######################################################################################
ind_11 = 2
par1 = par_1[ind_11]#'LDLPhoslip'#'IDLTGKorr'
indices = Par_idx_finder(par1,uc_proc_labels)
indices = indices[0]
# %%
# (0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# Generating cross-validation indices
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X3, Y3, test_size=0.3, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
#%%
# (1) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# Generating cross-validation indices
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X3, Y3, test_size=0.3, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val,0)))
idxy_tmp2 = list(np.where(idxy_tmp[1]==indices))
# Y_train_val[idxy_tmp[0][idxy_tmp2],indices]
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0], 
                                          num_iters=max_iter, seed=42)
algorithm = 1  
estimator = PLS(algorithm=algorithm, Signal_Type = 'NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val, 
                    scoring= rmse_output_variables(), 
                    cv=idx_generator, n_jobs=-1, verbose=1, 
                    fit_params=fit_params, return_estimator=False, 
                    error_score='raise')
# PLS2-based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
#                                 bins=range(1,Max_LV+2))[0].tolist())+1
# LV_no = 5
LV_no,_ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)*np.std(Y_train_val, axis=0)\
        + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)*np.std(Y_eval, axis=0)\
        + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)\
        + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)\
        + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:,indices], y_pred_te[:,indices],squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes', 
#                                 text='PLS2 Alg. & no imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], 
#                                 text='PLS2 Alg. & no imputation')
estimator.fit(X_train1,Y_train1,LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val,LV_no)*np.std(Y_val, axis=0)\
        + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val,LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:,indices],squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & no imputation')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
                                Y_train_val[:,indices], y_pred_tr[:,indices], 
                                rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
                                text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                                idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var'+str(indices+1)+'_rmse'],
#                                 bins=range(1,Max_LV+2))[0].tolist())+1
# LV_no = 10
_, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:,indices],axis=1), LV_no)
if Y_preprocess_mode == 'autoscale': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)\
        *np.std(Y_train_val[:,indices], axis=0)\
            + np.mean(Y_train_val[:,indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)*np.std(Y_eval[:,indices], axis=0)\
        + np.mean(Y_eval[:,indices], axis=0)
elif Y_preprocess_mode == 'mean_centering': 
    y_pred_tr = estimator.predict(X_train_val, LV_no)\
        + np.mean(Y_train_val[:,indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no)\
        + np.mean(Y_eval[:,indices], axis=0)
rmsep = mean_squared_error(Y_eval[:,indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes', 
#                                 text='PLS1 Alg. & no imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],  
#                                 text= 'PLS1 Alg. & no imputation')
estimator.fit(X_train1,np.expand_dims(Y_train1[:,indices],axis=1),LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val,LV_no)*np.std(Y_val[:,indices], axis=0)\
        + np.mean(Y_val[:,indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val,LV_no)\
        + np.mean(Y_val[:,indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val,squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & no imputation')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:,indices], y_pred_te, 
                                Y_train_val[:,indices], y_pred_tr, 
                                rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
                                text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                                idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (2-0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# Generating cross-validation indices
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X, Y, test_size=0.3, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))

estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (2) Evaluating the performance of a bootstrapping cross validated PLS2 algorithm
# with Generating cross-validation indices
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X, Y, test_size=0.3, random_state=Rnd_stat)
X_train_val = np.vstack((X_train1, X_val))
Y_train_val = np.vstack((Y_train1, Y_val))
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                          num_iters=max_iter, seed=42)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                    scoring=rmse_output_variables(),
                    cv=idx_generator, n_jobs=-1, verbose=1,
                    fit_params=fit_params, return_estimator=False,
                    error_score='raise')
# PLS2 based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
#                                bins=range(1, Max_LV + 2))[0].tolist()) + 1
# LV_no = 8
LV_no,_ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te[:, indices], squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
estimator.fit(X_train1, Y_train1, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:, indices], squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & PLS2-based Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1 based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
#                                bins=range(1, Max_LV + 2))[0].tolist()) + 1
# LV_no = 10
_, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval[:, indices], axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val, squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te,
                               Y_train_val[:, indices], y_pred_tr,
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & PLS2-based Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (3-0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
# Generating cross-validation indices
Y_train_val = np.vstack((Y_good, Y_MVM))
X_train_val = np.vstack((X_good, X_MV))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=Rnd_stat)
estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (3) Evaluating the performance of Mean value Imputation
# with Generating cross-validation indices
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
Y_train_val = np.vstack((Y_good, Y_MVM))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=Rnd_stat)
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                          num_iters=max_iter, seed=42)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                    scoring=rmse_output_variables(),
                    cv=idx_generator, n_jobs=-1, verbose=1,
                    fit_params=fit_params, return_estimator=False,
                    error_score='raise')
# PLS2 based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
                               # bins=range(1, Max_LV + 2))[0].tolist()) + 1
# LV_no = 5
LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te[:, indices], squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
estimator.fit(X_train1, Y_train1, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:, indices], squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Mean Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1 based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
#                                bins=range(1, Max_LV + 2))[0].tolist()) + 1
# LV_no = 10
_, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval[:, indices], axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val, squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te,
                               Y_train_val[:, indices], y_pred_tr,
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Mean Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (4-0) 
Y3 = np.vstack((Y_good, Y_MV))
X3 = np.vstack((X_good, X_MV))
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = Y3
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
# Generating cross-validation indices
Y_train_val = np.vstack((Y_good, Y_NMV))
X_train_val = np.vstack((X_good, X_MV))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X_train_val, Y_train_val, test_size=0.3, random_state=Rnd_stat)
estimator.fit(X_train1, Y_train1, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val, axis=0)
rmsecv_1 = np.sqrt(np.mean((Y_val - y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(np.mean(rmsecv_1, axis=1)) + 1
plt.plot(x_tmp, np.mean(rmsecv_1, axis=1))
plt.plot(LV_no, np.min(np.mean(rmsecv_1, axis=1)), 'o', color='red')
plt.title('Average PLS2-based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1, indices]

estimator.fit(X_train_val, Y_train_val, Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval) + np.mean(Y_eval, axis=0)

rmsep_1 = np.sqrt(np.mean((Y_eval - y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[np.argmin(np.mean(rmsecv_1, axis=1)), indices]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               y_pred_te[LV_no - 1, :, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1-based implementation
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv_1 = np.sqrt(np.mean((np.expand_dims(Y_val[:, indices], axis=1) -
                            y_pred_val) ** 2, axis=1))
x_tmp = range(1, Max_LV + 1)
LV_no = np.argmin(rmsecv_1) + 1
plt.plot(x_tmp, rmsecv_1)
plt.plot(LV_no, np.min(rmsecv_1), 'o', color='red')
plt.title('Average PLS1 -based RMSECV vs. No. LVs');
plt.xlabel('LVs');
plt.ylabel('Average RMSECV')
num_ticks = 10
step_size = (Max_LV + 1) // (num_ticks - 1)
plt.xticks(x_tmp[1::step_size])
plt.grid()
plt.show()
rmsecv = rmsecv_1[LV_no - 1].tolist()[0]

estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), Max_LV)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval) \
                + np.mean(Y_eval[:, indices], axis=0)

rmsep_1 = np.sqrt(np.mean((np.expand_dims(Y_eval[:, indices], axis=1) -
                           y_pred_te) ** 2, axis=1))
rmsep = rmsep_1[LV_no - 1].tolist()[0]
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices],
                               np.squeeze(y_pred_te[LV_no - 1, :]),
                               Y_train_val[:, indices],
                               np.squeeze(y_pred_tr[LV_no - 1, :]),
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. & Zero Imput.', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
# %%
# (4) Evaluating the performance of Output variables with Original values
# with Generating cross-validation indices
Y3 = np.vstack((Y_good, Y_NMV))
X3 = np.vstack((X_good, X_MV))
X_train1, X_val, Y_train1, Y_val = \
    train_test_split(X3, Y3, test_size=0.3, random_state=Rnd_stat)
# X_train_val = np.vstack((X_train1, X_val))
# Y_train_val = np.vstack((Y_train1, Y_val))
X_train_val = X3
Y_train_val = np.vstack((Y_good, Y_MV))
idxy_tmp = list(np.where(np.isin(Y_train_val, 0)))
idxy_tmp2 = list(np.where(idxy_tmp[1] == indices))
Y_train_val = np.vstack((Y_good, Y_NMV))
idx_generator = bootstrap_index_generator_new(X_train1.shape[0], X_val.shape[0],
                                          num_iters=max_iter, seed=42)
algorithm = 1
estimator = PLS(algorithm=algorithm, Signal_Type='NMR')
fit_params = {'A': Max_LV}
cv = cross_validate(estimator=estimator, X=X_train_val, y=Y_train_val,
                    scoring=rmse_output_variables(),
                    cv=idx_generator, n_jobs=-1, verbose=1,
                    fit_params=fit_params, return_estimator=False,
                    error_score='raise')
# PLS2 based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
#                                bins=range(1, Max_LV + 2))[0].tolist()) + 1
# LV_no = 5
LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, Y_train_val, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) * np.std(Y_train_val, axis=0) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval, axis=0) \
                + np.mean(Y_eval, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val, axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval, axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te[:, indices], squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS2 Alg. & suggested PLS2 imputation')
estimator.fit(X_train1, Y_train1, LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val, axis=0) \
                 + np.mean(Y_val, axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) + np.mean(Y_val, axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val[:, indices], squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te[:,indices], 
#                                 Y_train_val[:,indices], y_pred_tr[:,indices], 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS2 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                               Y_train_val[:, indices], y_pred_tr[:, indices],
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS2 Alg. for Real Values of MVs', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
## PLS1 based implementation
# LV_no = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
#                                bins=range(1, Max_LV + 2))[0].tolist()) + 1
# LV_no = 10
_, LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1,Max_LV,1)
estimator.fit(X_train_val, np.expand_dims(Y_train_val[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                * np.std(Y_train_val[:, indices], axis=0) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) * np.std(Y_eval[:, indices], axis=0) \
                + np.mean(Y_eval[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_tr = estimator.predict(X_train_val, LV_no) \
                + np.mean(Y_train_val[:, indices], axis=0)
    y_pred_te = estimator.predict(X_eval, LV_no) \
                + np.mean(Y_eval[:, indices], axis=0)
rmsep = mean_squared_error(Y_eval[:, indices], y_pred_te, squared=False)
# Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11], show_lr='yes',
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
estimator.fit(X_train1, np.expand_dims(Y_train1[:, indices], axis=1), LV_no)
if Y_preprocess_mode == 'autoscale':
    y_pred_val = estimator.predict(X_val, LV_no) * np.std(Y_val[:, indices], axis=0) \
                 + np.mean(Y_val[:, indices], axis=0)
elif Y_preprocess_mode == 'mean_centering':
    y_pred_val = estimator.predict(X_val, LV_no) \
                 + np.mean(Y_val[:, indices], axis=0)
rmsecv = mean_squared_error(Y_val[:, indices], y_pred_val, squared=False)
# Plot_Measured_Predicted_NMR(LV_no, Y_eval[:,indices], y_pred_te, 
#                                 Y_train_val[:,indices], y_pred_tr, 
#                                 rmsep, par_1[ind_11], par_2[ind_11],rmsecv,
#                                 text='PLS1 Alg. & suggested PLS2 imputed Val.s')
######################## Separate Display of MVs ##############################
Plot_Measured_Predicted_NMR_MV(LV_no, Y_eval[:, indices], y_pred_te,
                               Y_train_val[:, indices], y_pred_tr,
                               rmsep, par_1[ind_11], par_2[ind_11], rmsecv,
                               text='PLS1 Alg. for Real Values of MVs', idxy_tmp=idxy_tmp,
                               idxy_tmp2=idxy_tmp2)
###############################################################################
