# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:15:17 2023

@author: tsz874
"""
import numpy as np
from matplotlib import pyplot as plt
# import sys
# folder_path = r'C:\Users\tsz874\Downloads\FAB_Projects\Stuff_For_Ashkan\PLS-main'
# sys.path.append(folder_path)
Y_preprocess_mode = 'autoscale'  #'mean_centering' #  'none'
from sklearn.metrics import r2_score as Q2
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error as MAE
## Definition of applied Functions
###############################################################################
### Func 1 ###
def Plot_Measured_Predicted_NMR(LV_no, y_test, y_pred_te, y_train,
                                y_pred_tr, rmsep, par_1, par_2,
                                rmsecv=None, text=None):
    ulim = np.max([np.max(y_train), np.max(y_pred_tr), np.max(y_test), np.max(y_pred_te)]) * 1.1
    # dlim = np.min([np.min(y_train),np.min(y_pred_tr),np.min(y_test),np.min(y_pred_te)])*0.9
    dlim = -5e-1
    # Calculate the Q^2 factor using the r2_score function
    q2_p = Q2(y_test, y_pred_te, multioutput='variance_weighted')
    ## Calculate the coefficients of variation of the test set prediction model
    cv1 = 100 * rmsep / np.mean(y_test)
    plt.plot([], [], ' ', label='LVs={}'.format(LV_no))
    plt.plot([], [], ' ', label='RMSEP={}'.format(np.round(rmsep * 1e4) / 1e4))
    if rmsecv is not None:
        plt.plot([], [], ' ', label='RMSECV={}'.format(np.round(rmsecv * 1e4) / 1e4))
    plt.plot([], [], ' ', label='Q$^{{2}}$ ={}'.format(np.round(q2_p * 1e4) / 1e4))
    plt.plot([], [], ' ', label='CV ={}'.format(np.round(cv1 * 1e4) / 1e4))
    plt.plot(np.linspace(dlim, ulim, 100), np.linspace(dlim, ulim, 100), linestyle='--', linewidth=2, c='red',
             alpha=0.9)
    plt.plot(y_train, y_pred_tr, 'o', color='red', label=f'train set (# {len(y_train)})',
             alpha=1, markeredgewidth=.5, markeredgecolor='black')
    plt.plot(y_test, y_pred_te, 'o', color='blue', label=f'test set (# {len(y_test)})',
             alpha=0.9, markeredgewidth=.5, markeredgecolor='black')
    # Add more text to the legend
    plt.xlim([0.95 * dlim, 1.05 * ulim])
    plt.ylim([0.95 * dlim, 1.05 * ulim])
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    if text is not None:
        plt.title(f'Pred. vs. Measured for {par_2} based on {text}', fontweight="bold")
    else:
        plt.title(f'Scatter plot for Var {par_2}', fontweight="bold")
    plt.legend(title=r'$\bf{' + par_1 + '}$', fontsize="10")
    plt.show()


### Func 2 ###
def Plot_Measured_Predicted_NMR_PLS(LV_no, y_test, y_pred_te, y_train,
                                    y_pred_tr, rmsep, par_1, par_2, text=None, show_lr=None):
    ulim_M = np.max([np.max(y_train), np.max(y_test)])
    ulim_M += 0.1 * np.abs(ulim_M)
    # dlim_M = np.min([np.min(y_train),np.min(y_test)])
    # dlim_M -=0.1*np.abs(dlim_M)
    dlim_M = -5e-1
    ulim_P = np.max([np.max(y_pred_tr), np.max(y_pred_te)])
    ulim_P += 0.1 * np.abs(ulim_P)
    # dlim_P = np.min([np.min(y_pred_tr),np.min(y_pred_te)])
    # dlim_P -=0.1*np.abs(dlim_P)
    dlim_P = -5e-1
    # Calculate the Q^2 factor using the r2_score function
    q2_p = Q2(y_test, y_pred_te, multioutput='variance_weighted')
    cv1 = 100 * rmsep / np.mean(y_test)
    ## Displaying final plot
    # Add more text to the legend
    plt.figure(figsize=(8, 7))
    plt.plot([], [], ' ', label=f'LVs={LV_no}')
    plt.plot([], [], ' ', label=f'RMSEP={(np.round(rmsep * 1e4) / 1e4)}')
    plt.plot([], [], ' ', label=f'Q$^{{2}}$ ={(np.round(q2_p * 1e4) / 1e4)}')
    plt.plot([], [], ' ', label=f'CV ={(np.round(cv1 * 1e4) / 1e4)}')
    plt.plot(y_train, y_pred_tr, 'o', color='red', label=f'train set (# {len(y_train)})',
             alpha=1, markeredgewidth=.5, markeredgecolor='black')
    plt.plot(y_test, y_pred_te, 'o', color='blue', label=f'test set (# {len(y_test)})',
             alpha=0.85, markeredgewidth=.5, markeredgecolor='black')
    xx = np.linspace(np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P]), 100)
    if show_lr is not None:
        # Calculate the linear regression of test set
        plt.plot(xx, xx, linestyle='--', linewidth=2, color='red',
                 label='${y}$ = ${x}$')
        slope, intercept = np.polyfit(np.squeeze(y_test), np.squeeze(y_pred_te), 1)
        plt.plot(xx, xx * slope + intercept, linestyle='--', color='blue', linewidth=2,
                 label='LR for Test set')
    else:
        plt.plot(np.linspace(np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P]), 100),
                 np.linspace(np.min([dlim_M, dlim_P]),
                             np.max([ulim_M, ulim_P]), 100),
                 linestyle='--', linewidth=2, color='red')
    plt.xlim([np.min([dlim_M, dlim_P]), np.max([ulim_P, ulim_M])])
    plt.ylim([np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P])])
    plt.xlabel('Measured', fontsize="12")
    plt.ylabel('Predicted', fontsize="12")
    if text != None:
        plt.title(f'Pred. vs. Measured for {par_2} based on {text}', fontweight="bold")
    else:
        plt.title(f'Scatter plot for Var {par_2}', fontweight="bold")
    plt.legend(title=r'$\bf{' + par_1 + '}$', fontsize="14")
    plt.show()


### Func 3 ###
def q2_calc(y_real, y_pred):
    y_mean = np.mean(y_real)
    rss = np.sum(np.square(y_real - y_pred))
    tss = np.sum(np.square(y_real - y_mean))
    q2 = 1 - (rss / tss)
    return q2


### Func 4 ###
def Plot_Measured_vs_Predicted(y_true, y_pred, show_lr=None):
    ulim = np.max([np.max(y_true), np.max(y_pred)]) * 1.1
    dlim = np.min([np.min(y_true), np.min(y_pred)]) * 0.9
    ulim_M = np.max([np.max(y_true)])
    ulim_M += 0.1 * np.abs(ulim_M)
    # dlim_M = np.min([np.min(y_true)])
    # dlim_M -=0.1*np.abs(dlim_M)
    dlim_M = -5e-1
    ulim_P = np.max([np.max(y_pred)])
    ulim_P += 0.1 * np.abs(ulim_P)
    # dlim_P = np.min([np.min(y_pred)])
    # dlim_P -=0.1*np.abs(dlim_P)
    dlim_P = -5e-1
    # Calculate the Q^2 factor using the r2_score function
    q2_p = Q2(y_true, y_pred, multioutput='variance_weighted')
    ## Calculate RMSEC & RMSEP
    rmsep = mean_squared_error(y_true, y_pred, squared=False)
    # Calculate the coefficient of variation (CV)
    cv1 = 1e2 * rmsep / np.mean(y_true)
    plt.plot([], [], ' ', label='RMSEP={}'.format(np.round(rmsep * 1e4) / 1e4))
    plt.plot([], [], ' ', label='Q$^{{2}}$ ={}'.format(np.round(q2_p * 1e4) / 1e4))
    plt.plot([], [], ' ', label='CV ={}'.format(np.round(cv1 * 1e4) / 1e4))
    plt.plot(y_true, y_pred, 'o', color='blue', label=f'test set (# {len(y_true)})', alpha=0.8);
    # Add more text to the legend
    plt.plot(np.linspace(dlim, ulim, 100), np.linspace(dlim, ulim, 100), linestyle='--', linewidth=2);
    plt.xlim([0.95 * dlim, 1.05 * ulim]);
    plt.ylim([0.95 * dlim, 1.05 * ulim])
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Pred. vs. Measured for all uc missing Variables')
    xx = np.linspace(np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P]), 100)
    if show_lr is not None:
        # Calculate the linear regression of test set
        plt.plot(xx, xx, linestyle='--', linewidth=2, color='red',
                 label='${y}$ = ${x}$')
        slope, intercept = np.polyfit(np.squeeze(y_true), np.squeeze(y_pred), 1)
        plt.plot(xx, xx * slope + intercept, linestyle='--', color='blue', linewidth=2,
                 label='LR for Test set')
    else:
        plt.plot(np.linspace(np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P]), 100),
                 np.linspace(np.min([dlim_M, dlim_P]),
                             np.max([ulim_M, ulim_P]), 100),
                 linestyle='--', linewidth=2)
    plt.xlim([np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P])])
    plt.ylim([np.min([dlim_M, dlim_P]), np.max([ulim_M, ulim_P])])
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Pred. vs. Measured for All Variables')
    plt.legend(title='All Variables')
    plt.show()


### Func 5 ###
def Par_idx_finder(par_name, uc_proc_labels):
    indices = []
    for i in range(len(uc_proc_labels)):
        if par_name in uc_proc_labels[i]:
            indices.append(i)
    return indices


### Func 6 ###
def Plot_Measured_Predicted_NMR_MV(LV_no, y_test, y_pred_te, y_train,
                                   y_pred_tr, rmsep, par_1, par_2,
                                   rmsecv=None, text=None, idxy_tmp=None,
                                   idxy_tmp2=None):
    ulim = np.max([np.max(y_train), np.max(y_pred_tr), np.max(y_test), np.max(y_pred_te)]) * 1.1
    # dlim = np.min([np.min(y_train),np.min(y_pred_tr),np.min(y_test),np.min(y_pred_te)])*0.9
    dlim = -5e-1
    # Calculate the Q^2 factor using the r2_score function
    q2_p = Q2(y_test, y_pred_te, multioutput='variance_weighted')
    ## Calculate the coefficients of variation of the test set prediction model
    cv1 = 100 * rmsep / np.mean(y_test)
    plt.plot([], [], ' ', label='LVs={}'.format(LV_no))
    plt.plot([], [], ' ', label='RMSEP={}'.format(np.round(rmsep * 1e4) / 1e4))
    if rmsecv != None:
        plt.plot([], [], ' ', label='RMSECV={}'.format(np.round(rmsecv * 1e4) / 1e4))
    plt.plot([], [], ' ', label='Q$^{{2}}$ ={}'.format(np.round(q2_p * 1e4) / 1e4))
    plt.plot([], [], ' ', label='CV ={}'.format(np.round(cv1 * 1e4) / 1e4))
    plt.plot(np.linspace(dlim, ulim, 100), np.linspace(dlim, ulim, 100), linestyle='--', linewidth=2, c='red',
             alpha=0.9);
    plt.plot(y_train, y_pred_tr, 'o', color='red', label=f'train set (# {len(y_train)})',
             alpha=1, markeredgewidth=.5, markeredgecolor='black');
    plt.plot(y_test, y_pred_te, 'o', color='blue', label=f'test set (# {len(y_test)})',
             alpha=0.9, markeredgewidth=.5, markeredgecolor='black');
    if idxy_tmp!= None and idxy_tmp2!= None:
        plt.plot(np.squeeze(y_train[idxy_tmp[0][idxy_tmp2]]),
                 np.squeeze(y_pred_tr[idxy_tmp[0][idxy_tmp2]]),
                 'o', color='yellow',
                 label=f'MV samples (# {len(y_train[idxy_tmp[0][idxy_tmp2]].tolist()[0])})',
                 alpha=1, markeredgewidth=.5, markeredgecolor='black');
    # Add more text to the legend
    plt.xlim([dlim, 1.05 * ulim]);
    plt.ylim([dlim, 1.05 * ulim])
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    if text != None:
        plt.title(f'Pred. vs. Measured for {par_2} based on {text}', fontweight="bold")
    else:
        plt.title(f'Scatter plot for Var {par_2}', fontweight="bold")
    plt.legend(title=r'$\bf{' + par_1 + '}$', fontsize="7.5")
    plt.grid()
    plt.show()
  
### Func 7 ###
def Intermediate_Plot_PLS1_2(Max_LV,tmp,tmp1,max_iter,cv,estimator, X_train_val,
                             Y_train_val, X_eval, Y_eval, indices, par_1, par_2,
                             ind_11, Y_preprocess_mode, Show1 = None):
    if Show1 is not None:
        num_comp, bin_centers = np.histogram(tmp, bins=np.linspace(np.min(tmp),
                                                                   np.max(tmp), 100))
        num_comp = num_comp / np.sum(num_comp)
        seed_value = 15  # 50#40#20 #0
        np.random.seed(seed_value)
        for ii in range(0, Max_LV):
            tmp4 = tmp[np.argwhere(tmp1 == ii + 1)]
            if len(tmp4) != 0:
                hist11, bin_centers11 = np.histogram(tmp4, bins=np.linspace
                (0.9 * np.min(tmp4),
                 1.1 * np.max(tmp4), 101))
                hist11 = hist11 / np.sum(hist11)
                bar_color = np.random.rand(3, )  # Random RGB color
                border_color = np.random.rand(3, )  # Random RGB color
                plt.bar(bin_centers11[:-1], hist11, width=0.02,
                        color=bar_color, edgecolor=border_color, linewidth=1,
                        label=f'Hist. of LV #{ii + 1}', alpha=0.5);
                # M_RMSECV[ii] = np.mean(comp)
                plt.legend(fontsize='7.45')
        plt.xlabel('Average RMSECVs');
        plt.ylabel('Normalized Frequency of RMSCEVs');
        plt.title(f'Freq. of the RMSECVs among #{max_iter} bootstraps for all No.LVs');
        plt.show()
        bar_color = 'blue'
        border_color = 'lightblue'
        plt.bar(bin_centers[:-1], num_comp, width=0.02,
                color=bar_color, edgecolor=border_color, linewidth=0.1,
                label='Hist. of RMSECVs for all Var.s');
        # plt.hist(tmp , bins = np.linspace(np.min(tmp),np.max(tmp),100),
        #           color = "blue", ec="lightblue", lw=1);
        plt.xlabel('Average RMSECVs');
        plt.ylabel('Frequency of RMSCEVs');
        plt.title(f'Hist. of the Lowest RMSECVs among #{max_iter} bootstraps for All Var.s');
        plt.plot(np.repeat(np.mean(tmp), 100), 1.01 * np.linspace(0, np.max(num_comp), 100),
                 c='r', linestyle='-.', linewidth=2)
        plt.plot(np.repeat(np.std(tmp) + np.mean(tmp), 100), np.linspace(0, 0.75 * np.max(num_comp), 100),
                 c='black', linestyle='--', linewidth=3)
        plt.plot(np.repeat(-np.std(tmp) + np.mean(tmp), 100), np.linspace(0, 0.75 * np.max(num_comp), 100),
                 color='black', linestyle='--', linewidth=3)
        plt.text(np.mean(tmp), np.max(num_comp), '$\mu$', fontsize=10,
                 bbox=dict(facecolor='lightgreen', alpha=0.75))
        plt.text(np.mean(tmp) + 0.7 * np.std(tmp), 0.75 * np.max(num_comp), '$\mu+\sigma$', fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.75))
        plt.text(np.mean(tmp) - 1.3 * np.std(tmp), 0.75 * np.max(num_comp), '$\mu-\sigma$', fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.75))
        plt.show()
    ## PLS2-based prediction
    # LV_no1 = np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
    #                                 bins=range(1, Max_LV + 2))[0].tolist()) + 1
    # LV_no2, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices, Max_LV, 1)
    # LV_no = np.min([LV_no1, LV_no2])
    LV_no, _ = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1, Max_LV, 1)
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
    Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:, indices], y_pred_te[:, indices],
                                    Y_train_val[:, indices], y_pred_tr[:, indices],
                                    rmsep, par_1[ind_11], par_2[ind_11], show_lr='no',
                                    text='PLS2-based imp. Alg.')
    ## PLS1-based prediction
    # LV_no1 = np.argmax(np.histogram(cv['test_num_components_lowest_var' + str(indices + 1) + '_rmse'],
    #                                 bins=range(1, Max_LV + 2))[0].tolist()) + 1
    # _ , LV_no2 = Calc_Opt_LVs_PLS2_PLS1(cv,indices, Max_LV, 1)
    # LV_no = np.min([LV_no1, LV_no2])
    _ , LV_no = Calc_Opt_LVs_PLS2_PLS1(cv,indices+1, Max_LV, 1)
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
    Plot_Measured_Predicted_NMR_PLS(LV_no, Y_eval[:, indices], y_pred_te,
                                    Y_train_val[:, indices], y_pred_tr,
                                    rmsep, par_1[ind_11], par_2[ind_11], show_lr='no',
                                    text='PLS1 Prd. & PLS2-based imput.')
####################################################################################################
### Func_8 ####
def Calc_Opt_LVs_PLS2_PLS1(cv,var_no,Max_LV,no_show):
    LV_all = cv['test_num_components_lowest_mean_rmse']
    rmse_all = cv['test_lowest_mean_rmse'] 
    LV_v6 = cv[f'test_num_components_lowest_var{var_no}_rmse']
    rmse_v6 = cv[f'test_lowest_var{var_no}_rmse']
    id_LV = []
    for ii in range(1,Max_LV+1):
        id_LV.append(np.argwhere(LV_all==ii))
    rmse = np.zeros((Max_LV,1))
    for ii in range(Max_LV):
        rmse[ii] = np.mean(rmse_all[id_LV[ii]])
    non_nan_indices = [index for index, value in enumerate(rmse) if not np.isnan(value)]
    rmse = rmse[non_nan_indices]
    Best_LV_PLS2 = non_nan_indices[np.argmin(rmse)]+1
    if no_show!=1:
        plt.plot(np.array(non_nan_indices)+1, rmse);
        # plt.xticks(np.array(non_nan_indices)+1)
        plt.title('RMSE vs. LVs for all uc-variables based on PLS2 Implementation')
        plt.xlabel('LVs');plt.ylabel('RMSE');
        plt.plot(Best_LV_PLS2,rmse[np.argmin(rmse)],'o',color='green')
        plt.show()
        #######################################################################
        ## Create a dictionary to store LVs as keys and their corresponding RMSE values as values
        lv_to_rmse = {}
        for lv, rmse1 in zip(LV_all, rmse_all):
            if lv not in lv_to_rmse:
                lv_to_rmse[lv] = []
            lv_to_rmse[lv].append(rmse1)
        LV_id = np.sort([ii for ii in lv_to_rmse.keys()])
        lv_2_rmse = []
        for ii in LV_id:
            lv_2_rmse.append(lv_to_rmse[ii])
        mean_val = [np.mean(ii) for ii in lv_2_rmse]
        #######
        data = lv_2_rmse[np.argmin(rmse)]
        m1 = np.mean(data)
        upper_quartile = np.percentile(data, 75)
        lower_quartile = np.percentile(data, 25)
        iqr = upper_quartile - lower_quartile
        idx1=data<=upper_quartile+1.5*iqr
        upper_whisker = np.max([data[ii] for ii,jj in enumerate(idx1) if jj])
        idx1= data>=lower_quartile-1.5*iqr
        lower_whisker = np.min([data[ii] for ii,jj in enumerate(idx1) if jj])
        # np.arange(m1-0.1*m1,m1+0.1*m1,0.02*m1)
        #######################################################################
        m1 = Best_LV_PLS2
        xx = np.arange(m1-1e-2*m1,m1+1e-2*m1,1e-3*m1)
        if lower_whisker!=upper_whisker:
            yy = np.arange(lower_whisker, upper_whisker,
                           (upper_whisker-lower_whisker)*.02)
        else:
            lower_whisker = lower_whisker*0.9
            upper_whisker = upper_whisker*1.1
            yy = np.arange(lower_whisker, upper_whisker,
                           (upper_whisker-lower_whisker)*.02)   
        plt.plot(xx,np.ones((len(xx),1))*upper_whisker,'r')
        plt.plot(xx,np.ones((len(xx),1))*lower_whisker,'r')
        plt.plot(np.ones((len(yy),1))*xx[0], yy,'r')
        plt.plot(np.ones((len(yy),1))*xx[-1], yy,'r')
        #######################################################################
        # Create a boxplot with hidden boxes
        plt.boxplot(lv_2_rmse, positions=LV_id, vert=True, 
                    patch_artist=False, boxprops={'linewidth': 0.1}, 
                    medianprops={'linewidth':0}, showmeans=True)
                    # , flierprops={'linewidth':0.1},showfliers=True)  # Hide boxes
        plt.plot(LV_id,mean_val)
        plt.plot()
        #plt.xticks(LV_id)
        # Customize the appearance of the whiskers
        plt.xticks([])
        plt.title(f'RMSE vs. LVs for uv-var #{var_no} based on PLS2 Implementation')
        plt.xlabel('LVs');plt.ylabel('RMSE');
        # Show the plot
        plt.show()
        print('PLS2 results Based on all bootstraps:',non_nan_indices[np.argmin(rmse)]+1)
        print('PLS2 results Based on most frequent values:', np.argmax(np.histogram(cv['test_num_components_lowest_mean_rmse'],
                                       bins=range(1, Max_LV + 2))[0].tolist()) + 1)
    #########################################################
    id_LV_v6 = []
    for ii in range(1,Max_LV+1):
        id_LV_v6.append(np.argwhere(LV_v6==ii))
    rmsev6 = np.zeros((Max_LV,1))
    for ii in range(Max_LV):
        rmsev6[ii] = np.mean(rmse_v6[id_LV_v6[ii]])
    non_nan_indices = [index for index, value in enumerate(rmsev6) if not np.isnan(value)]
    rmsev6 = rmsev6[non_nan_indices]
    Best_LV_PLS1 = non_nan_indices[np.argmin(rmsev6)]+1
    if no_show !=1:
        plt.plot(np.array(non_nan_indices)+1, rmsev6,'red');
        plt.title(f'RMSE vs. LVs for uv-var #{var_no} based on PLS1 Implementation')
        plt.xlabel('LVs');plt.ylabel('RMSE');
        plt.plot(Best_LV_PLS1,rmsev6[np.argmin(rmsev6)],'o',color='green')
        plt.show()
        #######################################################################
        ## Create a dictionary to store LVs as keys and their corresponding RMSE values as values
        lv_to_rmse = {}
        for lv, rmse1 in zip(LV_v6, rmse_v6):
            if lv not in lv_to_rmse:
                lv_to_rmse[lv] = []
            lv_to_rmse[lv].append(rmse1)
        LV_id = np.sort([ii for ii in lv_to_rmse.keys()])
        LV_id = LV_id.astype(int)
        lv_2_rmse = []
        for ii in LV_id:
            lv_2_rmse.append(lv_to_rmse[ii])
        mean_val = [np.mean(ii) for ii in lv_2_rmse]
        #######
        data = (lv_2_rmse[np.argmin(rmsev6)])
        m1 = np.mean(data)
        upper_quartile = np.percentile(data, 75)
        lower_quartile = np.percentile(data, 25)
        iqr = upper_quartile - lower_quartile
        idx1=data<=upper_quartile+1.5*iqr
        upper_whisker = np.max([data[ii] for ii,jj in enumerate(idx1) if jj])
        idx1= data>=lower_quartile-1.5*iqr
        lower_whisker = np.min([data[ii] for ii,jj in enumerate(idx1) if jj])
        # np.arange(m1-0.1*m1,m1+0.1*m1,0.02*m1)
        #######################################################################
        m1 = Best_LV_PLS1
        xx = np.arange(m1-1e-2*m1,m1+1e-2*m1,1e-3*m1)
        if lower_whisker!=upper_whisker:
            yy = np.arange(lower_whisker, upper_whisker,
                           (upper_whisker-lower_whisker)*.02)
        else:
            lower_whisker = lower_whisker*0.9
            upper_whisker = upper_whisker*1.1
            yy = np.arange(lower_whisker, upper_whisker,
                           (upper_whisker-lower_whisker)*.02)   
        plt.plot(xx,np.ones((len(xx),1))*upper_whisker,'r')
        plt.plot(xx,np.ones((len(xx),1))*lower_whisker,'r')
        plt.plot(np.ones((len(yy),1))*xx[0], yy,'r')
        plt.plot(np.ones((len(yy),1))*xx[-1], yy,'r')
        #######################################################################       
        # Create a boxplot with hidden boxes
        plt.boxplot(lv_2_rmse, positions=LV_id, vert=True, 
                    patch_artist=False, boxprops={'linewidth': 0.1}, 
                    medianprops={'linewidth':0}, showmeans=True)
                    # , flierprops={'linewidth':0.1},showfliers=True)  # Hide boxes
        plt.plot(LV_id,mean_val)
        # plt.xticks(LV_id)
        plt.xticks([])
        # Customize the appearance of the whiskers
        plt.title(f'RMSE vs. LVs for uv-var #{var_no} based on PLS1 Implementation')
        plt.xlabel('LVs');plt.ylabel('RMSE');
        # plt.axis([range(LV_id) ])
        # plt.ylim(1.2, 2.25)
        # Show the plot
        plt.show()
        print('PLS1 results Based on all bootstraps:',Best_LV_PLS1)
        print('PLS1 results Based on most frequent values:',np.argmax(np.histogram(cv[f'test_num_components_lowest_var{var_no}_rmse'],
                                       bins=range(1, Max_LV + 2))[0].tolist()) + 1)
    return Best_LV_PLS2, Best_LV_PLS1

#### Func_9 ####
def Plot_Measured_Predicted_NMR_new(y_test, y_pred_te, y_train,
                                   y_pred_tr, rmsep, par_1, par_2,
                                   LV_no=None, rmsecv=None, text=None, 
                                   idxy_tmp=None, idxy_tmp2=None):
    ulim = np.max([np.max(y_train), np.max(y_pred_tr), np.max(y_test), np.max(y_pred_te)]) * 1.1
    # dlim = np.min([np.min(y_train),np.min(y_pred_tr),np.min(y_test),np.min(y_pred_te)])*0.9
    dlim = -5e-1
    # Calculate the Q^2 factor using the r2_score function
    q2_p = Q2(y_test, y_pred_te, multioutput='variance_weighted')
    ## Calculate the coefficients of variation of the test set prediction model
    cv1 = 100 * rmsep / np.mean(y_test)
    if LV_no!=None:
        plt.plot([], [], ' ', label='LVs={}'.format(LV_no))
    plt.plot([], [], ' ', label='RMSEP={}'.format(np.round(rmsep * 1e4) / 1e4))
    if rmsecv != None:
        plt.plot([], [], ' ', label='RMSECV={}'.format(np.round(rmsecv * 1e4) / 1e4))
    plt.plot([], [], ' ', label='Q$^{{2}}$ ={}'.format(np.round(q2_p * 1e4) / 1e4))
    plt.plot([], [], ' ', label='CV ={}'.format(np.round(cv1 * 1e4) / 1e4))
    plt.plot(np.linspace(dlim, ulim, 100), np.linspace(dlim, ulim, 100), linestyle='--', linewidth=2, c='red',
             alpha=0.9);
    plt.plot(y_train, y_pred_tr, 'o', color='red', label=f'train set (# {len(y_train)})',
             alpha=1, markeredgewidth=.5, markeredgecolor='black');
    plt.plot(y_test, y_pred_te, 'o', color='blue', label=f'test set (# {len(y_test)})',
             alpha=0.75, markeredgewidth=.5, markeredgecolor='black');
    if idxy_tmp!= None and idxy_tmp2!= None:
        plt.plot(np.squeeze(y_train[idxy_tmp[0][idxy_tmp2]]),
                 np.squeeze(y_pred_tr[idxy_tmp[0][idxy_tmp2]]),
                 'o', color='yellow',
                 label=f'MV samples (# {len(y_train[idxy_tmp[0][idxy_tmp2]].tolist()[0])})',
                 alpha=1, markeredgewidth=.5, markeredgecolor='black');
    # Add more text to the legend
    plt.xlim([dlim, 1.05 * ulim]);
    plt.ylim([dlim, 1.05 * ulim])
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    if text != None:
        plt.title(f'Pred. vs. Measured for {par_2} based on {text}', fontweight="bold")
    else:
        plt.title(f'Scatter plot for Var {par_2}', fontweight="bold")
    plt.legend(title=r'$\bf{' + par_1 + '}$', fontsize="10")
    plt.grid()
    plt.show()