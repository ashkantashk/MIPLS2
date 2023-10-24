# Baseline_PLS2_Model

# Introduction
This repository contains code and experimental data for a baseline PLS2-based imputation algorithm.

The directory, Original_data, contains the original NMR spectroscopy and ultracentrifugation (UC) measurements of the LipoProtein (LP) fractions and subfraction in human blood. 
The NMR spectra are from 1.4 to 0.6 p.p.m. range. The spectra originally contained 316 uniformly distributed samples and the spectral length is 3488. The UC dataset contains some values with no reliable references and therefore should be considered as missing values (MVs). The MVs are replaced with zero values in the current dataset. 

The codes will be added to the main directory ASAP. 

# Scripts
1- BaseLine_PLS2_based_Imputation_method_V03_Klavs_Suggestions.py:
This Python file includes the code lines for implementing the baseline PLS2 model and the trials for figuring out the impact of missing values on the covariance relationship between the prediction results.
The defined function Baseline_PLS2_Modeling_for_Calc_Normalized_RMSEs in the above code observes the aforementioned study of the missing values. 
Descriptions: 
The are also two scenarios for the written code. In the first scenario, I calculated the normalized RMSEC (RMSECN) and RMSECV (RMSECVN) for all uc variables (65 ultracentrifugation variables) by dividing the original RMSEs by the median of each variable and then calculated the average of each RMSE for each no. LVs. 
The only difference between the first and second scenarios is that in the second scenario, the normalized RMSEs for each sample are calculated in the absence of the missing values (MVs) for that sample. In the end, the procedure is completely like the one in the first scenario.

![image](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/acd2cc2d-0498-488e-b4e2-470e69865faa)
Fig. 1. Results of Normalized RMSEC and RMSECV (% per Avg. for each uc variable) per. LVs ∈ [1,24] in the presence of MVs

![image](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/e20aa68e-0764-4ded-8d55-50e3a2263e1b)
Fig. 2. Results of Normalized RMSEC and RMSECV (% per Avg. for each uc variable) per. LVs ∈ [1,24] by excluding MVs

