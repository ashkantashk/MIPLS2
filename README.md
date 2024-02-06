# Baseline_PLS2_Model

# Introduction
This repository contains code and experimental data for a baseline PLS2-based imputation algorithm.

<!--The directory, Original_data, contains the original NMR spectroscopy and ultracentrifugation (UC) measurements of the LipoProtein (LP) fractions and subfraction in human blood. 
The NMR spectra are from 1.4 to 0.6 p.p.m. range. The spectra originally contained 316 uniformly distributed samples and the spectral length is 3488. The UC dataset contains some values with no reliable references and therefore should be considered as missing values (MVs). The MVs are replaced with zero values in the current dataset. -->
<p align="justify">
The current Code provides chemometricians the ability to predict and impute the missing values existing in the measurements of target variables utilizing other mutual data such as spectroscopy measurements. If there are two multivariate datasets constructed based on two different measurements of similar samples or populations, then in the case that there are some unknown measurements or labile and unstable amounts in one of these two datasets, the current tool provides this ability that the missing values belonging to different target variables can be predicted and imputed efficiently and with the lowest rmsep. 
</p>
<!--![X_Y_Data](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/3e50bcda-5d95-49c0-8375-a29de74cc810)-->

The codes are available at both the main and the Revised_Versions Branches of the current repository. 

# Scripts
1- <!--BaseLine_PLS2_based_Imputation_method_V03_Klavs_Suggestions.py:-->
<p align="justify">
This Python file includes the code lines for implementing the baseline PLS2 model and the trials for figuring out the impact of missing values on the covariance relationship between the prediction results.
The defined function Baseline_PLS2_Modeling_for_Calc_Normalized_RMSEs in the above code observes the aforementioned study of the missing values. 
Descriptions: 
The are also two scenarios for the written code. In the first scenario, I calculated the normalized RMSEC (RMSECN) and RMSECV (RMSECVN) for all uc variables (65 ultracentrifugation variables) by dividing the original RMSEs by the median of each variable and then calculated the average of each RMSE for each no. LVs. 
The only difference between the first and second scenarios is that in the second scenario, the normalized RMSEs for each sample are calculated in the absence of the missing values (MVs) for that sample. In the end, the procedure is completely like the one in the first scenario.
</p>

<p align="center">

![Fig1](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/89ca87c1-12a9-4dab-abef-3caecdc5e08c)
Fig. 1. Results of Normalized RMSEC and RMSECV (% per Avg. for each uc variable) per. LVs ∈ [1,24] in the presence of MVs 
![Fig2](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/45a92290-520e-4045-bc3b-63c2d6b6511f)
Fig. 2. Results of Normalized RMSEC and RMSECV (% per Avg. for each uc variable) per. LVs ∈ [1,24] by excluding MVs
</p>
