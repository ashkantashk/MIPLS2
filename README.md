# PLS2-Based Imputation Method

# Introduction
This repository contains code and experimental data for a baseline PLS2-based imputation algorithm.

<p align="justify">
The current Code provides chemometricians the ability to predict and impute the missing values existing in the measurements of target variables utilizing other mutual data such as spectroscopy measurements. If there are two multivariate datasets constructed based on two different measurements of similar samples or populations, then in the case that there are some unknown measurements or labile and unstable amounts in one of these two datasets, the current tool provides this ability that the missing values belonging to different target variables can be predicted and imputed efficiently and with the lowest rmsep using the other flawless dataset. 
</p>

![Fig_PLS2_sample_Application_new](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/5b62eee7-70d4-4db6-b28d-c6c57bda8861)


The codes are available at both the main and the Revised_Versions Branches of the current repository. 

# Scripts
1- Baseline_PLS2_Imputation_Functions_new.py
<p align="justify">
This Python code script contains the code lines for implementing the baseline PLS2 model and the trials for figuring out the impact of missing values on the covariance relationship between the prediction results.

## Running Codes
> All the codes have been tested on Windows 10 using Anaconda.

> For running the code, you should first install all the libraries and frameworks mentioned in the requirements.txt file. For this goal, you just need to run the following code in your command window for your active environment:
>
>     pip install --user -r requirements.txt

> The other codes in this repository are:
> 1. 
> 2. 

* Note1: .
* Note2: .
* Note3: .
* Note4: .

</p>

## Descriptions: 
<p align="justify">
The are also two scenarios for the written code. In the first scenario, I calculated the normalized RMSEC (RMSECN) and RMSECV (RMSECVN) for all uc variables (65 ultracentrifugation variables) by dividing the original RMSEs by the median of each variable and then calculated the average of each RMSE for each no. LVs. 
The only difference between the first and second scenarios is that in the second scenario, the normalized RMSEs for each sample are calculated in the absence of the missing values (MVs) for that sample. In the end, the procedure is completely like the one in the first scenario.
</p>

![Fig1](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/89ca87c1-12a9-4dab-abef-3caecdc5e08c)

Fig. 1. Results of Normalized RMSEC and RMSECV (% per Avg. for each uc variable) per. LVs ∈ [1,24] in the presence of MVs 

![Fig2](https://github.com/ashkantashk/Baseline_PLS2_Model/assets/53473481/45a92290-520e-4045-bc3b-63c2d6b6511f)

Fig. 2. Results of Normalized RMSEC and RMSECV (% per Avg. for each uc variable) per. LVs ∈ [1,24] by excluding MVs
