# Multivariate Imputation by PLS2 (MIPLS2)

# Introduction
This repository contains code and experimental data for a baseline PLS2-based imputation algorithm.

<p align="justify">
The current Code provides chemometricians the ability to predict and impute the missing values existing in the measurements of target variables utilizing other mutual data such as spectroscopy measurements. If there are two multivariate datasets constructed based on two different measurements of similar samples or populations, then in the case that there are some unknown measurements or labile and unstable amounts in one of these two datasets, the current tool provides this ability that the missing values belonging to different target variables can be predicted and imputed efficiently and with the lowest rmsep using the other flawless dataset. 
</p>

![image](https://github.com/ashkantashk/MIPLS2/assets/53473481/d00a7829-e66e-4360-8585-c08dd16d08c9)
Figure 1. Graphical visualization of the proposed PLS2-based imputation

The codes are available at the current repository's main and the Revised_Versions Branches. 


## Running Codes
<p align="justify">
> All the codes have been tested on Windows 10 using Anaconda.

> For running the code, you should first install all the libraries and frameworks mentioned in the requirements.txt file. For this goal, you just need to run the following code in your command window for your active environment:
>
>     pip install --user -r requirements.txt

> The other codes in this repository are:

> 1. older version of the code with fewer list of input arguments 

* Note 1: The code contains essential functions for its complete and flawless running.
* Note 2: The hyperparameters are adjusted and tuned based on the implementation results of the evaluation datasets.
  
# Scripts
1- Baseline_PLS2_Imputation_Functions_v7.py
<p align="justify">
This Python script contains the code lines for implementing the baseline PLS2 model and the trials for figuring out the impact of missing values on the covariance relationship between the prediction results.

</p>

## Descriptions

<p align="justify">
The proposed PLS2-based imputation method has an implementation based on the visualized flow block diagram in Figure 2:
</p>

![image](https://github.com/ashkantashk/MIPLS2/assets/53473481/b99e111b-6477-4b70-a01d-01ca371c69d4)

<p align="justify">
Figure 2. Functional flow block diagram for the two phases of the proposed PLS2-based imputation algorithm
</p>
<p align="justify">
</p>
