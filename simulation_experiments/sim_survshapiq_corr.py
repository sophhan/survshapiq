# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
import shapiq
from sklearn.model_selection import train_test_split
import importlib
import sys

from survshapiq.survshapiq_repo.simulation.sim_survshapiq import X_test_linear_tdinter
sys.path.append("/survshapiq/")
import simulation.func as func
importlib.reload(func)

import os
print(os.getcwd())

# define paths
path_data = "/survshapiq/simulation/data"
path_plots = "/survshapiq/simulation/plots_corr"
path_plots_combined = "/survshapiq/simulation/plots_combined"

#---------------------------
# 1) Linear G(t|x), TI (no interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_1_0= pd.read_csv(f"{path_data}/1_simdata_linear_ti.csv")
print(simdata_1_0.head())

# convert eventtime and status columns to a structured array
data_y_1_0, data_x_1_0_df = func.prepare_survival_data(simdata_1_0)
print(data_y_1_0)
print(data_x_1_0_df.head())
data_x_1_0 = data_x_1_0_df.values
X_train_1_0, X_test_1_0, y_train_1_0, y_test_1_0 = train_test_split(
    data_x_1_0, data_y_1_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_1_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_1_0.fit(X_train_1_0, y_train_1_0)
print(f'C-index (train): {model_gbsa_1_0.score(X_test_1_0, y_test_1_0).item():0.3f}')
ibs_gbsa_1_0 = func.compute_integrated_brier(y_test_1_0, X_test_1_0, model_gbsa_1_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_1_0:0.3f}')

# fit CoxPH
model_cox_1_0 = CoxPHSurvivalAnalysis()
model_cox_1_0.fit(X_train_1_0, y_train_1_0)
print(f'C-index (train): {model_cox_1_0.score(X_test_1_0, y_test_1_0).item():0.3f}')
ibs_cox_1_0 = func.compute_integrated_brier(y_test_1_0, X_test_1_0, model_cox_1_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_1_0:0.3f}')

# create data point for explanation
idx =  7
x_new = data_x_1_0[[idx]]
print(x_new)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_1_0_loghaz = func.survshapiq_ground_truth(data_x_1_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti, 
                                                            times=model_gbsa_1_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_1_0_df.columns)

func.plot_interact(explanations_all = explanation_1_0_loghaz, 
                              model = None,
                              times=model_gbsa_1_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_1_0.pdf",
                              data_x = data_x_1_0,
                              survival_fn = func.log_hazard_wrap_linear_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 1) Linear G(t|x), TI (no interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_1_09 = pd.read_csv(f"{path_data}/1_simdata_linear_09.csv")
print(simdata_1_09.head())
simdata_1_09

# convert eventtime and status columns to a structured array
data_y_1_09, data_x_1_09_df = func.prepare_survival_data(simdata_1_09)
print(data_y_1_09)
print(data_x_1_09_df.head())
data_x_1_09 = data_x_1_09_df.values
X_train_1_09, X_test_1_09, y_train_1_09, y_test_1_09 = train_test_split(
    data_x_1_09, data_y_1_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_1_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_1_09.fit(X_train_1_09, y_train_1_09)
print(f'C-index (train): {model_gbsa_1_09.score(X_test_1_09, y_test_1_09).item():0.3f}')
ibs_gbsa_1_09 = func.compute_integrated_brier(y_test_1_09, X_test_1_09, model_gbsa_1_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_1_09:0.3f}')

# fit CoxPH
model_cox_1_09 = CoxPHSurvivalAnalysis()
model_cox_1_09.fit(X_train_1_09, y_train_1_09)
print(f'C-index (train): {model_cox_1_09.score(X_test_1_09, y_test_1_09).item():0.3f}')
ibs_cox_1_09 = func.compute_integrated_brier(y_test_1_09, X_test_1_09, model_cox_1_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_1_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_1_09 = data_x_1_09[[idx]]
#print(x_new_1_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_1_09_loghaz = func.survshapiq_ground_truth(data_x_1_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti, 
                                                            times=model_gbsa_1_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_1_09_df.columns)


func.plot_interact(explanations_all = explanation_1_09_loghaz, 
                              model = None,
                              times=model_gbsa_1_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_1_09.pdf",
                              data_x = data_x_1_09,
                              survival_fn = func.log_hazard_wrap_linear_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 1) Linear G(t|x), TI (no interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_1_05 = pd.read_csv(f"{path_data}/1_simdata_linear_05.csv")
print(simdata_1_05.head())
simdata_1_05

# convert eventtime and status columns to a structured array
data_y_1_05, data_x_1_05_df = func.prepare_survival_data(simdata_1_05)
print(data_y_1_05)
print(data_x_1_05_df.head())
data_x_1_05 = data_x_1_05_df.values
X_train_1_05, X_test_1_05, y_train_1_05, y_test_1_05 = train_test_split(
    data_x_1_05, data_y_1_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_1_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_1_05.fit(X_train_1_05, y_train_1_05)
print(f'C-index (train): {model_gbsa_1_05.score(X_test_1_05, y_test_1_05).item():0.3f}')
ibs_gbsa_1_05 = func.compute_integrated_brier(y_test_1_05, X_test_1_05, model_gbsa_1_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_1_05:0.3f}')

# fit CoxPH
model_cox_1_05= CoxPHSurvivalAnalysis()
model_cox_1_05.fit(X_train_1_05, y_train_1_05)
print(f'C-index (train): {model_cox_1_05.score(X_test_1_05, y_test_1_05).item():0.3f}')
ibs_cox_1_05 = func.compute_integrated_brier(y_test_1_05, X_test_1_05, model_cox_1_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_1_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_1_05 = data_x_1_05[[idx]]
#print(x_new_1_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_1_05_loghaz = func.survshapiq_ground_truth(data_x_1_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti, 
                                                            times=model_gbsa_1_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_1_05_df.columns)


func.plot_interact(explanations_all = explanation_1_05_loghaz, 
                              model = None,
                              times=model_gbsa_1_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_1_05.pdf",
                              data_x = data_x_1_05,
                              survival_fn = func.log_hazard_wrap_linear_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 1) Linear G(t|x), TI (no interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_1_02 = pd.read_csv(f"{path_data}/1_simdata_linear_02.csv")
print(simdata_1_02.head())
simdata_1_02

# convert eventtime and status columns to a structured array
data_y_1_02, data_x_1_02_df = func.prepare_survival_data(simdata_1_02)
print(data_y_1_02)
print(data_x_1_02_df.head())
data_x_1_02 = data_x_1_02_df.values
X_train_1_02, X_test_1_02, y_train_1_02, y_test_1_02 = train_test_split(
    data_x_1_02, data_y_1_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_1_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_1_02.fit(X_train_1_02, y_train_1_02)
print(f'C-index (train): {model_gbsa_1_02.score(X_test_1_02, y_test_1_02).item():0.3f}')
ibs_gbsa_1_02 = func.compute_integrated_brier(y_test_1_02, X_test_1_02, model_gbsa_1_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_1_02:0.3f}')

# fit CoxPH
model_cox_1_02 = CoxPHSurvivalAnalysis()
model_cox_1_02.fit(X_train_1_02, y_train_1_02)
print(f'C-index (train): {model_cox_1_02.score(X_test_1_02, y_test_1_02).item():0.3f}')
ibs_cox_1_02 = func.compute_integrated_brier(y_test_1_02, X_test_1_02, model_cox_1_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_1_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_1_02 = data_x_1_02[[idx]]
#print(x_new_1_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_1_02_loghaz = func.survshapiq_ground_truth(data_x_1_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti, 
                                                            times=model_gbsa_1_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_1_02_df.columns)
explanation_1_02_loghaz[0].mean()
explanation_1_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_1_02_loghaz, 
                              model = None,
                              times=model_gbsa_1_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_1_02.pdf",
                              data_x = data_x_1_02,
                              survival_fn = func.log_hazard_wrap_linear_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 2) Linear G(t|x), TD MAIN (no interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_2_0= pd.read_csv(f"{path_data}/2_simdata_linear_tdmain.csv")
print(simdata_2_0.head())

# convert eventtime and status columns to a structured array
data_y_2_0, data_x_2_0_df = func.prepare_survival_data(simdata_2_0)
print(data_y_2_0)
print(data_x_2_0_df.head())
data_x_2_0 = data_x_2_0_df.values
X_train_2_0, X_test_2_0, y_train_2_0, y_test_2_0 = train_test_split(
    data_x_2_0, data_y_2_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_2_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_2_0.fit(X_train_2_0, y_train_2_0)
print(f'C-index (train): {model_gbsa_2_0.score(X_test_2_0, y_test_2_0).item():0.3f}')
ibs_gbsa_2_0 = func.compute_integrated_brier(y_test_2_0, X_test_2_0, model_gbsa_2_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_2_0:0.3f}')

# fit CoxPH
model_cox_2_0 = CoxPHSurvivalAnalysis()
model_cox_2_0.fit(X_train_2_0, y_train_2_0)
print(f'C-index (train): {model_cox_2_0.score(X_test_2_0, y_test_2_0).item():0.3f}')
ibs_cox_2_0 = func.compute_integrated_brier(y_test_2_0, X_test_2_0, model_cox_2_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_2_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_2_0 = data_x_2_0[[idx]]
#print(x_new_2_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_2_0_loghaz = func.survshapiq_ground_truth(data_x_2_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_2_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_2_0_df.columns)

func.plot_interact(explanations_all = explanation_2_0_loghaz, 
                              model = None,
                              times=model_gbsa_2_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_2_0.pdf",
                              data_x = data_x_2_0,
                              survival_fn = func.log_hazard_wrap_linear_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 2) Linear G(t|x), TD MAIN (no interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_2_09 = pd.read_csv(f"{path_data}/2_simdata_linear_tdmain_09.csv")
print(simdata_2_09.head())
simdata_2_09

# convert eventtime and status columns to a structured array
data_y_2_09, data_x_2_09_df = func.prepare_survival_data(simdata_2_09)
print(data_y_2_09)
print(data_x_2_09_df.head())
data_x_2_09 = data_x_2_09_df.values
X_train_2_09, X_test_2_09, y_train_2_09, y_test_2_09 = train_test_split(
    data_x_2_09, data_y_2_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_2_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_2_09.fit(X_train_2_09, y_train_2_09)
print(f'C-index (train): {model_gbsa_2_09.score(X_test_2_09, y_test_2_09).item():0.3f}')
ibs_gbsa_2_09 = func.compute_integrated_brier(y_test_2_09, X_test_2_09, model_gbsa_2_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_2_09:0.3f}')

# fit CoxPH
model_cox_2_09 = CoxPHSurvivalAnalysis()
model_cox_2_09.fit(X_train_2_09, y_train_2_09)
print(f'C-index (train): {model_cox_2_09.score(X_test_2_09, y_test_2_09).item():0.3f}')
ibs_cox_2_09 = func.compute_integrated_brier(y_test_2_09, X_test_2_09, model_cox_2_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_2_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_2_09 = data_x_2_09[[idx]]
#print(x_new_2_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_2_09_loghaz = func.survshapiq_ground_truth(data_x_2_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_2_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_2_09_df.columns)


func.plot_interact(explanations_all = explanation_2_09_loghaz, 
                              model = None,
                              times=model_gbsa_2_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_2_09.pdf",
                              data_x = data_x_2_09,
                              survival_fn = func.log_hazard_wrap_linear_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 2) Linear G(t|x), TD MAIN (no interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_2_05 = pd.read_csv(f"{path_data}/2_simdata_linear_tdmain_05.csv")
print(simdata_2_05.head())
simdata_2_05

# convert eventtime and status columns to a structured array
data_y_2_05, data_x_2_05_df = func.prepare_survival_data(simdata_2_05)
print(data_y_2_05)
print(data_x_2_05_df.head())
data_x_2_05 = data_x_2_05_df.values
X_train_2_05, X_test_2_05, y_train_2_05, y_test_2_05 = train_test_split(
    data_x_2_05, data_y_2_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_2_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_2_05.fit(X_train_2_05, y_train_2_05)
print(f'C-index (train): {model_gbsa_2_05.score(X_test_2_05, y_test_2_05).item():0.3f}')
ibs_gbsa_2_05 = func.compute_integrated_brier(y_test_2_05, X_test_2_05, model_gbsa_2_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_2_05:0.3f}')

# fit CoxPH
model_cox_2_05= CoxPHSurvivalAnalysis()
model_cox_2_05.fit(X_train_2_05, y_train_2_05)
print(f'C-index (train): {model_cox_2_05.score(X_test_2_05, y_test_2_05).item():0.3f}')
ibs_cox_2_05 = func.compute_integrated_brier(y_test_2_05, X_test_2_05, model_cox_2_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_2_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_2_05 = data_x_2_05[[idx]]
#print(x_new_2_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_2_05_loghaz = func.survshapiq_ground_truth(data_x_2_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_2_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_2_05_df.columns)


func.plot_interact(explanations_all = explanation_2_05_loghaz, 
                              model = None,
                              times=model_gbsa_2_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_2_05.pdf",
                              data_x = data_x_2_05,
                              survival_fn = func.log_hazard_wrap_linear_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 2) Linear G(t|x), TD MAIN (no interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_2_02 = pd.read_csv(f"{path_data}/2_simdata_linear_tdmain_02.csv")
print(simdata_2_02.head())
simdata_2_02

# convert eventtime and status columns to a structured array
data_y_2_02, data_x_2_02_df = func.prepare_survival_data(simdata_2_02)
print(data_y_2_02)
print(data_x_2_02_df.head())
data_x_2_02 = data_x_2_02_df.values
X_train_2_02, X_test_2_02, y_train_2_02, y_test_2_02 = train_test_split(
    data_x_2_02, data_y_2_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_2_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_2_02.fit(X_train_2_02, y_train_2_02)
print(f'C-index (train): {model_gbsa_2_02.score(X_test_2_02, y_test_2_02).item():0.3f}')
ibs_gbsa_2_02 = func.compute_integrated_brier(y_test_2_02, X_test_2_02, model_gbsa_2_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_2_02:0.3f}')

# fit CoxPH
model_cox_2_02 = CoxPHSurvivalAnalysis()
model_cox_2_02.fit(X_train_2_02, y_train_2_02)
print(f'C-index (train): {model_cox_2_02.score(X_test_2_02, y_test_2_02).item():0.3f}')
ibs_cox_2_02 = func.compute_integrated_brier(y_test_2_02, X_test_2_02, model_cox_2_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_2_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_2_02 = data_x_2_02[[idx]]
#print(x_new_2_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_2_02_loghaz = func.survshapiq_ground_truth(data_x_2_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_2_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_2_02_df.columns)
explanation_2_02_loghaz[0].mean()
explanation_2_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_2_02_loghaz, 
                              model = None,
                              times=model_gbsa_2_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_2_02.pdf",
                              data_x = data_x_2_02,
                              survival_fn = func.log_hazard_wrap_linear_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

#---------------------------
# 3) Linear G(t|x), TI (interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_3_0= pd.read_csv(f"{path_data}/3_simdata_linear_ti_inter.csv")
print(simdata_3_0.head())

# convert eventtime and status columns to a structured array
data_y_3_0, data_x_3_0_df = func.prepare_survival_data(simdata_3_0)
print(data_y_3_0)
print(data_x_3_0_df.head())
data_x_3_0 = data_x_3_0_df.values
X_train_3_0, X_test_3_0, y_train_3_0, y_test_3_0 = train_test_split(
    data_x_3_0, data_y_3_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_3_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_3_0.fit(X_train_3_0, y_train_3_0)
print(f'C-index (train): {model_gbsa_3_0.score(X_test_3_0, y_test_3_0).item():0.3f}')
ibs_gbsa_3_0 = func.compute_integrated_brier(y_test_3_0, X_test_3_0, model_gbsa_3_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_3_0:0.3f}')

# fit CoxPH
model_cox_3_0 = CoxPHSurvivalAnalysis()
model_cox_3_0.fit(X_train_3_0, y_train_3_0)
print(f'C-index (train): {model_cox_3_0.score(X_test_3_0, y_test_3_0).item():0.3f}')
ibs_cox_3_0 = func.compute_integrated_brier(y_test_3_0, X_test_3_0, model_cox_3_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_3_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_3_0 = data_x_3_0[[idx]]
#print(x_new_3_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_3_0_loghaz = func.survshapiq_ground_truth(data_x_3_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti_inter, 
                                                            times=model_gbsa_3_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_3_0_df.columns)

func.plot_interact(explanations_all = explanation_3_0_loghaz, 
                              model = None,
                              times=model_gbsa_3_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_3_0.pdf",
                              data_x = data_x_3_0,
                              survival_fn = func.log_hazard_wrap_linear_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 3) Linear G(t|x), TI (interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_3_09 = pd.read_csv(f"{path_data}/3_simdata_linear_ti_inter_09.csv")
print(simdata_3_09.head())
simdata_3_09

# convert eventtime and status columns to a structured array
data_y_3_09, data_x_3_09_df = func.prepare_survival_data(simdata_3_09)
print(data_y_3_09)
print(data_x_3_09_df.head())
data_x_3_09 = data_x_3_09_df.values
X_train_3_09, X_test_3_09, y_train_3_09, y_test_3_09 = train_test_split(
    data_x_3_09, data_y_3_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_3_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_3_09.fit(X_train_3_09, y_train_3_09)
print(f'C-index (train): {model_gbsa_3_09.score(X_test_3_09, y_test_3_09).item():0.3f}')
ibs_gbsa_3_09 = func.compute_integrated_brier(y_test_3_09, X_test_3_09, model_gbsa_3_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_3_09:0.3f}')

# fit CoxPH
model_cox_3_09 = CoxPHSurvivalAnalysis()
model_cox_3_09.fit(X_train_3_09, y_train_3_09)
print(f'C-index (train): {model_cox_3_09.score(X_test_3_09, y_test_3_09).item():0.3f}')
ibs_cox_3_09 = func.compute_integrated_brier(y_test_3_09, X_test_3_09, model_cox_3_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_3_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_3_09 = data_x_3_09[[idx]]
#print(x_new_3_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_3_09_loghaz = func.survshapiq_ground_truth(data_x_3_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti_inter, 
                                                            times=model_gbsa_3_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_3_09_df.columns)


func.plot_interact(explanations_all = explanation_3_09_loghaz, 
                              model = None,
                              times=model_gbsa_3_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_3_09.pdf",
                              data_x = data_x_3_09,
                              survival_fn = func.log_hazard_wrap_linear_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 3) Linear G(t|x), TI (interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_3_05 = pd.read_csv(f"{path_data}/3_simdata_linear_ti_inter_05.csv")
print(simdata_3_05.head())
simdata_3_05

# convert eventtime and status columns to a structured array
data_y_3_05, data_x_3_05_df = func.prepare_survival_data(simdata_3_05)
print(data_y_3_05)
print(data_x_3_05_df.head())
data_x_3_05 = data_x_3_05_df.values
X_train_3_05, X_test_3_05, y_train_3_05, y_test_3_05 = train_test_split(
    data_x_3_05, data_y_3_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_3_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_3_05.fit(X_train_3_05, y_train_3_05)
print(f'C-index (train): {model_gbsa_3_05.score(X_test_3_05, y_test_3_05).item():0.3f}')
ibs_gbsa_3_05 = func.compute_integrated_brier(y_test_3_05, X_test_3_05, model_gbsa_3_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_3_05:0.3f}')

# fit CoxPH
model_cox_3_05= CoxPHSurvivalAnalysis()
model_cox_3_05.fit(X_train_3_05, y_train_3_05)
print(f'C-index (train): {model_cox_3_05.score(X_test_3_05, y_test_3_05).item():0.3f}')
ibs_cox_3_05 = func.compute_integrated_brier(y_test_3_05, X_test_3_05, model_cox_3_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_3_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_3_05 = data_x_3_05[[idx]]
#print(x_new_3_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_3_05_loghaz = func.survshapiq_ground_truth(data_x_3_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti_inter, 
                                                            times=model_gbsa_3_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_3_05_df.columns)


func.plot_interact(explanations_all = explanation_3_05_loghaz, 
                              model = None,
                              times=model_gbsa_3_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_3_05.pdf",
                              data_x = data_x_3_05,
                              survival_fn = func.log_hazard_wrap_linear_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 3) Linear G(t|x), TI (interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_3_02 = pd.read_csv(f"{path_data}/3_simdata_linear_ti_inter_02.csv")
print(simdata_3_02.head())
simdata_3_02

# convert eventtime and status columns to a structured array
data_y_3_02, data_x_3_02_df = func.prepare_survival_data(simdata_3_02)
print(data_y_3_02)
print(data_x_3_02_df.head())
data_x_3_02 = data_x_3_02_df.values
X_train_3_02, X_test_3_02, y_train_3_02, y_test_3_02 = train_test_split(
    data_x_3_02, data_y_3_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_3_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_3_02.fit(X_train_3_02, y_train_3_02)
print(f'C-index (train): {model_gbsa_3_02.score(X_test_3_02, y_test_3_02).item():0.3f}')
ibs_gbsa_3_02 = func.compute_integrated_brier(y_test_3_02, X_test_3_02, model_gbsa_3_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_3_02:0.3f}')

# fit CoxPH
model_cox_3_02 = CoxPHSurvivalAnalysis()
model_cox_3_02.fit(X_train_3_02, y_train_3_02)
print(f'C-index (train): {model_cox_3_02.score(X_test_3_02, y_test_3_02).item():0.3f}')
ibs_cox_3_02 = func.compute_integrated_brier(y_test_3_02, X_test_3_02, model_cox_3_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_3_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_3_02 = data_x_3_02[[idx]]
#print(x_new_3_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_3_02_loghaz = func.survshapiq_ground_truth(data_x_3_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_ti_inter, 
                                                            times=model_gbsa_3_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_3_02_df.columns)
explanation_3_02_loghaz[0].mean()
explanation_3_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_3_02_loghaz, 
                              model = None,
                              times=model_gbsa_3_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_3_02.pdf",
                              data_x = data_x_3_02,
                              survival_fn = func.log_hazard_wrap_linear_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_4_0= pd.read_csv(f"{path_data}/4_simdata_linear_tdmain_inter.csv")
print(simdata_4_0.head())

# convert eventtime and status columns to a structured array
data_y_4_0, data_x_4_0_df = func.prepare_survival_data(simdata_4_0)
print(data_y_4_0)
print(data_x_4_0_df.head())
data_x_4_0 = data_x_4_0_df.values
X_train_4_0, X_test_4_0, y_train_4_0, y_test_4_0 = train_test_split(
    data_x_4_0, data_y_4_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_4_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_4_0.fit(X_train_4_0, y_train_4_0)
print(f'C-index (train): {model_gbsa_4_0.score(X_test_4_0, y_test_4_0).item():0.3f}')
ibs_gbsa_4_0 = func.compute_integrated_brier(y_test_4_0, X_test_4_0, model_gbsa_4_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_4_0:0.3f}')

# fit CoxPH
model_cox_4_0 = CoxPHSurvivalAnalysis()
model_cox_4_0.fit(X_train_4_0, y_train_4_0)
print(f'C-index (train): {model_cox_4_0.score(X_test_4_0, y_test_4_0).item():0.3f}')
ibs_cox_4_0 = func.compute_integrated_brier(y_test_4_0, X_test_4_0, model_cox_4_0, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_4_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_4_0 = data_x_4_0[[idx]]
#print(x_new_4_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_4_0_loghaz = func.survshapiq_ground_truth(data_x_4_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain_inter, 
                                                            times=model_gbsa_4_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_4_0_df.columns)

func.plot_interact(explanations_all = explanation_4_0_loghaz, 
                              model = None,
                              times=model_gbsa_4_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_4_0.pdf",
                              data_x = data_x_4_0,
                              survival_fn = func.log_hazard_wrap_linear_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_4_09 = pd.read_csv(f"{path_data}/4_simdata_linear_tdmain_inter_09.csv")
print(simdata_4_09.head())
simdata_4_09

# convert eventtime and status columns to a structured array
data_y_4_09, data_x_4_09_df = func.prepare_survival_data(simdata_4_09)
print(data_y_4_09)
print(data_x_4_09_df.head())
data_x_4_09 = data_x_4_09_df.values
X_train_4_09, X_test_4_09, y_train_4_09, y_test_4_09 = train_test_split(
    data_x_4_09, data_y_4_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_4_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_4_09.fit(X_train_4_09, y_train_4_09)
print(f'C-index (train): {model_gbsa_4_09.score(X_test_4_09, y_test_4_09).item():0.3f}')
ibs_gbsa_4_09 = func.compute_integrated_brier(y_test_4_09, X_test_4_09, model_gbsa_4_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_4_09:0.3f}')

# fit CoxPH
model_cox_4_09 = CoxPHSurvivalAnalysis()
model_cox_4_09.fit(X_train_4_09, y_train_4_09)
print(f'C-index (train): {model_cox_4_09.score(X_test_4_09, y_test_4_09).item():0.3f}')
ibs_cox_4_09 = func.compute_integrated_brier(y_test_4_09, X_test_4_09, model_cox_4_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_4_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_4_09 = data_x_4_09[[idx]]
#print(x_new_4_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_4_09_loghaz = func.survshapiq_ground_truth(data_x_4_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain_inter, 
                                                            times=model_gbsa_4_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_4_09_df.columns)


func.plot_interact(explanations_all = explanation_4_09_loghaz, 
                              model = None,
                              times=model_gbsa_4_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_4_09.pdf",
                              data_x = data_x_4_09,
                              survival_fn = func.log_hazard_wrap_linear_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_4_05 = pd.read_csv(f"{path_data}/4_simdata_linear_tdmain_inter_05.csv")
print(simdata_4_05.head())
simdata_4_05

# convert eventtime and status columns to a structured array
data_y_4_05, data_x_4_05_df = func.prepare_survival_data(simdata_4_05)
print(data_y_4_05)
print(data_x_4_05_df.head())
data_x_4_05 = data_x_4_05_df.values
X_train_4_05, X_test_4_05, y_train_4_05, y_test_4_05 = train_test_split(
    data_x_4_05, data_y_4_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_4_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_4_05.fit(X_train_4_05, y_train_4_05)
print(f'C-index (train): {model_gbsa_4_05.score(X_test_4_05, y_test_4_05).item():0.3f}')
ibs_gbsa_4_05 = func.compute_integrated_brier(y_test_4_05, X_test_4_05, model_gbsa_4_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_4_05:0.3f}')

# fit CoxPH
model_cox_4_05= CoxPHSurvivalAnalysis()
model_cox_4_05.fit(X_train_4_05, y_train_4_05)
print(f'C-index (train): {model_cox_4_05.score(X_test_4_05, y_test_4_05).item():0.3f}')
ibs_cox_4_05 = func.compute_integrated_brier(y_test_4_05, X_test_4_05, model_cox_4_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_4_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_4_05 = data_x_4_05[[idx]]
#print(x_new_4_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_4_05_loghaz = func.survshapiq_ground_truth(data_x_4_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain_inter, 
                                                            times=model_gbsa_4_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_4_05_df.columns)


func.plot_interact(explanations_all = explanation_4_05_loghaz, 
                              model = None,
                              times=model_gbsa_4_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_4_05.pdf",
                              data_x = data_x_4_05,
                              survival_fn = func.log_hazard_wrap_linear_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_4_02 = pd.read_csv(f"{path_data}/4_simdata_linear_tdmain_inter_02.csv")
print(simdata_4_02.head())
simdata_4_02

# convert eventtime and status columns to a structured array
data_y_4_02, data_x_4_02_df = func.prepare_survival_data(simdata_4_02)
print(data_y_4_02)
print(data_x_4_02_df.head())
data_x_4_02 = data_x_4_02_df.values
X_train_4_02, X_test_4_02, y_train_4_02, y_test_4_02 = train_test_split(
    data_x_4_02, data_y_4_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_4_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_4_02.fit(X_train_4_02, y_train_4_02)
print(f'C-index (train): {model_gbsa_4_02.score(X_test_4_02, y_test_4_02).item():0.3f}')
ibs_gbsa_4_02 = func.compute_integrated_brier(y_test_4_02, X_test_4_02, model_gbsa_4_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_4_02:0.3f}')

# fit CoxPH
model_cox_4_02 = CoxPHSurvivalAnalysis()
model_cox_4_02.fit(X_train_4_02, y_train_4_02)
print(f'C-index (train): {model_cox_4_02.score(X_test_4_02, y_test_4_02).item():0.3f}')
ibs_cox_4_02 = func.compute_integrated_brier(y_test_4_02, X_test_4_02, model_cox_4_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_4_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_4_02 = data_x_4_02[[idx]]
#print(x_new_4_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_4_02_loghaz = func.survshapiq_ground_truth(data_x_4_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdmain_inter, 
                                                            times=model_gbsa_4_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_4_02_df.columns)
explanation_4_02_loghaz[0].mean()
explanation_4_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_4_02_loghaz, 
                              model = None,
                              times=model_gbsa_4_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_4_02.pdf",
                              data_x = data_x_4_02,
                              survival_fn = func.log_hazard_wrap_linear_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_5_0= pd.read_csv(f"{path_data}/5_simdata_linear_tdinter.csv")
print(simdata_0.head())

# convert eventtime and status columns to a structured array
data_y_5_0, data_x_5_0_df = func.prepare_survival_data(simdata_5_0)
print(data_y_5_0)
print(data_x_5_0_df.head())
data_x_5_0 = data_x_5_0_df.values
X_train_5_0, X_test_5_0, y_train_5_0, y_test_5_0 = train_test_split(
    data_x_5_0, data_y_5_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_5_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_5_0.fit(X_train_5_0, y_train_5_0)
print(f'C-index (train): {model_gbsa_5_0.score(X_test_5_0, y_test_5_0).item():0.3f}')
ibs_gbsa_5_0 = func.compute_integrated_brier(y_test_5_0, X_test_5_0, model_gbsa_5_0, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_5_0:0.3f}')

# fit CoxPH
model_cox_5_0 = CoxPHSurvivalAnalysis()
model_cox_5_0.fit(X_train_5_0, y_train_5_0)
print(f'C-index (train): {model_cox_5_0.score(X_test_5_0, y_test_5_0).item():0.3f}')
ibs_cox_5_0 = func.compute_integrated_brier(y_test_5_0, X_test_5_0, model_cox_5_0, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_5_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_5_0 = data_x_5_0[[idx]]
#print(x_new_5_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_5_0_loghaz = func.survshapiq_ground_truth(data_x_5_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_5_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_5_0_df.columns)

func.plot_interact(explanations_all = explanation_5_0_loghaz, 
                              model = None,
                              times=model_gbsa_5_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_5_0.pdf",
                              data_x = data_x_5_0,
                              survival_fn = func.log_hazard_wrap_linear_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_5_09 = pd.read_csv(f"{path_data}/5_simdata_corr_09.csv")
print(simdata_5_09.head())
simdata_5_09

# convert eventtime and status columns to a structured array
data_y_5_09, data_x_5_09_df = func.prepare_survival_data(simdata_5_09)
print(data_y_5_09)
print(data_x_5_09_df.head())
data_x_5_09 = data_x_5_09_df.values
X_train_5_09, X_test_5_09, y_train_5_09, y_test_5_09 = train_test_split(
    data_x_5_09, data_y_5_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_5_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_5_09.fit(X_train_5_09, y_train_5_09)
print(f'C-index (train): {model_gbsa_5_09.score(X_test_5_09, y_test_5_09).item():0.3f}')
ibs_gbsa_5_09 = func.compute_integrated_brier(y_test_5_09, X_test_5_09, model_gbsa_5_09, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_5_09:0.3f}')

# fit CoxPH
model_cox_5_09 = CoxPHSurvivalAnalysis()
model_cox_5_09.fit(X_train_5_09, y_train_5_09)
print(f'C-index (train): {model_cox_5_09.score(X_test_5_09, y_test_5_09).item():0.3f}')
ibs_cox_5_09 = func.compute_integrated_brier(y_test_5_09, X_test_5_09, model_cox_5_09, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_5_09:0.3f}')


# create data point for explanation
#idx =  7
#x_new_5_09 = data_x_5_09[[idx]]
#print(x_new_5_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_5_09_loghaz = func.survshapiq_ground_truth(data_x_5_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_5_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_5_09_df.columns)


func.plot_interact(explanations_all = explanation_5_09_loghaz, 
                              model = None,
                              times=model_gbsa_5_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_5_09.pdf",
                              data_x = data_x_5_09,
                              survival_fn = func.log_hazard_wrap_linear_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_5_05 = pd.read_csv(f"{path_data}/5_simdata_corr_05.csv")
print(simdata_5_05.head())
simdata_5_05

# convert eventtime and status columns to a structured array
data_y_5_05, data_x_5_05_df = func.prepare_survival_data(simdata_5_05)
print(data_y_5_05)
print(data_x_5_05_df.head())
data_x_5_05 = data_x_5_05_df.values
X_train_5_05, X_test_5_05, y_train_5_05, y_test_5_05 = train_test_split(
    data_x_5_05, data_y_5_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_5_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_5_05.fit(X_train_5_05, y_train_5_05)
print(f'C-index (train): {model_gbsa_5_05.score(X_test_5_05, y_test_5_05).item():0.3f}')
ibs_gbsa_5_05 = func.compute_integrated_brier(y_test_5_05, X_test_5_05, model_gbsa_5_05, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_5_05:0.3f}')

# fit CoxPH
model_cox_5_05= CoxPHSurvivalAnalysis()
model_cox_5_05.fit(X_train_5_05, y_train_5_05)
print(f'C-index (train): {model_cox_5_05.score(X_test_5_05, y_test_5_05).item():0.3f}')
ibs_cox_5_05 = func.compute_integrated_brier(y_test_5_05, X_test_5_05, model_cox_5_05, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_5_05:0.3f}')

# create data point for explanation
# create data point for explanation
#idx =  7
#x_new_5_05 = data_x_5_05[[idx]]
#print(x_new_5_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_5_05_loghaz = func.survshapiq_ground_truth(data_x_5_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_5_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_5_05_df.columns)


func.plot_interact(explanations_all = explanation_5_05_loghaz, 
                              model = None,
                              times=model_gbsa_5_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_5_05.pdf",
                              data_x = data_x_5_05,
                              survival_fn = func.log_hazard_wrap_linear_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_5_02 = pd.read_csv(f"{path_data}/5_simdata_corr_02.csv")
print(simdata_5_02.head())
simdata_5_02

# convert eventtime and status columns to a structured array
data_y_5_02, data_x_5_02_df = func.prepare_survival_data(simdata_5_02)
print(data_y_5_02)
print(data_x_5_02_df.head())
data_x_5_02 = data_x_5_02_df.values
X_train_5_02, X_test_5_02, y_train_5_02, y_test_5_02 = train_test_split(
    data_x_5_02, data_y_5_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_5_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_5_02.fit(X_train_5_02, y_train_5_02)
print(f'C-index (train): {model_gbsa_5_02.score(X_test_5_02, y_test_5_02).item():0.3f}')
ibs_gbsa_5_02 = func.compute_integrated_brier(y_test_5_02, X_test_5_02, model_gbsa_5_02, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_5_02:0.3f}')

# fit CoxPH
model_cox_5_02 = CoxPHSurvivalAnalysis()
model_cox_5_02.fit(X_train_5_02, y_train_5_02)
print(f'C-index (train): {model_cox_5_02.score(X_test_5_02, y_test_5_02).item():0.3f}')
ibs_cox_5_02 = func.compute_integrated_brier(y_test_5_02, X_test_5_02, model_cox_5_02, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_5_02:0.3f}')


# create data point for explanation
#idx =  7
#x_new_5_02 = data_x_5_02[[idx]]
#print(x_new_5_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_5_02_loghaz = func.survshapiq_ground_truth(data_x_5_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_5_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_5_02_df.columns)
explanation_5_02_loghaz[0].mean()
explanation_5_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_5_02_loghaz, 
                              model = None,
                              times=model_gbsa_5_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_5_02.pdf",
                              data_x = data_x_5_02,
                              survival_fn = func.log_hazard_wrap_linear_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_6_0= pd.read_csv(f"{path_data}/6_simdata_genadd_ti.csv")
print(simdata_6_0.head())

# convert eventtime and status columns to a structured array
data_y_6_0, data_x_6_0_df = func.prepare_survival_data(simdata_6_0)
print(data_y_6_0)
print(data_x_6_0_df.head())
data_x_6_0 = data_x_6_0_df.values
X_train_6_0, X_test_6_0, y_train_6_0, y_test_6_0 = train_test_split(
    data_x_6_0, data_y_6_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_6_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_6_0.fit(X_train_6_0, y_train_6_0)
print(f'C-index (train): {model_gbsa_6_0.score(X_test_6_0, y_test_6_0).item():0.3f}')
ibs_gbsa_6_0 = func.compute_integrated_brier(y_test_6_0, X_test_6_0, model_gbsa_6_0, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_6_0:0.3f}')

# fit CoxPH
model_cox_6_0 = CoxPHSurvivalAnalysis()
model_cox_6_0.fit(X_train_6_0, y_train_6_0)
print(f'C-index (train): {model_cox_6_0.score(X_test_6_0, y_test_6_0).item():0.3f}')
ibs_cox_6_0 = func.compute_integrated_brier(y_test_6_0, X_test_6_0, model_cox_6_0, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_6_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_6_0 = data_x_6_0[[idx]]
#print(x_new_6_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_6_0_loghaz = func.survshapiq_ground_truth(data_x_6_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_6_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_6_0_df.columns)

func.plot_interact(explanations_all = explanation_6_0_loghaz, 
                              model = None,
                              times=model_gbsa_6_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_6_0.pdf",
                              data_x = data_x_6_0,
                              survival_fn = func.log_hazard_wrap_genadd_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_6_09 = pd.read_csv(f"{path_data}/6_simdata_genadd_ti_09.csv")
print(simdata_6_09.head())
simdata_6_09

# convert eventtime and status columns to a structured array
data_y_6_09, data_x_6_09_df = func.prepare_survival_data(simdata_6_09)
print(data_y_6_09)
print(data_x_6_09_df.head())
data_x_6_09 = data_x_6_09_df.values
X_train_6_09, X_test_6_09, y_train_6_09, y_test_6_09 = train_test_split(
    data_x_6_09, data_y_6_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_6_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_6_09.fit(X_train_6_09, y_train_6_09)
print(f'C-index (train): {model_gbsa_6_09.score(X_test_6_09, y_test_6_09).item():0.3f}')
ibs_gbsa_6_09 = func.compute_integrated_brier(y_test_6_09, X_test_6_09, model_gbsa_6_09, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_6_09:0.3f}')

# fit CoxPH
model_cox_6_09 = CoxPHSurvivalAnalysis()
model_cox_6_09.fit(X_train_6_09, y_train_6_09)
print(f'C-index (train): {model_cox_6_09.score(X_test_6_09, y_test_6_09).item():0.3f}')
ibs_cox_6_09 = func.compute_integrated_brier(y_test_6_09, X_test_6_09, model_cox_6_09, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_6_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_6_09 = data_x_6_09[[idx]]
#print(x_new_6_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_6_09_loghaz = func.survshapiq_ground_truth(data_x_6_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_6_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_6_09_df.columns)


func.plot_interact(explanations_all = explanation_6_09_loghaz, 
                              model = None,
                              times=model_gbsa_6_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_6_09.pdf",
                              data_x = data_x_6_09,
                              survival_fn = func.log_hazard_wrap_genadd_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_6_05 = pd.read_csv(f"{path_data}/6_simdata_genadd_ti_05.csv")
print(simdata_6_05.head())
simdata_6_05

# convert eventtime and status columns to a structured array
data_y_6_05, data_x_6_05_df = func.prepare_survival_data(simdata_6_05)
print(data_y_6_05)
print(data_x_6_05_df.head())
data_x_6_05 = data_x_6_05_df.values
X_train_6_05, X_test_6_05, y_train_6_05, y_test_6_05 = train_test_split(
    data_x_6_05, data_y_6_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_6_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_6_05.fit(X_train_6_05, y_train_6_05)
print(f'C-index (train): {model_gbsa_6_05.score(X_test_6_05, y_test_6_05).item():0.3f}')
ibs_gbsa_6_05 = func.compute_integrated_brier(y_test_6_05, X_test_6_05, model_gbsa_6_05, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_6_05:0.3f}')

# fit CoxPH
model_cox_6_05= CoxPHSurvivalAnalysis()
model_cox_6_05.fit(X_train_6_05, y_train_6_05)
print(f'C-index (train): {model_cox_6_05.score(X_test_6_05, y_test_6_05).item():0.3f}')
ibs_cox_6_05 = func.compute_integrated_brier(y_test_6_05, X_test_6_05, model_cox_6_05, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_6_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_6_05 = data_x_6_05[[idx]]
#print(x_new_6_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_6_05_loghaz = func.survshapiq_ground_truth(data_x_6_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_6_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_6_05_df.columns)


func.plot_interact(explanations_all = explanation_6_05_loghaz, 
                              model = None,
                              times=model_gbsa_6_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_6_05.pdf",
                              data_x = data_x_6_05,
                              survival_fn = func.log_hazard_wrap_genadd_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_6_02 = pd.read_csv(f"{path_data}/6_simdata_genadd_ti_02.csv")
print(simdata_6_02.head())
simdata_6_02

# convert eventtime and status columns to a structured array
data_y_6_02, data_x_6_02_df = func.prepare_survival_data(simdata_6_02)
print(data_y_6_02)
print(data_x_6_02_df.head())
data_x_6_02 = data_x_6_02_df.values
X_train_6_02, X_test_6_02, y_train_6_02, y_test_6_02 = train_test_split(
    data_x_6_02, data_y_6_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_6_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_6_02.fit(X_train_6_02, y_train_6_02)
print(f'C-index (train): {model_gbsa_6_02.score(X_test_6_02, y_test_6_02).item():0.3f}')
ibs_gbsa_6_02 = func.compute_integrated_brier(y_test_6_02, X_test_6_02, model_gbsa_6_02, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_6_02:0.3f}')

# fit CoxPH
model_cox_6_02 = CoxPHSurvivalAnalysis()
model_cox_6_02.fit(X_train_6_02, y_train_6_02)
print(f'C-index (train): {model_cox_6_02.score(X_test_6_02, y_test_6_02).item():0.3f}')
ibs_cox_6_02 = func.compute_integrated_brier(y_test_6_02, X_test_6_02, model_cox_6_02, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_6_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_6_02 = data_x_6_02[[idx]]
#print(x_new_6_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_6_02_loghaz = func.survshapiq_ground_truth(data_x_6_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_6_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_6_02_df.columns)
explanation_6_02_loghaz[0].mean()
explanation_6_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_6_02_loghaz, 
                              model = None,
                              times=model_gbsa_6_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_6_02.pdf",
                              data_x = data_x_6_02,
                              survival_fn = func.log_hazard_wrap_genadd_ti,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_7_0= pd.read_csv(f"{path_data}/7_simdata_genadd_tdmain.csv")
print(simdata_7_0.head())

# convert eventtime and status columns to a structured array
data_y_7_0, data_x_7_0_df = func.prepare_survival_data(simdata_7_0)
print(data_y_7_0)
print(data_x_7_0_df.head())
data_x_7_0 = data_x_7_0_df.values
X_train_7_0, X_test_7_0, y_train_7_0, y_test_7_0 = train_test_split(
    data_x_7_0, data_y_7_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_7_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_7_0.fit(X_train_7_0, y_train_7_0)
print(f'C-index (train): {model_gbsa_7_0.score(X_test_7_0, y_test_7_0).item():0.3f}')
ibs_gbsa_7_0 = func.compute_integrated_brier(y_test_7_0, X_test_7_0, model_gbsa_7_0, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_7_0:0.3f}')

# fit CoxPH
model_cox_7_0 = CoxPHSurvivalAnalysis()
model_cox_7_0.fit(X_train_7_0, y_train_7_0)
print(f'C-index (train): {model_cox_7_0.score(X_test_7_0, y_test_7_0).item():0.3f}')
ibs_cox_7_0 = func.compute_integrated_brier(y_test_7_0, X_test_7_0, model_cox_7_0, min_time = 0.098, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_7_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_7_0 = data_x_7_0[[idx]]
#print(x_new_7_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_7_0_loghaz = func.survshapiq_ground_truth(data_x_7_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_7_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_7_0_df.columns)

func.plot_interact(explanations_all = explanation_7_0_loghaz, 
                              model = None,
                              times=model_gbsa_7_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_7_0.pdf",
                              data_x = data_x_7_0,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_7_09 = pd.read_csv(f"{path_data}/7_simdata_genadd_tdmain_09.csv")
print(simdata_7_09.head())
simdata_7_09

# convert eventtime and status columns to a structured array
data_y_7_09, data_x_7_09_df = func.prepare_survival_data(simdata_7_09)
print(data_y_7_09)
print(data_x_7_09_df.head())
data_x_7_09 = data_x_7_09_df.values
X_train_7_09, X_test_7_09, y_train_7_09, y_test_7_09 = train_test_split(
    data_x_7_09, data_y_7_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_7_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_7_09.fit(X_train_7_09, y_train_7_09)
print(f'C-index (train): {model_gbsa_7_09.score(X_test_7_09, y_test_7_09).item():0.3f}')
ibs_gbsa_7_09 = func.compute_integrated_brier(y_test_7_09, X_test_7_09, model_gbsa_7_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_7_09:0.3f}')

# fit CoxPH
model_cox_7_09 = CoxPHSurvivalAnalysis()
model_cox_7_09.fit(X_train_7_09, y_train_7_09)
print(f'C-index (train): {model_cox_7_09.score(X_test_7_09, y_test_7_09).item():0.3f}')
ibs_cox_7_09 = func.compute_integrated_brier(y_test_7_09, X_test_7_09, model_cox_7_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_7_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_7_09 = data_x_7_09[[idx]]
#print(x_new_7_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_7_09_loghaz = func.survshapiq_ground_truth(data_x_7_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_7_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_7_09_df.columns)


func.plot_interact(explanations_all = explanation_7_09_loghaz, 
                              model = None,
                              times=model_gbsa_7_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_7_09.pdf",
                              data_x = data_x_7_09,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_7_05 = pd.read_csv(f"{path_data}/7_simdata_genadd_tdmain_05.csv")
print(simdata_7_05.head())
simdata_7_05

# convert eventtime and status columns to a structured array
data_y_7_05, data_x_7_05_df = func.prepare_survival_data(simdata_7_05)
print(data_y_7_05)
print(data_x_7_05_df.head())
data_x_7_05 = data_x_7_05_df.values
X_train_7_05, X_test_7_05, y_train_7_05, y_test_7_05 = train_test_split(
    data_x_7_05, data_y_7_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_7_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_7_05.fit(X_train_7_05, y_train_7_05)
print(f'C-index (train): {model_gbsa_7_05.score(X_test_7_05, y_test_7_05).item():0.3f}')
ibs_gbsa_7_05 = func.compute_integrated_brier(y_test_7_05, X_test_7_05, model_gbsa_7_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_7_05:0.3f}')

# fit CoxPH
model_cox_7_05= CoxPHSurvivalAnalysis()
model_cox_7_05.fit(X_train_7_05, y_train_7_05)
print(f'C-index (train): {model_cox_7_05.score(X_test_7_05, y_test_7_05).item():0.3f}')
ibs_cox_7_05 = func.compute_integrated_brier(y_test_7_05, X_test_7_05, model_cox_7_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_7_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_7_05 = data_x_7_05[[idx]]
#print(x_new_7_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_7_05_loghaz = func.survshapiq_ground_truth(data_x_7_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_7_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_7_05_df.columns)


func.plot_interact(explanations_all = explanation_7_05_loghaz, 
                              model = None,
                              times=model_gbsa_7_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_7_05.pdf",
                              data_x = data_x_7_05,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_7_02 = pd.read_csv(f"{path_data}/7_simdata_genadd_tdmain_02.csv")
print(simdata_7_02.head())
simdata_7_02

# convert eventtime and status columns to a structured array
data_y_7_02, data_x_7_02_df = func.prepare_survival_data(simdata_7_02)
print(data_y_7_02)
print(data_x_7_02_df.head())
data_x_7_02 = data_x_7_02_df.values
X_train_7_02, X_test_7_02, y_train_7_02, y_test_7_02 = train_test_split(
    data_x_7_02, data_y_7_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_7_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_7_02.fit(X_train_7_02, y_train_7_02)
print(f'C-index (train): {model_gbsa_7_02.score(X_test_7_02, y_test_7_02).item():0.3f}')
ibs_gbsa_7_02 = func.compute_integrated_brier(y_test_7_02, X_test_7_02, model_gbsa_7_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_7_02:0.3f}')

# fit CoxPH
model_cox_7_02 = CoxPHSurvivalAnalysis()
model_cox_7_02.fit(X_train_7_02, y_train_7_02)
print(f'C-index (train): {model_cox_7_02.score(X_test_7_02, y_test_7_02).item():0.3f}')
ibs_cox_7_02 = func.compute_integrated_brier(y_test_7_02, X_test_7_02, model_cox_7_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_7_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_7_02 = data_x_7_02[[idx]]
#print(x_new_7_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_7_02_loghaz = func.survshapiq_ground_truth(data_x_7_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_7_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_7_02_df.columns)
explanation_7_02_loghaz[0].mean()
explanation_7_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_7_02_loghaz, 
                              model = None,
                              times=model_gbsa_7_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_7_02.pdf",
                              data_x = data_x_7_02,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_8_0= pd.read_csv(f"{path_data}/8_simdata_genadd_ti_inter.csv")
print(simdata_8_0.head())

# convert eventtime and status columns to a structured array
data_y_8_0, data_x_8_0_df = func.prepare_survival_data(simdata_8_0)
print(data_y_8_0)
print(data_x_8_0_df.head())
data_x_8_0 = data_x_8_0_df.values
X_train_8_0, X_test_8_0, y_train_8_0, y_test_8_0 = train_test_split(
    data_x_8_0, data_y_8_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_8_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_8_0.fit(X_train_8_0, y_train_8_0)
print(f'C-index (train): {model_gbsa_8_0.score(X_test_8_0, y_test_8_0).item():0.3f}')
ibs_gbsa_8_0 = func.compute_integrated_brier(y_test_8_0, X_test_8_0, model_gbsa_8_0, min_time = 0.12, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_8_0:0.3f}')

# fit CoxPH
model_cox_8_0 = CoxPHSurvivalAnalysis()
model_cox_8_0.fit(X_train_8_0, y_train_8_0)
print(f'C-index (train): {model_cox_8_0.score(X_test_8_0, y_test_8_0).item():0.3f}')
ibs_cox_8_0 = func.compute_integrated_brier(y_test_8_0, X_test_8_0, model_cox_8_0, min_time = 0.12, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_8_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_8_0 = data_x_8_0[[idx]]
#print(x_new_8_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_8_0_loghaz = func.survshapiq_ground_truth(data_x_8_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti_inter, 
                                                            times=model_gbsa_8_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_8_0_df.columns)

func.plot_interact(explanations_all = explanation_8_0_loghaz, 
                              model = None,
                              times=model_gbsa_8_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_8_0.pdf",
                              data_x = data_x_8_0,
                              survival_fn = func.log_hazard_wrap_genadd_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_8_09 = pd.read_csv(f"{path_data}/8_simdata_genadd_ti_inter_09.csv")
print(simdata_8_09.head())
simdata_8_09

# convert eventtime and status columns to a structured array
data_y_8_09, data_x_8_09_df = func.prepare_survival_data(simdata_8_09)
print(data_y_8_09)
print(data_x_8_09_df.head())
data_x_8_09 = data_x_8_09_df.values
X_train_8_09, X_test_8_09, y_train_8_09, y_test_8_09 = train_test_split(
    data_x_8_09, data_y_8_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_8_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_8_09.fit(X_train_8_09, y_train_8_09)
print(f'C-index (train): {model_gbsa_8_09.score(X_test_8_09, y_test_8_09).item():0.3f}')
ibs_gbsa_8_09 = func.compute_integrated_brier(y_test_8_09, X_test_8_09, model_gbsa_8_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_8_09:0.3f}')

# fit CoxPH
model_cox_8_09 = CoxPHSurvivalAnalysis()
model_cox_8_09.fit(X_train_8_09, y_train_8_09)
print(f'C-index (train): {model_cox_8_09.score(X_test_8_09, y_test_8_09).item():0.3f}')
ibs_cox_8_09 = func.compute_integrated_brier(y_test_8_09, X_test_8_09, model_cox_8_09, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_8_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_8_09 = data_x_8_09[[idx]]
#print(x_new_8_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_8_09_loghaz = func.survshapiq_ground_truth(data_x_8_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti_inter, 
                                                            times=model_gbsa_8_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_8_09_df.columns)


func.plot_interact(explanations_all = explanation_8_09_loghaz, 
                              model = None,
                              times=model_gbsa_8_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_8_09.pdf",
                              data_x = data_x_8_09,
                              survival_fn = func.log_hazard_wrap_genadd_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_8_05 = pd.read_csv(f"{path_data}/8_simdata_genadd_ti_inter_05.csv")
print(simdata_8_05.head())
simdata_8_05

# convert eventtime and status columns to a structured array
data_y_8_05, data_x_8_05_df = func.prepare_survival_data(simdata_8_05)
print(data_y_8_05)
print(data_x_8_05_df.head())
data_x_8_05 = data_x_8_05_df.values
X_train_8_05, X_test_8_05, y_train_8_05, y_test_8_05 = train_test_split(
    data_x_8_05, data_y_8_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_8_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_8_05.fit(X_train_8_05, y_train_8_05)
print(f'C-index (train): {model_gbsa_8_05.score(X_test_8_05, y_test_8_05).item():0.3f}')
ibs_gbsa_8_05 = func.compute_integrated_brier(y_test_8_05, X_test_8_05, model_gbsa_8_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_8_05:0.3f}')

# fit CoxPH
model_cox_8_05= CoxPHSurvivalAnalysis()
model_cox_8_05.fit(X_train_8_05, y_train_8_05)
print(f'C-index (train): {model_cox_8_05.score(X_test_8_05, y_test_8_05).item():0.3f}')
ibs_cox_8_05 = func.compute_integrated_brier(y_test_8_05, X_test_8_05, model_cox_8_05, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_8_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_8_05 = data_x_8_05[[idx]]
#print(x_new_8_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_8_05_loghaz = func.survshapiq_ground_truth(data_x_8_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti_inter, 
                                                            times=model_gbsa_8_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_8_05_df.columns)


func.plot_interact(explanations_all = explanation_8_05_loghaz, 
                              model = None,
                              times=model_gbsa_8_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_8_05.pdf",
                              data_x = data_x_8_05,
                              survival_fn = func.log_hazard_wrap_genadd_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_8_02 = pd.read_csv(f"{path_data}/8_simdata_genadd_ti_inter_02.csv")
print(simdata_8_02.head())
simdata_8_02

# convert eventtime and status columns to a structured array
data_y_8_02, data_x_8_02_df = func.prepare_survival_data(simdata_8_02)
print(data_y_8_02)
print(data_x_8_02_df.head())
data_x_8_02 = data_x_8_02_df.values
X_train_8_02, X_test_8_02, y_train_8_02, y_test_8_02 = train_test_split(
    data_x_8_02, data_y_8_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_8_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_8_02.fit(X_train_8_02, y_train_8_02)
print(f'C-index (train): {model_gbsa_8_02.score(X_test_8_02, y_test_8_02).item():0.3f}')
ibs_gbsa_8_02 = func.compute_integrated_brier(y_test_8_02, X_test_8_02, model_gbsa_8_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_8_02:0.3f}')

# fit CoxPH
model_cox_8_02 = CoxPHSurvivalAnalysis()
model_cox_8_02.fit(X_train_8_02, y_train_8_02)
print(f'C-index (train): {model_cox_8_02.score(X_test_8_02, y_test_8_02).item():0.3f}')
ibs_cox_8_02 = func.compute_integrated_brier(y_test_8_02, X_test_8_02, model_cox_8_02, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_8_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_8_02 = data_x_8_02[[idx]]
#print(x_new_8_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_8_02_loghaz = func.survshapiq_ground_truth(data_x_8_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_ti_inter, 
                                                            times=model_gbsa_8_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_8_02_df.columns)
explanation_8_02_loghaz[0].mean()
explanation_8_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_8_02_loghaz, 
                              model = None,
                              times=model_gbsa_8_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_8_02.pdf",
                              data_x = data_x_8_02,
                              survival_fn = func.log_hazard_wrap_genadd_ti_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 9) Generalized Additive G(t|x), TD Main (interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_9_0= pd.read_csv(f"{path_data}/9_simdata_genadd_tdmain_inter.csv")
print(simdata_9_0.head())

# convert eventtime and status columns to a structured array
data_y_9_0, data_x_9_0_df = func.prepare_survival_data(simdata_9_0)
print(data_y_9_0)
print(data_x_9_0_df.head())
data_x_9_0 = data_x_9_0_df.values
X_train_9_0, X_test_9_0, y_train_9_0, y_test_9_0 = train_test_split(
    data_x_9_0, data_y_9_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_9_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_9_0.fit(X_train_9_0, y_train_9_0)
print(f'C-index (train): {model_gbsa_9_0.score(X_test_9_0, y_test_9_0).item():0.3f}')
ibs_gbsa_9_0 = func.compute_integrated_brier(y_test_9_0, X_test_9_0, model_gbsa_9_0, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_9_0:0.3f}')

# fit CoxPH
model_cox_9_0 = CoxPHSurvivalAnalysis()
model_cox_9_0.fit(X_train_9_0, y_train_9_0)
print(f'C-index (train): {model_cox_9_0.score(X_test_9_0, y_test_9_0).item():0.3f}')
ibs_cox_9_0 = func.compute_integrated_brier(y_test_9_0, X_test_9_0, model_cox_9_0, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_9_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_9_0 = data_x_9_0[[idx]]
#print(x_new_9_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_9_0_loghaz = func.survshapiq_ground_truth(data_x_9_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain_inter, 
                                                            times=model_gbsa_9_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_9_0_df.columns)

func.plot_interact(explanations_all = explanation_9_0_loghaz, 
                              model = None,
                              times=model_gbsa_9_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_9_0.pdf",
                              data_x = data_x_9_0,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 9) Generalized Additive G(t|x), TD Main (interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_9_09 = pd.read_csv(f"{path_data}/9_simdata_genadd_tdmain_inter_09.csv")
print(simdata_9_09.head())
simdata_9_09

# convert eventtime and status columns to a structured array
data_y_9_09, data_x_9_09_df = func.prepare_survival_data(simdata_9_09)
print(data_y_9_09)
print(data_x_9_09_df.head())
data_x_9_09 = data_x_9_09_df.values
X_train_9_09, X_test_9_09, y_train_9_09, y_test_9_09 = train_test_split(
    data_x_9_09, data_y_9_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_9_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_9_09.fit(X_train_9_09, y_train_9_09)
print(f'C-index (train): {model_gbsa_9_09.score(X_test_9_09, y_test_9_09).item():0.3f}')
ibs_gbsa_9_09 = func.compute_integrated_brier(y_test_9_09, X_test_9_09, model_gbsa_9_09, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_9_09:0.3f}')

# fit CoxPH
model_cox_9_09 = CoxPHSurvivalAnalysis()
model_cox_9_09.fit(X_train_9_09, y_train_9_09)
print(f'C-index (train): {model_cox_9_09.score(X_test_9_09, y_test_9_09).item():0.3f}')
ibs_cox_9_09 = func.compute_integrated_brier(y_test_9_09, X_test_9_09, model_cox_9_09, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_9_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_9_09 = data_x_9_09[[idx]]
#print(x_new_9_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_9_09_loghaz = func.survshapiq_ground_truth(data_x_9_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain_inter, 
                                                            times=model_gbsa_9_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_9_09_df.columns)


func.plot_interact(explanations_all = explanation_9_09_loghaz, 
                              model = None,
                              times=model_gbsa_9_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_9_09.pdf",
                              data_x = data_x_9_09,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 9) Generalized Additive G(t|x), TD Main (interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_9_05 = pd.read_csv(f"{path_data}/9_simdata_genadd_tdmain_inter_05.csv")
print(simdata_9_05.head())
simdata_9_05

# convert eventtime and status columns to a structured array
data_y_9_05, data_x_9_05_df = func.prepare_survival_data(simdata_9_05)
print(data_y_9_05)
print(data_x_9_05_df.head())
data_x_9_05 = data_x_9_05_df.values
X_train_9_05, X_test_9_05, y_train_9_05, y_test_9_05 = train_test_split(
    data_x_9_05, data_y_9_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_9_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_9_05.fit(X_train_9_05, y_train_9_05)
print(f'C-index (train): {model_gbsa_9_05.score(X_test_9_05, y_test_9_05).item():0.3f}')
ibs_gbsa_9_05 = func.compute_integrated_brier(y_test_9_05, X_test_9_05, model_gbsa_9_05, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_9_05:0.3f}')

# fit CoxPH
model_cox_9_05= CoxPHSurvivalAnalysis()
model_cox_9_05.fit(X_train_9_05, y_train_9_05)
print(f'C-index (train): {model_cox_9_05.score(X_test_9_05, y_test_9_05).item():0.3f}')
ibs_cox_9_05 = func.compute_integrated_brier(y_test_9_05, X_test_9_05, model_cox_9_05, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_9_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_9_05 = data_x_9_05[[idx]]
#print(x_new_9_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_9_05_loghaz = func.survshapiq_ground_truth(data_x_9_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain_inter, 
                                                            times=model_gbsa_9_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_9_05_df.columns)


func.plot_interact(explanations_all = explanation_9_05_loghaz, 
                              model = None,
                              times=model_gbsa_9_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_9_05.pdf",
                              data_x = data_x_9_05,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 9) Generalized Additive G(t|x), TD Main (interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_9_02 = pd.read_csv(f"{path_data}/9_simdata_genadd_tdmain_inter_02.csv")
print(simdata_9_02.head())
simdata_9_02

# convert eventtime and status columns to a structured array
data_y_9_02, data_x_9_02_df = func.prepare_survival_data(simdata_9_02)
print(data_y_9_02)
print(data_x_9_02_df.head())
data_x_9_02 = data_x_9_02_df.values
X_train_9_02, X_test_9_02, y_train_9_02, y_test_9_02 = train_test_split(
    data_x_9_02, data_y_9_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_9_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_9_02.fit(X_train_9_02, y_train_9_02)
print(f'C-index (train): {model_gbsa_9_02.score(X_test_9_02, y_test_9_02).item():0.3f}')
ibs_gbsa_9_02 = func.compute_integrated_brier(y_test_9_02, X_test_9_02, model_gbsa_9_02, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_9_02:0.3f}')

# fit CoxPH
model_cox_9_02 = CoxPHSurvivalAnalysis()
model_cox_9_02.fit(X_train_9_02, y_train_9_02)
print(f'C-index (train): {model_cox_9_02.score(X_test_9_02, y_test_9_02).item():0.3f}')
ibs_cox_9_02 = func.compute_integrated_brier(y_test_9_02, X_test_9_02, model_cox_9_02, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_9_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_9_02 = data_x_9_02[[idx]]
#print(x_new_9_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_9_02_loghaz = func.survshapiq_ground_truth(data_x_9_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdmain_inter, 
                                                            times=model_gbsa_9_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_9_02_df.columns)
explanation_9_02_loghaz[0].mean()
explanation_9_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_9_02_loghaz, 
                              model = None,
                              times=model_gbsa_9_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_9_02.pdf",
                              data_x = data_x_9_02,
                              survival_fn = func.log_hazard_wrap_genadd_tdmain_inter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 10) Generalized Additive G(t|x), TD Inter (interactions)
# corr = 0
#---------------------------

# load simulated data DataFrame
simdata_10_0= pd.read_csv(f"{path_data}/10_simdata_genadd_tdinter.csv")
print(simdata_10_0.head())

# convert eventtime and status columns to a structured array
data_y_10_0, data_x_10_0_df = func.prepare_survival_data(simdata_10_0)
print(data_y_10_0)
print(data_x_10_0_df.head())
data_x_10_0 = data_x_10_0_df.values
X_train_10_0, X_test_10_0, y_train_10_0, y_test_10_0 = train_test_split(
    data_x_10_0, data_y_10_0, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_10_0 = GradientBoostingSurvivalAnalysis()
model_gbsa_10_0.fit(X_train_10_0, y_train_10_0)
print(f'C-index (train): {model_gbsa_10_0.score(X_test_10_0, y_test_10_0).item():0.3f}')
ibs_gbsa_10_0 = func.compute_integrated_brier(y_test_10_0, X_test_10_0, model_gbsa_10_0, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_10_0:0.3f}')

# fit CoxPH
model_cox_10_0 = CoxPHSurvivalAnalysis()
model_cox_10_0.fit(X_train_10_0, y_train_10_0)
print(f'C-index (train): {model_cox_10_0.score(X_test_10_0, y_test_10_0).item():0.3f}')
ibs_cox_10_0 = func.compute_integrated_brier(y_test_10_0, X_test_10_0, model_cox_10_0, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_10_0:0.3f}')

# create data point for explanation
#idx =  7
#x_new_10_0 = data_x_10_0[[idx]]
#print(x_new_10_0)


###### GROUND TRUTH LOG HAZARD
# exact
explanation_10_0_loghaz = func.survshapiq_ground_truth(data_x_10_0, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_10_0.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_10_0_df.columns)

func.plot_interact(explanations_all = explanation_10_0_loghaz, 
                              model = None,
                              times=model_gbsa_10_0.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_10_0.pdf",
                              data_x = data_x_10_0,
                              survival_fn = func.log_hazard_wrap_genadd_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 




#---------------------------
# 10) Generalized Additive G(t|x), TD Inter (interactions)
# corr = 0.9
#---------------------------

# load simulated data DataFrame
simdata_10_09 = pd.read_csv(f"{path_data}/10_simdata_genadd_tdinter_09.csv")
print(simdata_10_09.head())
simdata_10_09

# convert eventtime and status columns to a structured array
data_y_10_09, data_x_10_09_df = func.prepare_survival_data(simdata_10_09)
print(data_y_10_09)
print(data_x_10_09_df.head())
data_x_10_09 = data_x_10_09_df.values
X_train_10_09, X_test_10_09, y_train_10_09, y_test_10_09 = train_test_split(
    data_x_10_09, data_y_10_09, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_10_09 = GradientBoostingSurvivalAnalysis()
model_gbsa_10_09.fit(X_train_10_09, y_train_10_09)
print(f'C-index (train): {model_gbsa_10_09.score(X_test_10_09, y_test_10_09).item():0.3f}')
ibs_gbsa_10_09 = func.compute_integrated_brier(y_test_10_09, X_test_10_09, model_gbsa_10_09, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_10_09:0.3f}')

# fit CoxPH
model_cox_10_09 = CoxPHSurvivalAnalysis()
model_cox_10_09.fit(X_train_10_09, y_train_10_09)
print(f'C-index (train): {model_cox_10_09.score(X_test_10_09, y_test_10_09).item():0.3f}')
ibs_cox_10_09 = func.compute_integrated_brier(y_test_10_09, X_test_10_09, model_cox_10_09, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_10_09:0.3f}')

# create data point for explanation
#idx =  7
#x_new_10_09 = data_x_10_09[[idx]]
#print(x_new_10_09)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_10_09_loghaz = func.survshapiq_ground_truth(data_x_10_09, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_10_09.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_10_09_df.columns)


func.plot_interact(explanations_all = explanation_10_09_loghaz, 
                              model = None,
                              times=model_gbsa_10_09.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_10_09.pdf",
                              data_x = data_x_10_09,
                              survival_fn = func.log_hazard_wrap_genadd_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 10) Generalized Additive G(t|x), TD Inter (interactions)
# corr = 0.5
#---------------------------

# load simulated data DataFrame
simdata_10_05 = pd.read_csv(f"{path_data}/10_simdata_genadd_tdinter_05.csv")
print(simdata_10_05.head())
simdata_10_05

# convert eventtime and status columns to a structured array
data_y_10_05, data_x_10_05_df = func.prepare_survival_data(simdata_10_05)
print(data_y_10_05)
print(data_x_10_05_df.head())
data_x_10_05 = data_x_10_05_df.values
X_train_10_05, X_test_10_05, y_train_10_05, y_test_10_05 = train_test_split(
    data_x_10_05, data_y_10_05, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_10_05 = GradientBoostingSurvivalAnalysis()
model_gbsa_10_05.fit(X_train_10_05, y_train_10_05)
print(f'C-index (train): {model_gbsa_10_05.score(X_test_10_05, y_test_10_05).item():0.3f}')
ibs_gbsa_10_05 = func.compute_integrated_brier(y_test_10_05, X_test_10_05, model_gbsa_10_05, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_10_05:0.3f}')

# fit CoxPH
model_cox_10_05= CoxPHSurvivalAnalysis()
model_cox_10_05.fit(X_train_10_05, y_train_10_05)
print(f'C-index (train): {model_cox_10_05.score(X_test_10_05, y_test_10_05).item():0.3f}')
ibs_cox_10_05 = func.compute_integrated_brier(y_test_10_05, X_test_10_05, model_cox_10_05, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_10_05:0.3f}')

# create data point for explanation
#idx =  7
#x_new_10_05 = data_x_10_05[[idx]]
#print(x_new_10_05)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_10_05_loghaz = func.survshapiq_ground_truth(data_x_10_05, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_10_05.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_10_05_df.columns)


func.plot_interact(explanations_all = explanation_10_05_loghaz, 
                              model = None,
                              times=model_gbsa_10_05.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_10_05.pdf",
                              data_x = data_x_10_05,
                              survival_fn = func.log_hazard_wrap_genadd_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 10) Generalized Additive G(t|x), TD Inter (interactions)
# corr = 0.2
#---------------------------

# load simulated data DataFrame
simdata_10_02 = pd.read_csv(f"{path_data}/10_simdata_genadd_tdinter_02.csv")
print(simdata_10_02.head())
simdata_10_02

# convert eventtime and status columns to a structured array
data_y_10_02, data_x_10_02_df = func.prepare_survival_data(simdata_10_02)
print(data_y_10_02)
print(data_x_10_02_df.head())
data_x_10_02 = data_x_10_02_df.values
X_train_10_02, X_test_10_02, y_train_10_02, y_test_10_02 = train_test_split(
    data_x_10_02, data_y_10_02    , 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_10_02 = GradientBoostingSurvivalAnalysis()
model_gbsa_10_02.fit(X_train_10_02, y_train_10_02)
print(f'C-index (train): {model_gbsa_10_02.score(X_test_10_02, y_test_10_02).item():0.3f}')
ibs_gbsa_10_02 = func.compute_integrated_brier(y_test_10_02, X_test_10_02, model_gbsa_10_02, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_10_02:0.3f}')

# fit CoxPH
model_cox_10_02 = CoxPHSurvivalAnalysis()
model_cox_10_02.fit(X_train_10_02, y_train_10_02)
print(f'C-index (train): {model_cox_10_02.score(X_test_10_02, y_test_10_02).item():0.3f}')
ibs_cox_10_02 = func.compute_integrated_brier(y_test_10_02, X_test_10_02, model_cox_10_02, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_10_02:0.3f}')

# create data point for explanation
#idx =  7
#x_new_10_02 = data_x_10_02[[idx]]
#print(x_new_10_02)

###### GROUND TRUTH LOG HAZARD
# exact
explanation_10_02_loghaz = func.survshapiq_ground_truth(data_x_10_02, 
                                                            x_new, 
                                                            func.log_hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_10_02.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_10_02_df.columns)
explanation_10_02_loghaz[0].mean()
explanation_10_02_loghaz[0].var()

func.plot_interact(explanations_all = explanation_10_02_loghaz, 
                              model = None,
                              times=model_gbsa_10_02.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_10_02.pdf",
                              data_x = data_x_10_02,
                              survival_fn = func.log_hazard_wrap_genadd_tdinter,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

################################################################################################################

# ============================================================
# 1. Collect all simulation results into one structured object
# ============================================================

scenarios = {
    1: {
        0.0: explanation_1_0_loghaz[0],
        0.2: explanation_1_02_loghaz[0],
        0.5: explanation_1_05_loghaz[0],
        0.9: explanation_1_09_loghaz[0],
    },
    2: {
        0.0: explanation_2_0_loghaz[0],
        0.2: explanation_2_02_loghaz[0],
        0.5: explanation_2_05_loghaz[0],
        0.9: explanation_2_09_loghaz[0],
    },
    3: {
        0.0: explanation_3_0_loghaz[0],
        0.2: explanation_3_02_loghaz[0],
        0.5: explanation_3_05_loghaz[0],
        0.9: explanation_3_09_loghaz[0],
    },
    4: {
        0.0: explanation_4_0_loghaz[0],
        0.2: explanation_4_02_loghaz[0],
        0.5: explanation_4_05_loghaz[0],
        0.9: explanation_4_09_loghaz[0],
    },
    5: {
        0.0: explanation_5_0_loghaz[0],
        0.2: explanation_5_02_loghaz[0],
        0.5: explanation_5_05_loghaz[0],
        0.9: explanation_5_09_loghaz[0],
    },
    6: {
        0.0: explanation_6_0_loghaz[0],
        0.2: explanation_6_02_loghaz[0],
        0.5: explanation_6_05_loghaz[0],
        0.9: explanation_6_09_loghaz[0],
    },
    7: {
        0.0: explanation_7_0_loghaz[0],
        0.2: explanation_7_02_loghaz[0],
        0.5: explanation_7_05_loghaz[0],
        0.9: explanation_7_09_loghaz[0],
    },
    8: {
        0.0: explanation_8_0_loghaz[0],
        0.2: explanation_8_02_loghaz[0],
        0.5: explanation_8_05_loghaz[0],
        0.9: explanation_8_09_loghaz[0],
    },
    9: {
        0.0: explanation_9_0_loghaz[0],
        0.2: explanation_9_02_loghaz[0],
        0.5: explanation_9_05_loghaz[0],
        0.9: explanation_9_09_loghaz[0],
    },
    10: {
        0.0: explanation_10_0_loghaz[0],
        0.2: explanation_10_02_loghaz[0],
        0.5: explanation_10_05_loghaz[0],
        0.9: explanation_10_09_loghaz[0],
    },
}

# Build the full table for all scenarios

def summarize_scenario(scenario_id, datasets):
    rows = []

    for corr, df in datasets.items():
        means = df.mean()
        vars_ = df.var()

        row = {
            "scenario": scenario_id,
            "correlation": corr
        }

        # Create cells like: "mean (variance)"
        for col in df.columns:
            row[col] = f"{means[col]:.4f} ({vars_[col]:.4f})"

        rows.append(row)

    return pd.DataFrame(rows)

all_tables = []

for scen_id, scen_data in scenarios.items():
    scen_table = summarize_scenario(scen_id, scen_data)
    all_tables.append(scen_table)

final_table = pd.concat(all_tables, ignore_index=True)

print(final_table)