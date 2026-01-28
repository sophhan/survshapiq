# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
import shapiq
import xgboost
from sklearn.model_selection import train_test_split
import importlib
import sys

#from survshapiq.survshapiq_repo.simulation.sim_survshapiq import X_test_linear_tdinter
sys.path.append("/survshapiq/survshapiq/")
import simulation.func as func
importlib.reload(func)

import os
print(os.getcwd())

# define paths
path_data = "/survshapiq/simulation/data"
path_plots = "/survshapiq/simulation/plots_corr"
path_plots_combined = "/survshapiq/simulation/plots_combined"

#---------------------------
# Marginal SurvSHAP-IQ
#---------------------------

# load simulated data DataFrame
simdata_marg= pd.read_csv(f"{path_data}/simdata_marg_cond.csv")
print(simdata_marg.head())

# convert eventtime and status columns to a structured array
data_y_marg, data_x_marg_df = func.prepare_survival_data(simdata_marg)
print(data_y_marg)
print(data_x_marg_df.head())
data_x_marg = data_x_marg_df.values
X_train_marg, X_test_marg, y_train_marg, y_test_marg = train_test_split(
    data_x_marg, data_y_marg, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_marg = GradientBoostingSurvivalAnalysis()
model_gbsa_marg.fit(X_train_marg, y_train_marg)
print(f'C-index (train): {model_gbsa_marg.score(X_test_marg, y_test_marg).item():0.3f}')
ibs_gbsa_marg = func.compute_integrated_brier(y_test_marg, X_test_marg, model_gbsa_marg, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_marg:0.3f}')

# fit CoxPH
model_cox_marg = CoxPHSurvivalAnalysis()
model_cox_marg.fit(X_train_marg, y_train_marg)
print(f'C-index (train): {model_cox_marg.score(X_test_marg, y_test_marg).item():0.3f}')
ibs_cox_marg = func.compute_integrated_brier(y_test_marg, X_test_marg, model_cox_marg, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_marg:0.3f}')

# create data point for explanation
idx =  7
simdata_obs= pd.read_csv(f"{path_data}/1_simdata_linear_ti.csv")
data_y_obs, data_x_obs_df = func.prepare_survival_data(simdata_obs)
data_x_obs = data_x_obs_df.values
X_train_obs, X_test_obs, y_train_obs, y_test_obs = train_test_split(
    data_x_obs, data_y_obs, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)
x_new = data_x_obs[[idx]]
print(x_new)


###### GROUND TRUTH LOG HAZARD
# marginal 
explanation_marg_loghaz = func.survshapiq_ground_truth(data_x_marg, 
                                                            x_new, 
                                                            func.log_hazard_wrap_marg_cond, 
                                                            times=model_gbsa_marg.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_marg_df.columns)

func.plot_interact(explanations_all = explanation_marg_loghaz, 
                              model = None,
                              times=model_gbsa_marg.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_marg.pdf",
                              data_x = data_x_marg,
                              survival_fn = func.log_hazard_wrap_marg_cond,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


# conditional
explanation_cond_loghaz = func.survshapiq_ground_truth(data_x_marg, 
                                                            x_new, 
                                                            func.log_hazard_wrap_marg_cond, 
                                                            times=model_gbsa_marg.unique_times_[::5], 
                                                            budget=2**3, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            imputer="conditional",
                                                            feature_names = data_x_marg_df.columns)

func.plot_interact(explanations_all = explanation_cond_loghaz, 
                              model = None,
                              times=model_gbsa_marg.unique_times_[::5], 
                              x_new = x_new, 
                              save_path = f"{path_plots}/plot_loghaz_cond.pdf",
                              data_x = data_x_marg,
                              survival_fn = func.log_hazard_wrap_marg_cond,
                              compare_plots="Diff",
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 
