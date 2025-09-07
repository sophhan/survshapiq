# import libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
import shapiq
from sklearn.model_selection import train_test_split
import importlib
import simulation.survshapiq_func as survshapiq_func
importlib.reload(survshapiq_func)

################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-INDEPENDENCE 
# Load simulated data DataFrame
simdata_linear_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_ti.csv")
print(simdata_linear_ti.head())
simdata_linear_ti

# Convert eventtime and status columns to a structured array
data_y_linear_ti, data_x_linear_ti_df = survshapiq_func.prepare_survival_data(simdata_linear_ti)
print(data_y_linear_ti)
print(data_x_linear_ti_df.head())
data_x_linear_ti = data_x_linear_ti_df.values
X_train_linear_ti, X_test_linear_ti, y_train_linear_ti, y_test_linear_ti = train_test_split(
    data_x_linear_ti, data_y_linear_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_ti.fit(X_train_linear_ti, y_train_linear_ti)
print(f'C-index (train): {model_gbsa_linear_ti.score(X_test_linear_ti, y_test_linear_ti).item():0.3f}')
ibs_gbsa_linear_ti = survshapiq_func.compute_integrated_brier(y_test_linear_ti, X_test_linear_ti, model_gbsa_linear_ti, min_time = 0.02, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_ti:0.3f}')

# Fit CoxPH
model_cox_linear_ti = CoxPHSurvivalAnalysis()
model_cox_linear_ti.fit(X_train_linear_ti, y_train_linear_ti)
print(f'C-index (train): {model_cox_linear_ti.score(X_test_linear_ti, y_test_linear_ti).item():0.3f}')
ibs_cox_linear_ti = survshapiq_func.compute_integrated_brier(y_test_linear_ti, X_test_linear_ti, model_cox_linear_ti, min_time = 0.02, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_ti:0.3f}')

# Create data point for explanation
idx = 10
x_new_linear_ti = data_x_linear_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_linear_ti)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_linear_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x1 * x3))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_linear_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_linear_ti, t)
# exact
explanation_linear_ti_haz = survshapiq_func.survshapiq_ground_truth(data_x_linear_ti, 
                                                            x_new_linear_ti, 
                                                            hazard_wrap_linear_ti, 
                                                            times=model_gbsa_linear_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_ti_haz, 
                              model = None,
                              times=model_gbsa_linear_ti.unique_times_[::5], 
                              x_new = x_new_linear_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_ti_haz.pdf",
                              data_x = data_x_linear_ti,
                              survival_fn = hazard_wrap_linear_ti,
                              idx_plot=idx, 
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_linear_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1)))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_linear_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_linear_ti, t)
# exact
explanation_linear_ti_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_linear_ti, 
                                                            x_new_linear_ti, 
                                                            log_hazard_wrap_linear_ti, 
                                                            times=model_gbsa_linear_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_ti_loghaz, 
                              model = None,
                              times=model_gbsa_linear_ti.unique_times_[::5], 
                              x_new = x_new_linear_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_ti_loghaz.pdf",
                              data_x = data_x_linear_ti,
                              survival_fn = log_hazard_wrap_linear_ti,
                              idx_plot=idx, 
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1)  

######### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_linear_ti_wrap (X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_linear_ti, t)
# k-SII
explanation_linear_ti_surv = survshapiq_func.survshapiq_ground_truth(data_x_linear_ti, 
                                                            x_new_linear_ti, 
                                                            surv_from_hazard_linear_ti_wrap, 
                                                            times=model_gbsa_linear_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_ti_surv, 
                              model = None,
                              times=model_gbsa_linear_ti.unique_times_[::5], 
                              x_new = x_new_linear_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_ti_surv.pdf",
                              data_x = data_x_linear_ti,
                              survival_fn = surv_from_hazard_linear_ti_wrap,
                              idx_plot=idx, 
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

########### MODEL SURVIVAL
# gbsg 
explanation_gbsa_linear_ti = survshapiq_func.survshapiq(model_gbsa_linear_ti, 
                                                    X_train_linear_ti, 
                                                    x_new_linear_ti,  
                                                    time_stride=5,
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_linear_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_gbsa_linear_ti, 
                              model = model_gbsa_linear_ti,
                              x_new = x_new_linear_ti, 
                              times=model_gbsa_linear_ti.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_linear_ti_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_linear_ti,
                              idx_plot=idx,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_cox_linear_ti = survshapiq_func.survshapiq(model_cox_linear_ti, 
                                                    X_train_linear_ti, 
                                                    x_new_linear_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_linear_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_cox_linear_ti, 
                              model = model_cox_linear_ti,
                              x_new = x_new_linear_ti, 
                              times=model_cox_linear_ti.unique_times_[::10],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_linear_ti_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_linear_ti,
                              idx_plot=idx,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 



################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN MAIN EFFECTS
# Load simulated data DataFrame
simdata_linear_tdmain = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_main.csv")
print(simdata_linear_tdmain.head())
simdata_linear_tdmain

# Convert eventtime and status columns to a structured array
data_y_linear_tdmain, data_x_linear_tdmain_df = survshapiq_func.prepare_survival_data(simdata_linear_tdmain)
print(data_y_linear_tdmain)
print(data_x_linear_tdmain_df.head())
data_x_linear_tdmain = data_x_linear_tdmain_df.values
X_train_linear_tdmain, X_test_linear_tdmain, y_train_linear_tdmain, y_test_linear_tdmain = train_test_split(
    data_x_linear_tdmain, data_y_linear_tdmain, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_tdmain = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_tdmain.fit(X_train_linear_tdmain, y_train_linear_tdmain)
print(f'C-index (train): {model_gbsa_linear_tdmain.score(X_test_linear_tdmain, y_test_linear_tdmain).item():0.3f}')
ibs_gbsa_linear_tdmain = survshapiq_func.compute_integrated_brier(y_test_linear_tdmain, X_test_linear_tdmain, model_gbsa_linear_tdmain, min_time = 0.01, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_tdmain:0.3f}')

# Fit CoxPH
model_cox_linear_tdmain = CoxPHSurvivalAnalysis()
model_cox_linear_tdmain.fit(X_train_linear_tdmain, y_train_linear_tdmain)
print(f'C-index (train): {model_cox_linear_tdmain.score(X_test_linear_tdmain, y_test_linear_tdmain).item():0.3f}')
ibs_cox_linear_tdmain = survshapiq_func.compute_integrated_brier(y_test_linear_tdmain, X_test_linear_tdmain, model_cox_linear_tdmain, min_time = 0.01, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdmain:0.3f}')

# Create data point for explanation
idx =  10
x_new_linear_tdmain = data_x_linear_tdmain[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_linear_tdmain)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_linear_tdmain(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * x1) + (-1.2 * x1 * np.log(t+1)) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_linear_tdmain(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_linear_tdmain, t)
# exact
explanation_linear_tdmain_haz = survshapiq_func.survshapiq_ground_truth(data_x_linear_tdmain, 
                                                            x_new_linear_tdmain, 
                                                            hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_linear_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdmain_haz, 
                              model = None,
                              times=model_gbsa_linear_tdmain.unique_times_[::5], 
                              x_new = x_new_linear_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdmain_haz.pdf", # plot_gt_td_haz_sm_5
                              data_x = data_x_linear_tdmain,
                              survival_fn = hazard_wrap_linear_tdmain,
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_linear_tdmain(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * x1) + (-1.2 * x1 * np.log(t+1)) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1)))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_linear_tdmain(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_linear_tdmain, t)
# exact
explanation_linear_tdmain_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_linear_tdmain, 
                                                            x_new_linear_tdmain, 
                                                            log_hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_linear_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdmain_loghaz, 
                              model = None,
                              times=model_gbsa_linear_tdmain.unique_times_[::5], 
                              x_new = x_new_linear_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdmain_loghaz.pdf", 
                              data_x = data_x_linear_tdmain,
                              survival_fn = log_hazard_wrap_linear_tdmain,
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_linear_tdmain_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_linear_tdmain, t)
# exact
explanation_linear_tdmain_surv = survshapiq_func.survshapiq_ground_truth(data_x_linear_tdmain, 
                                                            x_new_linear_tdmain, 
                                                            surv_from_hazard_linear_tdmain_wrap, 
                                                            times=model_gbsa_linear_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_df.columns)
survshapiq_func.plot_interact(explanations_all = explanation_linear_tdmain_surv, 
                              model = None,
                              times=model_gbsa_linear_tdmain.unique_times_[::5], 
                              x_new = x_new_linear_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdmain_surv.pdf",
                              data_x = data_x_linear_tdmain,
                              survival_fn = surv_from_hazard_linear_tdmain_wrap,
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsa
explanation_linear_tdmain_gbsa = survshapiq_func.survshapiq(model_gbsa_linear_tdmain, 
                                                    X_train_linear_tdmain, 
                                                    x_new_linear_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdmain_gbsa, 
                              model = model_gbsa_linear_tdmain,
                              x_new = x_new_linear_tdmain, 
                              times = model_gbsa_linear_tdmain.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_linear_tdmain_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_linear_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_linear_tdmain_cox = survshapiq_func.survshapiq(model_cox_linear_tdmain, 
                                                    X_train_linear_tdmain, 
                                                    x_new_linear_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdmain_cox, 
                              model = model_cox_linear_tdmain,
                              times=model_cox_linear_tdmain.unique_times_[::5],
                              x_new = x_new_linear_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_linear_tdmain_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_linear_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN INTERACTION EFFECTS
# Load simulated data DataFrame
simdata_linear_tdinter = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_inter.csv")
print(simdata_linear_tdinter.head())
simdata_linear_tdinter

# Convert eventtime and status columns to a structured array
data_y_linear_tdinter, data_x_linear_tdinter_df = survshapiq_func.prepare_survival_data(simdata_linear_tdinter)
print(data_y_linear_tdinter)
print(data_x_linear_tdinter_df.head())
data_x_linear_tdinter = data_x_linear_tdinter_df.values
X_train_linear_tdinter, X_test_linear_tdinter, y_train_linear_tdinter, y_test_linear_tdinter = train_test_split(
    data_x_linear_tdinter, data_y_linear_tdinter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_tdinter = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_tdinter.fit(X_train_linear_tdmain, y_train_linear_tdinter)
print(f'C-index (train): {model_gbsa_linear_tdinter.score(X_test_linear_tdmain, y_test_linear_tdinter).item():0.3f}')
ibs_gbsa_linear_tdinter = survshapiq_func.compute_integrated_brier(y_test_linear_tdinter, X_test_linear_tdmain, model_gbsa_linear_tdinter, min_time = 0.01, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_tdinter:0.3f}')

# Fit CoxPH
model_cox_linear_tdinter = CoxPHSurvivalAnalysis()
model_cox_linear_tdinter.fit(X_train_linear_tdmain, y_train_linear_tdinter)
print(f'C-index (train): {model_cox_linear_tdinter.score(X_test_linear_tdmain, y_test_linear_tdinter).item():0.3f}')
ibs_cox_linear_tdinter = survshapiq_func.compute_integrated_brier(y_test_linear_tdinter, X_test_linear_tdmain, model_cox_linear_tdinter, min_time = 0.01, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdinter:0.3f}')


# Create data point for explanation
idx =  10
x_new_linear_tdinter = data_x_linear_tdinter[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_linear_tdinter)
x_new_linear_tdmain

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_linear_tdinter(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1) + (-0.4 * x1 * x3 * np.log(t+1)))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_linear_tdinter(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_linear_tdinter, t)
# exact
explanation_linear_tdinter_haz = survshapiq_func.survshapiq_ground_truth(data_x_linear_tdinter, 
                                                            x_new_linear_tdinter, 
                                                            hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_linear_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdinter_haz, 
                              model = None,
                              times=model_gbsa_linear_tdinter.unique_times_[::5], 
                              x_new = x_new_linear_tdinter, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdinter_haz.pdf", # plot_gt_td_haz_sm_5
                              data_x = data_x_linear_tdinter,
                              survival_fn = hazard_wrap_linear_tdinter,
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_linear_tdinter(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1) + (-0.4 * x1 * x3 * np.log(t+1))))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_linear_tdinter(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_linear_tdinter, t)
# exact
explanation_linear_tdinter_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_linear_tdinter, 
                                                            x_new_linear_tdinter, 
                                                            log_hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_linear_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdinter_loghaz, 
                              model = None,
                              times=model_gbsa_linear_tdinter.unique_times_[::5], 
                              x_new = x_new_linear_tdinter, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdinter_loghaz.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_linear_tdinter,
                              survival_fn = log_hazard_wrap_linear_tdinter,
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_linear_tdinter_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_linear_tdinter, t)
# exact
explanation_linear_tdinter_surv = survshapiq_func.survshapiq_ground_truth(data_x_linear_tdinter, 
                                                            x_new_linear_tdinter, 
                                                            surv_from_hazard_linear_tdinter_wrap, 
                                                            times=model_gbsa_linear_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_linear_tdinter_df.columns)
survshapiq_func.plot_interact(explanations_all = explanation_linear_tdinter_surv, 
                              model = None,
                              times=model_gbsa_linear_tdinter.unique_times_[::5], 
                              x_new = x_new_linear_tdinter, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdinter_surv.pdf",
                              data_x = data_x_linear_tdinter,
                              survival_fn = surv_from_hazard_linear_tdinter_wrap,
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsa
explanation_linear_tdinter_gbsa = survshapiq_func.survshapiq(model_gbsa_linear_tdinter, 
                                                    X_train_linear_tdinter, 
                                                    x_new_linear_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdinter_gbsa, 
                              model = model_gbsa_linear_tdinter,
                              x_new = x_new_linear_tdinter, 
                              times = model_gbsa_linear_tdinter.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_linear_tdinter_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_linear_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_linear_tdinter_cox = survshapiq_func.survshapiq(model_cox_linear_tdinter, 
                                                    X_train_linear_tdinter, 
                                                    x_new_linear_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_linear_tdinter_cox, 
                              model = model_cox_linear_tdinter,
                              x_new = x_new_linear_tdinter, 
                              times = model_cox_linear_tdinter.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_linear_tdinter_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_linear_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


################ ADDITIVE MAIN EFFECTS MODEL
###### TIME-INDEPENDENCE 
# Load simulated data DataFrame
simdata_add_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_add_ti.csv")
print(simdata_add_ti.head())
simdata_add_ti

# Convert eventtime and status columns to a structured array
data_y_add_ti, data_x_add_ti_df = survshapiq_func.prepare_survival_data(simdata_add_ti)
print(data_y_add_ti)
print(data_x_add_ti_df.head())
data_x_add_ti = data_x_add_ti_df.values
X_train_add_ti, X_test_add_ti, y_train_add_ti, y_test_add_ti = train_test_split(
    data_x_add_ti, data_y_add_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_add_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_add_ti.fit(X_train_add_ti, y_train_add_ti)
print(f'C-index (train): {model_gbsa_add_ti.score(X_test_add_ti, y_test_add_ti).item():0.3f}')
ibs_gbsa_add_ti = survshapiq_func.compute_integrated_brier(y_test_add_ti, X_test_add_ti, model_gbsa_add_ti, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_add_ti:0.3f}')

# Fit CoxPH
model_cox_add_ti = CoxPHSurvivalAnalysis()
model_cox_add_ti.fit(X_train_add_ti, y_train_add_ti)
print(f'C-index (train): {model_cox_add_ti.score(X_test_add_ti, y_test_add_ti).item():0.3f}')
ibs_cox_add_ti = survshapiq_func.compute_integrated_brier(y_test_add_ti, X_test_add_ti, model_cox_add_ti, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_add_ti:0.3f}')

# Create data point for explanation
idx = 10
x_new_add_ti = data_x_add_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_add_ti)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_add_ti(t, x1, x2, x3):
    
    # baseline hazard * exp(lp)
    return 0.015 * np.exp(-1.5 * ((x1 ** 2) - 1) + (2/math.pi) * np.arctan(0.5 * x2) + 0.6 * x3)

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_add_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_add_ti, t)
# exact
explanation_add_ti_haz = survshapiq_func.survshapiq_ground_truth(data_x_add_ti, 
                                                            x_new_add_ti, 
                                                            hazard_wrap_add_ti, 
                                                            times=model_gbsa_add_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_add_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_add_ti_haz, 
                              model = None,
                              times=model_gbsa_add_ti.unique_times_[::5], 
                              x_new = x_new_add_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_add_ti_haz.pdf",
                              data_x = data_x_add_ti,
                              survival_fn = hazard_wrap_add_ti,
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_add_ti(t, x1, x2, x3):

    # log(baseline hazard * exp(lp))
    return np.log(0.015 * np.exp(-1.5 * ((x1 ** 2) - 1) + (2/math.pi) * np.arctan(0.5 * x2) + 0.6 * x3))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_add_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_add_ti, t)
# exact
explanation_add_ti_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_add_ti, 
                                                            x_new_add_ti, 
                                                            log_hazard_wrap_add_ti, 
                                                            times=model_gbsa_add_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_add_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_add_ti_loghaz, 
                              model = None,
                              times=model_gbsa_add_ti.unique_times_[::5], 
                              x_new = x_new_add_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_add_ti_loghaz.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_add_ti,
                              survival_fn = log_hazard_wrap_add_ti,
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_add_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_add_ti, t)
# exact
explanation_add_ti_surv = survshapiq_func.survshapiq_ground_truth(data_x_add_ti, 
                                                            x_new_add_ti, 
                                                            surv_from_hazard_add_ti_wrap, 
                                                            times=model_gbsa_add_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_add_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_add_ti_surv, 
                              model = None,
                              times=model_gbsa_add_ti.unique_times_[::5], 
                              x_new = x_new_add_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_add_ti_surv.pdf",
                              data_x = data_x_add_ti,
                              survival_fn = surv_from_hazard_add_ti_wrap,
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsa 
explanation_add_ti_gbsa = survshapiq_func.survshapiq(model_gbsa_add_ti, 
                                                    X_train_add_ti, 
                                                    x_new_add_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_add_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_add_ti_gbsa, 
                              model = model_gbsa_add_ti,
                              x_new = x_new_add_ti, 
                              times = model_gbsa_add_ti.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_add_ti_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_add_ti,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_add_ti_cox = survshapiq_func.survshapiq(model_cox_add_ti, 
                                                    X_train_add_ti, 
                                                    x_new_add_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_add_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_add_ti_cox, 
                              model = model_cox_add_ti,
                              x_new = x_new_add_ti, 
                              times=model_cox_add_ti.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_add_ti_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_add_ti,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 



################ GENERAL ADDITIVE MODEL
###### TIME-INDEPENDENCE 
# Load simulated data DataFrame
simdata_genadd_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_ti.csv")
print(simdata_genadd_ti.head())
simdata_genadd_ti

# Convert eventtime and status columns to a structured array
data_y_genadd_ti, data_x_genadd_ti_df = survshapiq_func.prepare_survival_data(simdata_genadd_ti)
print(data_y_genadd_ti)
print(data_x_genadd_ti_df.head())
data_x_genadd_ti = data_x_genadd_ti_df.values
X_train_genadd_ti, X_test_genadd_ti, y_train_genadd_ti, y_test_genadd_ti = train_test_split(
    data_x_genadd_ti, data_y_genadd_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)     

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_ti.fit(X_train_genadd_ti, y_train_genadd_ti)
print(f'C-index (train): {model_gbsa_genadd_ti.score(X_train_genadd_ti, y_train_genadd_ti).item():0.3f}')
ibs_gbsa_genadd_ti = survshapiq_func.compute_integrated_brier(y_train_genadd_ti, X_train_genadd_ti, model_gbsa_genadd_ti, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_ti:0.3f}')

# Fit CoxPH
model_cox_genadd_ti = CoxPHSurvivalAnalysis()
model_cox_genadd_ti.fit(X_train_genadd_ti, y_train_genadd_ti)
print(f'C-index (train): {model_cox_genadd_ti.score(X_train_genadd_ti, y_train_genadd_ti).item():0.3f}')
ibs_cox_genadd_ti = survshapiq_func.compute_integrated_brier(y_train_genadd_ti, X_train_genadd_ti, model_cox_genadd_ti, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_genadd_ti:0.3f}')

# Create data point for explanation
idx = 10
x_new_genadd_ti = data_x_genadd_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_genadd_ti)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_genadd_ti(t, x1, x2, x3):
    # nonlinear transformations
    return 0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_genadd_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_genadd_ti, t)
# exact
explanation_genadd_ti_haz = survshapiq_func.survshapiq_ground_truth(data_x_genadd_ti, 
                                                            x_new_genadd_ti, 
                                                            hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_genadd_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_ti_haz, 
                              model = None,
                              times=model_gbsa_genadd_ti.unique_times_[::5], 
                              x_new = x_new_genadd_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_ti_haz.pdf",
                              data_x = data_x_genadd_ti,
                              survival_fn = hazard_wrap_genadd_ti,
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=200,
                              smooth_poly=1) 


###### GROUND TRUTH LOG HAZARD
# Define the hazard function
def log_hazard_func_genadd_ti(t, x1, x2, x3):
    # log(baseline hazard * exp(lp))
    return  np.log(0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3)))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_genadd_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_genadd_ti, t)
# exact
explanation_genadd_ti_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_genadd_ti, 
                                                            x_new_genadd_ti, 
                                                            log_hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_genadd_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_ti_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_ti.unique_times_[::5], 
                              x_new = x_new_genadd_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_ti_loghaz.pdf",
                              data_x = data_x_genadd_ti,
                              survival_fn = log_hazard_wrap_genadd_ti,
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Wrap the survival function
def surv_from_hazard_genadd_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_genadd_ti, t)
# exact
explanation_genadd_ti_surv = survshapiq_func.survshapiq_ground_truth(data_x_genadd_ti, 
                                                            x_new_genadd_ti, 
                                                            surv_from_hazard_genadd_ti_wrap, 
                                                            times=model_gbsa_genadd_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_df.columns)
survshapiq_func.plot_interact(explanations_all = explanation_genadd_ti_surv, 
                              model = None,
                              times=model_gbsa_genadd_ti.unique_times_[::5], 
                              x_new = x_new_genadd_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_ti_surv.pdf",
                              data_x = data_x_genadd_ti,
                              survival_fn = surv_from_hazard_genadd_ti_wrap,
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsa 
explanation_genadd_ti_gbsa = survshapiq_func.survshapiq(model_gbsa_genadd_ti, 
                                                    X_train_genadd_ti, 
                                                    x_new_genadd_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_genadd_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_ti_gbsa, 
                              model = model_gbsa_genadd_ti,
                              x_new = x_new_genadd_ti, 
                              times=model_gbsa_genadd_ti.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_genadd_ti_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_genadd_ti,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_cox_genadd_ti = survshapiq_func.survshapiq(model_cox_genadd_ti, 
                                                    X_train_genadd_ti, 
                                                    x_new_genadd_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_genadd_ti_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_cox_genadd_ti, 
                              model = model_cox_genadd_ti,
                              x_new = x_new_genadd_ti, 
                              times=model_cox_genadd_ti.unique_times_[::5] ,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_genadd_ti_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_genadd_ti,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


################ GENERAL ADDITIVE MODEL
###### TIME-DEPENDENCE IN MAIN EFFECTS
# Load simulated data DataFrame
simdata_genadd_tdmain = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_main.csv")
print(simdata_genadd_tdmain.head())
simdata_genadd_tdmain

# Convert eventtime and status columns to a structured array
data_y_genadd_tdmain, data_x_genadd_tdmain_df = survshapiq_func.prepare_survival_data(simdata_genadd_tdmain)
print(data_y_genadd_tdmain)
print(data_x_genadd_tdmain_df.head())
data_x_genadd_tdmain = data_x_genadd_tdmain_df.values
X_train_genadd_tdmain, X_test_genadd_tdmain, y_train_genadd_tdmain, y_test_genadd_tdmain = train_test_split(
    data_x_genadd_tdmain, data_y_genadd_tdmain, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_tdmain = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_tdmain.fit(X_train_genadd_tdmain, y_train_genadd_tdmain)
print(f'C-index (train): {model_gbsa_genadd_tdmain.score(X_test_genadd_tdmain, y_test_genadd_tdmain).item():0.3f}')
ibs_gbsa_genadd_tdmain = survshapiq_func.compute_integrated_brier(y_test_genadd_tdmain, X_test_genadd_tdmain, model_gbsa_genadd_tdmain, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_tdmain:0.3f}')

# Fit CoxPH
model_cox_genadd_tdmain = CoxPHSurvivalAnalysis()
model_cox_genadd_tdmain.fit(X_train_genadd_tdmain, y_train_genadd_tdmain)
print(f'C-index (train): {model_cox_genadd_tdmain.score(X_test_genadd_tdmain, y_test_genadd_tdmain).item():0.3f}')
ibs_cox_genadd_tdmain = survshapiq_func.compute_integrated_brier(y_test_genadd_tdmain, X_test_genadd_tdmain, model_cox_genadd_tdmain, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_genadd_tdmain:0.3f}')

# Create data point for explanation
idx = 10
x_new_genadd_tdmain = data_x_genadd_tdmain[[idx]]
#x_new_td = data_x_td[1:9]
print(x_new_genadd_tdmain)


###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_genadd_tdmain(t, x1, x2, x3):
    # baseline hazard * exp(lp)
    return 0.01 * np.exp(0.2 * x1 - 0.4 * (x1 * np.log(t + 1)) - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_genadd_tdmain(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_genadd_tdmain, t)
# exact
explanation_genadd_tdmain_haz = survshapiq_func.survshapiq_ground_truth(data_x_genadd_tdmain, 
                                                            x_new_genadd_tdmain, 
                                                            hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdmain_haz, 
                              model = None,
                              times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdmain_haz.pdf",
                              data_x = data_x_genadd_tdmain,
                              survival_fn = hazard_wrap_genadd_tdmain,
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=150,
                              smooth_poly=1) 


###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_genadd_tdmain(t, x1, x2, x3):
    # baseline hazard * exp(lp)
    return np.log(0.01 * np.exp(0.2 * x1 - 0.4 * (x1 * np.log(t + 1)) - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3)))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_genadd_tdmain(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_genadd_tdmain, t)
# exact
explanation_genadd_tdmain_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_genadd_tdmain, 
                                                            x_new_genadd_tdmain, 
                                                            log_hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdmain_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdmain_loghaz.pdf",
                              data_x = data_x_genadd_tdmain,
                              survival_fn = log_hazard_wrap_genadd_tdmain,
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Wrap the survival function
def surv_from_hazard_genadd_tdmain_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_genadd_tdmain, t)
# exact
explanation_genadd_tdmain_surv = survshapiq_func.survshapiq_ground_truth(data_x_genadd_tdmain, 
                                                            x_new_genadd_tdmain, 
                                                            surv_from_hazard_genadd_tdmain_wrap, 
                                                            times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_df.columns)
survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdmain_surv, 
                              model = None,
                              times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdmain_surv.pdf",
                              data_x = data_x_genadd_tdmain,
                              survival_fn = surv_from_hazard_genadd_tdmain_wrap,
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsg 
explanation_genadd_tdmain_gbsa = survshapiq_func.survshapiq(model_gbsa_genadd_tdmain, 
                                                    X_train_genadd_tdmain, 
                                                    x_new_genadd_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdmain_gbsa, 
                              model = model_gbsa_genadd_tdmain,
                              x_new = x_new_genadd_tdmain, 
                              times=model_gbsa_genadd_tdmain.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_genadd_tdmain_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_genadd_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_genadd_tdmain_cox = survshapiq_func.survshapiq(model_cox_genadd_tdmain, 
                                                    X_train_genadd_tdmain, 
                                                    x_new_genadd_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdmain_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdmain_cox, 
                              model = model_cox_genadd_tdmain,
                              x_new = x_new_genadd_tdmain, 
                              times=model_cox_genadd_tdmain.unique_times_[::5] ,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_genadd_tdmain_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_genadd_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


################ GENERAL ADDITIVE MODEL
###### TIME-DEPENDENCE IN INTERACTIONS
# Load simulated data DataFrame
simdata_genadd_tdinter = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_interaction.csv")
print(simdata_genadd_tdinter.head())
simdata_genadd_tdinter

# Convert eventtime and status columns to a structured array
data_y_genadd_tdinter, data_x_genadd_tdinter_df = survshapiq_func.prepare_survival_data(simdata_genadd_tdinter)
print(data_y_genadd_tdinter)
print(data_x_genadd_tdinter_df.head())
data_x_genadd_tdinter = data_x_genadd_tdinter_df.values
X_train_genadd_tdinter, X_test_genadd_tdinter, y_train_genadd_tdinter, y_test_genadd_tdinter = train_test_split(
    data_x_genadd_tdinter, data_y_genadd_tdinter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_tdinter = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_tdinter.fit(X_train_genadd_tdinter, y_train_genadd_tdinter)
print(f'C-index (train): {model_gbsa_genadd_tdinter.score(X_test_genadd_tdinter, y_test_genadd_tdinter).item():0.3f}')
ibs_gbsa_genadd_tdinter = survshapiq_func.compute_integrated_brier(y_test_genadd_tdinter, X_test_genadd_tdinter, model_gbsa_genadd_tdinter, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_tdinter:0.3f}')

# Fit CoxPH
model_cox_genadd_tdinter = CoxPHSurvivalAnalysis()
model_cox_genadd_tdinter.fit(X_train_genadd_tdinter, y_train_genadd_tdinter)
print(f'C-index (train): {model_cox_genadd_tdinter.score(X_test_genadd_tdinter, y_test_genadd_tdinter).item():0.3f}')
ibs_cox_genadd_tdinter = survshapiq_func.compute_integrated_brier(y_test_genadd_tdinter, X_test_genadd_tdinter, model_cox_genadd_tdinter, min_time = 0.04, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_genadd_tdinter:0.3f}')

# Create data point for explanation
idx = 10 #1,8
x_new_genadd_tdinter = data_x_genadd_tdinter[[idx]]
#x_new_td = data_x_td[1:9]
print(x_new_genadd_tdinter)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_genadd_tdinter(t, x1, x2, x3):
    # baseline hazard * exp(lp)
    return 0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 - 0.4 * (x1 * x2 * np.log(t + 1)) + 0.3 * ((x1 ** 2 - 1) * x3))

# Wrap the hazard function
def hazard_wrap_genadd_tdinter(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_genadd_tdinter, t)
# exact
explanation_genadd_tdinter_haz = survshapiq_func.survshapiq_ground_truth(data_x_genadd_tdinter, 
                                                            x_new_genadd_tdinter, 
                                                            hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdinter_haz, 
                              model = None,
                              times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                              x_new = x_new_genadd_tdinter, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdinter_haz.pdf",
                              data_x = data_x_genadd_tdinter,
                              survival_fn = hazard_wrap_genadd_tdinter,
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=150,
                              smooth_poly=1) 



###### GROUND TRUTH LOG HAZARD
# Define the hazard function
def log_hazard_func_genadd_tdinter(t, x1, x2, x3):
    # baseline hazard * exp(lp)
    return np.log(0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 - 0.4 * (x1 * x2 * np.log(t + 1)) + 0.3 * ((x1 ** 2 - 1) * x3)))

# Wrap the hazard function
def log_hazard_wrap_genadd_tdinter(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_genadd_tdinter, t)
# exact
explanation_genadd_tdinter_loghaz = survshapiq_func.survshapiq_ground_truth(data_x_genadd_tdinter, 
                                                            x_new_genadd_tdinter, 
                                                            log_hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdinter_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                              x_new = x_new_genadd_tdinter, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdinter_loghaz.pdf",
                              data_x = data_x_genadd_tdinter,
                              survival_fn = log_hazard_wrap_genadd_tdinter,
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Wrap the survival function
def surv_from_hazard_genadd_tdinter_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_genadd_tdinter, t)
# exact
explanation_genadd_tdinter_surv = survshapiq_func.survshapiq_ground_truth(data_x_genadd_tdinter, 
                                                            x_new_genadd_tdinter, 
                                                            surv_from_hazard_genadd_tdinter_wrap, 
                                                            times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdinter_df.columns)
survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdinter_surv, 
                              model = None,
                              times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                              x_new = x_new_genadd_tdinter, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdinter_surv.pdf",
                              data_x = data_x_genadd_tdinter,
                              survival_fn = surv_from_hazard_genadd_tdinter_wrap,
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsa
explanation_genadd_tdinter_gbsa = survshapiq_func.survshapiq(model_gbsa_genadd_tdinter, 
                                                    X_train_genadd_tdinter, 
                                                    x_new_genadd_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdinter_gbsa, 
                              model = model_gbsa_genadd_tdinter,
                              x_new = x_new_genadd_tdinter, 
                              times=model_gbsa_genadd_tdinter.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_genadd_tdinter_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_genadd_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_genadd_tdinter_cox = survshapiq_func.survshapiq(model_cox_genadd_tdinter, 
                                                    X_train_genadd_tdinter, 
                                                    x_new_genadd_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdinter_df.columns)

survshapiq_func.plot_interact(explanations_all = explanation_genadd_tdinter_cox, 
                              model = model_cox_genadd_tdinter,
                              x_new = x_new_genadd_tdinter, 
                              times = model_cox_genadd_tdinter.unique_times_[::5],
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_genadd_tdinter_surv.pdf",
                              compare_plots = True, 
                              data_x = data_x_genadd_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

#########################
t = 0
explanations_all = explanation_genadd_tdinter_cox
explanations_all[0]
row_sums = np.array(explanations_all[0].sum(axis=1))
row_sums
model = model_cox_genadd_tdinter
times = model.unique_times_[::5]
surv_funcs = model.predict_survival_function(data_x_genadd_tdinter)
surv_matrix = np.vstack([sf(times) for sf in surv_funcs]
surv_matrix.shape
idx_plot = 10
surv_matrix_id = surv_matrix[, t]
surv_matrix_id.shape
np.mean(surv_matrix_id - row_sums, axis=0)
surv_matrix_id - row_sums


mean_surv = np.sqrt(np.mean((surv_matrix_id - row_sums), axis=0)**2/np.mean((surv_matrix_id)**2, axis=0))
mean_surv



####### COMBINED PLOTS
#################################################################################################
############## TIME-INDEPENDENCE HAZARD
# Create figure
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_ti_haz[0],
     explanation_add_ti_haz[0],
     explanation_genadd_ti_haz[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_add_ti.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5]],
    [data_x_linear_ti, data_x_add_ti, data_x_genadd_ti],
    [hazard_wrap_linear_ti, hazard_wrap_add_ti, hazard_wrap_genadd_ti],
    ["GT: Linear RF", "GT: Additive RF", "GT: General Additive RF"]
    
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel="Attribution $h(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_ti_haz.pdf"
fig.savefig(save_path, bbox_inches="tight")

################### TIME-INDEPENDENCE LOG HAZARD
# Create figure
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_ti_loghaz[0],
     explanation_add_ti_loghaz[0],
     explanation_genadd_ti_loghaz[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_add_ti.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5]],
    [data_x_linear_ti, data_x_add_ti, data_x_genadd_ti],
    [log_hazard_wrap_linear_ti, log_hazard_wrap_add_ti, log_hazard_wrap_genadd_ti],
    ["GT: Linear RF", "GT: Additive RF", "GT: General Additive RF"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel="Attribution $\log(h(t|x))$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_ti_loghaz.pdf"
fig.savefig(save_path, bbox_inches="tight")

################### TIME-INDEPENDENCE SURVIVAL
# Create figure
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_ti_surv[0],
     explanation_add_ti_surv[0],
     explanation_genadd_ti_surv[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_add_ti.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5]],
    [data_x_linear_ti, data_x_add_ti, data_x_genadd_ti],
    [surv_from_hazard_linear_ti_wrap, surv_from_hazard_add_ti_wrap, surv_from_hazard_genadd_ti_wrap],
    ["GT: Linear RF", "GT: Additive RF", "GT: General Additive RF"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel="Attribution $S(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_ti_surv.pdf"
fig.savefig(save_path, bbox_inches="tight")

################### TIME-INDEPENDENCE GBSA
# Create figure
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, model, title in zip(
    axes,
    [explanation_gbsa_linear_ti[0],
     explanation_add_ti_gbsa[0],
     explanation_genadd_ti_gbsa[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_add_ti.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5]],
    [data_x_linear_ti, data_x_add_ti, data_x_genadd_ti],
    [model_gbsa_linear_ti, model_gbsa_add_ti, model_gbsa_genadd_ti],
    ["GBSA: Linear RF", "GBSA: Additive RF", "GBSA: General Additive RF"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        model=model,
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_ti_gbsa.pdf"
fig.savefig(save_path, bbox_inches="tight")

################### TIME-INDEPENDENCE COXPH
# Create figure
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, model, title in zip(
    axes,
    [explanation_cox_linear_ti[0],
     explanation_add_ti_cox[0],
     explanation_cox_genadd_ti[0]],
    [model_cox_linear_ti.unique_times_[::5],
     model_cox_add_ti.unique_times_[::5],
     model_cox_genadd_ti.unique_times_[::5]],
    [data_x_linear_ti, data_x_add_ti, data_x_genadd_ti],
    [model_cox_linear_ti, model_cox_add_ti, model_cox_genadd_ti],
    ["CoxPH: Linear RF", "CoxPH: Additive RF", "CoxPH: General Additive RF"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        model=model,
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_ti_cox.pdf"
fig.savefig(save_path, bbox_inches="tight")


############## TIME-DEPENDENCE HAZARD
# Create figure
fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_tdmain_haz[0],
     explanation_linear_tdinter_haz[0],
     explanation_genadd_tdmain_haz[0],
     explanation_genadd_tdinter_haz[0]],
    [model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_tdmain, data_x_linear_tdinter, data_x_genadd_tdmain, data_x_genadd_tdinter],
    [hazard_wrap_linear_tdmain, hazard_wrap_linear_tdinter, hazard_wrap_genadd_tdmain, hazard_wrap_genadd_tdinter],
    ["GT: Linear RF TD Main Effect", "GT: Linear RF TD Interaction", "General Additive RF TD Main Effect", "General Additive RF TD Interaction"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel="Attribution $h(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_haz.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## TIME-DEPENDENCE LOG HAZARD
# Create figure
fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_tdmain_loghaz[0],
     explanation_linear_tdinter_loghaz[0],
     explanation_genadd_tdmain_loghaz[0],
     explanation_genadd_tdinter_loghaz[0]],
    [model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_tdmain, data_x_linear_tdinter, data_x_genadd_tdmain, data_x_genadd_tdinter],
    [log_hazard_wrap_linear_tdmain, log_hazard_wrap_linear_tdinter, log_hazard_wrap_genadd_tdmain, log_hazard_wrap_genadd_tdinter],
    ["GT: Linear RF TD Main Effect", "GT: Linear RF TD Interaction", "GT: General Additive RF TD Main Effect", "GT: General Additive RF TD Interaction"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel="Attribution $\log(h(t|x))$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_loghaz.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## TIME-DEPENDENCE SURVIVAL
# Create figure
fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_tdmain_surv[0],
     explanation_linear_tdinter_surv[0],
     explanation_genadd_tdmain_surv[0],
     explanation_genadd_tdinter_surv[0]],
    [model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_tdmain, data_x_linear_tdinter, data_x_genadd_tdmain, data_x_genadd_tdinter],
    [surv_from_hazard_linear_tdmain_wrap, surv_from_hazard_linear_tdinter_wrap, surv_from_hazard_genadd_tdmain_wrap, surv_from_hazard_genadd_tdinter_wrap],
    ["GT: Linear RF TD Main Effect", "GT: Linear RF TD Interaction", "GT: General Additive RF TD Main Effect", "GT: General Additive RF TD Interaction"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel="Attribution $S(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_surv.pdf"
fig.savefig(save_path, bbox_inches="tight")

################### TIME-DEPENDENCE GBSA
# Create figure
fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, model, title in zip(
    axes,
    [explanation_linear_tdmain_gbsa[0],
     explanation_linear_tdinter_gbsa[0],
     explanation_genadd_tdmain_gbsa[0],
     explanation_genadd_tdinter_gbsa[0]],
    [model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_tdmain, data_x_linear_tdinter, data_x_genadd_tdmain, data_x_genadd_tdinter],
    [model_gbsa_linear_tdmain, model_gbsa_linear_tdinter, model_gbsa_genadd_tdmain, model_gbsa_genadd_tdinter],
    ["GBSA: Linear RF TD Main Effect", "GBSA: Linear RF TD Interaction", "GBSA: General Additive RF TD Main Effect", "GBSA: General Additive RF TD Interaction"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        model=model,
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_gbsa.pdf"
fig.savefig(save_path, bbox_inches="tight")

################### TIME-DEPENDENCE COXPH
# Create figure
fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, model, title in zip(
    axes,
    [explanation_linear_tdmain_cox[0],
     explanation_linear_tdinter_cox[0],
     explanation_genadd_tdmain_cox[0],
     explanation_genadd_tdinter_cox[0]],
    [model_cox_linear_tdmain.unique_times_[::5],
     model_cox_linear_tdinter.unique_times_[::5],
     model_cox_genadd_tdmain.unique_times_[::5],
     model_cox_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_tdmain, data_x_linear_tdinter, data_x_genadd_tdmain, data_x_genadd_tdinter],
    [model_cox_linear_tdmain, model_cox_linear_tdinter, model_cox_genadd_tdmain, model_cox_genadd_tdinter],
    ["CoxPH: Linear RF TD Main Effect", "CoxPH: Linear RF TD Interaction", "CoxPH: General Additive RF TD Main Effect", "CoxPH: General Additive RF TD Interaction"]
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        model=model,
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1, 
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_cox.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## TIME-INDEPENDENCE EXPERIMENTS
# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title, ytext in zip(
    axes.flat,
    [explanation_linear_ti_haz[0],
     explanation_genadd_ti_haz[0],
     explanation_add_ti_loghaz[0],
     explanation_linear_ti_surv[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5],
     model_gbsa_add_ti.unique_times_[::5],
     model_gbsa_linear_ti.unique_times_[::5]],
    [data_x_linear_ti, data_x_genadd_ti, data_x_add_ti, data_x_genadd_ti],
    [hazard_wrap_linear_ti, hazard_wrap_genadd_ti, log_hazard_wrap_add_ti, surv_from_hazard_linear_ti_wrap],
    ["GT: Linear RF", "GT: General Additive RF", "GT: Additive RF", "GT: Linear RF"],
    ["Attribution $h(t|x)$", "Attribution $h(t|x)$", "Attribution $\log(h(t|x))$", "Attribution $S(t|x)$"]
    
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel=ytext,
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_ti_exp.pdf"
fig.savefig(save_path, bbox_inches="tight")


############## TIME-DEPENDENCE EXPERIMENTS (LOG)HAZARD
# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title, ytext in zip(
    axes.flat,
    [explanation_linear_tdmain_haz[0],
     explanation_linear_tdmain_loghaz[0]],
    [model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_tdmain.unique_times_[::5]],
    [data_x_linear_tdmain, data_x_linear_tdmain],
    [hazard_wrap_linear_tdmain, log_hazard_wrap_genadd_tdmain],
    ["GT: Linear RF TD Main Effect", "GT: Linear RF TD Main Effect"],
    ["Attribution $h(t|x)$", "Attribution $\log(h(t|x))$"]
    
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel=ytext,
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.16, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_exp_loghaz.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## TIME-DEPENDENCE EXPERIMENTS (LOG)HAZARD
# Create figure
fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True)

# Initialize lists to collect all handles and labels
handles_all, labels_all = [], []

# Plot each subplot and collect handles/labels
for ax, expl, times, data_x, surv_fn, title, ytext, model in zip(
    axes.flat,
    [explanation_genadd_tdinter_surv[0],
     explanation_genadd_tdinter_gbsa[0],
     explanation_genadd_tdinter_cox[0]],
    [model_gbsa_genadd_tdinter.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_genadd_tdinter, data_x_genadd_tdinter, data_x_genadd_tdinter],
    [surv_from_hazard_genadd_tdinter_wrap, None, None],
    ["GT: General Additive RF TD Interaction", "GBSA: General Additive RF TD Interaction", "CoxPH: General Additive RF TD Interaction"],
    ["Attribution $S(t|x)$", "Attribution $\hat{S}(t|x)$", "Attribution $\hat{S}(t|x)$"],
    [None, model_gbsa_genadd_tdinter, model_cox_genadd_tdinter]
    
):
    h, l = survshapiq_func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        model=model,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        idx_plot=idx,
        ylabel=ytext,
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title
    )
    handles_all.extend(h)
    labels_all.extend(l)

# Remove duplicates
unique = dict(zip(labels_all, handles_all))

# One shared legend at the bottom
fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=4,
    fontsize=14
)

# Adjust layout to leave space for legend
plt.tight_layout(rect=[0, 0.16, 1, 1])

# Save the figure
save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/plot_td_exp_surv.pdf"
fig.savefig(save_path, bbox_inches="tight")