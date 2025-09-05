# import libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
from sksurv.metrics import integrated_brier_score
import shapiq
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
data_y_ti, data_x = survshapiq_func.prepare_survival_data(simdata_linear_ti)
print(data_y_ti)
print(data_x.head())
data_x_ti = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_gbsa_ti.score(data_x_ti, data_y_ti).item():0.3f}')
ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti, min_time = 0.004, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')

# Fit CoxPH
model_cox_ti = CoxPHSurvivalAnalysis()
model_cox_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_cox_ti.score(data_x_ti, data_y_ti).item():0.3f}')
ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti, min_time = 0.004, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')

# Create data point for explanation
idx = 17
x_new_ti = data_x_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_ti)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_ti(t, age, bmi, treatment):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * age) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)
# exact
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_ti_haz_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_ti(t, age, bmi, treatment):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * age) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age)))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)
# exact
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            log_hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_ti_loghaz_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = log_hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1)  

######### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_ti_wrap (X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)
# k-SII
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            surv_from_hazard_ti_wrap, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_ti_surv_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = surv_from_hazard_ti_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 
########### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_ti = survshapiq_func.survshapiq(model_gbsa_ti, 
                                                    data_x_ti, 
                                                    x_new_ti, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_ti, 
                              model = model_gbsa_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_linear_ti_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_df_cox_ti = survshapiq_func.survshapiq(model_cox_ti, 
                                                    data_x_ti, 
                                                    x_new_ti, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_ti, 
                              model = model_cox_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_coxph_linear_ti_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 



################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN MAIN EFFECTS
# Load simulated data DataFrame
simdata_main_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_main.csv")
print(simdata_main_td.head())
simdata_main_td

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_main_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_gbsa_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_gbsa_td, min_time = 0.004, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_td:0.3f}')

# Fit CoxPH
model_cox_td = CoxPHSurvivalAnalysis()
model_cox_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_cox_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_cox_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_cox_td, min_time = 0.004, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_td:0.3f}')


# Create data point for explanation
idx =  17
x_new_td = data_x_td[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_td)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_td(t, age, bmi, treatment):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * age) + (-1.2 * age * np.log(t+1)) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_td(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_td, t)
# exact
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdmain_haz_17.pdf", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_td(t, age, bmi, treatment):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * age) + (-1.2 * age * np.log(t+1)) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age)))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_td(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_td, t)
# exact
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            log_hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdmain_loghaz_17.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = log_hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_td_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_td, t)
# exact
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            surv_from_hazard_td_wrap, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdmain_surv_17.pdf",
                              data_x = data_x_td,
                              survival_fn = surv_from_hazard_td_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_td = survshapiq_func.survshapiq(model_gbsa_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_td, 
                              model = model_gbsa_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_linear_tdmain_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_df_cox_td = survshapiq_func.survshapiq(model_cox_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_td, 
                              model = model_cox_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_linear_tdmain_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN INTERACTION EFFECTS
# Load simulated data DataFrame
simdata_inter_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_inter.csv")
print(simdata_inter_td.head())
simdata_inter_td

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_inter_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_gbsa_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_gbsa_td, min_time = 0.004, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_td:0.3f}')

# Fit CoxPH
model_cox_td = CoxPHSurvivalAnalysis()
model_cox_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_cox_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_cox_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_cox_td, min_time = 0.004, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_td:0.3f}')


# Create data point for explanation
idx =  17
x_new_td = data_x_td[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_td)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_td(t, age, bmi, treatment):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * age) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age) + (-0.4 * age * treatment * np.log(t+1)))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_td(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_td, t)
# exact
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdinter_haz_17.pdf", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_td(t, age, bmi, treatment):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * age) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age) + (-0.4 * age * treatment * np.log(t+1))))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_td(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_td, t)
# exact
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            log_hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdinter_loghaz_17.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = log_hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_td_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_td, t)
# exact
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            surv_from_hazard_td_wrap, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_linear_tdinter_surv_17.pdf",
                              data_x = data_x_td,
                              survival_fn = surv_from_hazard_td_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_td = survshapiq_func.survshapiq(model_gbsa_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_td, 
                              model = model_gbsa_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_linear_tdinter_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# coxph
explanation_df_cox_td = survshapiq_func.survshapiq(model_cox_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_td, 
                              model = model_cox_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_linear_tdinter_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
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
data_y_ti, data_x = survshapiq_func.prepare_survival_data(simdata_add_ti)
print(data_y_ti)
print(data_x.head())
data_x_ti = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_gbsa_ti.score(data_x_ti, data_y_ti).item():0.3f}')
#ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')

# Fit CoxPH
model_cox_ti = CoxPHSurvivalAnalysis()
model_cox_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_cox_ti.score(data_x_ti, data_y_ti).item():0.3f}')
#ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')

# Create data point for explanation
idx = 1
x_new_ti = data_x_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_ti)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_add(t, x1, x2, x3):
    """
    Hazard function with nonlinear effects for age and bmi.
    - Age enters quadratically (linear + quadratic term).
    - BMI enters via a log transform.
    - Treatment has linear and interaction effects with age.
    """
    # nonlinear transformations
    x1_effect = -1.5 * (x1 ** 2) - 1
    x2_effect = 1 * (2/math.pi) * math.atan(0.5 * x2)
    
    # linear + nonlinear effects
    lp = (
        x1_effect +
        x2_effect +
        0.6 * x3
    )
    
    # baseline hazard * exp(lp)
    return 0.015 * np.exp(lp)

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_add(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_add, t)
# exact
explanation_df_add = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_add, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_add, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_add_ti_haz_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_add,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_add(t, x1, x2, x3):
    """
    Hazard function with nonlinear effects for age and bmi.
    - Age enters quadratically (linear + quadratic term).
    - BMI enters via a log transform.
    - Treatment has linear and interaction effects with age.
    """
    # nonlinear transformations
    x1_effect = -1.5 * ((x1 ** 2) - 1)      # quadratic, centered
    x2_effect = (2 / np.pi) * np.arctan(0.5 * x2)  # bounded S-shape

    # linear + nonlinear effects
    lp = (
        x1_effect +
        x2_effect +
        0.6 * x3
    )

    # baseline hazard * exp(lp)
    return 0.015 * lp

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_add(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_add, t)
# exact
explanation_df_add = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            log_hazard_wrap_add, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_add, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_add_ti_loghaz_17.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_ti,
                              survival_fn = log_hazard_wrap_add,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Explain the first row of x_new for every third time point
# Wrap the survival function
def surv_from_hazard_add_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_add, t)
# exact
explanation_df_add = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            surv_from_hazard_add_wrap, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_add, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_add_ti_surv_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = surv_from_hazard_add_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_add = survshapiq_func.survshapiq(model_gbsa_ti, 
                                                    data_x_ti, 
                                                    x_new_ti, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_add, 
                              model = model_gbsa_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_add_ti_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_df_cox_add = survshapiq_func.survshapiq(model_cox_ti, 
                                                    data_x_ti, 
                                                    x_new_ti, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_add, 
                              model = model_cox_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_add_ti_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 



################ GENERAL ADDITIVE MODEL
###### TIME-INDEPENDENCE 
# Load simulated data DataFrame
simdata_genadd_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_ti.csv")
print(simdata_genadd_ti.head())
simdata_genadd_ti

# Convert eventtime and status columns to a structured array
data_y_ti, data_x = survshapiq_func.prepare_survival_data(simdata_genadd_ti)
print(data_y_ti)
print(data_x.head())
data_x_ti = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_gbsa_ti.score(data_x_ti, data_y_ti).item():0.3f}')
#ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')

# Fit CoxPH
model_cox_ti = CoxPHSurvivalAnalysis()
model_cox_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_cox_ti.score(data_x_ti, data_y_ti).item():0.3f}')
#ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')

# Create data point for explanation
idx = 0
x_new_ti = data_x_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_ti)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_genadd(t, x1, x2, x3):
    # nonlinear transformations
    x1_lin   = x1
    x1_quad  = (x1 ** 2) - 1          # centered quadratic
    x2_s     = (2 / math.pi) * math.atan(0.7 * x2)  # bounded S-shape
    x3_lin   = x3

    # interactions
    x1x2_lin = x1_lin * x2
    x1x3_int = x1_quad * x3_lin
    #x2x3_nl  = x2_s * x3_lin

    # linear + nonlinear effects
    lp = (
        0.2 * x1_lin +
        -0.3 * x1_quad +
        0.5 * x2_s +
        -0.4 * x3_lin +
        0.2 * x1x2_lin +
        0.3 * x1x3_int 
        #-0.4 * x2x3_nl
    )

    # baseline hazard * exp(lp)
    return 0.01 * np.exp(lp)

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_genadd(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_genadd, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_ti_haz_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_genadd,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=200,
                              smooth_poly=1) 


###### GROUND TRUTH LOG HAZARD
# Define the hazard function
def log_hazard_func_genadd(t, x1, x2, x3):
    # nonlinear transformations
    x1_lin   = x1
    x1_quad  = (x1 ** 2) - 1          # centered quadratic
    x2_s     = (2 / math.pi) * math.atan(0.7 * x2)  # bounded S-shape
    x3_lin   = x3

    # interactions
    x1x2_lin = x1_lin * x2
    x1x3_int = x1_quad * x3_lin
    #x2x3_nl  = x2_s * x3_lin

    # linear + nonlinear effects
    lp = (
        0.2 * x1_lin +
        -0.3 * x1_quad +
        0.5 * x2_s +
        -0.4 * x3_lin +
        0.2 * x1x2_lin +
        0.3 * x1x3_int 
       # -0.4 * x2x3_nl
    )

    # baseline hazard * exp(lp)
    return 0.01 * lp

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_genadd(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            log_hazard_wrap_genadd, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_ti_loghaz_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = log_hazard_wrap_genadd,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Wrap the survival function
def surv_from_hazard_genadd_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            surv_from_hazard_genadd_wrap, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_ti_surv_17.pdf",
                              data_x = data_x_ti,
                              survival_fn = surv_from_hazard_genadd_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_genadd = survshapiq_func.survshapiq(model_gbsa_ti, 
                                                    data_x_ti, 
                                                    x_new_ti, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_genadd, 
                              model = model_gbsa_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_genadd_ti_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_df_cox_genadd = survshapiq_func.survshapiq(model_cox_ti, 
                                                    data_x_ti, 
                                                    x_new_ti, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_genadd, 
                              model = model_cox_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_genadd_ti_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


################ GENERAL ADDITIVE MODEL
###### TIME-DEPENDENCE IN MAIN EFFECTS
# Load simulated data DataFrame
simdata_genadd_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_main.csv")
print(simdata_genadd_td.head())
simdata_genadd_td

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_genadd_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_gbsa_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_gbsa_td, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_gbsa_td:0.3f}')

# Fit CoxPH
model_cox_td = CoxPHSurvivalAnalysis()
model_cox_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_cox_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')

# Create data point for explanation
idx = 0
x_new_td = data_x_td[[idx]]
#x_new_td = data_x_td[1:9]
print(x_new_td)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_genadd(t, x1, x2, x3):

    # nonlinear transformations
    x1_quad = (x1 ** 2) - 1
    x2_s    = (2 / np.pi) * np.arctan(0.7 * x2)
    x3_lin  = x3

    # time-dependent linear effect of x1
    x1_td = x1 * (0.2 - 0.1 * np.log(t + 1))

    # interactions
    x1x2_lin = x1 * x2
    x1x3_int = x1_quad * x3_lin

    # linear + nonlinear effects
    lp = (
        x1_td +
        -0.3 * x1_quad +
        0.5 * x2_s +
        -0.4 * x3_lin +
        0.2 * x1x2_lin +
        0.3 * x1x3_int
    )

    # baseline hazard * exp(lp)
    return 0.01 * np.exp(lp)

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def hazard_wrap_genadd(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_genadd, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdmain_haz_17.pdf",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_genadd,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH LOG HAZARD
# Define the log hazard function
def log_hazard_func_genadd(t, x1, x2, x3):

    # nonlinear transformations
    x1_quad = (x1 ** 2) - 1
    x2_s    = (2 / np.pi) * np.arctan(0.7 * x2)
    x3_lin  = x3

    # time-dependent linear effect of x1
    x1_td = x1 * (0.2 - 0.1 * np.log(t + 1))

    # interactions
    x1x2_lin = x1 * x2
    x1x3_int = x1_quad * x3_lin

    # linear + nonlinear effects
    lp = (
        x1_td +
        -0.3 * x1_quad +
        0.5 * x2_s +
        -0.4 * x3_lin +
        0.2 * x1x2_lin +
        0.3 * x1x3_int
    )

    # baseline hazard * exp(lp)
    return 0.01 * lp

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_genadd(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            log_hazard_wrap_genadd, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdmain_loghaz_17.pdf",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_genadd,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Wrap the survival function
def surv_from_hazard_genadd_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            surv_from_hazard_genadd_wrap, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdmain_surv_17.pdf",
                              data_x = data_x_td,
                              survival_fn = surv_from_hazard_genadd_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_genadd = survshapiq_func.survshapiq(model_gbsa_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_genadd, 
                              model = model_gbsa_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_genadd_tdmain_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_df_cox_genadd = survshapiq_func.survshapiq(model_cox_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_genadd, 
                              model = model_cox_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_genadd_tdmain_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


################ GENERAL ADDITIVE MODEL
###### TIME-DEPENDENCE IN MAIN EFFECTS
# Load simulated data DataFrame
simdata_genadd_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_interaction.csv")
print(simdata_genadd_td.head())
simdata_genadd_td

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_genadd_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_gbsa_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_gbsa_td, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_gbsa_td:0.3f}')

# Fit CoxPH
model_cox_td = CoxPHSurvivalAnalysis()
model_cox_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_cox_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti, min_time = 0.04, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')

# Create data point for explanation
idx = 10 #1,8
x_new_td = data_x_td[[idx]]
#x_new_td = data_x_td[1:9]
print(x_new_td)

###### GROUND TRUTH HAZARD
# Define the hazard function
def hazard_func_genadd(t, x1, x2, x3):
    # nonlinear transformations
    x1_quad = (x1 ** 2) - 1                        # centered quadratic
    x2_s    = (2 / np.pi) * np.arctan(0.7 * x2)    # bounded S-shape
    x3_lin  = x3

    # main effects (time-independent)
    x1_lin  = 0.2 * x1
    x3_eff  = -0.4 * x3_lin

    # interactions
    x1x2_lin = x1 * x2
    x1x3_int = x1_quad * x3_lin

    # time-dependent effect on x1*x2 interaction
    x1x2_td = x1x2_lin * (0.2 + (-0.1) * np.log(t + 1))

    # linear + nonlinear effects
    lp = (
        x1_lin +
        -0.3 * x1_quad +
        0.5 * x2_s +
        x3_eff +
        x1x2_td +            # time-dependent interaction
        0.3 * x1x3_int
    )

    # baseline hazard * exp(lp)
    return 0.01 * np.exp(lp)


# Wrap the hazard function
def hazard_wrap_genadd(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_genadd, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdinter_haz_17.pdf",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_genadd,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=100,
                              smooth_poly=1) 



###### GROUND TRUTH LOG HAZARD
# Define the hazard function
def log_hazard_func_genadd(t, x1, x2, x3):
    # nonlinear transformations
    x1_quad = (x1 ** 2) - 1                        # centered quadratic
    x2_s    = (2 / np.pi) * np.arctan(0.7 * x2)    # bounded S-shape
    x3_lin  = x3

    # main effects (time-independent)
    x1_lin  = 0.2 * x1
    x3_eff  = -0.4 * x3_lin

    # interactions
    x1x2_lin = x1 * x2
    x1x3_int = x1_quad * x3_lin

    # time-dependent effect on x1*x2 interaction
    x1x2_td = x1x2_lin * (0.2 + (-0.1) * np.log(t + 1))

    # linear + nonlinear effects
    lp = (
        x1_lin +
        -0.3 * x1_quad +
        0.5 * x2_s +
        x3_eff +
        x1x2_td +            # time-dependent interaction
        0.3 * x1x3_int
    )

    # baseline hazard * exp(lp)
    return 0.01 * lp


# Wrap the hazard function
def log_hazard_wrap_genadd(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            log_hazard_wrap_genadd, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdinter_loghaz_17.pdf",
                              data_x = data_x_td,
                              survival_fn = log_hazard_wrap_genadd,
                              idx_plot=idx, 
                              smooth=True, 
                              smooth_window=250,
                              smooth_poly=1) 


###### GROUND TRUTH SURVIVAL
# Wrap the survival function
def surv_from_hazard_genadd_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_genadd, t)
# exact
explanation_df_genadd = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            surv_from_hazard_genadd_wrap, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_genadd, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gt_genadd_tdinter_surv_17.pdf",
                              data_x = data_x_td,
                              survival_fn = surv_from_hazard_genadd_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### MODEL SURVIVAL
# gbsg 
explanation_df_gbsa_genadd = survshapiq_func.survshapiq(model_gbsa_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_genadd, 
                              model = model_gbsa_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_gbsa_genadd_tdinter_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_df_cox_genadd = survshapiq_func.survshapiq(model_cox_td, 
                                                    data_x_td, 
                                                    x_new_td, 
                                                    time_stride=10, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    exact=True, 
                                                    feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_cox_genadd, 
                              model = model_cox_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory/plot_cox_genadd_tdinter_surv_17.pdf",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

