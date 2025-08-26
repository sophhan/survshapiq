# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
from sksurv.metrics import integrated_brier_score
from scipy.special import expit
import shapiq
import importlib
import simulation.survshapiq_func as survshapiq_func
importlib.reload(survshapiq_func)

## Time-independent Interactions Hazard & Log-Hazard
# Load simulated data DataFrame
simdata_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_ti_haz.csv")
print(simdata_ti.head())
simdata_ti

# Convert eventtime and status columns to a structured array
data_y_ti, data_x = survshapiq_func.prepare_survival_data(simdata_ti)
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

# HAZARD
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_haz_sm_17.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_haz_17.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=False) 

# LOG HAZARD
# Create data point for explanation
idx = 17
x_new_ti = data_x_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_ti)

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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_log_sm_17.png",
                              data_x = data_x_ti,
                              survival_fn = log_hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_log_smooth_17.png",
                              data_x = data_x_ti,
                              survival_fn = log_hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=False) 

# SURVIVAL
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_surv_smooth_17.png",
                              data_x = data_x_ti,
                              survival_fn = surv_from_hazard_ti_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_surv_17.png",
                              data_x = data_x_ti,
                              survival_fn = surv_from_hazard_ti_wrap,
                              idx_plot=idx, 
                              smooth=False) 
# MODEL SURVIVAL
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gbsg_ti_surv_sm_17.png",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_ti, 
                              model = model_gbsa_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gbsg_ti_surv_17.png",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=False) 

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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_coxph_ti_surv_sm_17.png",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


survshapiq_func.plot_interact(explanations_all = explanation_df_cox_ti, 
                              model = model_cox_ti,
                              x_new = x_new_ti, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_coxph_ti_surv_17.png",
                              compare_plots = True, 
                              data_x = data_x_ti,
                              idx_plot=idx,
                              smooth=False) 



## Time-dependent Interactions Hazard & Log-Hazard
# Load simulated data DataFrame
simdata_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_td_haz.csv")
print(simdata_td.head())
simdata_td

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_td)
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
print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')


# Create data point for explanation
idx =  17
x_new_td = data_x_td[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_td)

# HAZARD
# Define the hazard function
def hazard_func_td(t, age, bmi, treatment):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * age) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age) + (-0.4 * treatment * age * np.log(t+1)))


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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_haz_sm_17.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_haz_17.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=False) 

# LOG HAZARD
# Create data point for explanation
idx = 17 # 5, 7, 10, 13, (14), 17
x_new_td = data_x_td[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_td)

# Define the log hazard function
def log_hazard_func_td(t, age, bmi, treatment):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * age) + (0.5 * bmi) + (0.9 * treatment) + (-0.6 * treatment * age) + (-0.4 * treatment * age * np.log(t+1))))

# Explain the first row of x_new for every third time point
# Wrap the hazard function
def log_hazard_wrap_td(X, t):
    return survshapiq_func.log_hazard_matrix(X, log_hazard_func_td, t)
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_log_haz_sm_17.png", #gt_td_log_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = log_hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_log_haz_17.png",
                              data_x = data_x_td,
                              survival_fn = log_hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=False) 


# SURVIVAL
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_surv_smooth_17.png",
                              data_x = data_x_td,
                              survival_fn = surv_from_hazard_td_wrap,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_surv_17.png",
                              data_x = data_x_td,
                              survival_fn = surv_from_hazard_td_wrap,
                              idx_plot=idx, 
                              smooth=False) 


# MODEL SURVIVAL
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gbsg_td_surv_sm_17.png",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


survshapiq_func.plot_interact(explanations_all = explanation_df_gbsa_td, 
                              model = model_gbsa_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gbsg_td_surv_17.png",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=False) 

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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_coxph_td_surv_sm_17.png",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 


survshapiq_func.plot_interact(explanations_all = explanation_df_cox_td, 
                              model = model_cox_td,
                              x_new = x_new_td, 
                              time_stride=10,
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_coxph_td_surv_17.png",
                              compare_plots = True, 
                              data_x = data_x_td,
                              idx_plot=idx,
                              smooth=False) 



## Time-independent Interactions Survival & Log-Survival additive survival 
# Load data
simdata_surv_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_surv_ti.csv")
print(simdata_surv_ti.head())
simdata_surv_ti

# Convert eventtime and status columns to a structured array
data_y_ti, data_x = survshapiq_func.prepare_survival_data(simdata_surv_ti)
print(data_y_ti)
print(data_x.head())
data_x_ti = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_gbsa_ti.score(data_x_ti, data_y_ti).item():0.3f}')
#ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti, min_time = 0.004, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')

# Fit CoxPH
#model_cox_ti = CoxPHSurvivalAnalysis()
#model_cox_ti.fit(data_x_ti, data_y_ti)
#print(f'C-index (train): {model_cox_ti.score(data_x_ti, data_y_ti).item():0.3f}')
#ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti, min_time = 0.004, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')


# Create data point for explanation
idx = 6 
x_new_ti = data_x_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_ti)

# logit survival
# Define the logit survival function
def logit_surv_func_ti(t, age, bmi, treatment):
    """
    Compute logit(S(t|x)) for given covariates and time.

    Parameters
    ----------
    t : float or array
        Time value(s).
    age : float
        Age covariate.
    bmi : float
        BMI covariate.
    treatment : int (0 or 1)
        Treatment indicator.

    Returns
    -------
    float or array
        Logit of survival probability at time t.
    """
    beta_age = 0.4
    beta_trt = -0.5
    beta_bmi = 0.7
    beta_int = -0.1   # interaction effect (constant, not time-dependent)
    baseline = 5.0

    # no multiplication with t here!
    return (baseline
            + beta_age * age
            + beta_trt * treatment
            + beta_bmi * bmi
            + beta_int * treatment * age)


# Explain the first row of x_new for every third time point
# Wrap the logit survival function
def logit_surv_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, logit_surv_func_ti, t)

explanation_df_logit_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            logit_surv_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)


survshapiq_func.plot_interact(explanations_all = explanation_df_logit_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_add_logit_surv_comp_6.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_ti,
                              survival_fn = logit_surv_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_logit_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_add_logit_surv_6.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_ti,
                              compare_plots=False, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

# survival function
def surv_func_ti(t, age, bmi, treatment):
    """
    Survival function S(t|x) without time-dependent interaction.

    Parameters
    ----------
    t : float or array
        Time value(s).
    age, bmi : float
        Covariates.
    treatment : int (0 or 1)
        Treatment indicator.

    Returns
    -------
    float or array
        Survival probability at time t.
    """
    beta_age = 0.4
    beta_trt = -0.5
    beta_bmi = 0.7
    beta_int = -0.1   # interaction effect (constant, not time-dependent)
    baseline = 5.0

    eta = baseline + beta_age * age + beta_trt * treatment + beta_bmi * bmi + beta_int * treatment * age
    p = expit(eta)        # baseline survival probability per unit time
    return p ** t         # survival decays exponentially with t

# Wrap the survival function
def surv_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, surv_func_ti, t)

explanation_df_surv_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            surv_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)


survshapiq_func.plot_interact(explanations_all = explanation_df_surv_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_add_surv_comp_6.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_ti,
                              survival_fn = surv_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_surv_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_ti_add_surv_6.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_ti,
                              compare_plots=False, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 



## Time-dependent Interactions Survival & Log-Survival additive survival 
# Load data
simdata_surv_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_surv_td.csv")
print(simdata_surv_td.head())
simdata_surv_td

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_surv_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values
#times_only = np.array([t for _, t in data_y_ti])
#unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_gbsa_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_gbsa_td, min_time = 0.004, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_gbsa_td:0.3f}')

# Fit CoxPH
#model_cox_td = CoxPHSurvivalAnalysis()
#model_cox_td.fit(data_x_td, data_y_td)
#print(f'C-index (train): {model_cox_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_cox_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_cox_td, min_time = 0.004, max_time = 69)
#print(f'Integrated Brier Score (train): {ibs_cox_td:0.3f}')


def logit_surv_func_td(t, age, bmi, treatment):
    """
    Compute logit(S(t|x)) for given covariates and time.
    
    Parameters
    ----------
    t : float or array
        Time value(s).
    age : float
        Age covariate.
    bmi : float
        BMI covariate.
    treatment : int (0 or 1)
        Treatment indicator.
    Returns
    -------
    float or array
        Logit of survival probability at time t.
    """
    # Example coefficients (you can adjust)
    beta_age = 0.4
    beta_trt = -0.5
    beta_bmi = 0.7
    beta_int = -0.1   # much stronger negative slope
    baseline = 5.0     # ensures S(0|x) ~ 1 at start     # time-dependent interaction


    return (baseline
            + beta_age * age
            + beta_trt * treatment
            + beta_bmi * bmi
            + beta_int * treatment * age * t)


idx = 2 #2
x_new_td = data_x_td[[idx]]
#x_new_td = data_x_td[1:9]
print(x_new_td)

# Wrap the survival function
def logit_surv_wrap_td(X, t):
    return survshapiq_func.hazard_matrix(X, logit_surv_func_td, t)

explanation_df_logit_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            logit_surv_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)

survshapiq_func.plot_interact(explanations_all = explanation_df_logit_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_add_logit_surv_smooth_2.png.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              compare_plots=False, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


survshapiq_func.plot_interact(explanations_all = explanation_df_logit_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_add_logit_surv_smooth_comp_2.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = logit_surv_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


# survival function with time-dependency
def surv_func_td(t, age, bmi, treatment):
    beta_age = 0.4
    beta_trt = -0.5
    beta_bmi = 0.7
    beta_int = -0.1
    baseline = 5.0

    eta = (baseline
           + beta_age * age
           + beta_trt * treatment
           + beta_bmi * bmi
           + beta_int * treatment * age * t)

    return expit(eta)  # equals 1/(1+exp(-eta))

# Wrap the survival function
def surv_wrap_td(X, t):
    return survshapiq_func.hazard_matrix(X, surv_func_td, t)

explanation_df_surv_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            surv_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            exact=True,
                                                            feature_names = data_x.columns)


survshapiq_func.plot_interact(explanations_all = explanation_df_surv_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_add_surv_smooth_2.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              survival_fn = surv_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

survshapiq_func.plot_interact(explanations_all = explanation_df_surv_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_gt_haz/plot_gt_td_add_surv_smooth_comp_2.png", # plot_gt_td_haz_sm_5
                              data_x = data_x_td,
                              compare_plots=False, 
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

