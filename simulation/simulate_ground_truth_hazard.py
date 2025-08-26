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
import shapiq
import importlib
import simulation.survshapiq_func as survshapiq_func
importlib.reload(survshapiq_func)

## Time-independent Interactions
# Load simulated data DataFrame
simdata_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_ti.csv")
print(simdata_ti.head())

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
#ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti)
#print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')


# Create data point for explanation
idx = 0
x_new_ti = data_x_ti[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_ti)

# Define the hazard function
def hazard_func_ti(t, age, bmi, treatment):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.01 * np.exp((0.3 * age) + (0.9 * bmi) + (-0.7 * treatment) + (-5 * treatment * age))


# Explain the first row of x_new for every third time point
# Wrap the hazard function
hazard_wrap_ti = lambda X, t: survshapiq_func.hazard_matrix(X, hazard_func_ti, t)
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_ti_exact_0.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# k-SII
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            approximator="auto", 
                                                            index = "k-SII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_ti_ksii_0_sm.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# FSII
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index = "FSII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_ti_fsii_0.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,,
                              smooth_window=200,
                              smooth_poly=1) 
# FBII
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index = "FBII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_ti_fbii_0_sm.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# STII
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index = "STII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_ti_stii_0.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# SII
explanation_df_ti = survshapiq_func.survshapiq_ground_truth(data_x_ti, 
                                                            x_new_ti, 
                                                            hazard_wrap_ti, 
                                                            times=model_gbsa_ti.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index = "SII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_ti, 
                              model = None,
                              times=model_gbsa_ti.unique_times_[::5], 
                              x_new = x_new_ti, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_ti_sii_0.png",
                              data_x = data_x_ti,
                              survival_fn = hazard_wrap_ti,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 


## Time-independent Interactions
# Load simulated data DataFrame
simdata_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_td.csv")
print(simdata_td.head())

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values
times_only = np.array([t for _, t in data_y_td])
unique_times = np.unique(times_only)

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
#ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti)
#print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')

# Create data point for explanation
idx = 79
x_new_td = data_x_td[[idx]]
#x_new_ti = data_x_ti[1:9]
print(x_new_td)

# Define the hazard function
def hazard_func_td(t, age, bmi, treatment):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.01 * np.exp((0.3 * age) + (0.9 * bmi) + (-0.7 * treatment) + (-5 * treatment * age) + (9 * treatment * age * np.log(t + 1)))

# Explain the first row of x_new for every third time point
# Wrap the survival function
hazard_wrap_td = lambda X, t: survshapiq_func.hazard_matrix(X, hazard_func_td, t)
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
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_exact_0_160.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1, 
                              time_interval=(0,50)) 
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_exact_nocomp_72.png",
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# k-SII
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index="k-SII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_ksii_0.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# FSII
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index="FSII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_fsii_0.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# FBII
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index="FBII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_fbii_0.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# FBII
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index="FBII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_fbii_0.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# STII
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index="STII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_stii_0.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 
# SII
explanation_df_td = survshapiq_func.survshapiq_ground_truth(data_x_td, 
                                                            x_new_td, 
                                                            hazard_wrap_td, 
                                                            times=model_gbsa_td.unique_times_[::5], 
                                                            budget=2**8, max_order=2, 
                                                            approximator="auto", 
                                                            index="SII",
                                                            feature_names = data_x.columns)
survshapiq_func.plot_interact(explanations_all = explanation_df_td, 
                              model = None,
                              times=model_gbsa_td.unique_times_[::5], 
                              x_new = x_new_td, 
                              save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots_ground_truth_hazard/plot_gt_td_sii_0.png",
                              data_x = data_x_td,
                              survival_fn = hazard_wrap_td,
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=200,
                              smooth_poly=1) 

