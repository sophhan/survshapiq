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
import simulation.func as func
importlib.reload(func)

# define paths
path_data = "/home/slangbei/survshapiq/survshapiq/simulation/data"
path_plots = "/home/slangbei/survshapiq/survshapiq/simulation/plots_theory"
path_plots_combined = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined"

#---------------------------
# 1) Linear G(t|x), TI (no interactions)
#---------------------------

# load simulated data dataframe
simdata_linear_ti = pd.read_csv(f"{path_data}/1_simdata_linear_ti.csv")
print(simdata_linear_ti.head())
simdata_linear_ti

# convert eventtime and status columns to a structured array
data_y_linear_ti, data_x_linear_ti_df = func.prepare_survival_data(simdata_linear_ti)
print(data_y_linear_ti)
print(data_x_linear_ti_df.head())
data_x_linear_ti = data_x_linear_ti_df.values
X_train_linear_ti, X_test_linear_ti, y_train_linear_ti, y_test_linear_ti = train_test_split(
    data_x_linear_ti, data_y_linear_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_ti.fit(X_train_linear_ti, y_train_linear_ti)
print(f'C-index (train): {model_gbsa_linear_ti.score(X_test_linear_ti, y_test_linear_ti).item():0.3f}')
ibs_gbsa_linear_ti = func.compute_integrated_brier(y_test_linear_ti, X_test_linear_ti, model_gbsa_linear_ti, min_time = 0.02, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_ti:0.3f}')

# fit CoxPH
model_cox_linear_ti = CoxPHSurvivalAnalysis()
model_cox_linear_ti.fit(X_train_linear_ti, y_train_linear_ti)
print(f'C-index (train): {model_cox_linear_ti.score(X_test_linear_ti, y_test_linear_ti).item():0.3f}')
ibs_cox_linear_ti = func.compute_integrated_brier(y_test_linear_ti, X_test_linear_ti, model_cox_linear_ti, min_time = 0.02, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_ti:0.3f}')

# create data point for explanation
idx = 7
x_new_linear_ti = data_x_linear_ti[[idx]]
print(x_new_linear_ti)

###### GROUND TRUTH HAZARD
# exact
explanation_linear_ti_haz = func.survshapiq_ground_truth(data_x_linear_ti, 
                                                            x_new_linear_ti, 
                                                            func.hazard_wrap_linear_ti, 
                                                            times=model_gbsa_linear_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_haz, 
                              model = None,
                              times=model_gbsa_linear_ti.unique_times_[::5], 
                              x_new = x_new_linear_ti, 
                              save_path = f"{path_plots}/1_linear_ti/plot_gt_linear_ti_haz.pdf",
                              data_x = data_x_linear_ti,
                              survival_fn = func.hazard_wrap_linear_ti,
                              compare_plots="Diff",
                              idx_plot=idx, 
                              ylabel="Attribution $h(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_linear_ti_loghaz = func.survshapiq_ground_truth(data_x_linear_ti, 
                                                            x_new_linear_ti, 
                                                            func.log_hazard_wrap_linear_ti, 
                                                            times=model_gbsa_linear_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_loghaz, 
                              model = None,
                              times=model_gbsa_linear_ti.unique_times_[::5], 
                              x_new = x_new_linear_ti, 
                              save_path = f"{path_plots}/1_linear_ti/plot_gt_linear_ti_loghaz.pdf",
                              data_x = data_x_linear_ti,
                              survival_fn = func.log_hazard_wrap_linear_ti,
                              compare_plots="Diff",
                              idx_plot=idx, 
                              ylabel="Attribution $\log(h(t|x))$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1)  

######### GROUND TRUTH SURVIVAL
# exact
explanation_linear_ti_surv = func.survshapiq_ground_truth(data_x_linear_ti, 
                                                            x_new_linear_ti, 
                                                            func.surv_from_hazard_linear_ti_wrap, 
                                                            times=model_gbsa_linear_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_surv, 
                              model = None,
                              times=model_gbsa_linear_ti.unique_times_[::5], 
                              x_new = x_new_linear_ti, 
                              save_path = f"{path_plots}/1_linear_ti/plot_gt_linear_ti_surv.pdf",
                              data_x = data_x_linear_ti,
                              survival_fn = func.surv_from_hazard_linear_ti_wrap,
                              compare_plots="Diff",
                              idx_plot=idx, 
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

########### MODEL SURVIVAL
# gbsa
explanation_linear_ti_gbsa = func.survshapiq(model_gbsa_linear_ti, 
                                                    X_train_linear_ti, 
                                                    x_new_linear_ti,  
                                                    time_stride=5,
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_ti_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_gbsa, 
                              model = model_gbsa_linear_ti,
                              x_new = x_new_linear_ti, 
                              times=model_gbsa_linear_ti.unique_times_[::5],
                              save_path = f"{path_plots}/1_linear_ti/plot_gbsa_linear_ti_surv.pdf",
                              data_x = data_x_linear_ti,
                              compare_plots="Diff",
                              idx_plot=idx,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_linear_ti_cox = func.survshapiq(model_cox_linear_ti, 
                                                    X_train_linear_ti, 
                                                    x_new_linear_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_ti_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_cox, 
                              model = model_cox_linear_ti,
                              x_new = x_new_linear_ti, 
                              times=model_cox_linear_ti.unique_times_[::10],
                              save_path = f"{path_plots}/1_linear_ti/plot_cox_linear_ti_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_ti,
                              idx_plot=idx,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 2) Linear G(t|x), TD MAIN (no interactions)
#---------------------------

# load simulated data DataFrame
simdata_linear_tdmain = pd.read_csv(f"{path_data}/2_simdata_linear_tdmain.csv")
print(simdata_linear_tdmain.head())
simdata_linear_tdmain

# convert eventtime and status columns to a structured array
data_y_linear_tdmain, data_x_linear_tdmain_df = func.prepare_survival_data(simdata_linear_tdmain)
print(data_y_linear_tdmain)
print(data_x_linear_tdmain_df.head())
data_x_linear_tdmain = data_x_linear_tdmain_df.values
X_train_linear_tdmain, X_test_linear_tdmain, y_train_linear_tdmain, y_test_linear_tdmain = train_test_split(
    data_x_linear_tdmain, data_y_linear_tdmain, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_tdmain = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_tdmain.fit(X_train_linear_tdmain, y_train_linear_tdmain)
print(f'C-index (train): {model_gbsa_linear_tdmain.score(X_test_linear_tdmain, y_test_linear_tdmain).item():0.3f}')
ibs_gbsa_linear_tdmain = func.compute_integrated_brier(y_test_linear_tdmain, X_test_linear_tdmain, model_gbsa_linear_tdmain, min_time = 0.09, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_tdmain:0.3f}')

# fit CoxPH
model_cox_linear_tdmain = CoxPHSurvivalAnalysis()
model_cox_linear_tdmain.fit(X_train_linear_tdmain, y_train_linear_tdmain)
print(f'C-index (train): {model_cox_linear_tdmain.score(X_test_linear_tdmain, y_test_linear_tdmain).item():0.3f}')
ibs_cox_linear_tdmain = func.compute_integrated_brier(y_test_linear_tdmain, X_test_linear_tdmain, model_cox_linear_tdmain, min_time = 0.09, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdmain:0.3f}')

# create data point for explanation
idx =  7
x_new_linear_tdmain = data_x_linear_tdmain[[idx]]
print(x_new_linear_tdmain)

###### GROUND TRUTH HAZARD
# exact
explanation_linear_tdmain_haz = func.survshapiq_ground_truth(data_x_linear_tdmain, 
                                                            x_new_linear_tdmain, 
                                                            func.hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_linear_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_haz, 
                              model = None,
                              times=model_gbsa_linear_tdmain.unique_times_[::5], 
                              x_new = x_new_linear_tdmain, 
                              save_path = f"{path_plots}/2_linear_tdmain/plot_gt_linear_tdmain_haz.pdf",
                              data_x = data_x_linear_tdmain,
                              survival_fn = func.hazard_wrap_linear_tdmain,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_linear_tdmain_loghaz = func.survshapiq_ground_truth(data_x_linear_tdmain, 
                                                            x_new_linear_tdmain, 
                                                            func.log_hazard_wrap_linear_tdmain, 
                                                            times=model_gbsa_linear_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_loghaz, 
                              model = None,
                              times=model_gbsa_linear_tdmain.unique_times_[::5], 
                              x_new = x_new_linear_tdmain, 
                              save_path = f"{path_plots}/2_linear_tdmain/plot_gt_linear_tdmain_loghaz.pdf", 
                              data_x = data_x_linear_tdmain,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_linear_tdmain_surv = func.survshapiq_ground_truth(data_x_linear_tdmain, 
                                                            x_new_linear_tdmain, 
                                                            func.surv_from_hazard_linear_tdmain_wrap, 
                                                            times=model_gbsa_linear_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_df.columns)
func.plot_interact(explanations_all = explanation_linear_tdmain_surv, 
                              model = None,
                              times=model_gbsa_linear_tdmain.unique_times_[::5], 
                              x_new = x_new_linear_tdmain, 
                              save_path = f"{path_plots}/2_linear_tdmain/plot_gt_linear_tdmain_surv.pdf",
                              data_x = data_x_linear_tdmain,
                              survival_fn = func.surv_from_hazard_linear_tdmain_wrap,
                              compare_plots="Diff",
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
explanation_linear_tdmain_gbsa = func.survshapiq(model_gbsa_linear_tdmain, 
                                                    X_train_linear_tdmain, 
                                                    x_new_linear_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_gbsa, 
                              model = model_gbsa_linear_tdmain,
                              x_new = x_new_linear_tdmain, 
                              times = model_gbsa_linear_tdmain.unique_times_[::5],
                              save_path = f"{path_plots}/2_linear_tdmain/plot_gbsa_linear_tdmain_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_linear_tdmain_cox = func.survshapiq(model_cox_linear_tdmain, 
                                                    X_train_linear_tdmain, 
                                                    x_new_linear_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_cox, 
                              model = model_cox_linear_tdmain,
                              times=model_cox_linear_tdmain.unique_times_[::5],
                              x_new = x_new_linear_tdmain, 
                              save_path = f"{path_plots}/2_linear_tdmain/plot_cox_linear_tdmain_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 3) Linear G(t|x), TI (interactions)
#---------------------------

# load simulated data DataFrame
simdata_linear_ti_inter = pd.read_csv(f"{path_data}/3_simdata_linear_ti_inter.csv")
print(simdata_linear_ti_inter.head())
simdata_linear_ti_inter

# Convert eventtime and status columns to a structured array
data_y_linear_ti_inter, data_x_linear_ti_inter_df = func.prepare_survival_data(simdata_linear_ti_inter)
print(data_y_linear_ti_inter)
print(data_x_linear_ti_inter_df.head())
data_x_linear_ti_inter = data_x_linear_ti_inter_df.values
X_train_linear_ti_inter, X_test_linear_ti_inter, y_train_linear_ti_inter, y_test_linear_ti_inter = train_test_split(
    data_x_linear_ti_inter, data_y_linear_ti_inter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_ti_inter = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_ti_inter.fit(X_train_linear_ti_inter, y_train_linear_ti_inter)
print(f'C-index (train): {model_gbsa_linear_ti_inter.score(X_test_linear_ti_inter, y_test_linear_ti_inter).item():0.3f}')
ibs_gbsa_linear_ti_inter = func.compute_integrated_brier(y_test_linear_ti_inter, X_test_linear_ti_inter, model_gbsa_linear_ti_inter, min_time = 0.06, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_ti_inter:0.3f}')

# fit CoxPH
model_cox_linear_ti_inter = CoxPHSurvivalAnalysis()
model_cox_linear_ti_inter.fit(X_train_linear_ti_inter, y_train_linear_ti_inter)
print(f'C-index (train): {model_cox_linear_ti_inter.score(X_test_linear_ti_inter, y_test_linear_ti_inter).item():0.3f}')
ibs_cox_linear_ti_inter = func.compute_integrated_brier(y_test_linear_ti_inter, X_test_linear_ti_inter, model_cox_linear_ti_inter, min_time = 0.06, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_ti_inter:0.3f}')


# create data point for explanation
idx =  7
x_new_linear_ti_inter = data_x_linear_ti_inter[[idx]]
print(x_new_linear_ti_inter)

###### GROUND TRUTH HAZARD
# exact
explanation_linear_ti_inter_haz = func.survshapiq_ground_truth(data_x_linear_ti_inter, 
                                                            x_new_linear_ti_inter, 
                                                            func.hazard_wrap_linear_ti_inter, 
                                                            times=model_gbsa_linear_ti_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_inter_haz, 
                              model = None,
                              times=model_gbsa_linear_ti_inter.unique_times_[::5], 
                              x_new = x_new_linear_ti_inter, 
                              save_path = f"{path_plots}/3_linear_ti_inter/plot_gt_linear_ti_inter_haz.pdf", 
                              data_x = data_x_linear_ti_inter,
                              survival_fn = func.hazard_wrap_linear_ti_inter,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=100,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_linear_ti_inter_loghaz = func.survshapiq_ground_truth(data_x_linear_ti_inter, 
                                                            x_new_linear_ti_inter, 
                                                            func.log_hazard_wrap_linear_ti_inter, 
                                                            times=model_gbsa_linear_ti_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_inter_loghaz, 
                              model = None,
                              times=model_gbsa_linear_ti_inter.unique_times_[::5], 
                              x_new = x_new_linear_ti_inter, 
                              save_path = f"{path_plots}/3_linear_ti_inter/plot_gt_linear_ti_inter_loghaz.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_linear_ti_inter,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_linear_ti_inter_surv = func.survshapiq_ground_truth(data_x_linear_ti_inter, 
                                                            x_new_linear_ti_inter, 
                                                            func.surv_from_hazard_linear_ti_inter_wrap, 
                                                            times=model_gbsa_linear_ti_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_inter_surv, 
                              model = None,
                              times=model_gbsa_linear_ti_inter.unique_times_[::5], 
                              x_new = x_new_linear_ti_inter, 
                              save_path = f"{path_plots}/3_linear_ti_inter/plot_gt_linear_ti_inter_surv.pdf",
                              data_x = data_x_linear_ti_inter,
                              survival_fn = func.surv_from_hazard_linear_ti_inter_wrap,
                              compare_plots="Diff",
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=30,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsa
explanation_linear_ti_inter_gbsa = func.survshapiq(model_gbsa_linear_ti_inter, 
                                                    X_train_linear_ti_inter, 
                                                    x_new_linear_ti_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_inter_gbsa, 
                              model = model_gbsa_linear_ti_inter,
                              x_new = x_new_linear_ti_inter, 
                              times = model_gbsa_linear_ti_inter.unique_times_[::5],
                              save_path = f"{path_plots}/3_linear_ti_inter/plot_gbsa_linear_ti_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_ti_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_linear_ti_inter_cox = func.survshapiq(model_cox_linear_ti_inter, 
                                                    X_train_linear_ti_inter, 
                                                    x_new_linear_ti_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_ti_inter_cox, 
                              model = model_cox_linear_ti_inter,
                              x_new = x_new_linear_ti_inter, 
                              times = model_cox_linear_ti_inter.unique_times_[::5],
                              save_path = f"{path_plots}/3_linear_ti_inter/plot_cox_linear_ti_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_ti_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
#---------------------------

# load simulated data DataFrame
simdata_linear_tdmain_inter = pd.read_csv(f"{path_data}/4_simdata_linear_tdmain_inter.csv")
print(simdata_linear_tdmain_inter.head())
simdata_linear_tdmain_inter

# Convert eventtime and status columns to a structured array
data_y_linear_tdmain_inter, data_x_linear_tdmain_inter_df = func.prepare_survival_data(simdata_linear_tdmain_inter)
print(data_y_linear_tdmain_inter)
print(data_x_linear_tdmain_inter_df.head())
data_x_linear_tdmain_inter = data_x_linear_tdmain_inter_df.values
X_train_linear_tdmain_inter, X_test_linear_tdmain_inter, y_train_linear_tdmain_inter, y_test_linear_tdmain_inter = train_test_split(
    data_x_linear_tdmain_inter, data_y_linear_tdmain_inter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_tdmain_inter = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_tdmain_inter.fit(X_train_linear_tdmain_inter, y_train_linear_tdmain_inter)
print(f'C-index (train): {model_gbsa_linear_tdmain_inter.score(X_test_linear_tdmain_inter, y_test_linear_tdmain_inter).item():0.3f}')
ibs_gbsa_linear_tdmain_inter = func.compute_integrated_brier(y_test_linear_tdmain_inter, X_test_linear_tdmain_inter, model_gbsa_linear_tdmain_inter, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_tdmain_inter:0.3f}')

# fit CoxPH
model_cox_linear_tdmain_inter = CoxPHSurvivalAnalysis()
model_cox_linear_tdmain_inter.fit(X_train_linear_tdmain_inter, y_train_linear_tdmain_inter)
print(f'C-index (train): {model_cox_linear_tdmain_inter.score(X_test_linear_tdmain_inter, y_test_linear_tdmain_inter).item():0.3f}')
ibs_cox_linear_tdmain_inter = func.compute_integrated_brier(y_test_linear_tdmain_inter, X_test_linear_tdmain_inter, model_cox_linear_tdmain_inter, min_time = 0.13, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdmain_inter:0.3f}')


# create data point for explanation
idx =  7
x_new_linear_tdmain_inter = data_x_linear_tdmain_inter[[idx]]
print(x_new_linear_tdmain_inter)

###### GROUND TRUTH HAZARD
# exact
explanation_linear_tdmain_inter_haz = func.survshapiq_ground_truth(data_x_linear_tdmain_inter, 
                                                            x_new_linear_tdmain_inter, 
                                                            func.hazard_wrap_linear_tdmain_inter, 
                                                            times=model_gbsa_linear_tdmain_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_inter_haz, 
                              model = None,
                              times=model_gbsa_linear_tdmain_inter.unique_times_[::5], 
                              x_new = x_new_linear_tdmain_inter, 
                              save_path = f"{path_plots}/4_linear_tdmain_inter/plot_gt_linear_tdmain_inter_haz.pdf", 
                              data_x = data_x_linear_tdmain_inter,
                              survival_fn = func.hazard_wrap_linear_tdmain_inter,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=40,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_linear_tdmain_inter_loghaz = func.survshapiq_ground_truth(data_x_linear_tdmain_inter, 
                                                            x_new_linear_tdmain_inter, 
                                                            func.log_hazard_wrap_linear_tdmain_inter, 
                                                            times=model_gbsa_linear_tdmain_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_inter_loghaz, 
                              model = None,
                              times=model_gbsa_linear_tdmain_inter.unique_times_[::5], 
                              x_new = x_new_linear_tdmain_inter, 
                              save_path = f"{path_plots}/4_linear_tdmain_inter/plot_gt_linear_tdmain_inter_loghaz.pdf", #gt_td_log_haz_sm_5
                              data_x = data_x_linear_tdmain_inter,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_linear_tdmain_inter_surv = func.survshapiq_ground_truth(data_x_linear_tdmain_inter, 
                                                            x_new_linear_tdmain_inter, 
                                                            func.surv_from_hazard_linear_tdmain_inter_wrap, 
                                                            times=model_gbsa_linear_tdmain_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_inter_surv, 
                              model = None,
                              times=model_gbsa_linear_tdmain_inter.unique_times_[::5], 
                              x_new = x_new_linear_tdmain_inter, 
                              save_path = f"{path_plots}/4_linear_tdmain_inter/plot_gt_linear_tdmain_inter_surv.pdf",
                              data_x = data_x_linear_tdmain_inter,
                              survival_fn = func.surv_from_hazard_linear_tdmain_inter_wrap,
                              compare_plots="Diff",
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=30,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsa
explanation_linear_tdmain_inter_gbsa = func.survshapiq(model_gbsa_linear_tdmain_inter, 
                                                    X_train_linear_tdmain_inter, 
                                                    x_new_linear_tdmain_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_inter_gbsa, 
                              model = model_gbsa_linear_tdmain_inter,
                              x_new = x_new_linear_tdmain_inter, 
                              times = model_gbsa_linear_tdmain_inter.unique_times_[::5],
                              save_path = f"{path_plots}/4_linear_tdmain_inter/plot_gbsa_linear_tdmain_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_tdmain_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_linear_tdmain_inter_cox = func.survshapiq(model_cox_linear_tdmain_inter, 
                                                    X_train_linear_tdmain_inter, 
                                                    x_new_linear_tdmain_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdmain_inter_cox, 
                              model = model_cox_linear_tdmain_inter,
                              x_new = x_new_linear_tdmain_inter, 
                              times = model_cox_linear_tdmain_inter.unique_times_[::5],
                              save_path = f"{path_plots}/4_linear_tdmain_inter/plot_cox_linear_tdmain_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_tdmain_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
#---------------------------

# load simulated data DataFrame
simdata_linear_tdinter = pd.read_csv(f"{path_data}/5_simdata_linear_tdinter.csv")
print(simdata_linear_tdinter.head())
simdata_linear_tdinter

# convert eventtime and status columns to a structured array
data_y_linear_tdinter, data_x_linear_tdinter_df = func.prepare_survival_data(simdata_linear_tdinter)
print(data_y_linear_tdinter)
print(data_x_linear_tdinter_df.head())
data_x_linear_tdinter = data_x_linear_tdinter_df.values
X_train_linear_tdinter, X_test_linear_tdinter, y_train_linear_tdinter, y_test_linear_tdinter = train_test_split(
    data_x_linear_tdinter, data_y_linear_tdinter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_linear_tdinter = GradientBoostingSurvivalAnalysis()
model_gbsa_linear_tdinter.fit(X_train_linear_tdinter, y_train_linear_tdinter)
print(f'C-index (train): {model_gbsa_linear_tdinter.score(X_test_linear_tdinter, y_test_linear_tdinter).item():0.3f}')
ibs_gbsa_linear_tdinter = func.compute_integrated_brier(y_test_linear_tdinter, X_test_linear_tdinter, model_gbsa_linear_tdinter, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_linear_tdinter:0.3f}')

# fit CoxPH
model_cox_linear_tdinter = CoxPHSurvivalAnalysis()
model_cox_linear_tdinter.fit(X_train_linear_tdinter, y_train_linear_tdinter)
print(f'C-index (train): {model_cox_linear_tdinter.score(X_test_linear_tdinter, y_test_linear_tdinter).item():0.3f}')
ibs_cox_linear_tdinter = func.compute_integrated_brier(y_test_linear_tdinter, X_test_linear_tdinter, model_cox_linear_tdinter, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdinter:0.3f}')


# create data point for explanation
idx =  7
x_new_linear_tdinter = data_x_linear_tdinter[[idx]]
print(x_new_linear_tdinter)

###### GROUND TRUTH HAZARD
# exact
explanation_linear_tdinter_haz = func.survshapiq_ground_truth(data_x_linear_tdinter, 
                                                            x_new_linear_tdinter, 
                                                            func.hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_linear_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdinter_haz, 
                              model = None,
                              times=model_gbsa_linear_tdinter.unique_times_[::5], 
                              x_new = x_new_linear_tdinter, 
                              save_path = f"{path_plots}/5_linear_tdinter/plot_gt_linear_tdinter_haz.pdf", 
                              data_x = data_x_linear_tdinter,
                              survival_fn = func.hazard_wrap_linear_tdinter,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=10,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_linear_tdinter_loghaz = func.survshapiq_ground_truth(data_x_linear_tdinter, 
                                                            x_new_linear_tdinter, 
                                                            func.log_hazard_wrap_linear_tdinter, 
                                                            times=model_gbsa_linear_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdinter_loghaz, 
                              model = None,
                              times=model_gbsa_linear_tdinter.unique_times_[::5], 
                              x_new = x_new_linear_tdinter, 
                              save_path = f"{path_plots}/5_linear_tdinter/plot_gt_linear_tdinter_loghaz.pdf",
                              data_x = data_x_linear_tdinter,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_linear_tdinter_surv = func.survshapiq_ground_truth(data_x_linear_tdinter, 
                                                            x_new_linear_tdinter, 
                                                            func.surv_from_hazard_linear_tdinter_wrap, 
                                                            times=model_gbsa_linear_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_linear_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdinter_surv, 
                              model = None,
                              times=model_gbsa_linear_tdinter.unique_times_[::5], 
                              x_new = x_new_linear_tdinter, 
                              save_path = f"{path_plots}/5_linear_tdinter/plot_gt_linear_tdinter_surv.pdf",
                              data_x = data_x_linear_tdinter,
                              survival_fn = func.surv_from_hazard_linear_tdinter_wrap,
                              compare_plots="Diff",
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=30,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsa
explanation_linear_tdinter_gbsa = func.survshapiq(model_gbsa_linear_tdinter, 
                                                    X_train_linear_tdinter, 
                                                    x_new_linear_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdinter_gbsa, 
                              model = model_gbsa_linear_tdinter,
                              x_new = x_new_linear_tdinter, 
                              times = model_gbsa_linear_tdinter.unique_times_[::5],
                              save_path = f"{path_plots}/5_linear_tdinter/plot_gbsa_linear_tdinter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_linear_tdinter_cox = func.survshapiq(model_cox_linear_tdinter, 
                                                    X_train_linear_tdinter, 
                                                    x_new_linear_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_linear_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_linear_tdinter_cox, 
                              model = model_cox_linear_tdinter,
                              x_new = x_new_linear_tdinter, 
                              times = model_cox_linear_tdinter.unique_times_[::5],
                              save_path = f"{path_plots}/5_linear_tdinter/plot_cox_linear_tdinter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_linear_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
#---------------------------

# load simulated data DataFrame
simdata_genadd_ti = pd.read_csv(f"{path_data}/6_simdata_genadd_ti.csv")
print(simdata_genadd_ti.head())
simdata_genadd_ti

# convert eventtime and status columns to a structured array
data_y_genadd_ti, data_x_genadd_ti_df = func.prepare_survival_data(simdata_genadd_ti)
print(data_y_genadd_ti)
print(data_x_genadd_ti_df.head())
data_x_genadd_ti = data_x_genadd_ti_df.values
X_train_genadd_ti, X_test_genadd_ti, y_train_genadd_ti, y_test_genadd_ti = train_test_split(
    data_x_genadd_ti, data_y_genadd_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_ti.fit(X_train_genadd_ti, y_train_genadd_ti)
print(f'C-index (train): {model_gbsa_genadd_ti.score(X_test_genadd_ti, y_test_genadd_ti).item():0.3f}')
ibs_gbsa_genadd_ti = func.compute_integrated_brier(y_test_genadd_ti, X_test_genadd_ti, model_gbsa_genadd_ti, min_time = 0.16, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_ti:0.3f}')

# fit CoxPH
model_cox_genadd_ti = CoxPHSurvivalAnalysis()
model_cox_genadd_ti.fit(X_train_genadd_ti, y_train_genadd_ti)
print(f'C-index (train): {model_cox_genadd_ti.score(X_test_genadd_ti, y_test_genadd_ti).item():0.3f}')
ibs_cox_linear_genadd_ti = func.compute_integrated_brier(y_test_genadd_ti, X_test_genadd_ti, model_cox_genadd_ti, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_genadd_ti:0.3f}')


# create data point for explanation
idx = 7
x_new_genadd_ti = data_x_genadd_ti[[idx]]
print(x_new_genadd_ti)

###### GROUND TRUTH HAZARD
# exact
explanation_genadd_ti_haz = func.survshapiq_ground_truth(data_x_genadd_ti, 
                                                            x_new_genadd_ti, 
                                                            func.hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_genadd_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_haz, 
                              model = None,
                              times=model_gbsa_genadd_ti.unique_times_[::5], 
                              x_new = x_new_genadd_ti, 
                              save_path = f"{path_plots}/6_genadd_ti/plot_gt_genadd_ti_haz.pdf", 
                              data_x = data_x_genadd_ti,
                              survival_fn = func.hazard_wrap_genadd_ti,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=120,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_genadd_ti_loghaz = func.survshapiq_ground_truth(data_x_genadd_ti, 
                                                            x_new_genadd_ti, 
                                                            func.log_hazard_wrap_genadd_ti, 
                                                            times=model_gbsa_genadd_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_ti.unique_times_[::5], 
                              x_new = x_new_genadd_ti, 
                              save_path = f"{path_plots}/6_genadd_ti/plot_gt_genadd_ti_loghaz.pdf",
                              data_x = data_x_genadd_ti,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_genadd_ti_surv = func.survshapiq_ground_truth(data_x_genadd_ti, 
                                                            x_new_genadd_ti, 
                                                            func.surv_from_hazard_genadd_ti_wrap, 
                                                            times=model_gbsa_genadd_ti.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_surv, 
                              model = None,
                              times=model_gbsa_genadd_ti.unique_times_[::5], 
                              x_new = x_new_genadd_ti, 
                              save_path = f"{path_plots}/6_genadd_ti/plot_gt_genadd_ti_surv.pdf",
                              data_x = data_x_genadd_ti,
                              survival_fn = func.surv_from_hazard_genadd_ti_wrap,
                              compare_plots="Diff",
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
explanation_genadd_ti_gbsa = func.survshapiq(model_gbsa_genadd_ti, 
                                                    X_train_genadd_ti, 
                                                    x_new_genadd_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_ti_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_gbsa, 
                              model = model_gbsa_genadd_ti,
                              x_new = x_new_genadd_ti, 
                              times = model_gbsa_genadd_ti.unique_times_[::5],
                              save_path = f"{path_plots}/6_genadd_ti/plot_gbsa_genadd_ti_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_ti,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=40,
                              smooth_poly=1) 

# coxph
explanation_genadd_ti_cox = func.survshapiq(model_cox_genadd_ti, 
                                                    X_train_genadd_ti, 
                                                    x_new_genadd_ti, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_ti_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_cox, 
                              model = model_cox_genadd_ti,
                              x_new = x_new_genadd_ti, 
                              times = model_cox_genadd_ti.unique_times_[::5],
                              save_path = f"{path_plots}/6_genadd_ti/plot_cox_genadd_ti_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_ti,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
#---------------------------

# load simulated data DataFrame
simdata_genadd_tdmain = pd.read_csv(f"{path_data}/7_simdata_genadd_tdmain.csv")
print(simdata_genadd_tdmain.head())
simdata_genadd_tdmain

# convert eventtime and status columns to a structured array
data_y_genadd_tdmain, data_x_genadd_tdmain_df = func.prepare_survival_data(simdata_genadd_tdmain)
print(data_y_genadd_tdmain)
print(data_x_genadd_tdmain_df.head())
data_x_genadd_tdmain = data_x_genadd_tdmain_df.values
X_train_genadd_tdmain, X_test_genadd_tdmain, y_train_genadd_tdmain, y_test_genadd_tdmain = train_test_split(
    data_x_genadd_tdmain, data_y_genadd_tdmain, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_tdmain = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_tdmain.fit(X_train_genadd_tdmain, y_train_genadd_tdmain)
print(f'C-index (train): {model_gbsa_genadd_tdmain.score(X_test_genadd_tdmain, y_test_genadd_tdmain).item():0.3f}')
ibs_gbsa_genadd_tdmain = func.compute_integrated_brier(y_test_genadd_tdmain, X_test_genadd_tdmain, model_gbsa_genadd_tdmain, min_time = 0.9, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_tdmain:0.3f}')

# fit CoxPH
model_cox_genadd_tdmain = CoxPHSurvivalAnalysis()
model_cox_genadd_tdmain.fit(X_train_genadd_tdmain, y_train_genadd_tdmain)
print(f'C-index (train): {model_cox_genadd_tdmain.score(X_test_genadd_tdmain, y_test_genadd_tdmain).item():0.3f}')
ibs_cox_genadd_tdmain = func.compute_integrated_brier(y_test_genadd_tdmain, X_test_genadd_tdmain, model_cox_genadd_tdmain, min_time = 0.09, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_genadd_tdmain:0.3f}')


# create data point for explanation
idx = 7
x_new_genadd_tdmain = data_x_genadd_tdmain[[idx]]
print(x_new_genadd_tdmain)

# exact
explanation_genadd_tdmain_haz = func.survshapiq_ground_truth(data_x_genadd_tdmain, 
                                                            x_new_genadd_tdmain, 
                                                            func.hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_haz, 
                              model = None,
                              times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain, 
                              save_path = f"{path_plots}/7_genadd_tdmain/plot_gt_genadd_tdmain_haz.pdf", 
                              data_x = data_x_genadd_tdmain,
                              survival_fn = func.hazard_wrap_genadd_tdmain,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=20,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_genadd_tdmain_loghaz = func.survshapiq_ground_truth(data_x_genadd_tdmain, 
                                                            x_new_genadd_tdmain, 
                                                            func.log_hazard_wrap_genadd_tdmain, 
                                                            times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain, 
                              save_path = f"{path_plots}/7_genadd_tdmain/plot_gt_genadd_tdmain_loghaz.pdf",
                              data_x = data_x_genadd_tdmain,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_genadd_tdmain_surv = func.survshapiq_ground_truth(data_x_genadd_tdmain, 
                                                            x_new_genadd_tdmain, 
                                                            func.surv_from_hazard_genadd_tdmain_wrap, 
                                                            times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_surv, 
                              model = None,
                              times=model_gbsa_genadd_tdmain.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain, 
                              save_path = f"{path_plots}/7_genadd_tdmain/plot_gt_genadd_tdmain_surv.pdf",
                              data_x = data_x_genadd_tdmain,
                              survival_fn = func.surv_from_hazard_genadd_tdmain_wrap,
                              compare_plots="Diff",
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
explanation_genadd_tdmain_gbsa = func.survshapiq(model_gbsa_genadd_tdmain, 
                                                    X_train_genadd_tdmain, 
                                                    x_new_genadd_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_gbsa, 
                              model = model_gbsa_genadd_tdmain,
                              x_new = x_new_genadd_tdmain, 
                              times = model_gbsa_genadd_tdmain.unique_times_[::5],
                              save_path = f"{path_plots}/7_genadd_tdmain/plot_gbsa_genadd_tdmain_surv.pdf",
                              compare_plots = "Diff", 
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
explanation_genadd_tdmain_cox = func.survshapiq(model_cox_genadd_tdmain, 
                                                    X_train_genadd_tdmain, 
                                                    x_new_genadd_tdmain, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdmain_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_cox, 
                              model = model_cox_genadd_tdmain,
                              x_new = x_new_genadd_tdmain, 
                              times = model_cox_genadd_tdmain.unique_times_[::5],
                              save_path = f"{path_plots}/7_genadd_tdmain/plot_cox_genadd_tdmain_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_tdmain,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
#---------------------------

# load simulated data DataFrame
simdata_genadd_ti_inter = pd.read_csv(f"{path_data}/8_simdata_genadd_ti_inter.csv")
print(simdata_genadd_ti_inter.head())
simdata_genadd_ti_inter

# convert eventtime and status columns to a structured array
data_y_genadd_ti_inter, data_x_genadd_ti_inter_df = func.prepare_survival_data(simdata_genadd_ti_inter)
print(data_y_genadd_ti_inter)
print(data_x_genadd_ti_inter_df.head())
data_x_genadd_ti_inter = data_x_genadd_ti_inter_df.values
X_train_genadd_ti_inter, X_test_genadd_ti_inter, y_train_genadd_ti_inter, y_test_genadd_ti_inter = train_test_split(
    data_x_genadd_ti_inter, data_y_genadd_ti_inter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_ti_inter = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_ti_inter.fit(X_train_genadd_ti_inter, y_train_genadd_ti_inter)
print(f'C-index (train): {model_gbsa_genadd_ti_inter.score(X_test_genadd_ti_inter, y_test_genadd_ti_inter).item():0.3f}')
ibs_gbsa_genadd_ti_inter = func.compute_integrated_brier(y_test_genadd_ti_inter, X_test_genadd_ti_inter, model_gbsa_genadd_ti_inter, min_time = 0.16, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_ti_inter:0.3f}')

# fit CoxPH
model_cox_genadd_ti_inter = CoxPHSurvivalAnalysis()
model_cox_genadd_ti_inter.fit(X_train_genadd_ti_inter, y_train_genadd_ti_inter)
print(f'C-index (train): {model_cox_genadd_ti_inter.score(X_test_genadd_ti_inter, y_test_genadd_ti_inter).item():0.3f}')
ibs_cox_linear_ti_inter = func.compute_integrated_brier(y_test_genadd_ti_inter, X_test_genadd_ti_inter, model_cox_genadd_ti_inter, min_time = 0.12, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_ti_inter:0.3f}')


# create data point for explanation
idx = 7
x_new_genadd_ti_inter = data_x_genadd_ti_inter[[idx]]
print(x_new_genadd_ti_inter)

###### GROUND TRUTH HAZARD
# exact
explanation_genadd_ti_inter_haz = func.survshapiq_ground_truth(data_x_genadd_ti_inter, 
                                                            x_new_genadd_ti_inter, 
                                                            func.hazard_wrap_genadd_ti_inter, 
                                                            times=model_gbsa_genadd_ti_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_inter_haz, 
                              model = None,
                              times=model_gbsa_genadd_ti_inter.unique_times_[::5], 
                              x_new = x_new_genadd_ti_inter, 
                              save_path = f"{path_plots}/8_genadd_ti_inter/plot_gt_genadd_ti_inter_haz.pdf", 
                              data_x = data_x_genadd_ti_inter,
                              survival_fn = func.hazard_wrap_genadd_ti_inter,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=20,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_genadd_ti_inter_loghaz = func.survshapiq_ground_truth(data_x_genadd_ti_inter, 
                                                            x_new_genadd_ti_inter, 
                                                            func.log_hazard_wrap_genadd_ti_inter, 
                                                            times=model_gbsa_genadd_ti_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_inter_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_ti_inter.unique_times_[::5], 
                              x_new = x_new_genadd_ti_inter, 
                              save_path = f"{path_plots}/8_genadd_ti_inter/plot_gt_genadd_ti_inter_loghaz.pdf",
                              data_x = data_x_genadd_ti_inter,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_genadd_ti_inter_surv = func.survshapiq_ground_truth(data_x_genadd_ti_inter, 
                                                            x_new_genadd_ti_inter, 
                                                            func.surv_from_hazard_genadd_ti_inter_wrap, 
                                                            times=model_gbsa_genadd_ti_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_inter_surv, 
                              model = None,
                              times=model_gbsa_genadd_ti_inter.unique_times_[::5], 
                              x_new = x_new_genadd_ti_inter, 
                              save_path = f"{path_plots}/8_genadd_ti_inter/plot_gt_genadd_ti_inter_surv.pdf",
                              data_x = data_x_genadd_ti_inter,
                              survival_fn = func.surv_from_hazard_genadd_ti_inter_wrap,
                              compare_plots="Diff",
                              ylabel="Attribution $S(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=60,
                              smooth_poly=1) 


###### MODEL SURVIVAL
# gbsa
explanation_genadd_ti_inter_gbsa = func.survshapiq(model_gbsa_genadd_ti_inter, 
                                                    X_train_genadd_ti_inter, 
                                                    x_new_genadd_ti_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_inter_gbsa, 
                              model = model_gbsa_genadd_ti_inter,
                              x_new = x_new_genadd_ti_inter, 
                              times = model_gbsa_genadd_ti_inter.unique_times_[::5],
                              save_path = f"{path_plots}/8_genadd_ti_inter/plot_gbsa_genadd_ti_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_ti_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=60,
                              smooth_poly=1) 

# coxph
explanation_genadd_ti_inter_cox = func.survshapiq(model_cox_genadd_ti_inter, 
                                                    X_train_genadd_ti_inter, 
                                                    x_new_genadd_ti_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_ti_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_ti_inter_cox, 
                              model = model_cox_genadd_ti_inter,
                              x_new = x_new_genadd_ti_inter, 
                              times = model_cox_genadd_ti_inter.unique_times_[::5],
                              save_path = f"{path_plots}/8_genadd_ti_inter/plot_cox_genadd_ti_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_ti_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=60,
                              smooth_poly=1) 


#---------------------------
# 9) Generalized Additive G(t|x), TD Main (interactions)
#---------------------------

# load simulated data DataFrame
simdata_genadd_tdmain_inter = pd.read_csv(f"{path_data}/9_simdata_genadd_tdmain_inter.csv")
print(simdata_genadd_tdmain_inter.head())
simdata_genadd_tdmain_inter

# convert eventtime and status columns to a structured array
data_y_genadd_tdmain_inter, data_x_genadd_tdmain_inter_df = func.prepare_survival_data(simdata_genadd_tdmain_inter)
print(data_y_genadd_tdmain_inter)
print(data_x_genadd_tdmain_inter_df.head())
data_x_genadd_tdmain_inter = data_x_genadd_tdmain_inter_df.values
X_train_genadd_tdmain_inter, X_test_genadd_tdmain_inter, y_train_genadd_tdmain_inter, y_test_genadd_tdmain_inter = train_test_split(
    data_x_genadd_tdmain_inter, data_y_genadd_tdmain_inter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_tdmain_inter = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_tdmain_inter.fit(X_train_genadd_tdmain_inter, y_train_genadd_tdmain_inter)
print(f'C-index (train): {model_gbsa_genadd_tdmain_inter.score(X_test_genadd_tdmain_inter, y_test_genadd_tdmain_inter).item():0.3f}')
ibs_gbsa_genadd_tdmain_inter = func.compute_integrated_brier(y_test_genadd_tdmain_inter, X_test_genadd_tdmain_inter, model_gbsa_genadd_tdmain_inter, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_tdmain_inter:0.3f}')

# fit CoxPH
model_cox_genadd_tdmain_inter = CoxPHSurvivalAnalysis()
model_cox_genadd_tdmain_inter.fit(X_train_genadd_tdmain_inter, y_train_genadd_tdmain_inter)
print(f'C-index (train): {model_cox_genadd_tdmain_inter.score(X_test_genadd_tdmain_inter, y_test_genadd_tdmain_inter).item():0.3f}')
ibs_cox_linear_tdmain_inter = func.compute_integrated_brier(y_test_genadd_tdmain_inter, X_test_genadd_tdmain_inter, model_cox_genadd_tdmain_inter, min_time = 0.17, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdmain_inter:0.3f}')


# create data point for explanation
idx = 7
x_new_genadd_tdmain_inter = data_x_genadd_tdmain_inter[[idx]]
print(x_new_genadd_tdmain_inter)

###### GROUND TRUTH HAZARD
# exact
explanation_genadd_tdmain_inter_haz = func.survshapiq_ground_truth(data_x_genadd_tdmain_inter, 
                                                            x_new_genadd_tdmain_inter, 
                                                            func.hazard_wrap_genadd_tdmain_inter, 
                                                            times=model_gbsa_genadd_tdmain_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_inter_haz, 
                              model = None,
                              times=model_gbsa_genadd_tdmain_inter.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain_inter, 
                              save_path = f"{path_plots}/9_genadd_tdmain_inter/plot_gt_genadd_tdmain_inter_haz.pdf", 
                              data_x = data_x_genadd_tdmain_inter,
                              survival_fn = func.hazard_wrap_genadd_tdmain_inter,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=30,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_genadd_tdmain_inter_loghaz = func.survshapiq_ground_truth(data_x_genadd_tdmain_inter, 
                                                            x_new_genadd_tdmain_inter, 
                                                            func.log_hazard_wrap_genadd_tdmain_inter, 
                                                            times=model_gbsa_genadd_tdmain_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_inter_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_tdmain_inter.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain_inter, 
                              save_path = f"{path_plots}/9_genadd_tdmain_inter/plot_gt_genadd_tdmain_inter_loghaz.pdf",
                              data_x = data_x_genadd_tdmain_inter,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_genadd_tdmain_inter_surv = func.survshapiq_ground_truth(data_x_genadd_tdmain_inter, 
                                                            x_new_genadd_tdmain_inter, 
                                                            func.surv_from_hazard_genadd_tdmain_inter_wrap, 
                                                            times=model_gbsa_genadd_tdmain_inter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_inter_surv, 
                              model = None,
                              times=model_gbsa_genadd_tdmain_inter.unique_times_[::5], 
                              x_new = x_new_genadd_tdmain_inter, 
                              save_path = f"{path_plots}/9_genadd_tdmain_inter/plot_gt_genadd_tdmain_inter_surv.pdf",
                              data_x = data_x_genadd_tdmain_inter,
                              survival_fn = func.surv_from_hazard_genadd_tdmain_inter_wrap,
                              compare_plots="Diff",
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
explanation_genadd_tdmain_inter_gbsa = func.survshapiq(model_gbsa_genadd_tdmain_inter, 
                                                    X_train_genadd_tdmain_inter, 
                                                    x_new_genadd_tdmain_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_inter_gbsa, 
                              model = model_gbsa_genadd_tdmain_inter,
                              x_new = x_new_genadd_tdmain_inter, 
                              times = model_gbsa_genadd_tdmain_inter.unique_times_[::5],
                              save_path = f"{path_plots}/9_genadd_tdmain_inter/plot_gbsa_genadd_tdmain_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_tdmain_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 

# coxph
explanation_genadd_tdmain_inter_cox = func.survshapiq(model_cox_genadd_tdmain_inter, 
                                                    X_train_genadd_tdmain_inter, 
                                                    x_new_genadd_tdmain_inter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdmain_inter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdmain_inter_cox, 
                              model = model_cox_genadd_tdmain_inter,
                              x_new = x_new_genadd_tdmain_inter, 
                              times = model_cox_genadd_tdmain_inter.unique_times_[::5],
                              save_path = f"{path_plots}/9_genadd_tdmain_inter/plot_cox_genadd_tdmain_inter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_tdmain_inter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 



#---------------------------
# 10) Generalized Additive G(t|x), TD Inter (interactions)
#---------------------------

# load simulated data DataFrame
simdata_genadd_tdinter = pd.read_csv(f"{path_data}/10_simdata_genadd_tdinter.csv")
print(simdata_genadd_tdinter.head())
simdata_genadd_tdinter

# convert eventtime and status columns to a structured array
data_y_genadd_tdinter, data_x_genadd_tdinter_df = func.prepare_survival_data(simdata_genadd_tdinter)
print(data_y_genadd_tdinter)
print(data_x_genadd_tdinter_df.head())
data_x_genadd_tdinter = data_x_genadd_tdinter_df.values
X_train_genadd_tdinter, X_test_genadd_tdinter, y_train_genadd_tdinter, y_test_genadd_tdinter = train_test_split(
    data_x_genadd_tdinter, data_y_genadd_tdinter, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

# fit GradientBoostingSurvivalAnalysis
model_gbsa_genadd_tdinter = GradientBoostingSurvivalAnalysis()
model_gbsa_genadd_tdinter.fit(X_train_genadd_tdinter, y_train_genadd_tdinter)
print(f'C-index (train): {model_gbsa_genadd_tdinter.score(X_test_genadd_tdinter, y_test_genadd_tdinter).item():0.3f}')
ibs_gbsa_genadd_tdinter = func.compute_integrated_brier(y_test_genadd_tdinter, X_test_genadd_tdinter, model_gbsa_genadd_tdinter, min_time = 0.16, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_gbsa_genadd_tdinter:0.3f}')

# fit CoxPH
model_cox_genadd_tdinter = CoxPHSurvivalAnalysis()
model_cox_genadd_tdinter.fit(X_train_genadd_tdinter, y_train_genadd_tdinter)
print(f'C-index (train): {model_cox_genadd_tdinter.score(X_test_genadd_tdinter, y_test_genadd_tdinter).item():0.3f}')
ibs_cox_linear_tdinter = func.compute_integrated_brier(y_test_genadd_tdinter, X_test_genadd_tdinter, model_cox_genadd_tdinter, min_time = 0.08, max_time = 69)
print(f'Integrated Brier Score (train): {ibs_cox_linear_tdinter:0.3f}')


# create data point for explanation
idx = 7
x_new_genadd_tdinter = data_x_genadd_tdinter[[idx]]
print(x_new_genadd_tdinter)

###### GROUND TRUTH HAZARD
# exact
explanation_genadd_tdinter_haz = func.survshapiq_ground_truth(data_x_genadd_tdinter, 
                                                            x_new_genadd_tdinter, 
                                                            func.hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdinter_haz, 
                              model = None,
                              times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                              x_new = x_new_genadd_tdinter, 
                              save_path = f"{path_plots}/10_genadd_tdinter/plot_gt_genadd_tdinter_haz.pdf", 
                              data_x = data_x_genadd_tdinter,
                              survival_fn = func.hazard_wrap_genadd_tdinter,
                              ylabel="Attribution $h(t|x)$",
                              compare_plots="Diff",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx, 
                              smooth=True,
                              smooth_window=20,
                              smooth_poly=1) 

###### GROUND TRUTH LOG HAZARD
# exact
explanation_genadd_tdinter_loghaz = func.survshapiq_ground_truth(data_x_genadd_tdinter, 
                                                            x_new_genadd_tdinter, 
                                                            func.log_hazard_wrap_genadd_tdinter, 
                                                            times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdinter_loghaz, 
                              model = None,
                              times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                              x_new = x_new_genadd_tdinter, 
                              save_path = f"{path_plots}/10_genadd_tdinter/plot_gt_genadd_tdinter_loghaz.pdf",
                              data_x = data_x_genadd_tdinter,
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


###### GROUND TRUTH SURVIVAL
# exact
explanation_genadd_tdinter_surv = func.survshapiq_ground_truth(data_x_genadd_tdinter, 
                                                            x_new_genadd_tdinter, 
                                                            func.surv_from_hazard_genadd_tdinter_wrap, 
                                                            times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                                                            budget=2**8, 
                                                            max_order=2, 
                                                            index= "k-SII",
                                                            exact=True,
                                                            feature_names = data_x_genadd_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdinter_surv, 
                              model = None,
                              times=model_gbsa_genadd_tdinter.unique_times_[::5], 
                              x_new = x_new_genadd_tdinter, 
                              save_path = f"{path_plots}/10_genadd_tdinter/plot_gt_genadd_tdinter_surv.pdf",
                              data_x = data_x_genadd_tdinter,
                              survival_fn = func.surv_from_hazard_genadd_tdinter_wrap,
                              compare_plots="Diff",
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
explanation_genadd_tdinter_gbsa = func.survshapiq(model_gbsa_genadd_tdinter, 
                                                    X_train_genadd_tdinter, 
                                                    x_new_genadd_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdinter_gbsa, 
                              model = model_gbsa_genadd_tdinter,
                              x_new = x_new_genadd_tdinter, 
                              times = model_gbsa_genadd_tdinter.unique_times_[::5],
                              save_path = f"{path_plots}/10_genadd_tdinter/plot_gbsa_genadd_tdinter_surv.pdf",
                              compare_plots = "Diff", 
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
explanation_genadd_tdinter_cox = func.survshapiq(model_cox_genadd_tdinter, 
                                                    X_train_genadd_tdinter, 
                                                    x_new_genadd_tdinter, 
                                                    time_stride=5, 
                                                    budget=2**8, 
                                                    max_order=2, 
                                                    index= "k-SII",
                                                    exact=True, 
                                                    feature_names = data_x_genadd_tdinter_df.columns)

func.plot_interact(explanations_all = explanation_genadd_tdinter_cox, 
                              model = model_cox_genadd_tdinter,
                              x_new = x_new_genadd_tdinter, 
                              times = model_cox_genadd_tdinter.unique_times_[::5],
                              save_path = f"{path_plots}/10_genadd_tdinter/plot_cox_genadd_tdinter_surv.pdf",
                              compare_plots = "Diff", 
                              data_x = data_x_genadd_tdinter,
                              ylabel="Attribution $\hat{S}(t|x)$",
                              label_fontsize=16,
                              tick_fontsize=14,
                              figsize=(10,6),
                              idx_plot=idx,
                              smooth=True,
                              smooth_window=50,
                              smooth_poly=1) 


####### COMBINED PLOTS
#################################################################################################
############## HAZARD
# Create figure
fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, smooth_w, expl, times, data_x, surv_fn, title in zip(
    axes,
    [100, 30, 100, 30, 15, 100, 25, 120, 25, 5],
    [explanation_linear_ti_haz[0],
     explanation_linear_tdmain_haz[0],
     explanation_linear_ti_inter_haz[0],
     explanation_linear_tdmain_inter_haz[0],
     explanation_linear_tdinter_haz[0],
     explanation_genadd_ti_haz[0],
     explanation_genadd_tdmain_haz[0],
     explanation_genadd_ti_inter_haz[0],
     explanation_genadd_tdmain_inter_haz[0],
     explanation_genadd_tdinter_haz[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_ti_inter.unique_times_[::5],
     model_gbsa_linear_tdmain_inter.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_ti_inter.unique_times_[::5],
     model_gbsa_genadd_tdmain_inter.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_ti, data_x_linear_tdmain, data_x_linear_ti_inter,
     data_x_linear_tdmain_inter, data_x_linear_tdinter,
     data_x_genadd_ti, data_x_genadd_tdmain, data_x_genadd_ti_inter,
     data_x_genadd_tdmain_inter, data_x_genadd_tdinter],
    [func.hazard_wrap_linear_ti, func.hazard_wrap_linear_tdmain,
     func.hazard_wrap_linear_ti_inter, func.hazard_wrap_linear_tdmain_inter,
     func.hazard_wrap_linear_tdinter, func.hazard_wrap_genadd_ti,
     func.hazard_wrap_genadd_tdmain, func.hazard_wrap_genadd_ti_inter,
     func.hazard_wrap_genadd_tdmain_inter, func.hazard_wrap_genadd_tdinter],
    ["(1) GT: Linear G(t|x) TI", "(2) GT: Linear G(t|x) TD Main",
     "(3) GT: Linear G(t|x) TI Inter", "(4) GT: Linear G(t|x) TD Main Inter",
     "(5) GT: Linear G(t|x) TD Inter", "(6) GT: General Additive G(t|x) TI",
     "(7) GT: General Additive G(t|x) TD Main",
     "(8) GT: General Additive G(t|x) TI Inter",
     "(9) GT: General Additive G(t|x) TD Main Inter",
     "(10) GT: General Additive G(t|x) TD Inter"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel="Attribution $h(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=smooth_w,
        smooth_poly=1,
        title=title,
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
plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save the figure
save_path = f"{path_plots_combined}/plot_haz.pdf"
fig.savefig(save_path, bbox_inches="tight")


############## LOG HAZARD
# Create figure
fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_ti_loghaz[0],
     explanation_linear_tdmain_loghaz[0],
     explanation_linear_ti_inter_loghaz[0],
     explanation_linear_tdmain_inter_loghaz[0],
     explanation_linear_tdinter_loghaz[0],
     explanation_genadd_ti_loghaz[0],
     explanation_genadd_tdmain_loghaz[0],
     explanation_genadd_ti_inter_loghaz[0],
     explanation_genadd_tdmain_inter_loghaz[0],
     explanation_genadd_tdinter_loghaz[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_ti_inter.unique_times_[::5],
     model_gbsa_linear_tdmain_inter.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_ti_inter.unique_times_[::5],
     model_gbsa_genadd_tdmain_inter.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_ti, data_x_linear_tdmain, data_x_linear_ti_inter,
     data_x_linear_tdmain_inter, data_x_linear_tdinter,
     data_x_genadd_ti, data_x_genadd_tdmain, data_x_genadd_ti_inter,
     data_x_genadd_tdmain_inter, data_x_genadd_tdinter],
    [func.log_hazard_wrap_linear_ti, func.log_hazard_wrap_linear_tdmain,
     func.log_hazard_wrap_linear_ti_inter, func.log_hazard_wrap_linear_tdmain_inter,
     func.log_hazard_wrap_linear_tdinter, func.log_hazard_wrap_genadd_ti,
     func.log_hazard_wrap_genadd_tdmain, func.log_hazard_wrap_genadd_ti_inter,
     func.log_hazard_wrap_genadd_tdmain_inter, func.log_hazard_wrap_genadd_tdinter],
    ["(1) GT: Linear G(t|x) TI", "(2) GT: Linear G(t|x) TD Main",
     "(3) GT: Linear G(t|x) TI Inter", "(4) GT: Linear G(t|x) TD Main Inter",
     "(5) GT: Linear G(t|x) TD Inter", "(6) GT: General Additive G(t|x) TI",
     "(7) GT: General Additive G(t|x) TD Main",
     "(8) GT: General Additive G(t|x) TI Inter",
     "(9) GT: General Additive G(t|x) TD Main Inter",
     "(10) GT: General Additive G(t|x) TD Inter"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel="Attribution $\log(h(t|x))$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=100,
        smooth_poly=1,
        title=title,
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
plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save the figure
save_path = f"{path_plots_combined}/plot_loghaz.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## SURVIVAL
# Create figure
fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, expl, times, data_x, surv_fn, title in zip(
    axes,
    [explanation_linear_ti_surv[0],
     explanation_linear_tdmain_surv[0],
     explanation_linear_ti_inter_surv[0],
     explanation_linear_tdmain_inter_surv[0],
     explanation_linear_tdinter_surv[0],
     explanation_genadd_ti_surv[0],
     explanation_genadd_tdmain_surv[0],
     explanation_genadd_ti_inter_surv[0],
     explanation_genadd_tdmain_inter_surv[0],
     explanation_genadd_tdinter_surv[0]],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_ti_inter.unique_times_[::5],
     model_gbsa_linear_tdmain_inter.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_ti_inter.unique_times_[::5],
     model_gbsa_genadd_tdmain_inter.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_ti, data_x_linear_tdmain, data_x_linear_ti_inter,
     data_x_linear_tdmain_inter, data_x_linear_tdinter,
     data_x_genadd_ti, data_x_genadd_tdmain, data_x_genadd_ti_inter,
     data_x_genadd_tdmain_inter, data_x_genadd_tdinter],
    [func.surv_from_hazard_linear_ti_wrap, func.surv_from_hazard_linear_tdmain_wrap,
     func.surv_from_hazard_linear_ti_inter_wrap, func.surv_from_hazard_linear_tdmain_inter_wrap,
     func.surv_from_hazard_linear_tdinter_wrap, func.surv_from_hazard_genadd_ti_wrap,
     func.surv_from_hazard_genadd_tdmain_wrap, func.surv_from_hazard_genadd_ti_inter_wrap,
     func.surv_from_hazard_genadd_tdmain_inter_wrap, func.surv_from_hazard_genadd_tdinter_wrap],
    ["(1) GT: Linear G(t|x) TI", "(2) GT: Linear G(t|x) TD Main",
     "(3) GT: Linear G(t|x) TI Inter", "(4) GT: Linear G(t|x) TD Main Inter",
     "(5) GT: Linear G(t|x) TD Inter", "(6) GT: General Additive G(t|x) TI",
     "(7) GT: General Additive G(t|x) TD Main",
     "(8) GT: General Additive G(t|x) TI Inter",
     "(9) GT: General Additive G(t|x) TD Main Inter",
     "(10) GT: General Additive G(t|x) TD Inter"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel="Attribution $S(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title,
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
plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save the figure
save_path = f"{path_plots_combined}/plot_surv.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## GBSA SURVIVAL
# Create figure
fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, expl, model, times, data_x, title in zip(
    axes,
    [explanation_linear_ti_gbsa[0],
     explanation_linear_tdmain_gbsa[0],
     explanation_linear_ti_inter_gbsa[0],
     explanation_linear_tdmain_inter_gbsa[0],
     explanation_linear_tdinter_gbsa[0],
     explanation_genadd_ti_gbsa[0],
     explanation_genadd_tdmain_gbsa[0],
     explanation_genadd_ti_inter_gbsa[0],
     explanation_genadd_tdmain_inter_gbsa[0],
     explanation_genadd_tdinter_gbsa[0]],
    [model_gbsa_linear_ti,
     model_gbsa_linear_tdmain,
     model_gbsa_linear_ti_inter,
     model_gbsa_linear_tdmain_inter,
     model_gbsa_linear_tdinter,
     model_gbsa_genadd_ti,
     model_gbsa_genadd_tdmain,
     model_gbsa_genadd_ti_inter,
     model_gbsa_genadd_tdmain_inter,
     model_gbsa_genadd_tdinter],
    [model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_linear_tdmain.unique_times_[::5],
     model_gbsa_linear_ti_inter.unique_times_[::5],
     model_gbsa_linear_tdmain_inter.unique_times_[::5],
     model_gbsa_linear_tdinter.unique_times_[::5],
     model_gbsa_genadd_ti.unique_times_[::5],
     model_gbsa_genadd_tdmain.unique_times_[::5],
     model_gbsa_genadd_ti_inter.unique_times_[::5],
     model_gbsa_genadd_tdmain_inter.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_ti, data_x_linear_tdmain, data_x_linear_ti_inter,
     data_x_linear_tdmain_inter, data_x_linear_tdinter,
     data_x_genadd_ti, data_x_genadd_tdmain, data_x_genadd_ti_inter,
     data_x_genadd_tdmain_inter, data_x_genadd_tdinter],
    ["(1) GT: Linear G(t|x) TI", "(2) GT: Linear G(t|x) TD Main",
     "(3) GT: Linear G(t|x) TI Inter", "(4) GT: Linear G(t|x) TD Main Inter",
     "(5) GT: Linear G(t|x) TD Inter", "(6) GT: General Additive G(t|x) TI",
     "(7) GT: General Additive G(t|x) TD Main",
     "(8) GT: General Additive G(t|x) TI Inter",
     "(9) GT: General Additive G(t|x) TD Main Inter",
     "(10) GT: General Additive G(t|x) TD Inter"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        model=model,
        times = times, 
        data_x=data_x,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title,
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
plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save the figure
save_path = f"{path_plots_combined}/plot_gbsa_surv.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## COXPH SURVIVAL
# Create figure
fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, expl, model, times, data_x, title in zip(
    axes,
    [explanation_linear_ti_cox[0],
     explanation_linear_tdmain_cox[0],
     explanation_linear_ti_inter_cox[0],
     explanation_linear_tdmain_inter_cox[0],
     explanation_linear_tdinter_cox[0],
     explanation_genadd_ti_cox[0],
     explanation_genadd_tdmain_cox[0],
     explanation_genadd_ti_inter_cox[0],
     explanation_genadd_tdmain_inter_cox[0],
     explanation_genadd_tdinter_cox[0]],
    [model_cox_linear_ti,
     model_cox_linear_tdmain,
     model_cox_linear_ti_inter,
     model_cox_linear_tdmain_inter,
     model_cox_linear_tdinter,
     model_cox_genadd_ti,
     model_cox_genadd_tdmain,
     model_cox_genadd_ti_inter,
     model_cox_genadd_tdmain_inter,
     model_cox_genadd_tdinter],
    [model_cox_linear_ti.unique_times_[::5],
     model_cox_linear_tdmain.unique_times_[::5],
     model_cox_linear_ti_inter.unique_times_[::5],
     model_cox_linear_tdmain_inter.unique_times_[::5],
     model_cox_linear_tdinter.unique_times_[::5],
     model_cox_genadd_ti.unique_times_[::5],
     model_cox_genadd_tdmain.unique_times_[::5],
     model_cox_genadd_ti_inter.unique_times_[::5],
     model_cox_genadd_tdmain_inter.unique_times_[::5],
     model_cox_genadd_tdinter.unique_times_[::5]],
    [data_x_linear_ti, data_x_linear_tdmain, data_x_linear_ti_inter,
     data_x_linear_tdmain_inter, data_x_linear_tdinter,
     data_x_genadd_ti, data_x_genadd_tdmain, data_x_genadd_ti_inter,
     data_x_genadd_tdmain_inter, data_x_genadd_tdinter],
    ["(1) GT: Linear G(t|x) TI", "(2) GT: Linear G(t|x) TD Main",
     "(3) GT: Linear G(t|x) TI Inter", "(4) GT: Linear G(t|x) TD Main Inter",
     "(5) GT: Linear G(t|x) TD Inter", "(6) GT: General Additive G(t|x) TI",
     "(7) GT: General Additive G(t|x) TD Main",
     "(8) GT: General Additive G(t|x) TI Inter",
     "(9) GT: General Additive G(t|x) TD Main Inter",
     "(10) GT: General Additive G(t|x) TD Inter"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        model=model,
        times = times, 
        data_x=data_x,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title,
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
plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save the figure
save_path = f"{path_plots_combined}/plot_coxph_surv.pdf"
fig.savefig(save_path, bbox_inches="tight")

######## OVERVIEW PLOT
# Create figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, smooth_w, expl, times, data_x, surv_fn, title, yaxis_lab in zip(
    axes,
    [100, 100, 50, 100, 100, 50],
    [explanation_genadd_tdmain_inter_loghaz[0],
     explanation_linear_ti_haz[0],
     explanation_linear_ti_surv[0],
     explanation_genadd_tdinter_loghaz[0],
     explanation_linear_tdmain_inter_haz[0],
     explanation_genadd_ti_inter_surv[0]],
    [model_gbsa_genadd_tdmain_inter.unique_times_[::5],
     model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_linear_ti.unique_times_[::5],
     model_gbsa_genadd_tdinter.unique_times_[::5],
     model_gbsa_linear_tdmain_inter.unique_times_[::5],
     model_gbsa_genadd_ti_inter.unique_times_[::5]],
    [data_x_genadd_tdmain_inter, data_x_linear_ti, data_x_linear_ti,
     data_x_genadd_tdinter, data_x_linear_tdmain_inter, data_x_genadd_ti_inter],
    [func.log_hazard_wrap_genadd_tdmain_inter, func.hazard_wrap_linear_ti, func.surv_from_hazard_linear_ti_wrap, 
     func.log_hazard_wrap_genadd_tdinter, func.hazard_wrap_linear_ti, func.surv_from_hazard_genadd_ti_inter_wrap],
    ["(9) GenAdd TD Main Inter", "(1) Linear TI, No Inter", "(1) Linear TI, No Inter", "(10) GenAdd TD Inter", "(4) Lin TD Main, Inter", "(8) GenAdd TI Inter"],
    ["Attribution $\log(h(t|x))$", "Attribution $h(t|x)$", "Attribution $S(t|x)$",
     "Attribution $\log(h(t|x))$", "Attribution $h(t|x)$", "Attribution $S(t|x)$"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        times=times,
        data_x=data_x,
        survival_fn=surv_fn,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel=yaxis_lab,
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=smooth_w,
        smooth_poly=1,
        title=title,
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
save_path = f"{path_plots_combined}/plot_overview_sim.pdf"
fig.savefig(save_path, bbox_inches="tight")

############## MODELS PLOTS
# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
axes = axes.flatten()  

handles_all, labels_all = [], []

for ax, expl, model, times, data_x, title in zip(
    axes,
    [explanation_genadd_ti_inter_cox[0],
     explanation_genadd_ti_inter_gbsa[0]],
    [model_cox_genadd_ti_inter,
     model_gbsa_genadd_ti_inter],
    [model_cox_genadd_ti_inter.unique_times_[::5],
     model_gbsa_genadd_ti_inter.unique_times_[::5]],
    [data_x_genadd_ti_inter, data_x_genadd_ti_inter],
    ["CoxPH: (8) GenAdd TI Inter",
     "GBSA: (8) GenAdd TI Inter"]
):
    h, l = func.plot_interact_ax(
        ax=ax,
        explanations_all=expl,
        model=model,
        times = times, 
        data_x=data_x,
        compare_plots="Diff",
        idx_plot=idx,
        ylabel="Attribution $\hat{S}(t|x)$",
        label_fontsize=16,
        tick_fontsize=14,
        smooth=True,
        smooth_window=50,
        smooth_poly=1,
        title=title,
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
plt.tight_layout(rect=[0, 0.14, 1, 1])

# Save the figure
save_path = f"{path_plots_combined}/plot_sim_models.pdf"
fig.savefig(save_path, bbox_inches="tight")
