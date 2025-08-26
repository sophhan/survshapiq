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

# Fit random survival forest model 
model_rf_ti = RandomSurvivalForest()
model_rf_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_rf_ti.score(data_x_ti, data_y_ti).item():0.3f}')
ibs_rf_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_rf_ti)
print(f'Integrated Brier Score (train): {ibs_rf_ti:0.3f}')

# Fit CoxPH
model_cox_ti = CoxPHSurvivalAnalysis()
model_cox_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_cox_ti.score(data_x_ti, data_y_ti).item():0.3f}')
ibs_cox_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_cox_ti)
print(f'Integrated Brier Score (train): {ibs_cox_ti:0.3f}')

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_ti = GradientBoostingSurvivalAnalysis()
model_gbsa_ti.fit(data_x_ti, data_y_ti)
print(f'C-index (train): {model_gbsa_ti.score(data_x_ti, data_y_ti).item():0.3f}')
ibs_gbsa_ti = survshapiq_func.compute_integrated_brier(data_y_ti, data_x_ti, model_gbsa_ti)
print(f'Integrated Brier Score (train): {ibs_gbsa_ti:0.3f}')

# Create data point for explanation
x_new_ti = data_x_ti[[3]]
print(x_new_ti)

# Predict survival function for new data points
models = [model_rf_ti, model_cox_ti, model_gbsa_ti]
model_names = ["Random Forest", "Cox PH", "GBSA"]
survshapiq_func.plot_multiple_survival_curves(models, model_names, x_new_ti, sample_idx=0)

# Explain the first row of x_new for every third time point
# k-SII
explanation_df_rf_ti_kSII = survshapiq_func.survshapiq(model_rf_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="k-SII", feature_names = data_x.columns)
explanation_df_cox_ti_kSII = survshapiq_func.survshapiq(model_cox_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="k-SII", feature_names = data_x.columns)
explanation_df_gbsa_ti_kSII = survshapiq_func.survshapiq(model_gbsa_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="k-SII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_ti_kSII, model_rf_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_ti_kSII.png") 
survshapiq_func.plot_interact(explanation_df_cox_ti_kSII, model_cox_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_ti_kSII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_ti_kSII, model_gbsa_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_ti_kSII.png") 

# FSII
explanation_df_rf_ti_FSII = survshapiq_func.survshapiq(model_rf_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FSII", feature_names = data_x.columns)
explanation_df_cox_ti_FSII = survshapiq_func.survshapiq(model_cox_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FSII", feature_names = data_x.columns)
explanation_df_gbsa_ti_FSII = survshapiq_func.survshapiq(model_gbsa_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FSII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_ti_FSII, model_rf_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_ti_FSII.png") 
survshapiq_func.plot_interact(explanation_df_cox_ti_FSII, model_cox_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_ti_FSII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_ti_FSII, model_gbsa_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_ti_FSII.png") 

# FBII
explanation_df_rf_ti_FBII = survshapiq_func.survshapiq(model_rf_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FBII", feature_names = data_x.columns)
explanation_df_cox_ti_FBII = survshapiq_func.survshapiq(model_cox_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FBII", feature_names = data_x.columns)
explanation_df_gbsa_ti_FBII = survshapiq_func.survshapiq(model_gbsa_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FBII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_ti_FBII, model_rf_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_ti_FBII.png") 
survshapiq_func.plot_interact(explanation_df_cox_ti_FBII, model_cox_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_ti_FBII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_ti_FBII, model_gbsa_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_ti_FBII.png") 

# STII
explanation_df_rf_ti_STII = survshapiq_func.survshapiq(model_rf_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, index="STII", feature_names = data_x.columns)
explanation_df_cox_ti_STII = survshapiq_func.survshapiq(model_cox_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, index="STII", feature_names = data_x.columns)
explanation_df_gbsa_ti_STII = survshapiq_func.survshapiq(model_gbsa_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2, index="STII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_ti_STII, model_rf_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_ti_STII.png") 
survshapiq_func.plot_interact(explanation_df_cox_ti_STII, model_cox_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_ti_STII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_ti_STII, model_gbsa_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_ti_STII.png") 

# SII
explanation_df_rf_ti_SII = survshapiq_func.survshapiq(model_rf_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2,  index="SII", feature_names = data_x.columns)
explanation_df_cox_ti_SII = survshapiq_func.survshapiq(model_cox_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2,  index="SII", feature_names = data_x.columns)
explanation_df_gbsa_ti_SII = survshapiq_func.survshapiq(model_gbsa_ti, data_x_ti, x_new_ti, time_stride=10, budget=2**8, max_order=2,  index="SII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_ti_SII, model_rf_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_ti_SII.png") 
survshapiq_func.plot_interact(explanation_df_cox_ti_SII, model_cox_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_ti_SII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_ti_SII, model_gbsa_ti, time_stride=10, x_new = x_new_ti, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_ti_SII.png") 


## Time-dependent Interactions
# Load simulated data DataFrame
simdata_td = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_td.csv")
print(simdata_td.head())

# Convert eventtime and status columns to a structured array
data_y_td, data_x = survshapiq_func.prepare_survival_data(simdata_td)
print(data_y_td)
print(data_x.head())
data_x_td = data_x.values

# Fit random survival forest model 
model_rf_td = RandomSurvivalForest()
model_rf_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_rf_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_rf_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_rf_td)
print(f'Integrated Brier Score (train): {ibs_rf_td:0.3f}')

# Fit CoxPH
model_cox_td = CoxPHSurvivalAnalysis()
model_cox_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_cox_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_cox_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_cox_td)
print(f'Integrated Brier Score (train): {ibs_cox_td:0.3f}')

# Fit GradientBoostingSurvivalAnalysis
model_gbsa_td = GradientBoostingSurvivalAnalysis()
model_gbsa_td.fit(data_x_td, data_y_td)
print(f'C-index (train): {model_gbsa_td.score(data_x_td, data_y_td).item():0.3f}')
ibs_gbsa_td = survshapiq_func.compute_integrated_brier(data_y_td, data_x_td, model_gbsa_td)
print(f'Integrated Brier Score (train): {ibs_gbsa_td:0.3f}')

# Create data point for explanation
x_new_td = data_x_td[[3]]
print(x_new_td)

# Predict survival function for new data points
models = [model_rf_td, model_cox_td, model_gbsa_td]
model_names = ["Random Forest", "Cox PH", "GBSA"]
survshapiq_func.plot_multiple_survival_curves(models, model_names, x_new_td, sample_idx=0)

# Explain the first row of x_new for every third time point
# k-SII
explanation_df_rf_td_kSII = survshapiq_func.survshapiq(model_rf_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="k-SII", feature_names = data_x.columns)
explanation_df_cox_td_kSII = survshapiq_func.survshapiq(model_cox_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="k-SII", feature_names = data_x.columns)
explanation_df_gbsa_td_kSII = survshapiq_func.survshapiq(model_gbsa_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="k-SII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_td_kSII, model_rf_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_td_kSII.png") 
survshapiq_func.plot_interact(explanation_df_cox_td_kSII, model_cox_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_td_kSII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_td_kSII, model_gbsa_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_td_kSII.png") 

# FSII
explanation_df_rf_td_FSII = survshapiq_func.survshapiq(model_rf_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FSII", feature_names = data_x.columns)
explanation_df_cox_td_FSII = survshapiq_func.survshapiq(model_cox_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FSII", feature_names = data_x.columns)
explanation_df_gbsa_td_FSII = survshapiq_func.survshapiq(model_gbsa_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, approximator="exact", index="FSII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_td_FSII, model_rf_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_td_FSII.png") 
survshapiq_func.plot_interact(explanation_df_cox_td_FSII, model_cox_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_td_FSII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_td_FSII, model_gbsa_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_td_FSII.png") 

# FBII
explanation_df_rf_td_FBII = survshapiq_func.survshapiq(model_rf_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="FBII", feature_names = data_x.columns)
explanation_df_cox_td_FBII = survshapiq_func.survshapiq(model_cox_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="FBII", feature_names = data_x.columns)
explanation_df_gbsa_td_FBII = survshapiq_func.survshapiq(model_gbsa_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="FBII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_td_FBII, model_rf_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_td_FBII.png") 
survshapiq_func.plot_interact(explanation_df_cox_td_FBII, model_cox_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_td_FBII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_td_FBII, model_gbsa_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_td_FBII.png") 

# STII
explanation_df_rf_td_STII = survshapiq_func.survshapiq(model_rf_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="STII", feature_names = data_x.columns)
explanation_df_cox_td_STII = survshapiq_func.survshapiq(model_cox_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="STII", feature_names = data_x.columns)
explanation_df_gbsa_td_STII = survshapiq_func.survshapiq(model_gbsa_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="STII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_td_STII, model_rf_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_td_STII.png") 
survshapiq_func.plot_interact(explanation_df_cox_td_STII, model_cox_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_td_STII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_td_STII, model_gbsa_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_td_STII.png") 

# SII
explanation_df_rf_td_SII = survshapiq_func.survshapiq(model_rf_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="SII", feature_names = data_x.columns)
explanation_df_cox_td_SII = survshapiq_func.survshapiq(model_cox_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="SII", feature_names = data_x.columns)
explanation_df_gbsa_td_SII = survshapiq_func.survshapiq(model_gbsa_td, data_x_td, x_new_td, time_stride=10, budget=2**8, max_order=2, index="SII", feature_names = data_x.columns)
survshapiq_func.plot_interact(explanation_df_rf_td_SII, model_rf_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_rf_td_SII.png") 
survshapiq_func.plot_interact(explanation_df_cox_td_SII, model_cox_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_cox_td_SII.png") 
survshapiq_func.plot_interact(explanation_df_gbsa_td_SII, model_gbsa_td, time_stride=10, x_new = x_new_td, save_path = "/home/slangbei/survshapiq/survshapiq/simulation/plots/plot_gbsa_td_SII.png") 

