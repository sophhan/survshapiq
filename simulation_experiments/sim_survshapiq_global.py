# load necessary packages
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score

import shapiq
import func as func


# global configurations
SEED = 1234
np.random.seed(SEED)
path_data = "/home/slangbei/survshapiq/survshapiq/simulation/data/"
path_explanations = "/home/slangbei/survshapiq/survshapiq/simulation/explanations/"

#---------------------------
# 1) Linear G(t|x), TI (no interactions)
#---------------------------

# load and prepare dataset
df = pd.read_csv(f"{path_data}1_simdata_linear_ti.csv")
print(df.head())

# traditional models
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model


model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# expülanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_linear_ti.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_linear_ti.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_linear_ti,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_linear_ti.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_linear_ti,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_linear_ti.csv", index=False)

# SURVIVAL
#  get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_linear_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_linear_ti.csv", index=False)


#---------------------------
# 2) Linear G(t|x), TD MAIN (no interactions)
#---------------------------

# load and prepare dataset
df = pd.read_csv(f"{path_data}2_simdata_linear_tdmain.csv")
print(df.head())

# traditional models
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model


model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configureation
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_linear_tdmain.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_linear_tdmain.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_linear_tdmain,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_linear_tdmain.csv", index=False)

# LOG HAZARD
#  get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_linear_tdmain,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_linear_tdmain.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_linear_tdmain_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_linear_tdmain.csv", index=False)


#---------------------------
# 3) Linear G(t|x), TI (interactions)
#---------------------------

# load and prepare dataset
df = pd.read_csv(f"{path_data}3_simdata_linear_ti_inter.csv")
print(df.head())

# traditional models
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model


model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_linear_ti_inter.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_linear_ti_inter.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_linear_ti_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_linear_ti_inter.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_linear_ti_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_linear_ti_inter.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_linear_ti_inter_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_linear_ti_inter.csv", index=False)

#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
#---------------------------

# === Load and Prepare Dataset ===
df = pd.read_csv(f"{path_data}4_simdata_linear_tdmain_inter.csv")
print(df.head())

# traditional models
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_linear_tdmain_inter.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_linear_tdmain_inter.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_linear_tdmain_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_linear_tdmain_inter.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_linear_tdmain_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_linear_tdmain_inter.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_linear_tdmain_inter_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_linear_tdmain_inter.csv", index=False)


#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
#---------------------------

# load and prepare dataset
df = pd.read_csv(f"{path_data}5_simdata_linear_tdinter.csv")
print(df.head())

# traditional models
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_linear_tdinter.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_linear_tdinter.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_linear_tdmain_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_linear_tdinter.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_linear_tdmain_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_linear_tdinter.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_linear_tdmain_inter_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_linear_tdinter.csv", index=False)


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
#---------------------------

# load and prepare dataset
df = pd.read_csv(f"{path_data}6_simdata_genadd_ti.csv")
print(df.head())

# traditional models
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_genadd_ti.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_genadd_ti.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_genadd_ti,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_genadd_ti.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_genadd_tdmain,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_genadd_ti.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_genadd_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_genadd_ti.csv", index=False)


#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
#---------------------------

# load and prepare dataset 
df = pd.read_csv(f"{path_data}7_simdata_genadd_tdmain.csv")
print(df.head())

# traditional models 
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_genadd_tdmain.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_genadd_tdmain.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_genadd_tdmain,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_genadd_tdmain.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_genadd_tdmain,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_genadd_tdmain.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_genadd_tdmain_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_genadd_tdmain.csv", index=False)


#---------------------------
# 8) Generalized Additive G(t|x), TI (with interactions)
#---------------------------

# load and prepare dataset 
df = pd.read_csv(f"{path_data}8_simdata_genadd_ti_inter.csv")
print(df.head())

# traditional models 
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_genadd_ti_inter.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_genadd_ti_inter.csv", index=False)

# HAZARD 
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_genadd_ti_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_genadd_ti_inter.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_genadd_ti_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_genadd_ti_inter.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_genadd_ti_inter_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_genadd_ti_inter.csv", index=False)


#---------------------------
# 9) Generalized Additive G(t|x), TD Main (with interactions)
#---------------------------

# load and prepare dataset 
df = pd.read_csv(f"{path_data}9_simdata_genadd_tdmain_inter.csv")
print(df.head())

# traditional models 
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_genadd_tdmain_inter.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv(f"{path_explanations}/gbsa_attributions_genadd_tdmain_inter.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_genadd_tdmain_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_genadd_tdmain_inter.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_genadd_tdmain_inter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_genadd_tdmain_inter.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_genadd_tdmain_inter_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_genadd_tdmain_inter.csv", index=False)


#---------------------------
# 10) Generalized Additive G(t|x), TD Inter (with interactions)
#---------------------------

# load and prepare dataset 
df = pd.read_csv(f"{path_data}10_simdata_genadd_tdinter.csv")
print(df.head())

# traditional models 
data_y, data_x_df = func.prepare_survival_data(df)
data_x_linear = data_x_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear, data_y, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.18, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# explanation configuration
time_stride = 5
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

# coxph
# get all explanations in parallel for all observations
explanations_cox = func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv(f"{path_explanations}/cox_attributions_genadd_tdinter.csv", index=False)

# gbsa
# get all explanations in parallel for all observations
explanations_gbsa = func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_genadd_tdinter.csv", index=False)

# HAZARD
# get all explanations in parallel for all observations
explanations_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.hazard_wrap_genadd_tdinter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard = func.annotate_explanations(explanations_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard.to_csv(f"{path_explanations}/hazard_attributions_genadd_tdinter.csv", index=False)

# LOG HAZARD
# get all explanations in parallel for all observations
explanations_log_hazard = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.log_hazard_wrap_genadd_tdinter,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard = func.annotate_explanations(explanations_log_hazard, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard.to_csv(f"{path_explanations}/log_hazard_attributions_genadd_tdinter.csv", index=False)

# SURVIVAL
# get all explanations in parallel for all observations
explanations_surv = func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=func.surv_from_hazard_genadd_tdinter_wrap,
    times=model_gbsa.unique_times_,
    time_stride=time_stride,
    budget=2**8,
    max_order=2,
    feature_names=df.columns[3:6],
    exact=True,
    n_jobs=20
)

# generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv = func.annotate_explanations(explanations_surv, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv.to_csv(f"{path_explanations}/survival_attributions_genadd_tdinter.csv", index=False)

