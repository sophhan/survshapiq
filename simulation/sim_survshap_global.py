# === Imports ===
print("test")
import os
import logging
import numpy as np
import pandas as pd
import torch
import importlib
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import logging
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from lifelines import CoxPHFitter

import torchtuples as tt
from pydataset import data
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shapiq
import simulation.survshapiq_func as survshapiq_func
importlib.reload(survshapiq_func)
dir_path = os.getcwd()
print(dir_path)
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp/joblib_temp'

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("script_output.log"),
        logging.StreamHandler()  # This prints to the terminal
    ]
)

# === Configuration ===
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-INDEPENDENCE 
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_ti.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

#model_rf = evaluate_model(RandomSurvivalForest(), "Random Forest")
model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_linear_ti.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_linear_ti.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x1 * x3))

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_linear_ti.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1)))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_linear_ti.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_linear_ti.csv", index=False)


################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN MAIN EFFECTS
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_main.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

#model_rf = evaluate_model(RandomSurvivalForest(), "Random Forest")
model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_linear_tdmain.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_linear_tdmain.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * x1) + (-1.2 * x1 * np.log(t+1)) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1))

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_linear_tdmain.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * x1) + (-1.2 * x1 * np.log(t+1)) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1)))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_linear_tdmain.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_linear_tdmain.csv", index=False)


################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN INTERACTIONS
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_inter.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

#model_rf = evaluate_model(RandomSurvivalForest(), "Random Forest")
model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_linear_tdinter.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_linear_tdinter.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1) + (-0.4 * x1 * x3 * np.log(t+1)))

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_linear_tdinter.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.03 * np.exp((0.8 * x1) + (0.5 * x2) + (0.9 * x3) + (-0.6 * x3 * x1) + (-0.4 * x1 * x3 * np.log(t+1))))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_linear_tdinter.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_linear_tdinter.csv", index=False)

################ ADDITIVE MAIN EFFECTS MODEL
###### TIME-INDEPENDENCE 
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_add_ti.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_add_ti.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_add_ti.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.015 * np.exp(-1.5 * ((x1 ** 2) - 1) + (2/np.pi) * np.arctan(0.5 * x2) + 0.6 * x3)

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_ladd_ti.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.015 * np.exp(-1.5 * ((x1 ** 2) - 1) + (2/np.pi) * np.arctan(0.5 * x2) + 0.6 * x3))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_add_ti.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_add_ti.csv", index=False)


################ GENERAL ADDITIVE MODEL
###### TIME-INDEPENDENCE 
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_ti.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_genadd_ti.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_genadd_ti.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3))

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_genadd_ti.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return  np.log(0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3)))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_genadd_ti.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_genadd_ti.csv", index=False)


################ GENERAL ADDITIVE MODEL
###### TIME-DEPENDENCE IN MAIN EFFECTS
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_main.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_genadd_tdmain.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_genadd_tdmain.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.01 * np.exp(0.2 * x1 - 0.4 * (x1 * np.log(t + 1)) - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3))

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_genadd_tdmain.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.01 * np.exp(0.2 * x1 - 0.4 * (x1 * np.log(t + 1)) - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 + 0.3 * ((x1 ** 2 - 1) * x3)))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_genadd_tdmain.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_genadd_tdmain.csv", index=False)


################ GENERAL ADDITIVE MODEL
###### TIME-DEPENDENCE IN INTERACTION EFFECTS
# === Load and Prepare Dataset ===
df_ti = pd.read_csv("/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_interaction.csv")
print(df_ti.head())

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_ti, data_x_ti_df = survshapiq_func.prepare_survival_data(df_ti)
data_x_linear_ti = data_x_ti_df.values
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
    data_x_linear_ti, data_y_ti, 
    test_size=0.2,   
    random_state=42, 
    stratify=None    
)

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model, min_time=0.05, max_time=69)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 30
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

## CoxPH
# Get all explanations in parallel for all observations
explanations_cox = survshapiq_func.survshapiq_parallel(
    model_cox, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_genadd_tdinter.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
explanations_gbsa = survshapiq_func.survshapiq_parallel(
    model_gbsa, 
    data_x_train, 
    data_x_full,
    time_stride=time_stride, 
    budget=2**8, 
    max_order=2,
    feature_names=df_ti.columns[3:6], 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_genadd_tdinter.csv", index=False)

# hazard
# Define the hazard function
# --- Hazard function (top-level) ---
def hazard_func_ti(t, x1, x2, x3):
    """
    Example hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual hazard function.
    """
    return 0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 - 0.4 * (x1 * x2 * np.log(t + 1)) + 0.3 * ((x1 ** 2 - 1) * x3))

# --- Picklable wrapper ---
def hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, hazard_func_ti, t)

explanations_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_hazard_ti = survshapiq_func.annotate_explanations(explanations_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_genadd_tdinter.csv", index=False)

# log hazard
# Define the hazard function
def log_hazard_func_ti(t, x1, x2, x3):
    """
    Example log hazard function that depends on time, age, bmi, and treatment.
    This is a placeholder; replace with the actual log hazard function.
    """
    return np.log(0.01 * np.exp(0.2 * x1 - 0.3 * ((x1 ** 2) - 1) + 0.5 * ((2 / np.pi) * np.arctan(0.7 * x2)) - 0.4 * x3 + 0.2 * x1 * x2 - 0.4 * (x1 * x2 * np.log(t + 1)) + 0.3 * ((x1 ** 2 - 1) * x3)))

# Wrap the hazard function
# Top-level picklable wrapper
def log_hazard_wrap_ti(X, t):
    return survshapiq_func.hazard_matrix(X, log_hazard_func_ti, t)

explanations_log_hazard_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=log_hazard_wrap_ti,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_log_hazard_ti = survshapiq_func.annotate_explanations(explanations_log_hazard_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_log_hazard_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_genadd_tdinter.csv", index=False)

# survival
# Wrap the survival function
def surv_from_hazard_ti_wrap(X, t):
    return survshapiq_func.survival_from_hazard(X, hazard_func_ti, t)

explanations_surv_ti = survshapiq_func.survshapiq_ground_truth_parallel(
    data_x=data_x_full,
    x_new_list=data_x_full,
    survival_from_hazard_func=surv_from_hazard_ti_wrap,
    times=model_gbsa.unique_times_,
    time_stride=5,
    budget=2**8,
    max_order=2,
    feature_names=df_ti.columns[3:6],
    exact=True,
    n_jobs=20
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_surv_ti = survshapiq_func.annotate_explanations(explanations_surv_ti, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
explanation_df_surv_ti.to_csv("/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_genadd_tdinter.csv", index=False)

