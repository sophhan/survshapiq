# === Imports ===
import os
import logging
import numpy as np
import pandas as pd
import torch
import importlib
from tqdm import tqdm
from datetime import datetime
import pickle
import os
import logging
from tqdm.contrib.concurrent import process_map

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from lifelines import CoxPHFitter

import torchtuples as tt
from pycox.datasets import gbsg
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shapiq
import support.survshapiq_func as survshapiq_func
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

# === Load and Prepare Dataset ===
df = gbsg.read_df()
df.columns = [
    "hormonal_therapy", "menopause", "tumor_grade", "age", "tumor_size", 
    "progesterone_receptors", "estrogen_receptors", "duration", "event"
]

# Train/Validation/Test Split
df_test = df.sample(frac=0.2, random_state=SEED)
df_remaining = df.drop(df_test.index)
df_val = df_remaining.sample(frac=0.2, random_state=SEED)
df_train = df_remaining.drop(df_val.index)

# === Preprocessing for CoxTime ===
cols_standardize = ["age", "progesterone_receptors", "estrogen_receptors"]
cols_leave = ["hormonal_therapy", "menopause", "tumor_grade", "tumor_size"]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

labtrans = CoxTime.label_transform()
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)

val = tt.tuplefy(x_val, y_val)

# === CoxTime Model ===
in_features = x_train.shape[1]
net = MLPVanillaCoxTime(in_features, [32, 32], batch_norm=True, dropout=0.1)
model_coxtime = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

# Learning Rate Finder
model_coxtime.lr_finder(x_train, y_train, batch_size=256, tolerance=2)
model_coxtime.optimizer.set_lr(0.05)
callbacks = [tt.callbacks.EarlyStopping()]

# Train CoxTime
model_coxtime.fit(x_train, y_train, batch_size=256, epochs=512, callbacks=callbacks,
                  verbose=True, val_data=val.repeat(10).cat())
model_coxtime.compute_baseline_hazards()

# Evaluation
surv = model_coxtime.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print(f'CoxTime Concordance Index: {ev.concordance_td():.3f}')
print("test0")

# === Traditional Models (RF, CoxPH, GBSA) ===
data_y_train, data_x_train_df = survshapiq_func.prepare_survival_data(df_train, 'duration', 'event')
data_y_test, data_x_test_df = survshapiq_func.prepare_survival_data(df_test, 'duration', 'event')

data_x_train = data_x_train_df.values
data_x_test = data_x_test_df.values

def evaluate_model(model, name):
    model.fit(data_x_train, data_y_train)
    c_index = model.score(data_x_test, data_y_test)
    ibs = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model)
    print(f'{name} - C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_rf = evaluate_model(RandomSurvivalForest(), "Random Forest")
model_cox = evaluate_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_gbsa = evaluate_model(GradientBoostingSurvivalAnalysis(), "GBSA")

# === SHAP Explanation Config ===
time_stride = 300
data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)
data_x_full_nn = np.concatenate((x_train, x_val, x_test), axis=0)

## Random Forest
# Get all explanations in parallel for all observations
#explanations_rf = survshapiq_func.survshapiq_parallel(
#    model_rf, data_x_train, data_x_full,
#    time_stride=time_stride, budget=2**8, max_order=2,
#    feature_names=df.columns,
#    n_jobs=20,  # or -1 for all cores
#    show_progress=True
#)

# Generate final annotated DataFrames, plots and save dataframes with explanations
#explanation_df_rf = survshapiq_func.annotate_explanations(explanations_rf, model_rf, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
#survshapiq_func.plot_samplewise_mean_abs_attributions(explanation_df_rf, model_name="Random Forest", save_path="/home/slangbei/survshapiq/survshapiq/gbsg/global_plots/rf_attributions.png")
#explanation_df_rf.to_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/rf_attributions.csv", index=False)

## CoxPH
# Get all explanations in parallel for all observations
#explanations_cox = survshapiq_func.survshapiq_parallel(
#    model_cox, data_x_train, data_x_full,
#    time_stride=time_stride, budget=2**8, max_order=2,
#    feature_names=df.columns, 
#    n_jobs=20,  # or -1 for all cores
#    show_progress=True
#)

# Generate final annotated DataFrames, plots and save dataframes with explanations
#explanation_df_cox = survshapiq_func.annotate_explanations(explanations_cox, model_cox, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
#survshapiq_func.plot_samplewise_mean_abs_attributions(explanation_df_cox, model_name="Cox Model", save_path="/home/slangbei/survshapiq/survshapiq/gbsg/global_plots/cox_attributions.png")
#explanation_df_cox.to_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/cox_attributions.csv", index=False)

## GBSA
# Get all explanations in parallel for all observations
#explanations_gbsa = survshapiq_func.survshapiq_parallel(
#    model_gbsa, data_x_train, data_x_full,
#    time_stride=time_stride, budget=2**8, max_order=2,
#    feature_names=df.columns, 
#    n_jobs=20,  # or -1 for all cores
#    show_progress=True
#)

# Generate final annotated DataFrames, plots and save dataframes with explanations
#explanation_df_gbsa = survshapiq_func.annotate_explanations(explanations_gbsa, model_gbsa, sample_idxs=range(len(data_x_full)), time_stride=time_stride)
#survshapiq_func.plot_samplewise_mean_abs_attributions(explanation_df_gbsa, model_name="GBSA Model", save_path="/home/slangbei/survshapiq/survshapiq/gbsg/global_plots/gbsa_attributions.png")
#explanation_df_gbsa.to_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/gbsa_attributions.csv", index=False)

## Coxtime
# Get all explanations in parallel for all observations
explanations_coxtime = survshapiq_func.survshapiq_pycox_parallel(
    model_coxtime, x_train, data_x_full_nn,
    time_stride=time_stride, budget=2**8, max_order=2,
    feature_names=df.columns, 
    n_jobs=20,  # or -1 for all cores
    show_progress=True
)

# Generate final annotated DataFrames, plots and save dataframes with explanations
explanation_df_coxtime = survshapiq_func.annotate_explanations(explanations_coxtime, model_coxtime, sample_idxs=range(len(data_x_full_nn)), time_stride=time_stride)
survshapiq_func.plot_samplewise_mean_abs_attributions(explanation_df_coxtime, model_name="GBSA Model", save_path="/home/slangbei/survshapiq/survshapiq/gbsg/global_plots/coxtime_attributions.png")
explanation_df_coxtime.to_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/coxtime_attributions.csv", index=False)


# Print top 5 sample_idx per feature for each model
# Print top 5 sample_idx per feature for each model
explanation_df_cox = pd.read_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/cox_attributions.csv")
explanation_df_gbsa = pd.read_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/gbsa_attributions.csv")
explanation_df_rf = pd.read_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/rf_attributions.csv")

# Print top 5 sample_idx per feature for each model
model_dfs = {
    "Random Forest": explanation_df_rf,
    "Cox PH": explanation_df_cox,
    "GBSA": explanation_df_gbsa
    #"CoxTime": explanation_df_coxtime
}

non_feature_cols = ["sample_idx", "time_idx"]
summary_rows = []

for model_name, df in model_dfs.items():
    print(f"\n==== {model_name} ====")

    # Group by sample and average absolute attributions across time
    grouped = df.drop(columns=non_feature_cols, errors="ignore").abs().groupby(df["sample_idx"]).mean()

    for feature in grouped.columns:
        top_samples = grouped[feature].nlargest(5)
        print(f"\nTop 5 sample_idx for feature '{feature}':")
        print("sample_idx:", top_samples.index.tolist())
        print("mean |attribution|:", top_samples.values.tolist())

        # Add to summary
        for idx, (sample_idx, attribution) in enumerate(top_samples.items(), start=1):
            summary_rows.append({
                "Model": model_name,
                "Feature": feature,
                "Rank": idx,
                "Sample_idx": sample_idx,
                "Mean_Abs_Attribution": attribution
            })

# Convert the summary to a DataFrame
summary_df = pd.DataFrame(summary_rows)

# Save to CSV
summary_df.to_csv("/home/slangbei/survshapiq/survshapiq/gbsg/explanations/top_feature_attributions.csv", index=False)
