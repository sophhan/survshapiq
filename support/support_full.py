# === Imports ===
import os
import logging
import numpy as np
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
from pycox.datasets import support
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv

import shapiq
import survshapiq_func
importlib.reload(survshapiq_func)

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
df = support.read_df()
df.columns = ["age", "sex", "race", "num_comorbidities", "diabetes", "dementia", "cancer", "mean_arterial_pressure",
              "heart_rate", "respiration_rate", "temperature", "white_blood_cell_count", "serum_sodium",     
              "serum_creatinine", "duration", "event"]

# Train/Validation/Test Split
df_test = df.sample(frac=0.2, random_state=SEED)
df_remaining = df.drop(df_test.index)
df_val = df_remaining.sample(frac=0.2, random_state=SEED)
df_train = df_remaining.drop(df_val.index)

# === Preprocessing for CoxTime ===
cols_standardize = ["age", "mean_arterial_pressure", "heart_rate", "respiration_rate", "temperature",
                    "white_blood_cell_count", "serum_sodium", "serum_creatinine"]
cols_leave = ["sex", "race",  "num_comorbidities", "diabetes", "dementia", "cancer"]

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
print("test1")

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
base_path = "/home/slangbei/survshapiq/survshapiq/support/plots"
os.makedirs(base_path, exist_ok=True)
random_indices = np.random.choice(data_x_test.shape[0], size=3, replace=False)
random_indices = np.array([2129, 235, 3480, 905])  

main_features = data_x_test_df.columns.tolist()
interactions = [
    ["age * sex", "age * race", "age * num_comorbidities", "age * diabetes", "age * dementia", "age * cancer", "age * mean_arterial_pressure", "age * heart_rate"],
    ["sex * race", "sex * num_comorbidities", "sex * diabetes", "sex * dementia", "sex * cancer", "sex * mean_arterial_pressure", "sex * heart_rate"],
    ["race * num_comorbidities", "race * diabetes", "race * dementia", "race * cancer", "race * mean_arterial_pressure", "race * heart_rate"],
    ["num_comorbidities * diabetes", "num_comorbidities * dementia", "num_comorbidities * cancer", "num_comorbidities * mean_arterial_pressure"],
    ["diabetes * dementia", "diabetes * cancer", "diabetes * mean_arterial_pressure"],
    ["dementia * cancer", "dementia * mean_arterial_pressure"],
    ["cancer * mean_arterial_pressure"]
]


# === Plot Generation ===
def generate_plots_for_instance(i, idx):
    try:
        x_new = data_x_test[[idx]]
        
        explanations = {
            #"rf": survshapiq_func.survshapiq(model_rf, data_x_test, x_new, 20, 256, 2, main_features),
            "cox": survshapiq_func.survshapiq(model_cox, data_x_test, x_new, 20, 256, 2, main_features),
            "gbsa": survshapiq_func.survshapiq(model_gbsa, data_x_test, x_new, 20, 256, 2, main_features)
            #"coxtime": survshapiq_func.survshapiq_pycox(model_coxtime, x_test, x_new, 20, 256, 2, main_features)
        }

        for name, explanation in explanations.items():
            plot_func = (
                survshapiq_func.plot_interact_pycox2
                if name == "coxtime"
                else survshapiq_func.plot_interact2
            )
            model = model_coxtime if name == "coxtime" else eval(f"model_{name}")
            x_input = x_test if name == "coxtime" else None
            print(f"Generating explanations for {name} on instance {i} (index {idx})")

            # Main feature plot
            plot_func(
                explanations_all=explanation,
                model=model,
                data_x=x_input,
                time_stride=20,
                x_new=x_new,
                save_path=os.path.join(base_path, f"plot_{name}_inst{i}_main.png"),
                include_features=main_features
            )
            print(f"Main feature plot for {name} on instance {i} (index {idx}) saved.")

            # Interaction plots
            for j, feature_set in enumerate(interactions):
                plot_func(
                    explanations_all=explanation,
                    model=model,
                    data_x=x_input,
                    time_stride=20,
                    x_new=x_new,
                    save_path=os.path.join(base_path, f"plot_{name}_inst{i}_x{j}inter.png"),
                    include_features=feature_set
                )
                print(f"Interaction plot {j} for {name} on instance {i} (index {idx}) saved.")

        logging.info(f"Generated plots for instance {i} (index {idx})")

    except Exception as e:
        logging.error(f"Error on instance {i} (index {idx}): {e}")

# === Execute Plotting not in Parallel with tqdm ===
# Sequential execution with progress bar
for i, idx in tqdm(enumerate(random_indices), total=len(random_indices), desc="Generating plots"):
    generate_plots_for_instance(i, idx)