# === Imports ===
import os
import logging
import numpy as np
import torch
import importlib
from tqdm import tqdm
from datetime import datetime
#from joblib import Parallel, delayed
import os
import logging
from tqdm.contrib.concurrent import process_map

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from lifelines import CoxPHFitter

import torchtuples as tt
from pydataset import data
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
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Load and Prepare Dataset ===
df = data('flchain')
df = df.rename(columns={'futime': 'duration', 'death': 'event'})
le = LabelEncoder()
#df['chapter'] = le.fit_transform(df['chapter'])
df['sex'] = le.fit_transform(df['sex'])
df = df.drop(columns=["sample.yr", "chapter"])
df = df.dropna()
print(df.columns)
print(df)

# Train/Validation/Test Split
df_test = df.sample(frac=0.2, random_state=SEED)
df_remaining = df.drop(df_test.index)
df_val = df_remaining.sample(frac=0.2, random_state=SEED)
df_train = df_remaining.drop(df_val.index)

# === Preprocessing for CoxTime ===
cols_standardize = ["age", "kappa", "lambda", "creatinine", "mgus"]
cols_leave = ["sex", "flc.grp"]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
x_test

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
base_path = "./flchain/plots"
os.makedirs(base_path, exist_ok=True)
random_indices = np.random.choice(data_x_test.shape[0], size=3, replace=False)

main_features = data_x_test_df.columns.tolist()
interactions = [
    ["age * sex", "age * sample.yr", "age * kappa", "age * lambda", "age * flc.grp", "age * creatinine", "age * mgus"],
    ["sex * sample.yr", "sex * kappa", "sex * lambda", "sex * flc.grp", "sex * creatinine", "sex * mgus"],
    ["sample.yr * kappa", "sample.yr * lambda", "sample.yr * flc.grp", "sample.yr * creatinine", "sample.yr * mgus"],
    ["kappa * lambda", "kappa * flc.grp", "kappa * creatinine", "kappa * mgus"],
    ["lambda * flc.grp", "lambda * creatinine", "lambda * mgus"],
    ["flc.grp * creatinine", "flc.grp * mgus"],
    ["creatinine * mgus"]
]

# === Plot Generation ===
def generate_plots_for_instance(i, idx):
    try:
        x_new = data_x_test[[idx]]
        
        explanations = {
            "rf": survshapiq_func.survshapiq(model_rf, data_x_test, x_new, 20, 256, 2, main_features),
            "cox": survshapiq_func.survshapiq(model_cox, data_x_test, x_new, 20, 256, 2, main_features),
            "gbsa": survshapiq_func.survshapiq(model_gbsa, data_x_test, x_new, 20, 256, 2, main_features),
            "coxtime": survshapiq_func.survshapiq_pycox(model_coxtime, x_test, x_new, 20, 256, 2, main_features)
        }

        for name, explanation in explanations.items():
            plot_func = (
                survshapiq_func.plot_interact_pycox2
                if name == "coxtime"
                else survshapiq_func.plot_interact2
            )
            model = model_coxtime if name == "coxtime" else eval(f"model_{name}")
            x_input = x_test if name == "coxtime" else None

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

        logging.info(f"Generated plots for instance {i} (index {idx})")

    except Exception as e:
        logging.error(f"Error on instance {i} (index {idx}): {e}")

# === Execute Plotting not in Parallel with tqdm ===
# Sequential execution with progress bar
for i, idx in tqdm(enumerate(random_indices), total=len(random_indices), desc="Generating plots"):
    generate_plots_for_instance(i, idx)

