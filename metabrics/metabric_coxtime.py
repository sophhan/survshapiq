import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
from sksurv.metrics import integrated_brier_score
import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import shapiq
import importlib
import survshapiq_func
import os
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime
importlib.reload(survshapiq_func)
np.random.seed(1234)
_ = torch.manual_seed(123)

# Load the METABRIC dataset
df_train = metabric.read_df()
print(df_train.columns)

# Define meaningful column names
custom_columns = [
     "MKI67",
     "EGFR",
     "PGR",
     "ERBB2",
     "hormonal",
     "radio",
     "chemo",
     "ER_positive",
     "age_diag",
    'duration', 
    'event'
]

# Rename the columns
df_train.columns = custom_columns
df_train.head()

# Split data into train, validation, and test sets
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
df_train.head()

## COXTIME EXPLANATIONS
# Define the columns to standardize and leave as is
cols_standardize = ["MKI67", "EGFR", "PGR", "ERBB2", 'age_diag']
cols_leave = ["hormonal", "radio", "chemo", "ER_positive"]

# Standardize the columns and transform the data for NN processing
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
val.shapes()
val.repeat(2).cat().shapes()

# Define the model
in_features = x_train.shape[1]
num_nodes = [32, 32]
batch_norm = True
dropout = 0.1
batch_size = 256
epochs = 512
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

# Define the learning rate finder
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
lrfinder.get_best_lr()
model.optimizer.set_lr(0.05)
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

# Train the model
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
ev.concordance_td()

# Select instance for explanation
x_new = x_test[[3]]
print(x_new)

# Explain the x_new for some time points
explanation_df = survshapiq_func.survshapiq_pycox(model, x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'])

# Plot the interaction values
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_main2.png", include_features = ["MKI67","EGFR","PGR","ERBB2","hormonal","radio","chemo","ER_positive","age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x0inter2.png", include_features = ["MKI67 * EGFR","MKI67 * PGR","MKI67 * ERBB2","MKI67 * hormonal","MKI67 * radio","MKI67 * chemo","MKI67 * ER_positive","MKI67 * age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x1inter2.png", include_features = ["EGFR * MKI67", "EGFR * PGR","EGFR * ERBB2","EGFR * hormonal","EGFR * radio","EGFR * chemo","EGFR * ER_positive","EGFR * age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x2inter2.png", include_features = ["PGR * ERBB2","PGR * hormonal","PGR * radio","PGR * chemo","PGR * ER_positive","PGR * age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x3inter2.png", include_features = ["hormonal * radio","hormonal * chemo","hormonal * ER_positive","hormonal * age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x4inter2.png", include_features = ["radio * chemo", "radio * ER_positive", "radio * age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x5inter2.png", include_features = ["chemo * ER_positive", "chemo * age_diag"])
survshapiq_func.plot_interact_pycox2(explanation_df, model, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_coxtime_metabrics_x6inter2.png", include_features = ["ER_positive * age_diag"])

## RF, GBSG, COX EXPLANATIONS
# Convert eventtime and status columns to a structured array
data_y_train, data_x_train_df = survshapiq_func.prepare_survival_data(df_train, event_col='duration', status_col='event', id_col=None)
data_y_test, data_x_test_df= survshapiq_func.prepare_survival_data(df_test, event_col='duration', status_col='event', id_col=None)
print(data_y_train)
print(data_x_train)
data_x_train = data_x_train_df.values
data_x_test = data_x_test_df.values

# Fit random survival forest model 
model_rf = RandomSurvivalForest()
model_rf.fit(data_x_train, data_y_train)
print(f'C-index (train): {model_rf.score(data_x_test, data_y_test).item():0.3f}')
ibs_rf = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model_rf)
print(f'Integrated Brier Score (train): {ibs_rf:0.3f}')

# Fit CoxPH
model_cox = CoxPHSurvivalAnalysis()
model_cox.fit(data_x_train, data_y_train)
print(f'C-index (train): {model_cox.score(data_x_test, data_y_test).item():0.3f}')
ibs_cox = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model_cox)
print(f'Integrated Brier Score (train): {ibs_cox:0.3f}')

# Fit GradientBoostingSurvivalAnalysis
model_gbsa = GradientBoostingSurvivalAnalysis()
model_gbsa.fit(data_x_train, data_y_train)
print(f'C-index (train): {model_gbsa.score(data_x_test, data_y_test).item():0.3f}')
ibs_gbsa = survshapiq_func.compute_integrated_brier(data_y_test, data_x_test, model_gbsa)
print(f'Integrated Brier Score (train): {ibs_gbsa:0.3f}')

# Create data point for explanation
x_new = data_x_test[[2]]
print(x_new)

# Explain the first row of x_new for every third time point
explanation_df_rf = survshapiq_func.survshapiq(model_rf, data_x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names = data_x_test_df.columns)
explanation_df_cox = survshapiq_func.survshapiq(model_cox, data_x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names = data_x_test_df.columns)
explanation_df_gbsa = survshapiq_func.survshapiq(model_gbsa, data_x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names = data_x_test_df.columns)

# Plot the interaction values rf
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_main2.png", include_features = ["MKI67","EGFR","PGR","ERBB2","hormonal","radio","chemo","ER_positive","age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x0inter2.png", include_features = ["MKI67 * EGFR","MKI67 * PGR","MKI67 * ERBB2","MKI67 * hormonal","MKI67 * radio","MKI67 * chemo","MKI67 * ER_positive","MKI67 * age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x1inter2.png", include_features = ["EGFR * MKI67", "EGFR * PGR","EGFR * ERBB2","EGFR * hormonal","EGFR * radio","EGFR * chemo","EGFR * ER_positive","EGFR * age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x2inter2.png", include_features = ["PGR * ERBB2","PGR * hormonal","PGR * radio","PGR * chemo","PGR * ER_positive","PGR * age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x3inter2.png", include_features = ["hormonal * radio","hormonal * chemo","hormonal * ER_positive","hormonal * age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x4inter2.png", include_features = ["radio * chemo", "radio * ER_positive", "radio * age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x5inter2.png", include_features = ["chemo * ER_positive", "chemo * age_diag"])
survshapiq_func.plot_interact2(explanation_df_rf, model_rf, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_rf_metabrics_x6inter2.png", include_features = ["ER_positive * age_diag"])

# Plot the interaction values gbsa
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_main2.png", include_features = ["MKI67","EGFR","PGR","ERBB2","hormonal","radio","chemo","ER_positive","age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x0inter2.png", include_features = ["MKI67 * EGFR","MKI67 * PGR","MKI67 * ERBB2","MKI67 * hormonal","MKI67 * radio","MKI67 * chemo","MKI67 * ER_positive","MKI67 * age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x1inter2.png", include_features = ["EGFR * MKI67", "EGFR * PGR","EGFR * ERBB2","EGFR * hormonal","EGFR * radio","EGFR * chemo","EGFR * ER_positive","EGFR * age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x2inter2.png", include_features = ["PGR * ERBB2","PGR * hormonal","PGR * radio","PGR * chemo","PGR * ER_positive","PGR * age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x3inter2.png", include_features = ["hormonal * radio","hormonal * chemo","hormonal * ER_positive","hormonal * age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x4inter2.png", include_features = ["radio * chemo", "radio * ER_positive", "radio * age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x5inter2.png", include_features = ["chemo * ER_positive", "chemo * age_diag"])
survshapiq_func.plot_interact2(explanation_df_gbsa, model_gbsa, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x6inter2.png", include_features = ["ER_positive * age_diag"])

# Plot the interaction values cox
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_main2.png", include_features = ["MKI67","EGFR","PGR","ERBB2","hormonal","radio","chemo","ER_positive","age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x0inter2.png", include_features = ["MKI67 * EGFR","MKI67 * PGR","MKI67 * ERBB2","MKI67 * hormonal","MKI67 * radio","MKI67 * chemo","MKI67 * ER_positive","MKI67 * age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x1inter2.png", include_features = ["EGFR * MKI67", "EGFR * PGR","EGFR * ERBB2","EGFR * hormonal","EGFR * radio","EGFR * chemo","EGFR * ER_positive","EGFR * age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x2inter2.png", include_features = ["PGR * ERBB2","PGR * hormonal","PGR * radio","PGR * chemo","PGR * ER_positive","PGR * age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x3inter2.png", include_features = ["hormonal * radio","hormonal * chemo","hormonal * ER_positive","hormonal * age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x4inter2.png", include_features = ["radio * chemo", "radio * ER_positive", "radio * age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x5inter2.png", include_features = ["chemo * ER_positive", "chemo * age_diag"])
survshapiq_func.plot_interact2(explanation_df_cox, model_cox, x_test, time_stride=10, x_new = x_new, save_path = "/home/slangbei/survshapiq/plots_metabrics/plot_gbsa_metabrics_x6inter2.png", include_features = ["ER_positive * age_diag"])


# === Configuration ===
np.random.seed(42)
base_path = "/home/slangbei/survshapiq/plots_coxtime"
os.makedirs(base_path, exist_ok=True)
num_instances = 2
random_indices = np.random.choice(data_x_test.shape[0], num_instances, replace=False)

main_features = ["MKI67", "EGFR", "PGR", "ERBB2", "hormonal", "radio", "chemo", "ER_positive", "age_diag"]
interactions = [
    ["MKI67 * EGFR", "MKI67 * PGR", "MKI67 * ERBB2", "MKI67 * hormonal", "MKI67 * radio", "MKI67 * chemo", "MKI67 * ER_positive", "MKI67 * age_diag"],
    ["EGFR * MKI67", "EGFR * PGR", "EGFR * ERBB2", "EGFR * hormonal", "EGFR * radio", "EGFR * chemo", "EGFR * ER_positive", "EGFR * age_diag"],
    ["PGR * ERBB2", "PGR * hormonal", "PGR * radio", "PGR * chemo", "PGR * ER_positive", "PGR * age_diag"],
    ["hormonal * radio", "hormonal * chemo", "hormonal * ER_positive", "hormonal * age_diag"],
    ["radio * chemo", "radio * ER_positive", "radio * age_diag"],
    ["chemo * ER_positive", "chemo * age_diag"],
    ["ER_positive * age_diag"]
]

# === Plotting Function ===
def generate_plots_for_instance(i, idx):
    try:
        x_new = data_x_test[[idx]]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate explanations
        explanation_rf = survshapiq_func.survshapiq(model_rf, data_x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names=data_x_test_df.columns)
        explanation_cox = survshapiq_func.survshapiq(model_cox, data_x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names=data_x_test_df.columns)
        explanation_gbsa = survshapiq_func.survshapiq(model_gbsa, data_x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names=data_x_test_df.columns)
        explanation_coxtime = survshapiq_func.survshapiq_pycox(model_coxtime, x_test, x_new, time_stride=10, budget=2**8, max_order=2, feature_names = data_x_test_df.columns)
        

        models = [
            ("rf", model_rf, explanation_rf),
            ("cox", model_cox, explanation_cox),
            ("gbsa", model_gbsa, explanation_gbsa),
            ("coxtime", model_coxtime, explanation_coxtime)
        ]

        for model_name, model, explanation in models:
            # Main feature plot
            save_main = os.path.join(base_path, f"plot_{model_name}_inst{i}_main.png")
            if model_name == "coxtime":
                survshapiq_func.plot_interact_pycox2(explanation, model, x_test, time_stride=10, x_new=x_new, save_path=save_main, include_features=main_features)
            else:
                survshapiq_func.plot_interact2(explanation, model, time_stride=10, x_new=x_new, save_path=save_main, include_features=main_features)

            # Interaction plots
            for j, feature_set in enumerate(interactions):
                save_inter = os.path.join(base_path, f"plot_{model_name}_inst{i}_x{j}inter.png")
                if model_name == "coxtime":
                    survshapiq_func.plot_interact_pycox2(explanation, model, x_test, time_stride=10, x_new=x_new, save_path=save_inter, include_features=feature_set)
                else:
                    survshapiq_func.plot_interact2(explanation, model, time_stride=10, x_new=x_new, save_path=save_inter, include_features=feature_set)

        logging.info(f"Successfully generated plots for instance {i} (index {idx})")

    except Exception as e:
        logging.error(f"Failed on instance {i} (index {idx}): {str(e)}")


# === Parallel Processing with Progress Bar ===
Parallel(n_jobs=-1)(
    delayed(generate_plots_for_instance)(i, idx)
    for i, idx in enumerate(tqdm(random_indices, desc="Generating plots"))
)
