# %% Imports
import shapiq
import numpy as np
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from pycox.datasets import gbsg

import src

SEED = 1234
np.random.seed(SEED)

# %% Load data
df = gbsg.read_df()
df.columns = [
    "hormonal_therapy", "menopause", "tumor_grade", "age", "tumor_size", 
    "progesterone_receptors", "estrogen_receptors", "duration", "event"
]

df_test = df.sample(frac=0.2, random_state=SEED)
df_train = df.drop(df_test.index)

y_train, x_train_df = src.prepare_survival_data(df_train, 'duration', 'event')
y_test, x_test_df = src.prepare_survival_data(df_test, 'duration', 'event')

x_train = x_train_df.values
x_test = x_test_df.values

# %% Train models
def train_model(model, name):
    model.fit(x_train, y_train)
    c_index = model.score(x_test, y_test)
    ibs = src.compute_integrated_brier(y_test, x_test, model)
    print(f'{name} | C-index: {c_index:.3f}, IBS: {ibs:.3f}')
    return model

model_cox = train_model(CoxPHSurvivalAnalysis(), "CoxPH")
model_rf = train_model(RandomSurvivalForest(), "Random Forest")
model_gbsa = train_model(GradientBoostingSurvivalAnalysis(), "Gradient Boosting")

# %% Define games
def loss_function_ibs(y_true, y_pred, times):
    pred_surv = np.asarray([[fn(t) for t in times] for fn in y_pred])
    ibs = integrated_brier_score(y_true, y_true, pred_surv, times)
    return ibs

times = np.linspace(
    np.percentile([y[1] for y in y_train], 5),
    np.percentile([y[1] for y in y_train], 80),
    51
)

game_cox = src.SageGame(
    data_x=x_test,
    data_y=y_test,
    model=lambda d: model_cox.predict_survival_function(d),
    loss_function=loss_function_ibs,
    loss_function_times=times
)

# %% Compute explanations
approximator = shapiq.KernelSHAPIQ(n=game_cox.n_players, max_order=2)
explanation = approximator.approximate(budget=2**7, game=game_cox)
print(explanation)

# %%
