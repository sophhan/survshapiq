# %% Imports
import pickle
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest

import matplotlib.pyplot as plt
import seaborn as sns

import src

SEED = 1234
np.random.seed(SEED)


# %% Data
from SurvSet.data import SurvLoader
loader = SurvLoader()
enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
sel_num = make_column_selector(pattern='^num\\_')
enc_df = ColumnTransformer(transformers=[('s', enc_num, sel_num)])
enc_df.set_output(transform='pandas')

ds_name = "Bergamaschi"
df = loader.load_dataset(ds_name=ds_name)['df'].set_index("pid")
senc = Surv()
So = senc.from_arrays(df['event'].astype(bool), df['time'])
enc_df.fit(df)
X_train = enc_df.transform(df)
X_train = X_train.loc[:, X_train.columns.str.startswith("s__num_")]
X_train.columns = X_train.columns.str.replace("s__num_", "")

# %%
model = RandomSurvivalForest(max_depth=6, n_estimators=200, oob_score=True, random_state=SEED)
model.fit(X=X_train.values, y=So)
print(model.oob_score_, model.score(X_train.values, So))

# %%
np.random.seed(SEED)
ground_truth = src.survshapiq(
    model, 
    X_train.values, 
    [X_train.iloc[[i]] for i in range(0, X_train.shape[0])],
    feature_names=X_train.columns,
    n_timepoints=21,
    exact=True, 
    index="k-SII"
)

# %%
filename = f'results/{ds_name}_approximators_gt.pkl'
with open(filename, 'wb') as f:
    pickle.dump(ground_truth, f)


# %%
def compute_error(exp, gt):
    return np.sum([(exp[i] - gt[i]).abs().sum().sum() for i in range(len(exp))]).item()

# %%
result = pd.DataFrame({'approximator': [], 'budget': [], 'error': []})

for approximator in ["montecarlo", "svarm", "permutation", "regression"]:
    for budget in [2**5, 2**6, 2**7, 2**8, 2**9]:
        np.random.seed(SEED)
        explanations = src.survshapiq(
            model, 
            X_train.values, 
            [X_train.iloc[[i]] for i in range(0, X_train.shape[0])], 
            feature_names=X_train.columns,
            n_timepoints=21,
            exact=False, 
            budget=budget,
            index="k-SII",
            approximator=approximator
        )

        error = compute_error(explanations, ground_truth)

        result = pd.concat([result, pd.DataFrame({
            'approximator': [approximator], 
            'budget': [budget], 
            'error': [error]
        })])

#%%
result.to_csv(f'results/{ds_name}_approximators.csv', index=False)

# %%
ax = sns.lineplot(result, x="budget", y="error", hue="approximator")
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
plt.grid(True, which="both", ls="--")
plt.title(f'dataset = {ds_name} | benchmark of approximators (k-SII)')
plt.tight_layout()
plt.savefig(f'results/{ds_name}_approximators.png', bbox_inches="tight")

# %%
plt.clf()