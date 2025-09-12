# %% Imports
import pickle
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sksurv.metrics import integrated_brier_score
from sklearn.compose import make_column_selector, ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

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

# grace (3), actg (4), vlbw (7), Pbc3 (7), whas500 (7), Bergamaschi (10)
# TODO: support2 (24)
# no/too little num: hdfail, zinc, pbc, TRACE (2)
for ds_name in ['grace', 'actg', 'vlbw', 'Pbc3', 'whas500', 'Bergamaschi', 'support2']:
    df = loader.load_dataset(ds_name=ds_name)['df'].set_index("pid")
    senc = Surv()
    So = senc.from_arrays(df['event'].astype(bool), df['time'])
    enc_df.fit(df)
    X_train = enc_df.transform(df)
    X_train = X_train.loc[:, X_train.columns.str.startswith("s__num_")]
    X_train.columns = X_train.columns.str.replace("s__num_", "")

    ## %%
    model = RandomSurvivalForest(max_depth=6, n_estimators=200, oob_score=True, random_state=SEED)
    model.fit(X=X_train.values, y=So)
    print(model.oob_score_, model.score(X_train.values, So))

    ## %%
    np.random.seed(SEED)
    n_samples = 100
    if X_train.shape[0] > n_samples:
        X_explain = X_train.sample(n_samples)
    else:
        n_samples = X_train.shape[0]
        X_explain = X_train
    explanations_rsf = src.survshapiq(
        model, 
        X_train.values, 
        [X_explain.iloc[[i]] for i in range(0, X_explain.shape[0])], 
        feature_names=X_train.columns,
        n_timepoints=21,
        exact=False, 
        budget=2**X_train.shape[1] if X_train.shape[1] <= 9 else 2**9,
        index="k-SII",
        approximator="auto"
    )

    filename = f'results/{ds_name}_explanations.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(explanations_rsf, f)

    ## %%
    e = explanations_rsf[0]
    for i in range(1, n_samples):
        e = e.add(explanations_rsf[i])
    topk = 6
    topk_names = e.sum(axis=0).sort_values(ascending=False)[:topk].index.append(e.sum(axis=0).sort_values(ascending=True)[:topk].index[::-1]).unique()
    e_topk = e.loc[:, topk_names]
    ax = sns.lineplot(e_topk, palette=sns.color_palette("tab10") + ['black', 'gray'])
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(f'dataset = {ds_name} | sum for top/bottom 6 terms out of {e.shape[1]}')
    plt.tight_layout()
    plt.savefig(f'results/{ds_name}_mean.png', bbox_inches="tight")
    plt.clf()

    ## %%
    e = explanations_rsf[0].abs()
    for i in range(1, n_samples):
        e = e.add(explanations_rsf[i].abs())
    topk = 12
    topk_names = e.sum(axis=0).sort_values(ascending=False)[:topk].index
    e_topk = e.loc[:, topk_names]
    ax = sns.lineplot(e_topk, palette=sns.color_palette("tab10") + ['black', 'gray'])
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(f'dataset = {ds_name} | sum absolute for top 12 terms out of {e.shape[1]}')
    plt.tight_layout()
    plt.savefig(f'results/{ds_name}_abs_mean.png', bbox_inches="tight")
    plt.clf()
# %%
