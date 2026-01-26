import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--id', default=0, type=int, help='observation id')
parser.add_argument('--output', default="/mnt/evafs/groups/mi2lab/hbaniecki/survshapiq", type=str, help='output path')
args = parser.parse_args()

import os
import pickle
import numpy as np

from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest
from SurvSet.data import SurvLoader

import src

SEED = 1234
np.random.seed(SEED)

### DATA
loader = SurvLoader()
enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
sel_num = make_column_selector(pattern='^num\\_')
enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'))])
sel_fac = make_column_selector(pattern='^fac\\_')
enc_df = ColumnTransformer(transformers=[('num', enc_num, sel_num), ('ohe', enc_fac, sel_fac)])
enc_df.set_output(transform='pandas')
df = loader.load_dataset(ds_name="nki70")['df'].set_index("pid")
senc = Surv()
So = senc.from_arrays(df['event'].astype(bool), df['time'])
enc_df.fit(df)
X_train = enc_df.transform(df)
X_train.columns = X_train.columns.str.replace("num__num_", "")
X_train.columns = X_train.columns.str.replace("ohe__fac_", "")

### MODEL
model = RandomSurvivalForest(max_depth=5, n_estimators=300, oob_score=True, random_state=SEED)
model.fit(X=X_train.values, y=So)
print(f'train score: {model.score(X_train.values, So):.04f} | OOB score: {model.oob_score_:.04f}')

### EXPLANATION
np.random.seed(SEED)
explanation_order1 = src.survshapiq(
    model, 
    X_train.values, 
    [X_train.iloc[[args.id]]], 
    feature_names=X_train.columns,
    n_timepoints=31,
    exact=False, 
    budget=2**15,
    index="SV",
    approximator="regression",
    max_order=1
)

filename = os.path.join(args.output, f'order1_id{args.id}.pkl')
with open(filename, 'wb') as f:
    pickle.dump(explanation_order1[0], f)

# np.random.seed(SEED)
# explanation_order2 = src.survshapiq(
#     model, 
#     X_train.values, 
#     [X_train.iloc[[args.id]]], 
#     feature_names=X_train.columns,
#     n_timepoints=31,
#     exact=False, 
#     budget=2**15,
#     index="k-SII",
#     approximator="regression",
#     max_order=2
# )

# filename = os.path.join(args.output, f'order2_id{args.id}.pkl')
# with open(filename, 'wb') as f:
#     pickle.dump(explanation_order2[0], f)