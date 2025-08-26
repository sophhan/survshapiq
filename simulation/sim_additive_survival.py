import numpy as np
import pandas as pd
from scipy.optimize import brentq


#### TIME-INDEPENDENT LOGIT ADDITIVE SURVIVAL FUNCTION
# --- user settings ---
# coefficients
beta_age = 0.4
beta_trt = -0.5
beta_bmi = 0.7
beta_int = -0.1   # now constant interaction effect (not time-dependent)
baseline = 5.0

n = 5000
np.random.seed(42)

# generate covariates
age = np.random.normal(0, 1, n)
treatment = np.random.normal(0, 1, n)
bmi = np.random.normal(0, 1, n)

# precompute covariate effects excluding interaction with time
c = baseline + beta_age * age + beta_trt * treatment + beta_bmi * bmi

# sample uniform
U = np.random.uniform(0, 1, n)

def survival(t, c, age, trt):
    # NO time-dependent term: just a constant interaction
    eta = c + beta_int * trt * age   # <- no "* t"
    return expit(eta) ** t           # ensure monotone decreasing survival

times = np.zeros(n)
maxt = 100
for i in range(n):
    try:
        times[i] = brentq(lambda tt: survival(tt, c[i], age[i], treatment[i]) - U[i], 0, maxt)
    except ValueError:
        times[i] = maxt

# apply administrative censoring
observed_time = np.minimum(times, maxt)
event = (times < maxt).astype(int)

# put into DataFrame
df = pd.DataFrame({
    "id": np.arange(n),
    "status": event,
    "eventtime": observed_time,
    "age": age,
    "treatment": treatment,
    "bmi": bmi
})

print(df.head())
print(df["status"].mean(), "proportion events")
print(df["eventtime"].describe())

csv_path = "/home/slangbei/survshapiq/survshapiq/simulation/simdata_surv_ti.csv"
df.to_csv(csv_path, index=False)




#### TIME-DEPENDENT LOGIT ADDITIVE SURVIVAL FUNCTION
# coefficients
beta_age = 0.4
beta_trt = -0.5
beta_bmi = 0.7
beta_int = -0.1   # much stronger negative slope -1.5
baseline = 5.0     # ensures S(0|x) ~ 1 at start

n = 5000
np.random.seed(42)

# generate covariates
age = np.random.normal(0, 1, n)
treatment = np.random.normal(0, 1, n)
bmi = np.random.normal(0, 1, n)

# precompute covariate effects excluding interaction with time
c = baseline + beta_age * age + beta_trt * treatment + beta_bmi * bmi

# sample uniform
U = np.random.uniform(0, 1, n)

def survival(t, c, age, trt):
    eta = c + beta_int * trt * age * t
    return 1 / (1 + np.exp(-eta))

times = np.zeros(n)
maxt = 100
for i in range(n):
    # root find S(t|x) - U = 0 in [0,maxt]
    try:
        times[i] = brentq(lambda tt: survival(tt, c[i], age[i], treatment[i]) - U[i], 0, maxt)
    except ValueError:
        # if no root found in [0,maxt], censor at maxt
        times[i] = maxt

# apply administrative censoring
observed_time = np.minimum(times, maxt)
event = (times < maxt).astype(int)

# put into DataFrame
df = pd.DataFrame({
    "id": np.arange(n),
    "status": event,
    "eventtime": observed_time,
    "age": age,
    "treatment": treatment,
    "bmi": bmi
})

print(df.head())
print(df["status"].mean(), "proportion events")
df["eventtime"].describe()

csv_path = "/home/slangbei/survshapiq/survshapiq/simulation/simdata_surv_td.csv"
df.to_csv(csv_path, index=False)

