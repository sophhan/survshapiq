import numpy as np
import pandas as pd
import shapiq
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.metrics import integrated_brier_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, Memory
memory = Memory("/home/slangbei/joblib_temp", verbose=0)
from tqdm import tqdm
from scipy.signal import savgol_filter
import math

def survshapiq(model, data_x, x_new_list, time_stride=3, budget=2**8, max_order=2, approximator=None, index=None, exact = True,  feature_names = None):
    """
    Explain interaction effects at different time points for a survival model.

    Parameters:
    - model: fitted survival model with .predict_survival_function() and .unique_times_
    - data_x: DataFrame of training covariates
    - x_new: DataFrame or array of new observations to explain (only the first row will be used)
    - time_stride: interval for selecting time points (every 3rd by default)
    - budget: computational budget for Shapiq explainer
    - max_order: maximum order of interactions (default is 2)
    - approximator: type of approximator to use (default is "auto")
    - index: type of index to use (default is "k-SII")
    - exact: whether to use exact mode (default is True)

    Returns:
    - explanation_df: a DataFrame with interaction values over selected time points
    """

    explanations_all = []

    for x_new in x_new_list:
        explanations = {}

        for t in model.unique_times_[::time_stride].tolist():
            which_timepoint_equals_t = np.where(model.unique_times_ == t)[0][0].item()

            model_at_time_t = lambda d: model.predict_survival_function(d, return_array=True)[:, which_timepoint_equals_t]

            if index is not None and approximator is not None:
                explainer = shapiq.TabularExplainer(model=model_at_time_t,
                                                    data=data_x,
                                                    max_order=max_order,
                                                    approximator=approximator,
                                                    index=index)
            elif exact:
                explainer = shapiq.TabularExplainer(model=model_at_time_t,
                                                    data=data_x,
                                                    max_order=max_order,
                                                    exact=True)
            else:
                raise ValueError("Must either provide both 'index' and 'approximator', or set exact=True.")

            interaction_values = explainer.explain(x_new, budget=budget)
            explanations[t] = interaction_values

        # Aggregate
        explanation_dict = {}

        for features, _ in interaction_values.dict_values.items():
            if len(features) == 0:
                continue
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]] = []
            if len(features) == 2:
                new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[new_feature_name] = []

        for t, iv in explanations.items():
            for features, value in iv.dict_values.items():
                if len(features) == 0:
                    continue
                if len(features) == 1:
                    explanation_dict[feature_names[features[0]]].append(value)
                if len(features) == 2:
                    new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                    explanation_dict[new_feature_name].append(value)

        explanation_df = pd.DataFrame(explanation_dict)
        explanations_all.append(explanation_df)

    return explanations_all

def survshapiq_pycox(model, data_x, x_new_list, time_stride=3, budget=2**8, max_order=2, feature_names = None):
    """
    Explain interaction effects at different time points for a survival model.

    Parameters:
    - model: fitted survival model with .predict_survival_function() and .unique_times_
    - data_x: DataFrame of training covariates
    - x_new: DataFrame or array of new observations to explain (only the first row will be used)
    - time_stride: interval for selecting time points (every 3rd by default)
    - budget: computational budget for Shapiq explainer
    - max_order: maximum order of interactions (default is 2)

    Returns:
    - explanation_df: a DataFrame with interaction values over selected time points
    """

    explanations_all = []
    data_x = data_x.astype('float32')
    surv = model.predict_surv_df(data_x)

    for x_new in x_new_list:
        explanations = {}

        for t in surv.index.values[::time_stride].tolist():
            which_timepoint_equals_t = np.where(surv.index.values == t)[0][0].item()

            model_at_time_t = lambda d: model.predict_surv_df(d.astype('float32')).iloc[which_timepoint_equals_t, :].values

            explainer = shapiq.TabularExplainer(
                model=model_at_time_t,
                data=data_x,
                max_order=2
            )

            interaction_values = explainer.explain(x_new.astype('float32'), budget=budget)
            explanations[t] = interaction_values

        # Aggregate
        explanation_dict = {}

        for features, _ in interaction_values.dict_values.items():
            if len(features) == 0:
                continue
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]] = []
            if len(features) == 2:
                new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[new_feature_name] = []

        for t, iv in explanations.items():
            for features, value in iv.dict_values.items():
                if len(features) == 0:
                    continue
                if len(features) == 1:
                    explanation_dict[feature_names[features[0]]].append(value)
                if len(features) == 2:
                    new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                    explanation_dict[new_feature_name].append(value)

        explanation_df = pd.DataFrame(explanation_dict)
        explanations_all.append(explanation_df)

    return explanations_all


def prepare_survival_data(df, event_col='eventtime', status_col='status', id_col='id'):
    """
    Prepares survival analysis data: structured array for survival outcome and covariate DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing survival data
    - event_col: name of the event time column
    - status_col: name of the event status column
    - id_col: name of the ID column (to drop from covariates), can be None or absent
    
    Returns:
    - data_y: structured numpy array with fields 'status' (bool) and 'eventtime' (float)
    - data_x: pandas DataFrame containing covariates only
    """

    # Step 1: Create structured array for survival outcome
    data_y = np.array(
        list(zip(df[status_col].astype(bool), df[event_col])),
        dtype=[('status', '?'), ('eventtime', 'f8')]
    )
    
    # Step 2: Drop outcome columns and optionally the ID column
    drop_cols = [status_col, event_col]
    if id_col is not None and id_col in df.columns:
        drop_cols.append(id_col)
    
    data_x = df.drop(columns=drop_cols)
    
    return data_y, data_x


def smooth_series(y, window=7, poly=3):
    """Savitzky–Golay smoothing."""
    if window >= len(y):
        return y
    if window % 2 == 0:  # must be odd
        window += 1
    return savgol_filter(y, window, poly)

def plot_interact(
    explanations_all,
    model=None,
    times=None,
    x_new=None,
    time_stride=1,
    save_path=None,
    compare_plots=True,
    survival_fn=None,
    data_x=None,
    idx_plot=None,
    smooth=False,
    smooth_window=11,
    smooth_poly=3
):
    # --- Determine time grid ---
    if model is not None:
        all_times = model.unique_times_
    elif times is not None:
        all_times = np.asarray(times)
    else:
        raise ValueError("Either 'model' or 'times' must be provided.")

    n_obs = len(explanations_all)
    n_cols = 1 if n_obs == 1 else 2
    n_rows = int(np.ceil(n_obs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows), squeeze=False)

    for idx, explanation_df in enumerate(explanations_all):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # --- Plot SHAP interaction curves ---
        for feature_name, feature_values in explanation_df.items():
            y_vals = np.array(feature_values)
            if smooth and len(y_vals) >= smooth_window:
                y_vals = savgol_filter(y_vals, smooth_window, smooth_poly)

            # Align y_vals length with time grid
            if len(y_vals) == len(all_times):
                plot_times = all_times
                plot_vals = y_vals
            else:
                plot_times = all_times[::time_stride]
                plot_vals = y_vals

            n = min(len(plot_times), len(plot_vals))
            ax.step(plot_times[:n], plot_vals[:n], where="post", label=f"{feature_name}", alpha=0.7)

        # --- Plot survival curves ---
        if compare_plots and data_x is not None:
            if model is not None:
                surv_funcs = model.predict_survival_function(data_x)
                surv_matrix = np.vstack([sf(all_times) for sf in surv_funcs])
            elif survival_fn is not None:
                surv_matrix = survival_fn(data_x, all_times)
            else:
                raise ValueError("Need either 'model' or 'survival_fn' when data_x is provided.")

            surv_times = all_times[::time_stride]
            surv_matrix = surv_matrix[:, ::time_stride]

            # Mean survival
            mean_surv = np.mean(surv_matrix, axis=0)
            ax.step(surv_times, mean_surv, where="post", color="blue", lw=2, label="Mean f(t|x)")

            # Individual survival
            if idx_plot is not None and 0 <= idx_plot < len(data_x):
                indiv_surv = surv_matrix[idx_plot]
                ax.step(surv_times, indiv_surv, where="post", color="red", lw=2, label=f"Indiv {idx_plot}")

                diff_curve = indiv_surv - mean_surv
                ax.step(surv_times, diff_curve, where="post", color="green", lw=2, linestyle="--",
                        label="Diff f(t|x) - mean f(t|x)")

        # --- Styling ---
        ax.set_ylabel("Value")
        ax.set_xlabel("Time $t$")
        if x_new is not None:
            sample_str = ", ".join([str(val) for val in x_new[idx]])
            ax.set_title(f"Sample {idx}: [{sample_str}]")
        else:
            ax.set_title(f"Sample {idx}")
        ax.legend(loc="best")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        print(f"Combined plot saved to: {save_path}")

    plt.show()
    
def plot_interact_pycox2(
    explanations_all,
    model,
    data_x,
    x_new=None,
    time_stride=3,
    save_path=None,
    include_features="all", 
):
    """
    Plot interaction explanations over time on a single plot.

    Parameters:
    - explanations_all: list of DataFrames (from explain_interactions_over_time)
    - model: the survival model (must have model.unique_times_)
    - data_x: input data
    - x_new: optional, the samples for which explanations were generated
    - time_stride: same time stride used in explanation
    - save_path: where to save the plot (optional)
    - exclude_features: list of feature names to exclude from plotting
    """

    if include_features == "all":
        include_features = explanations_all[0].columns

    data_x = data_x.astype('float32')
    surv = model.predict_surv_df(data_x)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, explanation_df in enumerate(explanations_all):
        for feature_name, feature_values in explanation_df.items():
            if feature_name not in include_features:
                continue
            label = f"{feature_name} (sample {idx})" if len(explanations_all) > 1 else feature_name
            ax.step(
                surv.index.values[::time_stride].tolist(),
                feature_values,
                where="post",
                label=label
            )

    ax.set_ylabel(r"Order 2 values for the est. probability of survival $\hat{S}(t)$")
    ax.set_xlabel("time $t$")
    ax.set_title("Interaction Explanations Over Time")
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()

    
    
def compute_integrated_brier(data_y, data_x, model, min_time = 0.05, max_time = 200):
    """_summary_

    Args:
        data_y (_type_): _description_
        data_x (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    #max_time = np.max([y[1] for y in data_y])
    #times = model.unique_times_[model.unique_times_ < max_time]
    #test_times = np.array([y[1] for y in data_y])  # Extract event/censoring times
    #min_time = np.percentile(test_times, 5)
    #max_time = np.percentile(test_times, 95)
    times = np.linspace(min_time, max_time, 100)
    surv_funcs = model.predict_survival_function(data_x)
    pred_surv = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
    ibs = integrated_brier_score(data_y, data_y, pred_surv, times)
    
    return ibs

def plot_survival_function(model, x_new, time_stride=3):
    """
    Plot survival function for new data points.

    Parameters:
    - model: fitted survival model
    - x_new: array of new observations to explain
    - time_stride: interval for selecting time points (every 3rd by default)
    """

    pred_surv = model.predict_survival_function(x_new, return_array=True)

    plt.figure(figsize=(4, 3))
    for i in range(pred_surv.shape[0]):
        label = f"{x_new[i]}"
        plt.step(model.unique_times_[::time_stride], pred_surv[i, ::time_stride], where="post", label=label)

    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    plt.title("Survival Functions for x_new Samples")
    plt.tight_layout()
    plt.show()
    
    
def plot_multiple_survival_curves(models, model_names, x_new, sample_idx=0, colors=None):
    """
    Plot survival curves from multiple models for a given sample.

    Parameters:
    - models: list of fitted survival models (each must have .predict_survival_function and .unique_times_)
    - model_names: list of names corresponding to the models (same order as `models`)
    - x_new: input data (e.g., from x_train or x_test), should be 2D (n_samples x n_features)
    - sample_idx: index of the sample in x_new to plot (default: 0)
    - colors: optional list of colors for the plots
    """

    if colors is None:
        colors = ["blue", "green", "red", "orange", "purple"]  # Extend if needed

    plt.figure(figsize=(8, 6))

    for i, model in enumerate(models):
        # Get survival times
        times = model.unique_times_

        # Predict survival for the specified sample
        surv = model.predict_survival_function(x_new, return_array=True)[sample_idx]

        # Plot
        plt.step(times, surv, where="post", label=model_names[i], color=colors[i % len(colors)])

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title(f"Survival Curves for Sample {sample_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_samplewise_mean_abs_attributions(explanation_df, model_name="Model", save_path=None):
    # Exclude metadata columns
    non_feature_cols = ['sample_idx', 'time_idx']
    feature_cols = [col for col in explanation_df.columns if col not in non_feature_cols]

    # Compute mean absolute attributions per feature per sample
    agg_df = (
        explanation_df
        .copy()
        .groupby(['sample_idx'])[feature_cols]
        .apply(lambda df: df.abs().mean())
        .reset_index()
        .melt(id_vars='sample_idx', var_name='feature', value_name='mean_abs_attribution')
    )

    # Plot as scatter points
    plt.figure(figsize=(10, len(feature_cols) * 0.4))
    sns.scatterplot(
        data=agg_df,
        x='mean_abs_attribution',
        y='feature',
        hue='sample_idx',
        palette='tab10',
        s=60,
        legend='full'
    )

    plt.title(f"{model_name}: Sample-wise Mean Absolute Attributions per Feature")
    plt.xlabel("Mean Absolute Attribution")
    plt.ylabel("Feature")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory if not showing
    else:
        plt.show()
        
        
        
def explain_single_instance(model, data_x, x_new, time_indices, budget, max_order, feature_names):
    explanations = {}

    # Compute interaction values for each time index
    for time_idx in time_indices:
        model_at_time = lambda d: model.predict_survival_function(d, return_array=True)[:, time_idx]
        explainer = shapiq.TabularExplainer(model=model_at_time, data=data_x, max_order=max_order)
        interaction_values = explainer.explain(x_new, budget=budget)
        explanations[model.unique_times_[time_idx]] = interaction_values

    # Initialize explanation dict with empty lists
    explanation_dict = {}

    sample_iv = next(iter(explanations.values()))  # Get one sample interaction value set

    for features in sample_iv.dict_values:
        if len(features) == 1:
            explanation_dict[feature_names[features[0]]] = []
        elif len(features) == 2:
            key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
            explanation_dict[key] = []

    # Fill in values
    for iv in explanations.values():
        for features, value in iv.dict_values.items():
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]].append(value)
            elif len(features) == 2:
                key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[key].append(value)

    return pd.DataFrame(explanation_dict)

# Cache the large shared data
@memory.cache
def load_shared_data(model, data_x):
    return model, data_x

def survshapiq_parallel(model, data_x, x_new_list, time_stride=3, budget=2**8, max_order=2, feature_names=None, n_jobs=-1, show_progress=True):
    time_indices = np.arange(0, len(model.unique_times_), time_stride)

    # Cache the shared inputs
    model_cached, data_x_cached = load_shared_data(model, data_x)

    # Optionally convert data_x to numpy (if not already)
    if isinstance(data_x_cached, pd.DataFrame):
        data_x_cached = data_x_cached.to_numpy()

    # Prepare iterator
    iterator = tqdm(x_new_list) if show_progress else x_new_list

    # Define wrapped function for delayed execution
    def wrapped_func(x_new):
        return explain_single_instance(
            model_cached,
            data_x_cached,
            x_new,
            time_indices,
            budget,
            max_order,
            feature_names
        )

    # Run in parallel
    explanations_all = Parallel(n_jobs=n_jobs, temp_folder="/home/slangbei/joblib_temp", max_nbytes='50M')(
        delayed(wrapped_func)(x_new) for x_new in iterator
    )

    return explanations_all


# Annotate and flatten
import pandas as pd
import numpy as np

def annotate_explanations(explanations, model=None, sample_idxs=None, time_stride=1, data_x_train=None):
    """
    Annotate and flatten a list of explanation DataFrames (from survshapiq or ground-truth simulations).
    
    Parameters
    ----------
    explanations : list of pd.DataFrame
        List of per-sample explanation DataFrames (rows = time points, columns = features or interactions).
    model : optional
        Survival model object with `unique_times_` attribute (optional if explanations are ground-truth).
    sample_idxs : list or iterable, optional
        Indices of samples corresponding to explanations. Defaults to range(len(explanations)).
    time_stride : int, optional
        Stride used when generating explanations. Default is 1.
    data_x_train : pd.DataFrame, optional
        Training data (required if model does not have `unique_times_`).
    
    Returns
    -------
    pd.DataFrame
        Flattened DataFrame with columns: features, sample_idx, time.
    """
    
    all_rows = []

    if sample_idxs is None:
        sample_idxs = range(len(explanations))
    
    # Determine time points once if possible
    if model is not None and hasattr(model, 'unique_times_'):
        time_points = model.unique_times_[::time_stride]
    
    for i, df in enumerate(explanations):
        df = df.copy()
        df["sample_idx"] = sample_idxs[i]

        # Assign time column
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            df["time"] = idx.values
        elif np.issubdtype(idx.dtype, np.number):
            df["time"] = idx.values
        else:
            # fallback
            if model is not None and hasattr(model, "unique_times_"):
                df["time"] = time_points
            elif data_x_train is not None:
                surv_df = model.predict_surv_df(data_x_train)
                surv_times = surv_df.index.values
                time_indices = np.arange(0, len(surv_times), time_stride)
                df["time"] = surv_times[time_indices]
            else:
                raise ValueError("Cannot determine time values. Provide model with unique_times_ or data_x_train.")

        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def explain_single_instance_pycox(model, data_x, x_new, time_indices, surv_times, budget, max_order, feature_names):
    explanations = {}

    for time_idx in time_indices:
        t = surv_times[time_idx]

        model_at_time = lambda d: model.predict_surv_df(d.astype('float32')).iloc[time_idx, :].values

        explainer = shapiq.TabularExplainer(
            model=model_at_time,
            data=data_x,
            max_order=max_order
        )

        interaction_values = explainer.explain(x_new.astype('float32'), budget=budget)
        explanations[t] = interaction_values

    explanation_dict = {}

    sample_iv = next(iter(explanations.values()))
    for features in sample_iv.dict_values:
        if len(features) == 1:
            explanation_dict[feature_names[features[0]]] = []
        elif len(features) == 2:
            key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
            explanation_dict[key] = []

    for iv in explanations.values():
        for features, value in iv.dict_values.items():
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]].append(value)
            elif len(features) == 2:
                key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[key].append(value)

    return pd.DataFrame(explanation_dict)


def survshapiq_pycox_parallel(model, data_x, x_new_list, time_stride=3, budget=2**8, max_order=2, feature_names=None, n_jobs=-1, show_progress=True):
    data_x = data_x.astype('float32')
    surv_df = model.predict_surv_df(data_x)
    surv_times = surv_df.index.values
    time_indices = np.arange(0, len(surv_times), time_stride)

    iterator = tqdm(x_new_list) if show_progress else x_new_list

    explanations_all = Parallel(n_jobs=n_jobs, temp_folder="/home/slangbei/joblib_temp", max_nbytes='50M')(
        delayed(explain_single_instance_pycox)(
            model, data_x, x_new, time_indices, surv_times, budget, max_order, feature_names
        ) for x_new in iterator
    )

    return explanations_all


# Compute survival function from ground truth hazard (any hazard function)
def survival_from_hazard(df, hazard_func, t_values):
    """
    df : np.ndarray of shape (n_individuals, 3)
        Columns: [age, treatment, bmi]
    hazard_func(t, age, bmi, treatment) -> np.ndarray over t
    """
    df = np.asarray(df)
    n_people = df.shape[0]
    t_values = np.asarray(t_values)
    dt = np.diff(t_values, prepend=0)

    age = df[:, 0]
    treatment = df[:, 1]
    bmi = df[:, 2]

    survival_matrix = np.zeros((n_people, len(t_values)))

    for i in range(n_people):
        h_vals = hazard_func(t_values, age[i], bmi[i], treatment[i])
        H = np.cumsum(h_vals * dt)
        survival_matrix[i, :] = np.exp(-H)

    return survival_matrix

# Compute hazard matrix
def hazard_matrix(df, hazard_func, t_values):
    """
    df : np.ndarray of shape (n_individuals, 3)
        Columns: [age, treatment, bmi]
    hazard_func(t, age, bmi, treatment) -> np.ndarray over t
    """
    df = np.asarray(df)
    n_people = df.shape[0]
    t_values = np.asarray(t_values)
    dt = np.diff(t_values, prepend=0)

    age = df[:, 0]
    treatment = df[:, 1]
    bmi = df[:, 2]

    hazard_matrix = np.zeros((n_people, len(t_values)))

    for i in range(n_people):
        hazard_matrix[i, :] = hazard_func(t_values, age[i], bmi[i], treatment[i])

    return hazard_matrix


# compute survshapiq from ground truth survival values 
def survshapiq_ground_truth(data_x, x_new_list, survival_from_hazard_func, times, time_stride=1, 
                            budget=2**8, max_order=2, approximator=None, index=None, exact = True, feature_names=None):
    explanations_all = []

    # Compute survival for all individuals in x_new_list
    #S_matrix = survival_from_hazard_func(x_new_list, times)

    for idx, x_new in enumerate(x_new_list):
        explanations = {}
        for i in range(0, len(times), time_stride):
            # Returns survival for *all* individuals at time index i
            survival_at_time_t = lambda d: survival_from_hazard_func(d, times)[:, i]

            if index is not None and approximator is not None:
                explainer = shapiq.TabularExplainer(model=survival_at_time_t,
                                                    data=data_x,
                                                    max_order=max_order,
                                                    approximator=approximator,
                                                    index=index)
            elif exact:
                explainer = shapiq.TabularExplainer(model=survival_at_time_t,
                                                    data=data_x,
                                                    max_order=max_order,
                                                    exact=True)
            else:
                raise ValueError("Must either provide both 'index' and 'approximator', or set exact=True.")


            interaction_values = explainer.explain(x_new, budget=budget)
            explanations[i] = interaction_values

        # Initialize feature names in dict
        explanation_dict = {}
        for features, _ in interaction_values.dict_values.items():
            if not features:
                continue
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]] = []
            elif len(features) == 2:
                new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[new_feature_name] = []

        # Fill values
        for t, iv in explanations.items():
            for features, value in iv.dict_values.items():
                if not features:
                    continue
                if len(features) == 1:
                    explanation_dict[feature_names[features[0]]].append(value)
                elif len(features) == 2:
                    new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                    explanation_dict[new_feature_name].append(value)

        explanation_df = pd.DataFrame(explanation_dict, index=[times[t] for t in explanations.keys()])
        explanations_all.append(explanation_df)

    return explanations_all

# Compute survival matrix from time-independent hazard function (exact)
def survival_matrix_ti(X, times):
    """
    Non-vectorized: Loop over individuals to compute survival matrix.
    
    Parameters
    ----------
    X : array-like, shape (n_individuals, 3)
        Columns: [age, treatment, bmi]  (adjust if your order differs)
    times : array-like
        Time points at which to compute survival.
    
    Returns
    -------
    np.ndarray
        Survival matrix of shape (n_individuals, len(times)).
    """
    X = np.asarray(X)
    times = np.asarray(times)
    n_people = X.shape[0]
    n_times = len(times)

    survival_matrix = np.zeros((n_people, n_times))

    for i in range(n_people):
        age = X[i, 0]
        treatment = X[i, 1]
        bmi = X[i, 2]

        # constant hazard for this person
        h = 0.01 * np.exp(0.3*age + 0.9*bmi - 0.7*treatment - 5*treatment*age)

        for j, t in enumerate(times):
            survival_matrix[i, j] = np.exp(-h * t)

    return survival_matrix


## GROUND THRUTH PARALLELIZATION
# Optional: cache large shared data
memory = Memory("/home/slangbei/joblib_temp", verbose=0)

@memory.cache
def load_shared_data_ground_truth(data_x, survival_from_hazard_func, times):
    return data_x, survival_from_hazard_func, times

def explain_single_instance_ground_truth(x_new, data_x, survival_from_hazard_func, times, time_indices, budget, max_order, feature_names, index=None, approximator=None, exact=True):
    explanations = {}

    # Compute survival contributions for each time index
    for i in time_indices:
        survival_at_time_t = lambda d: survival_from_hazard_func(d, times)[:, i]

        if index is not None and approximator is not None:
            explainer = shapiq.TabularExplainer(
                model=survival_at_time_t,
                data=data_x,
                max_order=max_order,
                approximator=approximator,
                index=index
            )
        elif exact:
            explainer = shapiq.TabularExplainer(
                model=survival_at_time_t,
                data=data_x,
                max_order=max_order,
                exact=True
            )
        else:
            raise ValueError("Must provide both 'index' and 'approximator', or set exact=True.")

        interaction_values = explainer.explain(x_new, budget=budget)
        explanations[i] = interaction_values

    # Initialize feature names in dict
    explanation_dict = {}
    sample_iv = next(iter(explanations.values()))
    for features, _ in sample_iv.dict_values.items():
        if not features:
            continue
        if len(features) == 1:
            explanation_dict[feature_names[features[0]]] = []
        elif len(features) == 2:
            new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
            explanation_dict[new_feature_name] = []

    # Fill values
    for t, iv in explanations.items():
        for features, value in iv.dict_values.items():
            if not features:
                continue
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]].append(value)
            elif len(features) == 2:
                new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[new_feature_name].append(value)

    return pd.DataFrame(explanation_dict, index=[times[t] for t in explanations.keys()])


def survshapiq_ground_truth_parallel(
    data_x,
    x_new_list,
    survival_from_hazard_func,
    times,
    time_stride=1,
    budget=2**8,
    max_order=2,
    approximator=None,
    index=None,
    exact=True,
    feature_names=None,
    n_jobs=-1,
    show_progress=True
):
    time_indices = np.arange(0, len(times), time_stride)

    # Cache large shared inputs
    data_x_cached, survival_func_cached, times_cached = load_shared_data_ground_truth(
        data_x, survival_from_hazard_func, times
    )

    # Optionally convert to numpy
    if isinstance(data_x_cached, pd.DataFrame):
        data_x_cached = data_x_cached.to_numpy()

    # Prepare iterator with progress bar
    iterator = tqdm(x_new_list) if show_progress else x_new_list

    # Wrapper function
    def wrapped_func(x_new):
        return explain_single_instance_ground_truth(
            x_new,
            data_x_cached,
            survival_func_cached,
            times_cached,
            time_indices,
            budget,
            max_order,
            feature_names,
            index=index,
            approximator=approximator,
            exact=exact
        )

    # Run in parallel
    explanations_all = Parallel(
        n_jobs=n_jobs,
        temp_folder="/home/slangbei/joblib_temp",
        max_nbytes='50M'
    )(delayed(wrapped_func)(x_new) for x_new in iterator)

    return explanations_all

