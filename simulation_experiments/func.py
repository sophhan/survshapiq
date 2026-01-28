import numpy as np
import pandas as pd
import shapiq
import matplotlib.pyplot as plt
from sksurv.metrics import integrated_brier_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from matplotlib.ticker import ScalarFormatter

def explain_single_instance(model, data_x, x_new, time_indices, budget, max_order, feature_names):
    """
    Compute Shapley interaction explanations for a single instance across multiple time points.

    Parameters
    ----------
    model : object
        Fitted survival model implementing `predict_survival_function(X, return_array=True)` 
        and providing `unique_times_`.
    data_x : pd.DataFrame or np.ndarray
        Background dataset for Shapley computation (used as the reference distribution).
    x_new : array-like
        Single instance to explain (shape must match data_x columns/features).
    time_indices : array-like of int
        Indices of time points (in model.unique_times_) at which to compute explanations.
    budget : int
        Computational budget for Shapley explainer (higher = more accurate).
    max_order : int
        Maximum order of interactions (1 = main effects, 2 = pairwise).
    feature_names : list of str
        Names of the features used for readable output.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per time point and columns for features and interactions.
    """
    explanations = {}

    # Compute interaction values for each time index
    for time_idx in time_indices:
        model_at_time = lambda d: model.predict_survival_function(d, return_array=True)[:, time_idx]
        explainer = shapiq.TabularExplainer(model=model_at_time, data=data_x, max_order=max_order, exact=True)
        interaction_values = explainer.explain(x_new, budget=budget)
        explanations[model.unique_times_[time_idx]] = interaction_values

    explanation_dict = {}
    sample_iv = next(iter(explanations.values()))  # Get one sample set of interaction values

    # Initialize columns for main effects and pairwise interactions
    for features in sample_iv.dict_values:
        if len(features) == 1:
            explanation_dict[feature_names[features[0]]] = []
        elif len(features) == 2:
            key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
            explanation_dict[key] = []

    # Fill in time-resolved values
    for iv in explanations.values():
        for features, value in iv.dict_values.items():
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]].append(value)
            elif len(features) == 2:
                key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[key].append(value)

    return pd.DataFrame(explanation_dict)


def survshapiq_parallel(model, data_x, x_new_list, time_stride=3, budget=2**8, max_order=2,
                        feature_names=None, n_jobs=-1, show_progress=True):
    """
    Compute Shapley interaction explanations for multiple instances in parallel.

    Parameters
    ----------
    model : object
        Fitted survival model with `predict_survival_function` and `unique_times_`.
    data_x : pd.DataFrame or np.ndarray
        Background dataset used for Shapley computation.
    x_new_list : list of array-like
        List of new instances to explain.
    time_stride : int, optional
        Step size for selecting time points (default: every 3rd time point).
    budget : int, optional
        Computational budget for the Shapley explainer.
    max_order : int, optional
        Maximum order of interactions to compute (default = 2).
    feature_names : list of str, optional
        List of feature names for readable output.
    n_jobs : int, optional
        Number of parallel jobs (default = -1 uses all cores).
    show_progress : bool, optional
        If True, shows a progress bar.

    Returns
    -------
    list of pd.DataFrame
        List of explanation DataFrames, one per instance.
    """
    time_indices = np.arange(0, len(model.unique_times_), time_stride)

    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()

    iterator = tqdm(x_new_list) if show_progress else x_new_list

    def wrapped_func(x_new):
        return explain_single_instance(
            model,
            data_x,
            x_new,
            time_indices,
            budget,
            max_order,
            feature_names
        )

    explanations_all = Parallel(n_jobs=n_jobs, max_nbytes='50M')(
        delayed(wrapped_func)(x_new) for x_new in iterator
    )

    return explanations_all


def annotate_explanations(explanations, model=None, sample_idxs=None, time_stride=1, data_x_train=None):
    """
    Annotate and flatten a list of explanation DataFrames with sample indices and time values.

    Parameters
    ----------
    explanations : list of pd.DataFrame
        List of per-sample explanation DataFrames.
    model : object, optional
        Survival model with `unique_times_`. Used to assign time points if needed.
    sample_idxs : list, optional
        Indices of samples. Defaults to range(len(explanations)).
    time_stride : int, optional
        Stride used to sample time points.
    data_x_train : pd.DataFrame, optional
        Training data, required if model does not provide `unique_times_`.

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame containing feature attributions, sample indices, and times.
    """
    all_rows = []

    if sample_idxs is None:
        sample_idxs = range(len(explanations))

    if model is not None and hasattr(model, 'unique_times_'):
        time_points = model.unique_times_[::time_stride]

    for i, df in enumerate(explanations):
        df = df.copy()
        df["sample_idx"] = sample_idxs[i]

        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            df["time"] = idx.values
        elif np.issubdtype(idx.dtype, np.number):
            df["time"] = idx.values
        else:
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


def survival_from_hazard(df, hazard_func, t_values):
    """
    Compute survival curves from a hazard function for multiple individuals.

    Parameters
    ----------
    df : np.ndarray of shape (n_individuals, n_features)
        Covariate matrix (rows = individuals).
    hazard_func : callable
        Function hazard_func(t, x1, ..., xk) → hazard value at each t.
    t_values : array-like
        Time points at which to evaluate hazards and survival.

    Returns
    -------
    np.ndarray of shape (n_individuals, len(t_values))
        Survival curves for each individual.
    """
    df = np.asarray(df)
    n_people, n_features = df.shape
    t_values = np.asarray(t_values)
    dt = np.diff(t_values, prepend=0)

    survival_matrix = np.zeros((n_people, len(t_values)))

    for i in range(n_people):
        h_vals = hazard_func(t_values, *df[i, :])
        H = np.cumsum(h_vals * dt)
        survival_matrix[i, :] = np.exp(-H)

    return survival_matrix


def survshapiq(model, data_x, x_new_list, time_stride=3, budget=2**8,
               max_order=2, approximator=None, index=None, exact=True, feature_names=None):
    """
    Compute Shapley interaction explanations for multiple instances over survival times.

    Parameters
    ----------
    model : object
        Fitted survival model with `predict_survival_function` and `unique_times_`.
    data_x : pd.DataFrame or np.ndarray
        Background dataset used for Shapley computation.
    x_new_list : list of array-like
        List of new instances to explain.
    time_stride : int, optional
        Interval for selecting time points (every k-th time point).
    budget : int, optional
        Computational budget for Shapley explainer.
    max_order : int, optional
        Maximum order of interactions to compute.
    approximator : str, optional
        Approximation method for interaction index.
    index : str, optional
        Interaction index (e.g., "k-SII").
    exact : bool, optional
        Whether to use exact computation (if False, approximator & index required).
    feature_names : list of str, optional
        Feature names for readable output.

    Returns
    -------
    list of pd.DataFrame
        List of explanation DataFrames, one per instance.
    """
    explanations_all = []

    for x_new in x_new_list:
        explanations = {}
        for t in model.unique_times_[::time_stride].tolist():
            which_timepoint_equals_t = np.where(model.unique_times_ == t)[0][0].item()
            model_at_time_t = lambda d: model.predict_survival_function(d, return_array=True)[:, which_timepoint_equals_t]

            if index is not None and approximator is not None:
                explainer = shapiq.TabularExplainer(model=model_at_time_t, data=data_x,
                                                    max_order=max_order, approximator=approximator, index=index)
            elif exact:
                explainer = shapiq.TabularExplainer(model=model_at_time_t, data=data_x,
                                                    max_order=max_order, exact=True)
            else:
                raise ValueError("Must either provide both 'index' and 'approximator', or set exact=True.")

            interaction_values = explainer.explain(x_new, budget=budget)
            explanations[t] = interaction_values

        explanation_dict = {}
        for features, _ in interaction_values.dict_values.items():
            if len(features) == 0:
                continue
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]] = []
            if len(features) == 2:
                key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[key] = []

        for t, iv in explanations.items():
            for features, value in iv.dict_values.items():
                if len(features) == 0:
                    continue
                if len(features) == 1:
                    explanation_dict[feature_names[features[0]]].append(value)
                if len(features) == 2:
                    key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                    explanation_dict[key].append(value)

        explanation_df = pd.DataFrame(explanation_dict)
        explanations_all.append(explanation_df)

    return explanations_all


def prepare_survival_data(df, event_col='eventtime', status_col='status', id_col='id'):
    """
    Prepare survival analysis data: structured array (y) and covariate DataFrame (X).

    Parameters
    ----------
    df : pd.DataFrame
        Data containing event time, status, and features.
    event_col : str
        Column containing event/censoring times.
    status_col : str
        Column containing event status (1=event, 0=censored).
    id_col : str, optional
        Column containing sample IDs (dropped from covariates if present).

    Returns
    -------
    data_y : np.ndarray
        Structured array with fields ('status', 'eventtime').
    data_x : pd.DataFrame
        DataFrame of covariates (features only).
    """
    data_y = np.array(
        list(zip(df[status_col].astype(bool), df[event_col])),
        dtype=[('status', '?'), ('eventtime', 'f8')]
    )

    drop_cols = [status_col, event_col]
    if id_col is not None and id_col in df.columns:
        drop_cols.append(id_col)

    data_x = df.drop(columns=drop_cols)

    return data_y, data_x


def smooth_series(y, window=7, poly=3):
    """
    Apply Savitzky–Golay smoothing to a time series.

    Parameters
    ----------
    y : array-like
        Input series.
    window : int, optional
        Window length (must be odd, adjusted automatically if even).
    poly : int, optional
        Polynomial order for smoothing.

    Returns
    -------
    np.ndarray
        Smoothed series.
    """
    if window >= len(y):
        return y
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window, poly)


# Color-blind friendly palette (Tableau 10 / Okabe-Ito)
CBF_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#F0E442", "#56B4E9", "#999999"
]


def plot_interact_ax(
    ax,
    explanations_all,
    times,
    model=None,
    data_x=None,
    survival_fn=None,
    idx_plot=None,
    compare_plots=False,
    time_stride=1,
    smooth=False,
    smooth_window=11,
    smooth_poly=3,
    ylabel="Attribution $S(t|x)$",
    label_fontsize=16,
    tick_fontsize=14,
    title=None,
    title_fontsize=18,
    add_to_legend=None
):
    """
    Plot feature attributions (and optionally survival curves) into a given Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to draw on.
    explanations_all : dict
        Mapping feature/interaction → array of values over time.
    times : array-like
        Time points for x-axis.
    model : object, optional
        Survival model with predict_survival_function (if plotting survival curves).
    data_x : pd.DataFrame, optional
        Dataset for computing mean/individual survival curves.
    survival_fn : callable, optional
        Custom survival function for computing curves.
    idx_plot : int, optional
        Index of individual to compare against population mean.
    compare_plots : bool or {"All", "Diff"}, optional
        Whether to plot survival curves: 
        - False = no curves
        - "All" = mean, individual, diff
        - "Diff" = difference only
    time_stride : int, optional
        Stride for survival curves.
    smooth : bool, optional
        Whether to apply smoothing.
    smooth_window : int, optional
        Window size for smoothing.
    smooth_poly : int, optional
        Polynomial order for smoothing.
    ylabel, label_fontsize, tick_fontsize, title, title_fontsize : various
        Plot styling options.

    Returns
    -------
    handles, labels : tuple
        Handles and labels for legends (can be used for global legends).
    """
    cb_palette = ["#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    handles, labels = [], []

    for i, (feature_name, feature_values) in enumerate(explanations_all.items()):
        y_vals = np.atleast_1d(feature_values)
        if smooth and len(y_vals) >= smooth_window:
            y_vals = savgol_filter(y_vals, smooth_window, smooth_poly)

        n = min(len(times), len(y_vals))
        (h,) = ax.step(
            times[:n],
            y_vals[:n],
            where="post",
            color=cb_palette[i % len(cb_palette)],
            lw=2,
            alpha=0.9,
            label=feature_name,
        )
        handles.append(h)
        labels.append(feature_name)

    if compare_plots and (model is not None or survival_fn is not None) and data_x is not None:
        if survival_fn is not None:
            surv_matrix = survival_fn(data_x, times)
        else:
            surv_funcs = model.predict_survival_function(data_x)
            surv_matrix = np.vstack([sf(times) for sf in surv_funcs])

        surv_times = times[::time_stride]
        surv_matrix = surv_matrix[:, ::time_stride]
        mean_surv = np.mean(surv_matrix, axis=0)

        if idx_plot is None:
            raise ValueError("idx_plot must be provided when using compare_plots='All' or 'Diff'.")

        indiv_surv = surv_matrix[idx_plot]
        diff_curve = indiv_surv - mean_surv

        if compare_plots == "All":
            (h_mean,) = ax.step(surv_times, mean_surv, color="grey", lw=2, linestyle=":", label="Mean")
            handles.append(h_mean); labels.append("Mean")

            (h_indiv,) = ax.step(surv_times, indiv_surv, color="grey", lw=3, linestyle="-", label="Individual")
            handles.append(h_indiv); labels.append("Individual")

            (h_diff,) = ax.step(surv_times, diff_curve, color="grey", lw=2, linestyle="--", label="Diff")
            handles.append(h_diff); labels.append("Diff")

        elif compare_plots == "Diff":
            (h_diff,) = ax.step(surv_times, diff_curve, color="grey", lw=2, linestyle="--", label="Diff")
            handles.append(h_diff); labels.append("Diff")
        else:
            raise ValueError("compare_plots must be False, 'All', or 'Diff'.")

    ax.set_xlabel("Time", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.yaxis.get_offset_text().set_x(-0.11)
    ax.yaxis.get_offset_text().set_y(1.1)
    ax.yaxis.get_offset_text().set_horizontalalignment('left')

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, loc="left")

    return handles, labels

def plot_interact(
    explanations_all,
    model=None,
    times=None,
    x_new=None,
    time_stride=1,
    save_path=None,
    compare_plots=False,  # Options: False, "All", "Diff"
    survival_fn=None,
    data_x=None,
    idx_plot=None,
    smooth=False,
    smooth_window=11,
    smooth_poly=3,
    ylabel="Attribution $S(t|x)$",
    label_fontsize=16,
    tick_fontsize=14,
    figsize=(20, 7)
):
    """
    Plot feature attribution explanations over time, with optional survival curve comparison.

    Parameters
    ----------
    explanations_all : pd.DataFrame, dict, pd.Series, or list
        Feature attribution results to plot. Each element should map feature names to their
        attribution values over time.
    model : optional
        Survival model object with a `predict_survival_function` method and `unique_times_` attribute.
        Required if survival comparison plots are requested and `survival_fn` is not provided.
    times : array-like, optional
        Time points corresponding to the attributions. If None, uses index positions.
    x_new : unused (reserved for future use)
    time_stride : int, default=1
        Stride for sampling survival times when plotting survival curves.
    save_path : str, optional
        If provided, saves the plot to this file path.
    compare_plots : bool or {"All", "Diff"}, default=False
        Whether to plot survival curves for comparison.
        - False: no survival curves
        - "All": plot mean, individual, and difference survival curves
        - "Diff": plot only the difference curve
    survival_fn : callable, optional
        Function to compute survival curves: survival_fn(data_x, times) -> survival matrix.
    data_x : array-like, optional
        Data for which to compute survival curves if compare_plots is enabled.
    idx_plot : int, optional
        Index of the individual in `data_x` to plot survival curves for.
        Required if compare_plots is "All" or "Diff".
    smooth : bool, default=False
        Whether to smooth attribution curves using Savitzky-Golay filter.
    smooth_window : int, default=11
        Window length for smoothing (must be odd).
    smooth_poly : int, default=3
        Polynomial order for smoothing.
    ylabel : str, default="Attribution $S(t|x)$"
        Label for the y-axis.
    label_fontsize : int, default=16
        Font size for axis labels.
    tick_fontsize : int, default=14
        Font size for tick labels.
    figsize : tuple, default=(20, 7)
        Figure size for the plot.
    """

    # --- Convert input into a list of explanations ---
    if isinstance(explanations_all, pd.DataFrame):
        explanations_list = [row.to_dict() for _, row in explanations_all.iterrows()]
    elif isinstance(explanations_all, (dict, pd.Series)):
        explanations_list = [explanations_all]
    elif isinstance(explanations_all, list):
        explanations_list = explanations_all
    else:
        raise ValueError("explanations_all must be DataFrame, dict, Series, or list of these.")

    n_obs = len(explanations_list)
    n_cols = 1 if n_obs == 1 else 2
    n_rows = int(np.ceil(n_obs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # --- Color-blind friendly palette (Okabe–Ito) ---
    cb_palette = ["#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    for idx, explanation in enumerate(explanations_list):
        ax = axes_flat[idx]

        # --- Plot each feature attribution curve ---
        for i, (feature_name, feature_values) in enumerate(explanation.items()):
            y_vals = np.atleast_1d(feature_values)
            if smooth and len(y_vals) >= smooth_window:
                from scipy.signal import savgol_filter
                y_vals = savgol_filter(y_vals, smooth_window, smooth_poly)

            plot_times = times if times is not None else np.arange(len(y_vals))
            n = min(len(plot_times), len(y_vals))

            ax.step(
                plot_times[:n],
                y_vals[:n],
                where="post",
                color=cb_palette[i % len(cb_palette)],
                alpha=0.9,
                lw=2,
                label=feature_name
            )

        # --- Plot survival comparison curves if requested ---
        if compare_plots and data_x is not None:
            # Compute survival matrix either from a custom function or model
            if survival_fn is not None:
                surv_matrix = survival_fn(data_x, times)
            elif model is not None:
                times = model.unique_times_
                surv_funcs = model.predict_survival_function(data_x)
                surv_matrix = np.vstack([sf(times) for sf in surv_funcs])
            else:
                raise ValueError("Need either model or survival_fn when data_x is provided.")

            surv_times = times[::time_stride]
            surv_matrix = surv_matrix[:, ::time_stride]

            mean_surv = np.mean(surv_matrix, axis=0)

            if idx_plot is None:
                raise ValueError("idx_plot must be provided when using compare_plots='All' or 'Diff'.")

            indiv_surv = surv_matrix[idx_plot]
            diff_curve = indiv_surv - mean_surv

            if compare_plots == "All":
                ax.step(surv_times, mean_surv, color='grey', lw=2,
                        linestyle=':', label="Mean")
                ax.step(surv_times, indiv_surv, color='grey', lw=3,
                        linestyle='-', label=f"Individual {idx_plot}")
                ax.step(surv_times, diff_curve, color='grey', lw=2,
                        linestyle='--', label="Diff")
            elif compare_plots == "Diff":
                ax.step(surv_times, diff_curve, color='grey', lw=2,
                        linestyle='--', label="Diff")
            else:
                raise ValueError("compare_plots must be False, 'All', or 'Diff'.")

        # --- Axis styling ---
        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(fontsize=tick_fontsize, loc="best", frameon=False)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")


def compute_integrated_brier(data_y, data_x, model, min_time=0.05, max_time=200):
    """
    Compute the Integrated Brier Score (IBS) for a survival model.

    Parameters
    ----------
    data_y : structured array or list of tuples
        Survival outcomes for the evaluation dataset, e.g. [(time, event), ...].
    data_x : array-like or pd.DataFrame
        Covariate matrix of the evaluation dataset.
    model : object
        Survival model with `predict_survival_function` method.
    min_time : float, default=0.05
        Minimum time to start evaluation.
    max_time : float, default=200
        Maximum time to stop evaluation.

    Returns
    -------
    float
        Integrated Brier Score over the specified time range.
    """
    # Generate equally spaced time grid for evaluation
    times = np.linspace(min_time, max_time, 100)

    # Predict survival functions for each individual
    surv_funcs = model.predict_survival_function(data_x)
    pred_surv = np.asarray([[fn(t) for t in times] for fn in surv_funcs])

    # Compute the Integrated Brier Score
    ibs = integrated_brier_score(data_y, data_y, pred_surv, times)
    
    return ibs

        
def explain_single_instance(model, data_x, x_new, time_indices, budget, max_order, feature_names):
    """
    Compute SurvSHAP-IQ explanations for a single sample using a fitted survival model.

    Parameters
    ----------
    model : object
        Trained survival model with `predict_survival_function()` and `unique_times_`.
    data_x : array-like of shape (n_samples, n_features)
        Background dataset used for Shapley value computation.
    x_new : array-like of shape (n_features,)
        Single instance to explain.
    time_indices : array-like
        Indices of time points to compute explanations for.
    budget : int
        Monte Carlo budget for Shapley value approximation.
    max_order : int
        Maximum order of feature interactions to compute.
    feature_names : list of str
        Names of features (used as column names in the output DataFrame).

    Returns
    -------
    pd.DataFrame
        DataFrame where each column is a feature or interaction term, and rows correspond
        to the selected time points.
    """
    explanations = {}

    # compute interaction values for each time index
    for time_idx in time_indices:
        model_at_time = lambda d: model.predict_survival_function(d, return_array=True)[:, time_idx]
        explainer = shapiq.TabularExplainer(model=model_at_time, data=data_x, max_order=max_order, exact=True)
        interaction_values = explainer.explain(x_new, budget=budget)
        explanations[model.unique_times_[time_idx]] = interaction_values

    # initialize explanation dict with empty lists
    explanation_dict = {}
    sample_iv = next(iter(explanations.values()))  # get one sample interaction value set

    for features in sample_iv.dict_values:
        if len(features) == 1:
            explanation_dict[feature_names[features[0]]] = []
        elif len(features) == 2:
            key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
            explanation_dict[key] = []

    # fill in values
    for iv in explanations.values():
        for features, value in iv.dict_values.items():
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]].append(value)
            elif len(features) == 2:
                key = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[key].append(value)

    return pd.DataFrame(explanation_dict)


def survshapiq_parallel(
    model,
    data_x,
    x_new_list,
    time_stride=3,
    budget=2**8,
    max_order=2,
    feature_names=None,
    n_jobs=-1,
    show_progress=True
):
    """
    Compute SurvSHAP-IQ explanations for multiple samples in parallel.

    Parameters
    ----------
    model : object
        Trained survival model with `predict_survival_function()` and `unique_times_`.
    data_x : array-like or pd.DataFrame
        Background dataset used for Shapley value computation.
    x_new_list : list of array-like
        List of samples to explain.
    time_stride : int, default=3
        Step size for selecting time points from `model.unique_times_`.
    budget : int, default=2**8
        Monte Carlo budget for Shapley value approximation.
    max_order : int, default=2
        Maximum order of feature interactions to compute.
    feature_names : list of str, optional
        Names of features (used as column names in the output DataFrames).
    n_jobs : int, default=-1
        Number of parallel jobs to run (use -1 to use all available cores).
    show_progress : bool, default=True
        Whether to display a progress bar during computation.

    Returns
    -------
    list of pd.DataFrame
        List of DataFrames containing feature contributions across time for each sample.
    """
    time_indices = np.arange(0, len(model.unique_times_), time_stride)

    # optionally convert data_x to numpy
    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()

    # prepare iterator
    iterator = tqdm(x_new_list) if show_progress else x_new_list

    # define wrapped function for delayed execution
    def wrapped_func(x_new):
        return explain_single_instance(
            model,
            data_x,
            x_new,
            time_indices,
            budget,
            max_order,
            feature_names
        )

    # run in parallel
    explanations_all = Parallel(n_jobs=n_jobs, max_nbytes='50M')(
        delayed(wrapped_func)(x_new) for x_new in iterator
    )

    return explanations_all


def annotate_explanations(explanations, model=None, sample_idxs=None, time_stride=1, data_x_train=None):
    """
    Annotate and flatten a list of explanation DataFrames (from survshapiq or ground-truth simulations).

    Parameters
    ----------
    explanations : list of pd.DataFrame
        List of per-sample explanation DataFrames (rows = time points, columns = features/interactions).
    model : object, optional
        Survival model with `unique_times_` attribute. Used to infer time points if missing.
    sample_idxs : list or iterable, optional
        Indices of samples corresponding to explanations. Defaults to range(len(explanations)).
    time_stride : int, default=1
        Stride used when generating explanations.
    data_x_train : pd.DataFrame, optional
        Training data (required if model does not have `unique_times_`).

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame with columns for features, `sample_idx`, and `time`.
    """
    all_rows = []

    if sample_idxs is None:
        sample_idxs = range(len(explanations))

    # determine time points if available
    if model is not None and hasattr(model, 'unique_times_'):
        time_points = model.unique_times_[::time_stride]

    for i, df in enumerate(explanations):
        df = df.copy()
        df["sample_idx"] = sample_idxs[i]

        # assign time column
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            df["time"] = idx.values
        elif np.issubdtype(idx.dtype, np.number):
            df["time"] = idx.values
        else:
            # fallback method
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


def survival_from_hazard(df, hazard_func, t_values):
    """
    Compute survival curves from a hazard function for multiple individuals.

    Parameters
    ----------
    df : np.ndarray of shape (n_individuals, n_features)
        Each row corresponds to an individual, each column to a feature.
    hazard_func : callable
        Function hazard_func(t, x1, x2, ..., xk) returning hazard values over time t.
    t_values : array-like
        Time points at which to evaluate hazard and compute survival.

    Returns
    -------
    np.ndarray of shape (n_individuals, len(t_values))
        Survival curves for each individual over time.
    """
    df = np.asarray(df)
    n_people, n_features = df.shape
    t_values = np.asarray(t_values)
    dt = np.diff(t_values, prepend=0)

    survival_matrix = np.zeros((n_people, len(t_values)))

    for i in range(n_people):
        # compute hazard values for this individual
        h_vals = hazard_func(t_values, *df[i, :])
        H = np.cumsum(h_vals * dt)  # cumulative hazard
        survival_matrix[i, :] = np.exp(-H)  # survival function

    return survival_matrix        


def hazard_matrix(df, hazard_func, t_values):
    """
    Compute a hazard matrix for multiple individuals with flexible number of covariates.

    Parameters
    ----------
    df : array-like of shape (n_individuals, n_features)
        Feature matrix, each row is an individual, each column is a feature.
    hazard_func : callable
        Function hazard_func(t, x1, x2, ..., xk) returning hazard values for times t.
    t_values : array-like
        Time points at which to evaluate the hazard function.

    Returns
    -------
    np.ndarray of shape (n_individuals, len(t_values))
        Hazard values for each individual over time.
    """
    df = np.asarray(df)
    n_people, n_features = df.shape
    t_values = np.asarray(t_values)

    hazard_mat = np.zeros((n_people, len(t_values)))

    for i in range(n_people):
        # unpack row i into arguments for hazard_func
        hazard_mat[i, :] = hazard_func(t_values, *df[i, :])

    return hazard_mat


def survshapiq_ground_truth(
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
    feature_names=None
):
    """
    Compute SurvSHAP-IQ explanations from ground-truth survival curves.

    Parameters
    ----------
    data_x : array-like or DataFrame
        Training feature matrix used for background distribution.
    x_new_list : list of array-like
        List of samples to explain.
    survival_from_hazard_func : callable
        Function returning survival curves given features and time grid.
    times : array-like
        Time points for evaluation.
    time_stride : int, default=1
        Step size for selecting time indices.
    budget : int, default=2**8
        Monte Carlo sampling budget for Shapley approximation.
    max_order : int, default=2
        Maximum interaction order to compute.
    approximator : object, optional
        Approximation method for Shapley values (if exact=False).
    index : int, optional
        Reference index for conditional Shapley values.
    exact : bool, default=True
        If True, compute exact Shapley values (no approximation).
    feature_names : list of str, optional
        Names of features for labeling.

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame contains feature contributions across time for one sample.
    """
    explanations_all = []

    for idx, x_new in enumerate(x_new_list):
        explanations = {}

        for i in range(0, len(times), time_stride):
            # returns survival for all individuals at time index i
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
                    index=index,
                    max_order=max_order,
                    exact=True
                )
            else:
                raise ValueError("Must either provide both 'index' and 'approximator', or set exact=True.")

            # compute interaction values
            interaction_values = explainer.explain(x_new, budget=budget)
            explanations[i] = interaction_values

        # initialize feature names in dict
        explanation_dict = {}
        for features, _ in interaction_values.dict_values.items():
            if not features:
                continue
            if len(features) == 1:
                explanation_dict[feature_names[features[0]]] = []
            elif len(features) == 2:
                new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                explanation_dict[new_feature_name] = []

        # fill values
        for t, iv in explanations.items():
            for features, value in iv.dict_values.items():
                if not features:
                    continue
                if len(features) == 1:
                    explanation_dict[feature_names[features[0]]].append(value)
                elif len(features) == 2:
                    new_feature_name = f'{feature_names[features[0]]} * {feature_names[features[1]]}'
                    explanation_dict[new_feature_name].append(value)

        explanation_df = pd.DataFrame(
            explanation_dict, index=[times[t] for t in explanations.keys()]
        )
        explanations_all.append(explanation_df)

    return explanations_all


def explain_single_instance_ground_truth(
    x_new,
    data_x,
    survival_from_hazard_func,
    times,
    time_indices,
    budget,
    max_order,
    feature_names,
    index=None,
    approximator=None,
    exact=True
):
    """
    Compute SurvSHAP-IQ explanations for a single sample using ground-truth survival.

    Parameters
    ----------
    x_new : array-like
        Sample to explain.
    data_x : array-like
        Training feature matrix used as background.
    survival_from_hazard_func : callable
        Function returning survival curves given features and time grid.
    times : array-like
        Time points for evaluation.
    time_indices : array-like
        Indices of times to compute explanations for.
    budget : int
        Monte Carlo budget for approximation.
    max_order : int
        Maximum interaction order.
    feature_names : list of str
        Names of features.
    index : int, optional
        Reference index for conditional Shapley values.
    approximator : object, optional
        Approximation method for Shapley values.
    exact : bool, default=True
        Whether to compute exact Shapley values.

    Returns
    -------
    pd.DataFrame
        DataFrame of feature contributions across time for this sample.
    """
    explanations = {}

    # compute survival contributions for each time index
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

        # compute interaction values
        interaction_values = explainer.explain(x_new, budget=budget)
        explanations[i] = interaction_values

    # initialize feature names in dict
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

    # fill values
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
    """
    Compute SurvSHAP-IQ explanations for multiple samples in parallel.

    Parameters
    ----------
    data_x : array-like or DataFrame
        Training feature matrix used as background.
    x_new_list : list of array-like
        Samples to explain.
    survival_from_hazard_func : callable
        Function returning survival curves given features and time grid.
    times : array-like
        Time points for evaluation.
    time_stride : int, default=1
        Step size for selecting time indices.
    budget : int, default=2**8
        Monte Carlo budget for approximation.
    max_order : int, default=2
        Maximum interaction order.
    approximator : object, optional
        Approximation method for Shapley values.
    index : int, optional
        Reference index for conditional Shapley values.
    exact : bool, default=True
        Whether to compute exact Shapley values.
    feature_names : list of str, optional
        Names of features.
    n_jobs : int, default=-1
        Number of parallel jobs.
    show_progress : bool, default=True
        Whether to show a progress bar.

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame contains feature contributions across time for one sample.
    """
    time_indices = np.arange(0, len(times), time_stride)

    # optionally convert to numpy
    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()

    # prepare iterator with progress bar
    iterator = tqdm(x_new_list) if show_progress else x_new_list

    # wrapper function
    def wrapped_func(x_new):
        return explain_single_instance_ground_truth(
            x_new,
            data_x,
            survival_from_hazard_func,
            times,
            time_indices,
            budget,
            max_order,
            feature_names,
            index=index,
            approximator=approximator,
            exact=exact
        )

    # run in parallel
    explanations_all = Parallel(
        n_jobs=n_jobs,
        max_nbytes='50M'
    )(delayed(wrapped_func)(x_new) for x_new in iterator)

    return explanations_all


def compute_local_accuracy(
    explanations_all: pd.DataFrame,
    data_df: pd.DataFrame,
    survival_fn=None,
    model=None,
    test_size: float = 0.2,
    random_state: int = 42,
    time_stride: int = 1
):
    """
    Compute local accuracy of SurvSHAP-IQ explanations by comparing reconstructed
    survival curves with true or predicted survival curves.

    Parameters
    ----------
    explanations_all : pd.DataFrame
        DataFrame of feature attributions with columns 'sample_idx' and 'time'.
    data_df : pd.DataFrame
        Original survival dataset containing covariates and survival outcomes.
    survival_fn : callable, optional
        Function computing survival curves: survival_fn(X, times).
    model : str or object, optional
        Trained survival model ("Cox" or "GBSA") or pre-fitted estimator.
    test_size : float, default=0.2
        Fraction of dataset used for test split.
    random_state : int, default=42
        Random seed for reproducibility.
    time_stride : int, default=1
        Step size for selecting time points when using model predictions.

    Returns
    -------
    local_accuracy : np.ndarray
        Local accuracy values for each time point.
    times : np.ndarray
        Corresponding time points.
    mean_local_accuracy : float
        Mean local accuracy across all time points.
    """
    # extract attribution matrix
    value_cols = [c for c in explanations_all.columns if c not in ["sample_idx", "time"]]
    explanations_all["sum_all"] = explanations_all[value_cols].sum(axis=1)
    exp_matrix = explanations_all.pivot(index="sample_idx", columns="time", values="sum_all")
    exp_matrix = exp_matrix.values
    times = explanations_all["time"].unique()

    # prepare survival data
    data_y, data_x_df = prepare_survival_data(data_df)
    data_x = data_x_df.values

    # split and reassemble to match above
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, test_size=test_size, random_state=random_state
    )
    data_x_full = np.concatenate((data_x_train, data_x_test), axis=0)

    # compute survival matrix
    if survival_fn is not None:
        surv_matrix = survival_fn(data_x_full, times)
    elif model is not None:
        if model == "Cox":
            model = CoxPHSurvivalAnalysis()
            model.fit(data_x_train, data_y_train)
        elif model == "GBSA":
            model = GradientBoostingSurvivalAnalysis()
            model.fit(data_x_train, data_y_train)
        times = model.unique_times_[::time_stride]
        surv_funcs = model.predict_survival_function(data_x_full)
        surv_matrix = np.vstack([sf(times) for sf in surv_funcs])
    else:
        raise ValueError("Either survival_fn or model must be provided.")

    # mean-center predictions
    col_means = surv_matrix.mean(axis=0)
    surv_centered = surv_matrix - col_means

    # compute local accuracy
    local_accuracy = np.sqrt(
        ((surv_centered - exp_matrix) ** 2).mean(axis=0)
        / ((surv_matrix ** 2).mean(axis=0))
    )

    # mean local accuracy
    mean_local_accuracy = np.mean(local_accuracy)

    return local_accuracy, times, mean_local_accuracy


# --- Ground Truth Hazard (Time-Invariant) ---
def hazard_func_linear_ti(t, x1, x2, x3):
    """Linear time-invariant hazard function."""
    return 0.03 * np.exp((0.4 * x1) - (0.8 * x2) - (0.6 * x3))


def hazard_wrap_linear_ti(X, t):
    """Wrapper returning hazard matrix for hazard_func_linear_ti."""
    return hazard_matrix(X, hazard_func_linear_ti, t)


def log_hazard_func_linear_ti(t, x1, x2, x3):
    """Log of linear time-invariant hazard function."""
    return np.log(0.03 * np.exp((0.4 * x1) - (0.8 * x2) - (0.6 * x3)))


def log_hazard_wrap_linear_ti(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_linear_ti."""
    return hazard_matrix(X, log_hazard_func_linear_ti, t)


def surv_from_hazard_linear_ti_wrap(X, t):
    """Compute survival function from hazard_func_linear_ti."""
    return survival_from_hazard(X, hazard_func_linear_ti, t)


# --- Ground Truth Hazard (Time-Dependent Main Effect) ---
def hazard_func_linear_tdmain(t, x1, x2, x3):
    """Linear hazard function with time-dependent main effect on x1."""
    return 0.03 * np.exp((0.4 * x1) * np.log(t + 1) - (0.8 * x2) - (0.6 * x3))


def hazard_wrap_linear_tdmain(X, t):
    """Wrapper returning hazard matrix for hazard_func_linear_tdmain."""
    return hazard_matrix(X, hazard_func_linear_tdmain, t)


def log_hazard_func_linear_tdmain(t, x1, x2, x3):
    """Log of linear hazard function with time-dependent main effect."""
    return np.log(0.03 * np.exp((0.4 * x1) * np.log(t + 1) - (0.8 * x2) - (0.6 * x3)))


def log_hazard_wrap_linear_tdmain(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_linear_tdmain."""
    return hazard_matrix(X, log_hazard_func_linear_tdmain, t)


def surv_from_hazard_linear_tdmain_wrap(X, t):
    """Compute survival function from hazard_func_linear_tdmain."""
    return survival_from_hazard(X, hazard_func_linear_tdmain, t)


# --- Ground Truth Hazard (Time-Invariant Interaction) ---
def hazard_func_linear_ti_inter(t, x1, x2, x3):
    """Linear time-invariant hazard function with x1*x3 interaction."""
    return 0.03 * np.exp((0.4 * x1) - (0.8 * x2) - (0.6 * x3) - (0.9 * x1 * x3))


def hazard_wrap_linear_ti_inter(X, t):
    """Wrapper returning hazard matrix for hazard_func_linear_ti_inter."""
    return hazard_matrix(X, hazard_func_linear_ti_inter, t)


def log_hazard_func_linear_ti_inter(t, x1, x2, x3):
    """Log of linear time-invariant hazard function with interaction."""
    return np.log(0.03 * np.exp((0.4 * x1) - (0.8 * x2) - (0.6 * x3) - (0.9 * x1 * x3)))


def log_hazard_wrap_linear_ti_inter(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_linear_ti_inter."""
    return hazard_matrix(X, log_hazard_func_linear_ti_inter, t)


def surv_from_hazard_linear_ti_inter_wrap(X, t):
    """Compute survival function from hazard_func_linear_ti_inter."""
    return survival_from_hazard(X, hazard_func_linear_ti_inter, t)


# --- Ground Truth Hazard (Time-Dependent Main Effect + Interaction) ---
def hazard_func_linear_tdmain_inter(t, x1, x2, x3):
    """Linear hazard function with time-dependent x1 and x1*x3 interaction."""
    return 0.03 * np.exp((0.4 * x1) * np.log(t + 1) - (0.8 * x2) - (0.6 * x3) - (0.9 * x1 * x3))


def hazard_wrap_linear_tdmain_inter(X, t):
    """Wrapper returning hazard matrix for hazard_func_linear_tdmain_inter."""
    return hazard_matrix(X, hazard_func_linear_tdmain_inter, t)


def log_hazard_func_linear_tdmain_inter(t, x1, x2, x3):
    """Log of hazard function with time-dependent x1 and interaction."""
    return np.log(0.03 * np.exp((0.4 * x1) * np.log(t + 1) - (0.8 * x2) - (0.6 * x3) - (0.9 * x1 * x3)))


def log_hazard_wrap_linear_tdmain_inter(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_linear_tdmain_inter."""
    return hazard_matrix(X, log_hazard_func_linear_tdmain_inter, t)


def surv_from_hazard_linear_tdmain_inter_wrap(X, t):
    """Compute survival function from hazard_func_linear_tdmain_inter."""
    return survival_from_hazard(X, hazard_func_linear_tdmain_inter, t)


# --- Ground Truth Hazard (Time-Dependent Interaction Only) ---
def hazard_func_linear_tdinter(t, x1, x2, x3):
    """Hazard function with time-dependent interaction term x1*x3."""
    return 0.03 * np.exp((0.4 * x1) - (0.8 * x2) - (0.6 * x3) - (0.9 * x1 * x3) * np.log(t + 1))


def hazard_wrap_linear_tdinter(X, t):
    """Wrapper returning hazard matrix for hazard_func_linear_tdinter."""
    return hazard_matrix(X, hazard_func_linear_tdinter, t)


def log_hazard_func_linear_tdinter(t, x1, x2, x3):
    """Log of hazard function with time-dependent interaction term."""
    return np.log(0.03 * np.exp((0.4 * x1) - (0.8 * x2) - (0.6 * x3) - (0.9 * x1 * x3) * np.log(t + 1)))


def log_hazard_wrap_linear_tdinter(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_linear_tdinter."""
    return hazard_matrix(X, log_hazard_func_linear_tdinter, t)


def surv_from_hazard_linear_tdinter_wrap(X, t):
    """Compute survival function from hazard_func_linear_tdinter."""
    return survival_from_hazard(X, hazard_func_linear_tdinter, t)


# --- Ground Truth Hazard (Generalized Additive Model - TI) ---
def hazard_func_genadd_ti(t, x1, x2, x3):
    """Nonlinear time-invariant hazard function (x1^2, arctan(x2))."""
    return 0.03 * np.exp((0.4 * x1**2) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2)) - (0.6 * x3))


def hazard_wrap_genadd_ti(X, t):
    """Wrapper returning hazard matrix for hazard_func_genadd_ti."""
    return hazard_matrix(X, hazard_func_genadd_ti, t)


def log_hazard_func_genadd_ti(t, x1, x2, x3):
    """Log of nonlinear time-invariant hazard function."""
    return np.log(0.03 * np.exp((0.4 * x1**2) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2)) - (0.6 * x3)))


def log_hazard_wrap_genadd_ti(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_genadd_ti."""
    return hazard_matrix(X, log_hazard_func_genadd_ti, t)


def surv_from_hazard_genadd_ti_wrap(X, t):
    """Compute survival function from hazard_func_genadd_ti."""
    return survival_from_hazard(X, hazard_func_genadd_ti, t)


# --- Ground Truth Hazard (Generalized Additive Model - TD Main) ---
def hazard_func_genadd_tdmain(t, x1, x2, x3):
    """Nonlinear hazard function with time-dependent x1^2 term."""
    return 0.03 * np.exp((0.4 * x1**2 * np.log(t + 1)) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2)) - (0.6 * x3))


def hazard_wrap_genadd_tdmain(X, t):
    """Wrapper returning hazard matrix for hazard_func_genadd_tdmain."""
    return hazard_matrix(X, hazard_func_genadd_tdmain, t)


def log_hazard_func_genadd_tdmain(t, x1, x2, x3):
    """Log of nonlinear hazard function with time-dependent x1^2."""
    return np.log(0.03 * np.exp((0.4 * x1**2) * np.log(t + 1) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2)) - (0.6 * x3)))


def log_hazard_wrap_genadd_tdmain(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_genadd_tdmain."""
    return hazard_matrix(X, log_hazard_func_genadd_tdmain, t)


def surv_from_hazard_genadd_tdmain_wrap(X, t):
    """Compute survival function from hazard_func_genadd_tdmain."""
    return survival_from_hazard(X, hazard_func_genadd_tdmain, t)


# --- Ground Truth Hazard (Generalized Additive Model - TI Interaction) ---
def hazard_func_genadd_ti_inter(t, x1, x2, x3):
    """Nonlinear TI hazard with x1*x2 and x1*x3^2 interactions."""
    return 0.03 * np.exp((0.4 * x1**2) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2))
                         - (0.6 * x3) - (0.5 * x1 * x2) + (0.2 * x1 * x3**2))


def hazard_wrap_genadd_ti_inter(X, t):
    """Wrapper returning hazard matrix for hazard_func_genadd_ti_inter."""
    return hazard_matrix(X, hazard_func_genadd_ti_inter, t)


def log_hazard_func_genadd_ti_inter(t, x1, x2, x3):
    """Log of nonlinear TI hazard with interactions."""
    return np.log(0.03 * np.exp((0.4 * x1**2) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2))
                                - (0.6 * x3) - (0.5 * x1 * x2) + (0.2 * x1 * x3**2)))


def log_hazard_wrap_genadd_ti_inter(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_genadd_ti_inter."""
    return hazard_matrix(X, log_hazard_func_genadd_ti_inter, t)


def surv_from_hazard_genadd_ti_inter_wrap(X, t):
    """Compute survival function from hazard_func_genadd_ti_inter."""
    return survival_from_hazard(X, hazard_func_genadd_ti_inter, t)


# --- Ground Truth Hazard (Generalized Additive Model - TD Main + Interaction) ---
def hazard_func_genadd_tdmain_inter(t, x1, x2, x3):
    """Nonlinear hazard with time-dependent x1^2 and interactions."""
    return 0.03 * np.exp((0.4 * x1**2 * np.log(t + 1)) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2))
                         - (0.6 * x3) - (0.5 * x1 * x2) + (0.2 * x1 * x3**2))


def hazard_wrap_genadd_tdmain_inter(X, t):
    """Wrapper returning hazard matrix for hazard_func_genadd_tdmain_inter."""
    return hazard_matrix(X, hazard_func_genadd_tdmain_inter, t)


def log_hazard_func_genadd_tdmain_inter(t, x1, x2, x3):
    """Log of nonlinear hazard with time-dependent x1^2 and interactions."""
    return np.log(0.03 * np.exp((0.4 * x1**2) * np.log(t + 1) - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2))
                                - (0.6 * x3) - (0.5 * x1 * x2) + (0.2 * x1 * x3**2)))


def log_hazard_wrap_genadd_tdmain_inter(X, t):
    """Wrapper returning hazard matrix for log_hazard_func_genadd_tdmain_inter."""
    return hazard_matrix(X, log_hazard_func_genadd_tdmain_inter, t)


def surv_from_hazard_genadd_tdmain_inter_wrap(X, t):
    """Compute survival function from hazard_func_genadd_tdmain_inter."""
    return survival_from_hazard(X, hazard_func_genadd_tdmain_inter, t)


# --- Ground Truth Hazard (Generalized Additive Model - TD Interaction Only) ---
def hazard_func_genadd_tdinter(t, x1, x2, x3):
    """Hazard with quadratic, nonlinear, and time-dependent interaction terms."""
    return 0.03 * np.exp(
        (0.4 * x1**2)
        - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2))
        - (0.6 * x3)
        - (0.5 * x1 * x2)
        + (0.2 * x1 * x3**2 * np.log(t + 1))
    )


def hazard_wrap_genadd_tdinter(X, t):
    """Hazard matrix wrapper for hazard_func_genadd_tdinter."""
    return hazard_matrix(X, hazard_func_genadd_tdinter, t)


def log_hazard_func_genadd_tdinter(t, x1, x2, x3):
    """Log-hazard version of hazard_func_genadd_tdinter."""
    return np.log(
        0.03 * np.exp(
            (0.4 * x1**2)
            - (0.8 * (2 / np.pi) * np.arctan(0.7 * x2))
            - (0.6 * x3)
            - (0.5 * x1 * x2)
            + (0.2 * x1 * x3**2 * np.log(t + 1))
        )
    )


def log_hazard_wrap_genadd_tdinter(X, t):
    """Log-hazard matrix wrapper for log_hazard_func_genadd_tdinter."""
    return hazard_matrix(X, log_hazard_func_genadd_tdinter, t)


def surv_from_hazard_genadd_tdinter_wrap(X, t):
    """Compute survival from hazard_func_genadd_tdinter."""
    return survival_from_hazard(X, hazard_func_genadd_tdinter, t)
