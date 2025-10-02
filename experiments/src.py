import shapiq
import numpy as np
import pandas as pd
from sksurv.metrics import integrated_brier_score
import tqdm

def prepare_survival_data(df, event_col='eventtime', status_col='status', id_col='id'):
    data_y = np.array(
        list(zip(df[status_col].astype(bool), df[event_col])),
        dtype=[('status', '?'), ('eventtime', 'f8')]
    )
    drop_cols = [status_col, event_col]
    if id_col is not None and id_col in df.columns:
        drop_cols.append(id_col)
    data_x = df.drop(columns=drop_cols)
    return data_y, data_x


def compute_integrated_brier(data_y, data_x, model):
    max_time = np.max([y[1] for y in data_y])
    #times = model.unique_times_[model.unique_times_ < max_time]
    test_times = np.array([y[1] for y in data_y])  # Extract event/censoring times
    min_time = np.percentile(test_times, 5)
    max_time = np.percentile(test_times, 95)
    times = np.linspace(min_time, max_time, 100)
    surv_funcs = model.predict_survival_function(data_x)
    pred_surv = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
    ibs = integrated_brier_score(data_y, data_y, pred_surv, times)
    return ibs


def get_evenly_spaced_integers(numbers: list[int], k: int) -> list[int]:
    """
    Selects k evenly spaced integers from a list of unique sorted integers.
    """
    n = len(numbers)
    # --- Handle Edge Cases ---
    if k <= 0:
        return []
    if k >= n:
        # If k is greater than or equal to the number of available elements,
        # the most "evenly spaced" selection is the entire list itself.
        return numbers
    if k == 1:
        # If only one number is requested, return the first one.
        return [numbers[0]]
    # --- Core Logic ---
    # We want to select k items, which means we will have k-1 intervals
    # spanning the total index range of n-1 (from index 0 to n-1).
    result = []
    step = (n - 1) / (k - 1)
    for i in range(k):
        # Calculate the ideal index as a float.
        float_index = i * step
        # Round to the nearest integer index to find the element.
        # This approach ensures the first (i=0) and last (i=k-1) elements
        # of the input list are always included in the result.
        index = int(round(float_index))
        result.append(numbers[index].item())
    return result


def survshapiq(
        model, 
        data_x, 
        x_new_list, 
        n_timepoints=20, 
        budget=2**8, 
        max_order=2, 
        approximator=None, 
        index=None, 
        exact=True, 
        feature_names=None, 
        imputer="marginal",
        sample_size=80
    ):
    """
    Explain interaction effects at different time points for a survival model.

    Parameters:
    - model: fitted survival model with .predict_survival_function() and .unique_times_
    - data_x: DataFrame of training covariates
    - x_new: DataFrame or array of new observations to explain (only the first row will be used)
    - n_timepoints: number of time points (20 by default)
    - budget: computational budget for Shapiq explainer
    - max_order: maximum order of interactions (default is 2)
    - approximator: type of approximator to use (default is "auto")
    - index: type of index to use (default is "k-SII")
    - exact: whether to use exact mode (default is True)

    Returns:
    - explanation_df: a DataFrame with interaction values over selected time points
    """

    explanations_all = []
    timepoints = get_evenly_spaced_integers(model.unique_times_, n_timepoints)

    for x_new in tqdm.tqdm(x_new_list):
        explanations = {}
        for t in timepoints:
            which_timepoint_equals_t = np.where(model.unique_times_ == t)[0][0].item()
            model_at_time_t = lambda d: model.predict_survival_function(d, return_array=True)[:, which_timepoint_equals_t]

            if imputer == "baseline":
                if index is not None and approximator is not None:
                    explainer = shapiq.TabularExplainer(model=model_at_time_t,
                                                        data=data_x,
                                                        max_order=max_order,
                                                        approximator=approximator,
                                                        index=index,
                                                        imputer=imputer
                                                        )
                elif exact:
                    explainer = shapiq.TabularExplainer(model=model_at_time_t,
                                                        data=data_x,
                                                        max_order=max_order,
                                                        exact=True,
                                                        imputer=imputer
                                                        )
                else:
                    raise ValueError("Must either provide both 'index' and 'approximator', or set exact=True.")
            else:
                if index is not None and approximator is not None:
                    explainer = shapiq.TabularExplainer(model=model_at_time_t,
                                                        data=data_x,
                                                        max_order=max_order,
                                                        approximator=approximator,
                                                        index=index,
                                                        imputer=imputer,
                                                        sample_size=sample_size
                                                        )
                elif exact:
                    explainer = shapiq.TabularExplainer(model=model_at_time_t,
                                                        data=data_x,
                                                        max_order=max_order,
                                                        exact=True,
                                                        imputer=imputer,
                                                        sample_size=sample_size
                                                        )
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