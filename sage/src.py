import copy 

import shapiq
import numpy as np
from sksurv.metrics import integrated_brier_score


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


class SageGame(shapiq.Game):
    """
    Based on https://github.com/mmschlk/shapiq/blob/main/src/shapiq_games/benchmark/global_xai/base.py
    """
    def __init__(
        self,
        data_x,
        data_y,
        model,
        loss_function,
        loss_function_times,
        n_samples_eval=200,
        normalize=True,
        random_state=0,
        verbose=False
    ):
        
        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)
        self.n_samples_eval = n_samples_eval  # how many samples to evaluate for each coalition

        self.data_x = copy.deepcopy(data_x)
        self.data_y = copy.deepcopy(data_y)
        self._n_samples = self.data_x.shape[0]
        # shuffle the data column wise (shuffle each column independently)
        self._data_x_shuffled = copy.deepcopy(self.data_x)
        for i in range(self._data_x_shuffled.shape[1]):
            self._rng.shuffle(self._data_x_shuffled[:, i])

        self.model = model
        self.loss_function = loss_function
        self.loss_function_times = loss_function_times
        self._predictions = self.model(self.data_x)
        self._predictions_empty = self.model(self._data_x_shuffled)
        self._loss_empty = self.loss_function(self.data_y, self._predictions_empty, times=self.loss_function_times)

        super().__init__(
            n_players=data_x.shape[1],
            normalize=normalize,
            normalization_value=self._loss_empty,
            verbose=verbose,
        )

    def value_function(self, coalitions):
        values = []
        for i, coalition in enumerate(coalitions):
            if not any(coalition):
                values.append(self._loss_empty)
                continue
            # get the subset of the data
            idx = self._rng.choice(self._n_samples, size=self.n_samples_eval, replace=False)
            data_subset = self.data_x[idx].copy()
            # replace the features not part of the subset
            data_subset[:, ~coalition] = self._data_x_shuffled[idx][:, ~coalition]
            # get the predictions of the model on the subset
            predictions_new = self.model(data_subset)
            # get the loss of the model on the subset
            values.append(self.loss_function(self.data_y[idx], predictions_new, times=self.loss_function_times))
        return values