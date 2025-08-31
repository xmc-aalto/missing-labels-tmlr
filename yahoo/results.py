"""
This file contains utility functions that can be used to transform the json files written by the runner
into plottable data.
"""
import itertools

import pandas as pd
import numpy as np
from losses import LossMode, DataMode


def make_setting(data: str, mode: str, loss: str, norm: bool) -> str:
    """
    Given the training data, loss mode, base loss, and normalization setting, makes a short string identifying the
    setting of this run.
    :param data: The data setting. Either `DataMode.CLEAN` or `DataMode.NOISY`. For clean data, the identifier get the suffix `*`
    :param mode: The loss variant to use, i.e. whether this is a vanilla, unbiased, or bound run. Identified using `V`, `U` and `B` respectively.
    :param loss: A string identifier for the name of the base loss.
    :param norm: Whether this is the normalized variation of the loss composition, in which case the identifier is prefixed with `N`.
    :return: The constructed identifier.
    """
    assert data in {DataMode.CLEAN, DataMode.NOISY}
    assert mode in {LossMode.VANILLA, LossMode.UNBIASED, LossMode.BOUND}
    loss_prefixes = {LossMode.VANILLA: "V", LossMode.UNBIASED: "U", LossMode.BOUND: "B"}
    is_clean = "*" if data == DataMode.CLEAN else ""
    prefix = loss_prefixes[mode]
    return f"{prefix}-{'N' if norm else ''}{loss.upper()}{is_clean}"


def load_data(json_files: list) -> pd.DataFrame:
    """
    Loads and merges the data from the given json files. Also adds additional helper columns for the `Setting` based
    on `make_setting`, the `LossDecomposition` (base loss and decomposition normalization) and `LossVariant` (data mode
    and loss mode).
    :param json_files:
    :return:
    """
    raw_data = pd.concat([pd.read_json(f, orient="records").transpose() for f in json_files])
    data = raw_data['config/data'].tolist()
    mode = raw_data['config/mode'].tolist()
    loss = raw_data['config/loss'].tolist()
    norm = raw_data['config/normalized'].tolist()

    # add a column that summarizes the setting
    raw_data.loc[:, 'Setting'] = [make_setting(d, m, l, n) for d, m, l, n, in zip(data, mode, loss, norm)]
    raw_data.loc[:, 'LossDecomposition'] = [f"{l}{('-norm' if n else '')}" for l, n, in zip(loss, norm)]
    raw_data.loc[:, 'LossVariant'] = [f"{l}{('-clean' if n == DataMode.CLEAN else '-noisy')}" for l, n, in zip(mode, data)]

    print(raw_data.loc[:, 'noisy-test/tsPSP@1'])
    for metric, k in itertools.product(["P", "R", "trPSP", "tsPSP"], [1, 3, 5]):
        for kind, split in itertools.product(["noisy"], ["train", "test", "val"]):
            raw_data[f"{kind}-{split}/{metric}@{k}"] *= 100

    return raw_data


def extract_best_hyperparams(data):
    """
    Performs a reduction over the dataset by selecting for each (Setting, index) combination only the single
    datapoint for which the validation loss on noisy data is lowest.
    :param data:
    :return:
    """
    grouped = data.groupby(["Setting", "config/index"])

    def select_best(x):
        # TODO This is very wrong, fix as soon as we get rid of the NaNs
        criterion = x['noisy-test/unbiased_test_loss'].to_numpy()
        best = np.argmin(criterion)
        return x.iloc[best]

    return grouped.apply(select_best)


def iter_metrics():
    yield "loss"
    yield "vanilla_loss"
    yield "unbiased_loss"
    yield "unbiased_test_loss"    
    for metric, k in itertools.product(["P", "R", "trPSP", "tsPSP"], [1, 3, 5]):
        yield f"{metric}@{k}"


def iter_metrics_path():
    for kind, split in itertools.product(["noisy"], ["train", "test", "val"]):
        for metric in iter_metrics():
            yield f"{kind}-{split}/{metric}"


def remove_metrics_from_data(data: pd.DataFrame):
    for m in iter_metrics_path():
        del data[m]


def split_by_evaluation(data: pd.DataFrame) -> pd.DataFrame:
    base_data = data.copy()
    remove_metrics_from_data(base_data)

    eval_splits = []
    for kind, split in itertools.product(["noisy"], ["train", "test", "val"]):
        new_eval = base_data.copy()
        for metric in iter_metrics():
            new_eval[metric] = data[f"{kind}-{split}/{metric}"]
        new_eval["eval-on"] = f"{kind}-{split}"
        eval_splits.append(new_eval)

    data = pd.concat(eval_splits)
    return data


def split_by_metric(data: pd.DataFrame) -> pd.DataFrame:
    base_data = data.copy()
    remove_metrics_from_data(base_data)

    eval_splits = []
    for metric in iter_metrics():
        new_eval = base_data.copy()
        for kind, split in itertools.product(["noisy"], ["train", "test", "val"]):
            new_eval[f"{kind}-{split}/value"] = data[f"{kind}-{split}/{metric}"]
        new_eval["metric"] = f"{metric}"
        eval_splits.append(new_eval)

    data = pd.concat(eval_splits)
    return data


def make_average_and_std(data: pd.DataFrame):
    data["_Setting"] = data["Setting"]
    grouped = data.groupby(["_Setting"], as_index=False)

    def summarize(x):
        # this is the index that we average over
        del x["config/index"]
        #
        head = x.head(1)
        mean = x.mean(numeric_only=True)
        std = x.std(numeric_only=True)
        metrics = list(iter_metrics_path()) + ["config/l2_reg"]
        for metric in metrics:
            del head[metric]
            head[f"{metric}/mean"] = mean[metric]
            head[f"{metric}/std"] = std[metric]

        return head.copy()

    result = grouped.apply(summarize).reset_index(drop=True)
    result["Setting"] = result["_Setting"]
    del result["_Setting"]
    return result
