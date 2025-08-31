import json
import hashlib
from typing import List

import numpy as np
from pathlib import Path
import pickle
import argparse

from train import generate_datasets, run_experiment
from losses import LossMode, DataMode, get_base_loss


def flatten_dict(data):
    result = {}
    for key in data:
        if isinstance(data[key], dict):
            for entry in data[key]:
                result[key + "/" + entry] = data[key][entry]
        else:
            result[key] = data[key]
    return result


def run_and_cache_experiment(cache: dict, data, train_mode: str, loss_mode: str, normalized: bool, base_loss: str,
                             l2_reg: float, index: int, num_epochs: int, pretraining: int):
    config = {
        "data": train_mode,
        "mode": loss_mode,
        "loss": base_loss,
        "normalized": normalized,
        "l2_reg": l2_reg,
        "index": index
    }

    identifier = hashlib.sha256(json.dumps(config).encode()).hexdigest()
    if identifier in cache:
        return

    results = run_experiment(data=data, train_mode=train_mode, loss_mode=loss_mode, base_loss=base_loss,
                             normalized=normalized, l2_reg=l2_reg, num_epochs=num_epochs, pretraining=pretraining)
    results["config"] = config
    results = flatten_dict(results)
    cache[identifier] = results


def run_all_settings(index: int, losses: List[str]):
    propensities = 1.0 / np.linspace(2.0, 20.0, num=source_train_data.num_labels)
    data = generate_datasets(source_train_data, source_test_data, propensities)

    def rce(train_mode: str, loss_mode: str, normalized: bool, base_loss: str, pretraining: int = 0):
        kind, _ = get_base_loss(base_loss)
        # multiclass bound == multiclass unbiased, so no need to run twice
        if kind == "multiclass" and loss_mode == LossMode.BOUND and not normalized:
            return

        regs = np.logspace(0.0, -7, 11, endpoint=True)
        if base_loss == "sqh":
            regs = np.logspace(1.4, -7+1.4, 11, endpoint=True)
        if base_loss == "sqh" and loss_mode == LossMode.UNBIASED and not normalized:
            regs = np.concatenate([regs, np.logspace(-0.7, -2.8, 7, endpoint=True)])
        for reg in regs:
            run_and_cache_experiment(result_cache, data, train_mode, loss_mode, normalized, base_loss, reg, index,
                                     num_epochs=20, pretraining=0)
            result_path.write_text(json.dumps(result_cache, indent=2))

    def run_all(base_loss, normalized):
        rce(train_mode=DataMode.CLEAN, loss_mode=LossMode.VANILLA, base_loss=base_loss, normalized=normalized)
        rce(train_mode=DataMode.NOISY, loss_mode=LossMode.VANILLA, base_loss=base_loss, normalized=normalized)
        rce(train_mode=DataMode.NOISY, loss_mode=LossMode.UNBIASED, base_loss=base_loss, normalized=normalized)
        rce(train_mode=DataMode.NOISY, loss_mode=LossMode.BOUND, base_loss=base_loss, normalized=normalized)

    for loss in losses:
        run_all(base_loss=loss, normalized=False)
        run_all(base_loss=loss, normalized=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("runner")
    parser.add_argument("base_path", type=str)
    parser.add_argument("--result-path", type=str, default="results.json")
    parser.add_argument("--losses", type=str, default="bce,sqh,cce")
    args = parser.parse_args()

    base_path = args.base_path
    source_train_data = pickle.loads(Path(f"{base_path}/original-train.pkl").read_bytes())
    source_test_data = pickle.loads(Path(f"{base_path}/original-test.pkl").read_bytes())
    print("Loaded raw data")

    losses = [x.strip() for x in args.losses.split(",")]

    result_path = Path(args.result_path)
    if result_path.exists():
        result_cache = json.loads(result_path.read_text())
    else:
        result_cache = {}
    for i in range(5):
        run_all_settings(index=i, losses=losses)
