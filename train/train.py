import dataclasses
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from losses import build_loss, LossMode, DataMode
import mlds
import metrics


@dataclasses.dataclass
class ExperimentData:
    propensities: "np.ndarray"
    clean_train: "tf.data.Dataset"
    noisy_train: "tf.data.Dataset"
    clean_val: "tf.data.Dataset"
    noisy_val: "tf.data.Dataset"
    clean_test: "tf.data.Dataset"
    noisy_test: "tf.data.Dataset"


def run_experiment(data: ExperimentData, train_mode: str,
                   loss_mode: str, base_loss: str, normalized: bool, l2_reg: float = 1e-6,
                   num_epochs: int = 20, pretraining=0):
    num_features = data.clean_test.element_spec[0].shape[0]
    num_labels = data.clean_test.element_spec[1].shape[0]
    propensities = data.propensities

    if train_mode == DataMode.CLEAN:
        train_data = data.clean_train
        val_data = data.clean_val
    else:
        train_data = data.noisy_train
        val_data = data.noisy_val

    model = keras.Sequential([keras.layers.InputLayer(input_shape=(num_features,), sparse=True),
                              keras.layers.Dense(num_labels, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))])

    train_loss = build_loss(base_loss, loss_mode, normalized, propensities)
    vanilla_loss = build_loss(base_loss, LossMode.VANILLA, normalized, propensities)
    vanilla_loss.name = "vanilla_loss"
    unbiased_loss = build_loss(base_loss, LossMode.UNBIASED, normalized, propensities)
    unbiased_loss.name = "unbiased_loss"

    metric_list = []
    inv_prop = 1.0 / propensities
    for k in (1, 3, 5):
        metric_list.append(metrics.PrecisionAtK(k=k, name=f"P@{k}"))
        metric_list.append(metrics.PrecisionAtK(k=k, label_weights=inv_prop, normalize=False, name=f"PSP@{k}"))
        metric_list.append(metrics.RecallAtK(k=k, name=f"R@{k}"))
    metric_list.append(vanilla_loss)
    metric_list.append(unbiased_loss)

    optimizer = keras.optimizers.Adam(1e-4)

    def schedule(epoch, lr):
        if epoch <= 15:
            return 1e-4
        else:
            return 1e-5

    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

    if pretraining > 0:
        model.compile(optimizer, unbiased_loss)
        model.fit(train_data.shuffle(8192 * 64).batch(512), epochs=pretraining)

    model.compile(optimizer, train_loss, metrics=metric_list)
    model.fit(train_data.shuffle(8192*64).batch(512), validation_data=val_data.batch(2048), epochs=num_epochs,
              callbacks=[lr_callback])

    def evaluate_to_dict(data, mode):
        values = model.evaluate(data.batch(2048))
        # need to skip the first value here, because that is the implicitly added loss metric
        result = {m.name: v for m, v in zip(metric_list, values[1:])}
        if mode == DataMode.CLEAN:
            result["loss"] = result["vanilla_loss"]
        else:
            result["loss"] = result["unbiased_loss"]
        return result

    results = {"clean-train": evaluate_to_dict(data.clean_train, DataMode.CLEAN),
               "noisy-train": evaluate_to_dict(data.noisy_train, DataMode.NOISY),
               "clean-test": evaluate_to_dict(data.clean_test, DataMode.CLEAN),
               "noisy-test": evaluate_to_dict(data.noisy_test, DataMode.NOISY),
               "clean-val": evaluate_to_dict(data.clean_val, DataMode.CLEAN),
               "noisy-val": evaluate_to_dict(data.noisy_val, DataMode.NOISY)}
    return results


def to_tf_pipeline(data: mlds.SparseDataset, propensities: Optional[np.ndarray] = None):
    features, labels = mlds.to_tensors(data)
    features = tf.data.Dataset.from_tensor_slices(features)
    dense_labels = tf.sparse.to_dense(labels)
    if propensities is not None:
        dense_labels = dense_labels * tf.cast(tf.less(tf.random.uniform(shape=tf.shape(dense_labels)), propensities[None, :]), tf.float32)
    labels = tf.data.Dataset.from_tensor_slices(dense_labels)
    dataset = tf.data.Dataset.zip((features, labels))
    return dataset


def generate_datasets(train: mlds.TextLineDataset, test: mlds.TextLineDataset, propensities: np.ndarray):
    train_ds, val_ds = mlds.split_dataset(train, 0.7)

    train_ds = mlds.to_sparse(train_ds)
    val_ds = mlds.to_sparse(val_ds)
    test_ds = mlds.to_sparse(test)

    clean = to_tf_pipeline
    noisy = lambda x: to_tf_pipeline(x, propensities)
    return ExperimentData(
        propensities=propensities,
        clean_train=clean(train_ds),
        clean_val=clean(val_ds),
        clean_test=clean(test_ds),
        noisy_train=noisy(train_ds),
        noisy_val=noisy(val_ds),
        noisy_test=noisy(test_ds)
    )
