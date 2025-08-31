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
    train_prop: "np.ndarray"
    test_prop: "np.ndarray"
    noisy_train: "tf.data.Dataset"
    noisy_val: "tf.data.Dataset"
    noisy_test: "tf.data.Dataset"


def set_name(obj, name):
    try:
        obj.name = name
    except AttributeError:
        obj._name = name


def run_experiment(data: ExperimentData, train_mode: str,
                   loss_mode: str, base_loss: str, normalized: bool, l2_reg: float = 1e-6,
                   num_epochs: int = 50, pretraining=0):
    num_features = data.noisy_test.element_spec[0].shape[0]
    num_labels = data.noisy_test.element_spec[1].shape[0]

    train_data = data.noisy_train
    val_data = data.noisy_val

    model = keras.Sequential([keras.layers.InputLayer(input_shape=(num_features,), sparse=True),
                              keras.layers.Dense(num_labels, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))])

    train_loss = build_loss(base_loss, loss_mode, normalized, data.train_prop)
    vanilla_loss = build_loss(base_loss, LossMode.VANILLA, normalized, data.train_prop)
    set_name(vanilla_loss, "vanilla_loss")
    unbiased_loss = build_loss(base_loss, LossMode.UNBIASED, normalized, data.train_prop)
    set_name(unbiased_loss, "unbiased_loss")
    unbiased_test_loss = build_loss(base_loss, LossMode.UNBIASED, normalized, data.test_prop)
    set_name(unbiased_test_loss, "unbiased_test_loss")

    metric_list = []
    for k in (1, 3, 5):
        metric_list.append(metrics.PrecisionAtK(k=k, name=f"P@{k}"))
        metric_list.append(metrics.PrecisionAtK(k=k, label_weights=1.0 / data.train_prop, normalize=False, name=f"trPSP@{k}"))
        metric_list.append(metrics.PrecisionAtK(k=k, label_weights=1.0 / data.test_prop, normalize=False, name=f"tsPSP@{k}"))
        metric_list.append(metrics.RecallAtK(k=k, name=f"R@{k}"))
    metric_list.append(vanilla_loss)
    metric_list.append(unbiased_loss)
    metric_list.append(unbiased_test_loss)

    if base_loss == "cce":
        base_lr = 2e-3
        batch_size = 512
    else:
        base_lr = 2e-3
        batch_size = 128
        #assert False
    optimizer = keras.optimizers.Adam(base_lr)

    def schedule(epoch, lr):
        if epoch <= 45:
            return base_lr
        else:
            return 5e-4

    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

    if pretraining > 0:
        model.compile(optimizer, unbiased_loss)
        model.fit(train_data.shuffle(8192 * 64).batch(512), epochs=pretraining)

    model.compile(optimizer, train_loss, metrics=metric_list, run_eagerly=True)
    tf.config.run_functions_eagerly(True)
    model.fit(train_data.shuffle(8192*64).batch(batch_size), validation_data=val_data.batch(2048), epochs=num_epochs,
              callbacks=[lr_callback], verbose=2)

    def evaluate_to_dict(data, mode):
        values = model.evaluate(data.batch(2048))
        # need to skip the first value here, because that is the implicitly added loss metric
        result = {m.name: v for m, v in zip(metric_list, values[1:])}
        if mode == DataMode.CLEAN:
            result["loss"] = result["vanilla_loss"]
        else:
            result["loss"] = result["unbiased_loss"]
        return result

    results = {"noisy-train": evaluate_to_dict(data.noisy_train, DataMode.NOISY),
               "noisy-val": evaluate_to_dict(data.noisy_val, DataMode.NOISY),
               "noisy-test": evaluate_to_dict(data.noisy_test, DataMode.NOISY),}
    all_preds = []
    for pred in model.predict(data.noisy_test.batch(2048)):
        all_preds.append(pred)
    np.savez("predictions.npz", np.asarray(all_preds))
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


def generate_datasets(train: mlds.TextLineDataset, test: mlds.TextLineDataset, train_prop: np.ndarray, test_prop: np.ndarray):
    train_ds, val_ds = mlds.split_dataset(train, 0.7)

    train_ds = mlds.to_sparse(train_ds)
    val_ds = mlds.to_sparse(val_ds)
    test_ds = mlds.to_sparse(test)

    return ExperimentData(
        train_prop=train_prop,
        test_prop=test_prop,
        noisy_train=to_tf_pipeline(train_ds),
        noisy_val=to_tf_pipeline(val_ds),
        noisy_test=to_tf_pipeline(test_ds)
    )
