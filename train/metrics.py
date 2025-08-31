import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import Metric


def topk_as_matrix(data, k):
    """
    Calculates the top_k among each entry in `data`, and returns a matrix with values in
    `{0, 1}`, which indicates for each entry whether it is in the topk.
    :param data: Data to rank.
    :param k: Number of top indices to consider.
    :return: A SparseTensor of the same (dense) shape as data.
    """

    with tf.name_scope("topk_as_matrix"):
        _, tops = tf.nn.top_k(data, k=k)        # tops: [BATCH x K]
        batch_size = tf.cast(tf.shape(data)[0], tf.int32, name="batch_size")
        batch_ids = tf.range(0, batch_size, dtype=tf.int32)
        batch_ids = tf.tile(batch_ids[:, None], (1, k), name="batch_ids")
        idx = tf.stack([batch_ids, tops], axis=-1)  # [B, K, 2]
        idx = tf.reshape(idx, (-1, 2), name="indices")

        result = tf.sparse.SparseTensor(tf.cast(idx, tf.int64), tf.ones(batch_size * k, dtype=tf.float32),
                                        tf.shape(data, out_type=tf.int64))
        return tf.sparse.reorder(result)


def sparse_dense_cwise_op(a: tf.sparse.SparseTensor, b: tf.Tensor, op: callable):
    with tf.name_scope("sparse_dense_cwise_op"):
        tf.debugging.assert_equal(tf.shape(a), tf.shape(b))
        gathered = tf.gather_nd(b, a.indices)
        return tf.sparse.SparseTensor(a.indices, op(a.values, gathered), a.dense_shape)


@tf.function
def _mul_sample_weight(x, s):
    if s is None:
        return x
    elif x.shape.rank == 2:
        return x * s[:, None]
    else:
        return x * s


class PrecisionAtK(Metric):
    """
    Calculates Precision@K, i.e. how often *all* top-k predictions are
    among the true labels. Given a score vector $y$, a threshold is chosen
    such that only $k$ labels are predicted as true, and the corresponding
    precision is calculated. This means that if the dataset contains examples
    with less than $k$ labels, no score function can reach a P@k of 1.
    """
    def __init__(self, k=1, label_weights=None, name=None, normalize: bool = False, **kwargs):
        """

        :param k: Number of highest ranking predictions to consider.
        :param label_weights: If supplied, these will be used to weight the contributions of different labels. Expects a
        vector of shape `(NUM_LABELS,)`. Typical use: Inverse propensities.
        :param normalize: If `label_weights` is given, then the resulting values may not lie in the interval `[0, 1]`.
        If `normalize` is `True`, then the metric will be divided by the highest possible value that could be achieved for any prediction,
        thus ensuring values in `[0, 1]`.
        :param name: A name for this metric. If `None` is given, defaults to `P_at_{k}`.
        """
        name = name or "P_at_{}".format(k)
        super().__init__(name=name, **kwargs)
        self._k = k
        self._correct = self.add_weight("NumCorrect", ())       # type: tf.Variable
        self._total = self.add_weight("NumTotal", ())           # type: tf.Variable
        if label_weights is not None:
            label_weights = tf.convert_to_tensor(label_weights, dtype=tf.float32)
        self._label_weights = label_weights
        self._normalize = normalize

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v.value()))

    def update_state(self, y_true, y_pred, sample_weight=None):
        # get data, ensure these are tensors, and verify shape constraints
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        shape_constraints = [(y_true, 'BL'), (y_pred, 'BL')]
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float32, name="sample_weight")
            shape_constraints.append((sample_weight, 'B'))

        tf.debugging.assert_shapes(shape_constraints)

        # apply the weights if necessary
        if self._label_weights is not None:
            y_true = tf.multiply(y_true, self._label_weights[None, :], name="weighted_y_true")

        # get the top k predictions and a matrix that determines where they are correct
        top_indices = topk_as_matrix(y_pred, self._k)
        is_correct = sparse_dense_cwise_op(top_indices, y_true, tf.multiply)

        total_correct = tf.sparse.reduce_sum(is_correct, axis=1)
        num_correct = tf.reduce_sum(_mul_sample_weight(total_correct, sample_weight), name="num_correct")

        self._correct.assign_add(num_correct)
        if self._label_weights is None or not self._normalize:
            self._add_total_unbiased(y_true, sample_weight)
        else:
            # this is the maximum score that could be achieved in this metric with the given weighting.
            # TODO address SAMPLE_WEIGHT
            val, _ = tf.nn.top_k(y_true, self._k, name="top_k_of_true")
            self._total.assign_add(tf.reduce_sum(val))

    def _add_total_unbiased(self, y_true: tf.Tensor, sample_weight):
        if sample_weight is None:
            self._total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32, name="batch_size") * self._k)
        else:
            self._total.assign_add(tf.reduce_sum(sample_weight, name="reduce_sample_weight") * self._k)

    def result(self):
        return tf.math.divide_no_nan(self._correct.value(), self._total.value())


class RecallAtK(Metric):
    """
    Calculates Recall@K, i.e. how often many of the ground-truth labels are
    covered by the top-k predictions. Given a score vector $y$, a threshold is chosen
    such that exactly $k$ labels are predicted as true, and the corresponding
    recalls is calculated. This means that if the dataset contains examples
    with more than $k$ labels, no score function can reach a Rec@k of 1.

    There are two different ways of interpreting this:
     1) Calculate the recall for each sample individually, and then report the average. This seems
        to be the more prevalent approach, e.g. in [Lapin]. In this setting, few samples with
        number of labels >> k do not change the recall metric by much. In this setting, we use the convention
        0/0=1, i.e. examples without any ground-truth data are considered as 100% recalled.
     2) Accumulate true positives and false negatives over the entire training set, and
        calculate the recall as the ratio. In this setting, few samples that have large
        number of labels >> k can cause a strong decrease of the recall metric.
    """
    def __init__(self, k=1, name=None, num_labels=None, **kwargs):
        """
        :param k: Number of highest ranking predictions to consider.
        :param name: A name for this metric. If `None` is given, defaults to `R_at_{k}`.
        """
        name = name or "R_at_{}".format(k)
        super().__init__(name=name, **kwargs)

        self._k = k
        self._num_labels = num_labels
        self._correct = self.add_weight("NumCorrect", ())  # type: tf.Variable
        self._total = self.add_weight("NumTotal", ())  # type: tf.Variable

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        y_true.shape.assert_has_rank(2)
        y_pred.shape.assert_has_rank(2)

        top_indices = topk_as_matrix(y_pred, self._k)

        shape_constraints = [(y_true, 'BL'), (y_pred, 'BL')]
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
            tf.debugging.assert_non_negative(sample_weight)
            shape_constraints.append((sample_weight, 'B'))

        tf.debugging.assert_shapes(shape_constraints)

        is_correct = sparse_dense_cwise_op(top_indices, y_true, tf.multiply)

        is_correct = tf.sparse.reduce_sum(is_correct, axis=1)
        num_true = tf.reduce_sum(y_true, axis=1)

        no_true = tf.equal(num_true, 0)
        div_to_1 = tf.where(no_true, tf.ones_like(is_correct), is_correct / num_true)
        num_correct = tf.reduce_sum(_mul_sample_weight(div_to_1, sample_weight))
        tf.debugging.assert_non_negative(num_correct, "negative number of correct labels")
        self._correct.assign_add(num_correct)
        if sample_weight is None:
            self._total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        else:
            self._total.assign_add(tf.reduce_sum(sample_weight))

    def result(self):
        return tf.math.divide_no_nan(self._correct.value(), self._total.value())

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v.value()))
