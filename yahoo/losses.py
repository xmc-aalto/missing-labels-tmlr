import tensorflow as tf


@tf.function
def ova_base_loss(base_loss: callable, y_true: tf.Tensor, y_pred: tf.Tensor):
    pos = base_loss(tf.ones_like(y_pred), y_pred)
    neg = base_loss(tf.zeros_like(y_pred), y_pred)
    return y_true * (pos - neg) + neg


def make_ova_loss(base_loss: callable):
    """
    Makes the OvA decomposition for `base_loss` as the underlying binary loss.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        return ova_base_loss(base_loss, y_true, y_pred)
    return loss


@tf.function
def squared_hinge_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    t = 2.0 * y_true - 1.0
    return tf.square(tf.nn.relu(1 - t * y_pred))


def make_normalized_ova(base_loss: callable):
    @tf.function
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        y_true_norm = tf.math.xdivy(y_true, tf.reduce_sum(y_true, axis=-1, keepdims=True))
        return ova_base_loss(base_loss, y_true_norm, y_pred)

    return loss


def make_naive_ova_normalized(base_loss: callable, propensities):
    p = propensities[None, :]

    @tf.function
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        scaled_y = y_true / p
        weighted_true = tf.reduce_sum(scaled_y / p, axis=-1, keepdims=True) + 1.0 - scaled_y
        y_true_norm = tf.math.xdivy(scaled_y, weighted_true)

        return ova_base_loss(base_loss, y_true_norm, y_pred)

    return loss


def make_unbiased_ova(base_loss: callable, propensities):
    p = propensities[None, :]

    def loss(y_true, y_pred):
        return ova_base_loss(base_loss, y_true / p, y_pred)
    return loss


def make_upper_bound_ova(base_loss: callable, propensities):
    p = propensities[None, :]

    @tf.function
    def loss(y_true, y_pred):
        pos = base_loss(tf.ones_like(y_pred), y_pred)
        neg = base_loss(tf.zeros_like(y_pred), y_pred)
        return y_true * (2.0 / p - 1) * pos + (1.0 - y_true) * neg
    return loss


def eutms_factors_for_present(y_true, pattern, propensity):
    subsample = y_true * pattern
    # if subsample == 1, then present = p, else present = 1
    present = propensity[None, :] * subsample + (1 - subsample)
    return 1.0 / tf.reduce_prod(present, axis=1, keepdims=True)


def eutms_factors_for_absent(y_true, pattern, propensity):
    rejected = y_true * (1.0 - pattern)
    # if rejected == 1, then absent = 1 - 1/p, else 1
    absent = (1.0 - 1.0 / propensity[None, :]) * rejected + (1 - rejected)
    return tf.reduce_prod(absent, axis=1, keepdims=True)


def evaluate_unbiased_true_multilabel_sample(base_loss, y_true, y_pred, propensities, pattern, n_samples, clip: float):
    tf.debugging.assert_shapes([(y_true, 'BL'), (y_pred, 'BL'), (pattern, 'BL'), (propensities, 'L')])

    # decide at which places we want to "forget" labels
    subsample = y_true * pattern

    # the code is written so that it can handle per-label losses. If the base loss function doesn't have per
    # label data, we insert a dimension to prevent errors due to broadcasting below
    base_loss_value = base_loss(subsample, y_pred)
    if len(base_loss_value.shape) == 1:
        base_loss_value = tf.expand_dims(base_loss_value, axis=-1)

    present = eutms_factors_for_present(y_true, pattern, propensities)
    absent = eutms_factors_for_absent(y_true, pattern, propensities)
    scale = present * absent
    tf.print(tf.reduce_sum(y_true, axis=-1, keepdims=True))
    factor = (tf.math.pow(2.0, tf.reduce_sum(y_true, axis=-1, keepdims=True)) / n_samples) * scale
    tf.debugging.assert_equal(tf.shape(base_loss_value)[0], tf.shape(y_true)[0], "Base loss has to preserve batch size")
    return factor * base_loss_value #, -1e20, 1e20)


def make_unbiased_true_multilabel(base_loss: callable, propensities, n_samples=1, clip: float = 100.0):
    @tf.function
    def loss(y_true, y_pred):
        y_true_expanded = tf.repeat(y_true, repeats=n_samples, axis=0)
        y_pred_expanded = tf.repeat(y_pred, repeats=n_samples, axis=0)
        pattern = tf.cast(tf.less(tf.random.uniform(shape=tf.shape(y_true_expanded)), 0.5), tf.float32)
        attempts = evaluate_unbiased_true_multilabel_sample(
            base_loss, y_true_expanded, y_pred_expanded, propensities, pattern=pattern, n_samples=n_samples, clip=clip)
        attempts = tf.reshape(attempts, tf.concat([(-1, n_samples), tf.shape(attempts)[1:]], axis=0))
        return tf.clip_by_value(tf.reduce_sum(attempts, axis=1, keepdims=False), -clip, clip)
    return loss


def _sum_over_positives(values, batch_ids, batch_size):
    summed = tf.math.segment_sum(values, batch_ids)
    padding = batch_size - tf.shape(summed)[0]
    return tf.pad(summed, [[0, padding]])


def make_pal_loss(multiclass_loss):
    """
    Makes the PaL decomposition for `multiclass_loss`. The `multiclass_loss` function
    expects a one-hot vector as ground-truth and a prediction vector as second argument.
    """
    @tf.function
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        positives = tf.where(tf.greater(y_true, 0.5))
        one_hots = tf.one_hot(positives[:, 1], depth=tf.shape(y_true)[1])
        ground_truths = tf.gather(y_pred, positives[:, 0])
        losses = multiclass_loss(one_hots, ground_truths)
        return _sum_over_positives(losses, positives[:, 0], tf.shape(y_true)[0])

    return loss


def make_normalized_pal(multiclass_loss: callable):
    @tf.function
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        batch_size = tf.shape(y_true)[0]
        positives = tf.where(tf.greater(y_true, 0.5))
        one_hots = tf.one_hot(positives[:, 1], depth=tf.shape(y_true)[1])
        ground_truths = tf.gather(y_pred, positives[:, 0])
        num_true = tf.reduce_sum(y_true, axis=-1)
        weights = tf.math.divide_no_nan(1.0, tf.gather(num_true, positives[:, 0]))
        losses = weights * multiclass_loss(one_hots, ground_truths)
        return _sum_over_positives(losses, positives[:, 0], batch_size)

    return loss


def make_unbiased_pal_loss(multiclass_loss: callable, propensities):
    """
    Makes the PaL decomposition for `multiclass_loss`. The `multiclass_loss` function
    expects a one-hot vector as ground-truth and a prediction vector as second argument.
    """
    inv_p = 1.0 / propensities

    @tf.function
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        batch_size = tf.shape(y_true)[0]
        positives = tf.where(tf.greater(y_true, 0.5))
        # positives contains pairs (batch-id, label)
        one_hots = tf.one_hot(positives[:, 1], depth=tf.shape(y_true)[1])
        weights = tf.gather(tf.cast(inv_p, tf.float32), positives[:, 1])
        ground_truths = tf.gather(y_pred, positives[:, 0])
        losses = multiclass_loss(one_hots, ground_truths)
        return _sum_over_positives(weights * losses, positives[:, 0], batch_size)

    return loss


def make_normalized_pal_bound(multiclass_loss: callable, propensities):
    inv_p = 1.0 / propensities

    @tf.function
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        batch_size = tf.shape(y_true)[0]
        positives = tf.where(tf.greater(y_true, 0.5))
        one_hots = tf.one_hot(positives[:, 1], depth=tf.shape(y_true)[1])
        ground_truths = tf.gather(y_pred, positives[:, 0])

        pos_weights = tf.gather(tf.cast(inv_p, tf.float32), positives[:, 1])
        weighted_true = tf.reduce_sum(y_true * inv_p, axis=-1)
        denominator = tf.gather(weighted_true, positives[:, 0]) - pos_weights + 1.0
        weights = tf.math.divide_no_nan(pos_weights, denominator)
        tf.debugging.assert_non_negative(weights)
        losses = weights * multiclass_loss(one_hots, ground_truths)
        return _sum_over_positives(losses, positives[:, 0], batch_size)

    return loss


def get_base_loss(loss: str):
    if loss in ["softmax_cross_entropy", "categorical_cross_entropy", "cce"]:
        return LossType.MULTICLASS, tf.nn.softmax_cross_entropy_with_logits
    elif loss in ["sigmoid_cross_entropy", "binary_cross_entropy", "bce"]:
        return LossType.BINARY, tf.nn.sigmoid_cross_entropy_with_logits
    elif loss in ["sqh", "squared_hinge"]:
        return LossType.BINARY, squared_hinge_loss


class DataMode:
    CLEAN: str = "clean"
    NOISY: str = "noisy"


class LossType:
    BINARY: str = "binary"
    MULTICLASS: str = "multiclass"


class LossMode:
    VANILLA: str = "vanilla"
    UNBIASED: str = "unbiased"
    BOUND: str = "bound"


def build_loss(loss: str, loss_mode: str, normalized: bool, propensities: "np.ndarray", n_samples: int = 32):
    loss_kind, base_loss = get_base_loss(loss.lower())
    if loss_mode == LossMode.VANILLA:
        if loss_kind == LossType.MULTICLASS:
            if normalized:
                return make_normalized_pal(base_loss)
            else:
                return make_pal_loss(base_loss)
        elif loss_kind == LossType.BINARY:
            if normalized:
                return make_normalized_ova(base_loss)
            else:
                return make_ova_loss(base_loss)
        else:
            raise NotImplementedError()
    elif loss_mode == LossMode.UNBIASED:
        if loss_kind == LossType.MULTICLASS:
            if normalized:
                return make_unbiased_true_multilabel(make_normalized_pal(base_loss), propensities, n_samples=n_samples)
            else:
                return make_unbiased_pal_loss(base_loss, propensities)
        elif loss_kind == LossType.BINARY:
            if normalized:
                return make_unbiased_true_multilabel(make_normalized_ova(base_loss), propensities, n_samples=n_samples)
            else:
                return make_unbiased_ova(base_loss, propensities)
        else:
            raise NotImplementedError()
    elif loss_mode == LossMode.BOUND:
        if normalized:
            if loss_kind == LossType.MULTICLASS:
                return make_normalized_pal_bound(base_loss, propensities)
            elif loss_kind == LossType.BINARY:
                return make_naive_ova_normalized(base_loss, propensities)
            raise NotImplementedError()
        if loss_kind == LossType.MULTICLASS:
            return make_unbiased_pal_loss(base_loss, propensities)
        elif loss_kind == LossType.BINARY:
            return make_upper_bound_ova(base_loss, propensities)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
