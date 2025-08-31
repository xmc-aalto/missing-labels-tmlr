import pytest
from losses import *
import numpy as np


def test_ova():
    gt = tf.constant([[1, 0, 1], [0,  0, 1], [0, 0, 0]], dtype=tf.float32)
    pr = tf.constant([[3, 2, 1], [1, -1, 2], [0, 1, 1]], dtype=tf.float32)

    ova_loss = make_ova_loss(tf.nn.sigmoid_cross_entropy_with_logits)

    ova = ova_loss(gt, pr).numpy()
    ref = tf.nn.sigmoid_cross_entropy_with_logits(gt, pr).numpy()

    assert ova == pytest.approx(ref)


def test_pal_all_negative():
    """
    Without positives, all PAL variants should give 0.
    """
    propensity = tf.constant([0.9, 0.7, 0.6], dtype=tf.float32)
    gt = tf.constant([[0, 0, 0]], dtype=tf.float32)
    pr = tf.constant([[0, 1, 1]], dtype=tf.float32)

    loss = make_pal_loss(tf.nn.softmax_cross_entropy_with_logits)
    unbiased_loss = make_unbiased_true_multilabel(loss, propensity, n_samples=10)

    assert loss(gt, pr).numpy() == 0.0
    assert unbiased_loss(gt, pr).numpy() == 0.0

    loss = make_normalized_pal(tf.nn.softmax_cross_entropy_with_logits)
    unbiased_loss = make_unbiased_true_multilabel(loss, propensity, n_samples=10)
    bound = make_normalized_pal_bound(tf.nn.softmax_cross_entropy_with_logits, propensity)

    assert loss(gt, pr).numpy() == 0.0
    assert unbiased_loss(gt, pr).numpy() == 0.0
    assert bound(gt, pr).numpy() == 0.0


def test_generic_unbiased_factors():
    gt = tf.constant([[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=tf.float32)
    pattern = tf.constant([[1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=tf.float32)
    propensity = tf.constant([0.9, 0.7, 0.6], dtype=tf.float32)

    pos_factors = eutms_factors_for_present(gt, pattern, propensity).numpy()
    expect = [[1.0 / 0.9], [1.0 / 0.6], [1.0], [1.0 / 0.9], [1.0 / 0.7 / 0.6]]
    assert pos_factors == pytest.approx(np.array(expect))

    neg_factors = eutms_factors_for_absent(gt, pattern, propensity).numpy()
    expect = [[1 - 1.0/0.6], [1.0], [1.0], [1 - 1.0/0.7], [1.0]]
    assert neg_factors == pytest.approx(np.array(expect))


def test_unbiased_generic_ova():
    propensity = tf.constant([0.9, 0.7, 0.6], dtype=tf.float32)
    gt = tf.constant([[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=tf.float32)
    pr = tf.constant([[3, 2, 1], [1, -1, 2], [0, 1, 1], [1, 0, 2], [1, 1, -1]], dtype=tf.float32)

    loss = make_ova_loss(tf.nn.sigmoid_cross_entropy_with_logits)
    unbiased_loss = make_unbiased_true_multilabel(loss, propensity, n_samples=10)
    samples = []
    for sample in range(1000):
        keep = tf.cast(tf.less(tf.random.uniform(shape=tf.shape(gt)), propensity), tf.float32)
        samples.append(tf.reduce_sum(unbiased_loss(keep * gt, pr), axis=1).numpy())

    true_loss = tf.reduce_sum(loss(gt, pr), axis=1).numpy()
    print(true_loss)

    estimate = (tf.add_n(samples) / len(samples)).numpy()
    print(estimate)
    assert estimate == pytest.approx(true_loss, rel=0.1)


def test_unbiased_generic_pal():
    propensity = tf.constant([0.9, 0.7, 0.6], dtype=tf.float32)
    gt = tf.constant([[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=tf.float32)
    pr = tf.constant([[3, 2, 1], [1, -1, 2], [0, 1, 1], [1, 0, 2], [1, 1, -1]], dtype=tf.float32)

    loss = make_normalized_pal(tf.nn.softmax_cross_entropy_with_logits)
    unbiased_loss = make_unbiased_true_multilabel(loss, propensity, n_samples=10)
    samples = []
    for sample in range(1000):
        keep = tf.cast(tf.less(tf.random.uniform(shape=tf.shape(gt)), propensity), tf.float32)
        samples.append(tf.reduce_sum(unbiased_loss(keep * gt, pr), axis=1).numpy())

    true_loss = loss(gt, pr).numpy()

    estimate = (tf.add_n(samples) / len(samples)).numpy()
    assert estimate == pytest.approx(true_loss, rel=0.1)


def test_normalized_pal_bound():
    propensity = tf.constant([0.9, 0.7, 0.6], dtype=tf.float32)
    gt = tf.constant([[1, 0, 1], [0, 0, 1]], dtype=tf.float32)
    pr = tf.constant([[3, 2, 1], [1, -1, 2]], dtype=tf.float32)

    bound = make_normalized_pal_bound(tf.nn.softmax_cross_entropy_with_logits, propensity)

    l0 = tf.nn.softmax_cross_entropy_with_logits([1, 0, 0], pr[0])
    l1 = tf.nn.softmax_cross_entropy_with_logits([0, 0, 1], pr[0])
    l2 = tf.nn.softmax_cross_entropy_with_logits([0, 0, 1], pr[1])
    expected = [(1.0 / 0.9) / (1 + 1 / 0.6) * l0 + (1.0 / 0.6) / (1 + 1 / 0.9) * l1, 1.0 / 0.6 * l2]
    calculated = bound(gt, pr).numpy()
    assert calculated == pytest.approx(np.array(expected))
