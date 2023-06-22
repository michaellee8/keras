# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""F-Score metrics for binary classification."""

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric


# Adapted from original F-score implementation.
@keras_export("keras.metrics.BinaryFBetaScore")
class BinaryFBetaScore(base_metric.Metric):
    """Computes F-Beta score for a binary classification problem.

    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It is designed for binary classification only.

    It is defined as:

    ```python
    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```

    Args:
        beta: Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Default value is 1.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        Binary F-Beta Score: float.

    Example:

    >>> metric = tf.keras.metrics.BinaryFBetaScore(beta=2.0, threshold=0.5)
    >>> y_true = np.array([[1], [0], [0], [1], [1]], np.int32)
    >>> y_pred = np.array([[0.2], [0.2], [0.7], [0.6], [0.55]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    0.66666666666667
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        threshold,
        beta=1.0,
        name="binary_fbeta_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        if beta <= 0.0:
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be > 0. "
                f"Received: beta={beta}"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should be a Python float. "
                    f"Received: threshold={threshold} "
                    f"of type '{type(threshold)}'"
                )
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should verify 0 < threshold <= 1. "
                    f"Received: threshold={threshold}"
                )
        else:
            raise ValueError(
                "Invalid `threshold` argument value. "
                "It should not be None. "
                f"Received: threshold={threshold}"
            )

        self.beta = beta
        self.threshold = threshold
        self.built = False

    def build(self, y_true_shape, y_pred_shape):
        if (
            len(y_pred_shape) != 2
            or len(y_true_shape) != 2
            or y_pred_shape[1] != 1
            or y_true_shape[1] != 1
        ):
            raise ValueError(
                "BinaryFBetaScore expects 2D inputs with shape "
                "(batch_size, 1). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "BinaryFBetaScore expects 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )

        def _add_zeros_weight(name):
            return self.add_weight(
                name,
                shape=(1,),
                initializer="zeros",
                dtype=self.dtype,
            )

        self.true_positives = _add_zeros_weight("true_positives")
        self.false_positives = _add_zeros_weight("false_positives")
        self.false_negatives = _add_zeros_weight("false_negatives")
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        if not self.built:
            self.build(y_true.shape, y_pred.shape)

        y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        self.true_positives.assign_add(
            tf.reduce_sum(y_pred * y_true, axis=0)
        )
        self.false_positives.assign_add(
            tf.reduce_sum(y_pred * (1 - y_true), axis=0)
        )
        self.false_negatives.assign_add(
            tf.reduce_sum((1 - y_pred) * y_true, axis=0)
        )

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))


@keras_export("keras.metrics.BinaryF1Score")
class BinaryF1Score(BinaryFBetaScore):
    r"""Computes F-1 Score for a binary classification problem.

    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It is designed for binary classification only.

    It is defined as:

    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```

    Args:
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-1 Score: float.

    Example:

    >>> metric = tf.keras.metrics.F1Score(threshold=0.5)
    >>> y_true = np.array([[1], [0], [0], [1], [1]], np.int32)
    >>> y_pred = np.array([[0.2], [0.2], [0.7], [0.6], [0.55]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    0.6666666666667
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        threshold,
        name="f1_score",
        dtype=None,
    ):
        super().__init__(
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
