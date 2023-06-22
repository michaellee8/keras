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
"""Tests for Binary F-score metrics."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.metrics import binary_f_score_metrics
from keras.testing_infra import test_utils


@test_utils.run_v2_only
class BinaryFBetaScoreTest(parameterized.TestCase, tf.test.TestCase):
    def _run_test(
        self,
        y_true,
        y_pred,
        beta,
        threshold,
        reference_result,
    ):
        y_true = tf.constant(y_true, dtype="float32")
        y_pred = tf.constant(y_pred, dtype="float32")
        fbeta = binary_f_score_metrics.BinaryFBetaScore(
            beta=beta, threshold=threshold
        )
        fbeta.update_state(y_true, y_pred)
        result = fbeta.result().numpy()
        self.assertAllClose(result, reference_result, atol=1e-6)

    def test_config(self):
        fbeta_obj = binary_f_score_metrics.BinaryFBetaScore(
            beta=0.5, threshold=0.3
        )
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.threshold, 0.3)
        self.assertEqual(fbeta_obj.dtype, tf.float32)

        # Check save and restore config
        fbeta_obj2 = binary_f_score_metrics.BinaryFBetaScore.from_config(
            fbeta_obj.get_config()
        )
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.threshold, 0.3)
        self.assertEqual(fbeta_obj2.dtype, tf.float32)

    def test_correctness(self):
        self._run_test(
            [[1], [0], [0], [1], [1]],
            [[0.2], [0.2], [0.7], [0.6], [0.55]],
            2.0,
            0.5,
            0.6666666666666666666,
        )


@test_utils.run_v2_only
class BinaryF1ScoreTest(tf.test.TestCase):
    def test_config(self):
        f1_obj = binary_f_score_metrics.BinaryF1Score(threshold=0.5)
        config = f1_obj.get_config()
        self.assertNotIn("beta", config)

        # Check save and restore config
        f1_obj = binary_f_score_metrics.BinaryF1Score.from_config(config)
        self.assertEqual(f1_obj.dtype, tf.float32)

    def test_correctness(self):
        f1 = binary_f_score_metrics.BinaryF1Score(threshold=0.5)
        fbeta = binary_f_score_metrics.BinaryFBetaScore(beta=1.0, threshold=0.5)

        y_true = [[1], [0], [0], [1], [1]]
        y_pred = [[0.2], [0.2], [0.7], [0.6], [0.55]]

        fbeta.update_state(y_true, y_pred)
        f1.update_state(y_true, y_pred)
        self.assertAllClose(
            fbeta.result().numpy(), f1.result().numpy(), atol=1e-6
        )


if __name__ == "__main__":
    tf.test.main()
