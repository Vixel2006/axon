import numpy as np
import pytest
from axon.core.tensor import Tensor
from axon.functions import from_data, zeros, ones
from axon.metrics import bce, mse, nll_loss


class TestMetrics:
    def test_bce_basic(self):
        pred_data = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        truth_data = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        pred = from_data(pred_data.shape, pred_data)
        truth = from_data(truth_data.shape, truth_data)

        loss = bce(pred, truth)
        expected_loss = np.mean(
            -(truth_data * np.log(pred_data))
            - ((1 - truth_data) * np.log(1 - pred_data))
        )

        assert np.isclose(loss.realize().data, expected_loss)

    def test_bce_with_gradients(self):
        pred_data = np.array([0.1, 0.9], dtype=np.float32)
        truth_data = np.array([0.0, 1.0], dtype=np.float32)

        pred = from_data(pred_data.shape, pred_data, requires_grad=True)
        truth = from_data(truth_data.shape, truth_data, requires_grad=True)

        loss = bce(pred, truth)
        loss.backward()

        # Manual gradient calculation for BCE: dL/dp = -(y/p) + (1-y)/(1-p)
        expected_grad_pred = -(truth_data / pred_data) + (1 - truth_data) / (
            1 - pred_data
        )
        expected_grad_pred = expected_grad_pred / len(
            pred_data
        )  # Because of mean reduction

        assert np.allclose(pred.grad, expected_grad_pred)

    def test_mse_basic(self):
        pred_data = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        truth_data = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        pred = from_data(pred_data.shape, pred_data)
        truth = from_data(truth_data.shape, truth_data)

        loss = mse(pred, truth)
        expected_loss = np.mean((pred_data - truth_data) ** 2)

        assert np.isclose(loss.realize().data, expected_loss)

    def test_mse_with_gradients(self):
        pred_data = np.array([0.1, 0.9], dtype=np.float32)
        truth_data = np.array([0.0, 1.0], dtype=np.float32)

        pred = from_data(pred_data.shape, pred_data, requires_grad=True)
        truth = from_data(truth_data.shape, truth_data, requires_grad=True)

        loss = mse(pred, truth)
        loss.backward()

        # Manual gradient calculation for MSE: dL/dp = 2 * (p - y)
        expected_grad_pred = 2 * (pred_data - truth_data)
        expected_grad_pred = expected_grad_pred / len(pred_data)

        assert np.allclose(pred.grad, expected_grad_pred)

    def test_nll_loss_basic(self):
        predictions_data = np.array(
            [[0.2, 0.7, 0.1], [0.5, 0.3, 0.2]], dtype=np.float32
        )
        targets_data = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)

        predictions = from_data(predictions_data.shape, predictions_data)
        targets = from_data(targets_data.shape, targets_data)

        loss = nll_loss(predictions, targets)
        expected_loss = np.mean(-(predictions_data * targets_data))

        assert np.isclose(loss.realize().data, expected_loss)

    def test_nll_loss_with_gradients(self):
        predictions_data = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)
        targets_data = np.array([[0, 1, 0]], dtype=np.float32)

        predictions = from_data(
            predictions_data.shape, predictions_data, requires_grad=True
        )
        targets = from_data(targets_data.shape, targets_data, requires_grad=True)

        loss = nll_loss(predictions, targets)
        loss.backward()

        # Manual gradient calculation for NLL: dL/dp = -t
        expected_grad_predictions = -targets_data
        expected_grad_predictions = expected_grad_predictions / 3

        assert np.allclose(predictions.grad, expected_grad_predictions)
