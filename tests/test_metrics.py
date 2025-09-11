import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.metrics.mse import mse
from idrak.metrics.bce import bce
from idrak.metrics.cce import categorical_cross_entropy
from idrak.functions import log_softmax, sum, mean # Import for manual calculation

def test_mse_mean_reduction():
    pred_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    truth_np = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    pred_idrak = Tensor(data=pred_np, shape=(2, 2))
    truth_idrak = Tensor(data=truth_np, shape=(2, 2))

    mse_idrak = mse(pred_idrak, truth_idrak)
    mse_idrak.realize()

    expected_mse_np = np.mean((pred_np - truth_np) ** 2)

    assert mse_idrak.shape == (1,)
    assert np.allclose(mse_idrak.data.item(), expected_mse_np, rtol=1e-4, atol=1e-6)

def test_bce_mean_reduction():
    pred_np = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
    truth_np = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    pred_idrak = Tensor(data=pred_np, shape=(2, 2))
    truth_idrak = Tensor(data=truth_np, shape=(2, 2))

    bce_idrak = bce(pred_idrak, truth_idrak)
    bce_idrak.realize()

    # Manual BCE calculation
    # -(truth * log(pred)) - ((1 - truth) * log(1 - pred))
    term1 = -truth_np * np.log(pred_np)
    term2 = -(1 - truth_np) * np.log(1 - pred_np)
    expected_bce_np = np.mean(term1 + term2)

    assert bce_idrak.shape == (1,)
    assert np.allclose(bce_idrak.data.item(), expected_bce_np, rtol=1e-4, atol=1e-6)

def test_categorical_cross_entropy():
    logits_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    truth_np = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) # One-hot encoded

    logits_idrak = Tensor(data=logits_np, shape=(2, 3))
    truth_idrak = Tensor(data=truth_np, shape=(2, 3))

    cce_idrak = categorical_cross_entropy(logits_idrak, truth_idrak)
    cce_idrak.realize()

    # Manual CCE calculation
    # 1. log_softmax
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    log_probs_np = logits_np - np.max(logits_np, axis=1, keepdims=True) - np.log(np.sum(exp_logits, axis=1, keepdims=True))

    # 2. nll = -sum(truth * log_probs, dim=1)
    nll_np = -np.sum(truth_np * log_probs_np, axis=1)

    # 3. mean(nll)
    expected_cce_np = np.mean(nll_np)

    assert cce_idrak.shape == (1,)
    assert np.allclose(cce_idrak.data.item(), expected_cce_np, rtol=1e-4, atol=1e-6)
