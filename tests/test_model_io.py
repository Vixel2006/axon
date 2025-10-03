import pytest
import os
import numpy as np
from fajr.nn.pipeline import Pipeline
from fajr.nn.linear import Linear
from fajr.nn.activations import ReLU, Sigmoid
from fajr.functions import from_data
from fajr.utils.model_io import save_model, load_model


class TestModelIO:

    def setup_method(self):
        self.test_dir = "test_models"
        os.makedirs(self.test_dir, exist_ok=True)
        self.model_path = os.path.join(self.test_dir, "test_model.pkl")

    def teardown_method(self):
        if os.path.exists(self.test_dir):
            for f in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, f))
            os.rmdir(self.test_dir)

    def create_simple_model(self):
        linear1 = Linear(2, 3)
        relu = ReLU()
        linear2 = Linear(3, 1)
        model = Pipeline(linear1, relu, linear2)
        return model

    def test_save_model(self):
        model = self.create_simple_model()
        # Perform a forward pass to ensure all tensors are realized
        dummy_input = from_data((1, 2), np.array([[0.0, 0.0]], dtype=np.float32))
        _ = model(dummy_input)
        save_model(model, self.model_path)
        # The file should exist if pickling succeeded
        assert os.path.exists(self.model_path)

    def test_load_model(self):
        model = self.create_simple_model()
        # Modify parameters to ensure loading restores them correctly
        model.layers[0].W = from_data(
            model.layers[0].W.shape, np.ones(model.layers[0].W.shape, dtype=np.float32)
        )
        model.layers[0].B = from_data(
            model.layers[0].B.shape, np.ones(model.layers[0].B.shape, dtype=np.float32)
        )
        model.layers[2].W = from_data(
            model.layers[2].W.shape, np.zeros(model.layers[2].W.shape, dtype=np.float32)
        )
        model.layers[2].B = from_data(
            model.layers[2].B.shape, np.zeros(model.layers[2].B.shape, dtype=np.float32)
        )

        save_model(model, self.model_path)
        loaded_model = load_model(self.model_path)

        assert isinstance(loaded_model, Pipeline)
        assert len(loaded_model.layers) == 3
        assert isinstance(loaded_model.layers[0], Linear)
        assert isinstance(loaded_model.layers[1], ReLU)
        assert isinstance(loaded_model.layers[2], Linear)

        # Verify parameters are restored
        assert np.array_equal(
            loaded_model.layers[0].W.data, np.ones(model.layers[0].W.shape)
        )
        assert np.array_equal(
            loaded_model.layers[0].B.data, np.ones(model.layers[0].B.shape)
        )
        assert np.array_equal(
            loaded_model.layers[2].W.data, np.zeros(model.layers[2].W.shape)
        )
        assert np.array_equal(
            loaded_model.layers[2].B.data, np.zeros(model.layers[2].B.shape)
        )

    def test_model_round_trip(self):
        original_model = self.create_simple_model()
        # Set specific weights for predictable output
        original_model.layers[0].W = from_data(
            (3, 2), np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        )
        original_model.layers[0].B = from_data(
            (1, 3), np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        )
        original_model.layers[2].W = from_data(
            (1, 3), np.array([[0.7, 0.8, 0.9]], dtype=np.float32)
        )
        original_model.layers[2].B = from_data(
            (1, 1), np.array([[0.0]], dtype=np.float32)
        )

        input_np = np.array([[1.0, 2.0]], dtype=np.float32)
        input_tensor = from_data(input_np.shape, input_np)

        original_output = original_model(input_tensor).realize().data

        save_model(original_model, self.model_path)
        loaded_model = load_model(self.model_path)

        loaded_output = loaded_model(input_tensor).realize().data

        assert np.allclose(original_output, loaded_output)
