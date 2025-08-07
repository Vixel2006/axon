import pytest
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nawah_api as nw

@nw.pipe
def relu_factory():
    def relu_fn(x):
        return nw.relu(x)
    return {"name": "ReLU", "params": {}, "fn": relu_fn}

@nw.pipe
def tanh_factory():
    def tanh_fn(x):
        return nw.tanh(x)
    return {"name": "Tanh", "params": {}, "fn": tanh_fn}


class TestLayers:
    """Tests for individual layer implementations."""

    def setup_method(self):
        """Initialize common inputs."""
        self.batch_size = 4
        self.in_features = 10
        self.out_features = 5
        self.seq_len = 8
        self.hidden_dim = 12

        self.linear_input = nw.randn([self.batch_size, self.in_features])
        self.rnn_input = nw.randn([self.batch_size, self.seq_len, self.in_features])

    def test_linear_output_shape(self):
        """Test that the linear layer produces the correct output shape."""
        layer = nw.layers.linear(self.in_features, self.out_features, has_bias=True)
        output = layer['fn'](self.linear_input)
        
        expected_shape = (self.batch_size, self.out_features)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert 'w' in layer['params']
        assert 'b' in layer['params']

    def test_linear_no_bias(self):
        """Test linear layer parameter registration without bias."""
        layer = nw.layers.linear(self.in_features, self.out_features, has_bias=False)
        assert 'b' not in layer['params'], "Bias parameter should not exist when has_bias=False"

    def test_linear_input_validation(self):
        """Test that linear layer raises errors for invalid input shapes."""
        layer = nw.layers.linear(self.in_features, self.out_features)
        with pytest.raises(ValueError):
            layer['fn'](nw.randn([self.batch_size, self.in_features + 1])) # Wrong feature dim
        with pytest.raises(ValueError):
            layer['fn'](nw.randn([self.batch_size])) # Wrong number of dims

    def test_rnn_output_shape(self):
        """Test that the RNN layer produces the correct output shape."""
        layer = nw.layers.rnn(self.in_features, self.hidden_dim)
        output = layer['fn'](self.rnn_input)
        
        expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    def test_rnn_with_initial_state(self):
        """Test RNN layer with a provided initial hidden state."""
        layer = nw.layers.rnn(self.in_features, self.hidden_dim)
        h0 = nw.zeros([self.batch_size, self.hidden_dim])
        output = layer['fn'](self.rnn_input, h0=h0)
        
        expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

class TestNet:
    """Tests for the main Net class functionality."""

    def setup_method(self):
        """Initialize a new Net for each test."""
        self.net = nw.Net()

    def test_add_module(self):
        """Test adding a module and check parameter registration."""
        self.net.add("fc1", nw.layers.linear(10, 5))
        assert "fc1" in self.net.modules
        assert "fc1.w" in self.net.params
        assert "fc1.b" in self.net.params
        assert len(self.net.buffers) == 0

    def test_add_existing_module_error(self):
        """Test that adding a module with a duplicate name raises an error."""
        self.net.add("fc1", nw.layers.linear(10, 5))
        with pytest.raises(ValueError):
            self.net.add("fc1", linear(10, 5)) # Add again with the same name

    def test_forward_pass(self):
        """Test a full forward pass through a sequential network."""
        self.net.add("fc1", nw.layers.linear(20, 10))
        self.net.add("act1", relu_factory())
        self.net.add("fc2", nw.layers.linear(10, 5))

        input_tensor = nw.randn([4, 20])
        output = self.net(input_tensor) # Use __call__ which triggers forward

        expected_shape = (4, 5)
        assert output.shape == expected_shape, f"Expected final output shape {expected_shape}, got {output.shape}"

    def test_state_dict(self):
        """Test the creation of a state_dict."""
        self.net.add("fc1", nw.layers.linear(10, 8))
        self.net.add("fc2", nw.layers.linear(8, 6, has_bias=False))
        
        state = self.net.state_dict()
        
        expected_keys = {"fc1.w", "fc1.b", "fc2.w"}
        assert set(state.keys()) == expected_keys
        # Check that the values are numpy arrays, not Tensors
        assert isinstance(state["fc1.w"], np.ndarray)

    def test_load_state_dict(self):
        """Test loading a state_dict into another model."""
        # Create a source model
        net1 = Net()
        net1.add("fc1", nw.layers.linear(10, 5))
        net1.add("fc2", nw.layers.linear(5, 2))
        
        # Get its state
        state = net1.state_dict()

        # Create a destination model with the same architecture
        net2 = Net()
        net2.add("fc1", nw.layers.linear(10, 5))
        net2.add("fc2", nw.layers.linear(5, 2))
        
        # Make sure weights are different before loading
        net2.params["fc1.w"].data = np.zeros((10, 5))
        assert not np.array_equal(net1.params["fc1.w"].data, net2.params["fc1.w"].data)

        # Load the state
        net2.load_state_dict(state)
        
        # Assert that the weights are now identical
        np.testing.assert_array_equal(net1.params["fc1.w"].data, net2.params["fc1.w"].data)
        np.testing.assert_array_equal(net1.params["fc2.b"].data, net2.params["fc2.b"].data)

    def test_load_state_dict_shape_mismatch_error(self):
        """Test that loading a state_dict with incorrect shapes raises an error."""
        net1 = Net()
        net1.add("layer1", nw.layers.linear(10, 5))
        state = net1.state_dict()

        # Create a model with a different architecture
        net2 = Net()
        net2.add("layer1", nw.layers.linear(10, 4)) # out_dims is 4 instead of 5

        with pytest.raises(ValueError):
            net2.load_state_dict(state)

    def test_module_replacement(self):
        """Test the replacement of a module using __setitem__."""
        self.net.add("fc1", nw.layers.linear(10, 8, has_bias=True))
        self.net.add("output", nw.layers.linear(8, 4))
        
        # Check initial params
        assert "fc1.b" in self.net.params

        # Replace "fc1" with a new layer that has no bias
        new_fc1 = nw.layers.linear(10, 8, has_bias=False)
        self.net["fc1"] = new_fc1

        # Check that the old bias parameter is gone and the module is replaced
        assert "fc1.b" not in self.net.params
        assert self.net.modules["fc1"] == new_fc1
        
        # Ensure a forward pass still works
        output = self.net(nw.randn([2, 10]))
        assert output.shape == (2, 4)
