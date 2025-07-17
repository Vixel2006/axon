import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nawah
import pytest


class TestTensor:
    def setup_method(self):
        """Initialize sample tensors for testing."""
        self.tensor_1d = nawah.Tensor(data=[1, 2, 3, 4, 5, 6])
        self.tensor_2d = nawah.Tensor(data=[[1, 2, 3], [4, 5, 6]])
        self.tensor_3d = nawah.Tensor(data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_view(self):
        """Test the view() method for reshaping tensors."""
        t = self.tensor_1d.view((2, 3))
        assert t.shape == [2, 3], f"Expected shape (2, 3), got {t.shape}"
        assert t.strides == [3, 1], f"Expected strides (3, 1), got {t.strides}"
        assert t.tolist() == [[1, 2, 3], [4, 5, 6]], f"Data mismatch: {t.tolist()}"

        t2 = self.tensor_1d.view((3, 2))
        assert t2.shape == [3, 2], f"Expected shape (3, 2), got {t2.shape}"
        assert t2.strides == [2, 1], f"Expected strides (2, 1), got {t2.strides}"
        assert t2.data == [[1, 2], [3, 4], [5, 6]], f"Data mismatch: {t2.tolist()}"

    def test_view_data_shared(self):
        """Test that view() shares data with the original tensor."""
        t = self.tensor_1d
        t_view = t.view((2, 3))
        t_view[0, 0] = 10
        assert t[0] == 10, "Modification in view did not affect original tensor"

    def test_unsqueeze(self):
        """Test the unsqueeze() method for adding dimensions."""
        t = self.tensor_1d.unsqueeze(0)
        assert t.shape == [1, 6], f"Expected shape (1, 6), got {t.shape}"
        assert t.strides == [6, 1], f"Expected strides (6, 1), got {t.strides}"
        assert t.data == [[1, 2, 3, 4, 5, 6]], f"Data mismatch: {t.tolist()}"

        t2 = self.tensor_1d.unsqueeze(1)
        assert t2.shape == [6, 1], f"Expected shape (6, 1), got {t2.shape}"
        assert t2.strides == [1, 1], f"Expected strides (1, 1), got {t2.strides}"
        assert t2.data == [[1], [2], [3], [4], [5], [6]], (
            f"Data mismatch: {t2.tolist()}"
        )

    def test_squeeze(self):
        """Test the squeeze() method for removing dimensions of size 1."""
        t = nawah.Tensor([[1, 2, 3]])  # shape (1, 3)
        t_squeezed = t.squeeze()
        assert t_squeezed.shape == [3], f"Expected shape (3,), got {t_squeezed.shape}"
        assert t_squeezed.strides == [1], (
            f"Expected strides (1,), got {t_squeezed.strides}"
        )
        assert t_squeezed.tolist() == [1, 2, 3], f"Data mismatch: {t_squeezed.tolist()}"

        t2 = nawah.Tensor([[1], [2], [3]])  # shape (3, 1)
        t2_squeezed = t2.squeeze()
        assert t2_squeezed.shape == [3], f"Expected shape (3,), got {t2_squeezed.shape}"
        assert t2_squeezed.strides == [1], (
            f"Expected strides (1,), got {t2_squeezed.strides}"
        )
        assert t2_squeezed.tolist() == [1, 2, 3], (
            f"Data mismatch: {t2_squeezed.tolist()}"
        )

    def test_squeeze_error(self):
        """Test that squeeze() raises an error for dimensions not of size 1."""
        with pytest.raises(ValueError):
            self.tensor_2d.squeeze(axis=0)  # Shape (2, 3), dim 0 is 2

    def test_broadcast(self):
        """Test the broadcast() method for expanding tensor dimensions."""
        t = nawah.Tensor([[1], [2], [3]])  # shape (3, 1)
        t_broadcasted = t.broadcast((3, 4))
        assert t_broadcasted.shape == [3, 4], (
            f"Expected shape (3, 4), got {t_broadcasted.shape}"
        )
        expected = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        assert t_broadcasted.tolist() == expected, (
            f"Data mismatch: {t_broadcasted.tolist()}"
        )

    def test_permute(self):
        """Test the permute() method for rearranging dimensions."""
        t = self.tensor_3d.permute(1, 0, 2)
        assert t.shape == [2, 2, 2], f"Expected shape (2, 2, 2), got {t.shape}"
        assert t[0, 0, 0] == 1, "Permutation data mismatch at [0, 0, 0]"
        assert t[0, 1, 0] == 5, "Permutation data mismatch at [0, 1, 0]"

    def test_transpose(self):
        """Test the transpose() method for swapping dimensions."""
        t = self.tensor_2d.transpose()
        assert t.shape == [3, 2], f" ortam Expected shape (3, 2), got {t.shape}"
        assert t.strides == [1, 3], f"Expected strides (1, 3), got {t.strides}"
        assert t.data == [[1, 4], [2, 5], [3, 6]], f"Data mismatch: {t.tolist()}"

    def test_expand(self):
        """Test the expand() method for replicating data along dimensions."""
        t = nawah.Tensor([[1], [2], [3]])  # shape (3, 1)
        t_expanded = t.expand(3, 4)
        assert t_expanded.shape == [3, 4], (
            f"Expected shape (3, 4), got {t_expanded.shape}"
        )
        expected = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        assert t_expanded.data == expected, f"Data mismatch: {t_expanded.tolist()}"

    """
    def test_getitem(self):
        Test the __getitem__() method for indexing and slicing
        t = self.tensor_2d
        assert t[0, 0] == 1, "Indexing failed at [0, 0]"
        assert t[1, 2] == 6, "Indexing failed at [1, 2]"
        assert t[0, :].tolist() == [1, 2, 3], "Slicing failed for row 0"
        assert t[:, 1].tolist() == [2, 5], "Slicing failed for column 1"
        t_slice = t[0:2, 1:3]
        assert t_slice.shape == [2, 2], (
            f"Expected slice shape (2, 2), got {t_slice.shape}"
        )
        assert t_slice.tolist() == [[2, 3], [5, 6]], (
            f"Slice data mismatch: {t_slice.tolist()}"
        )

    def test_getitem_offset(self):
        Test that __getitem__() adjusts offset correctly when slicing
        t = self.tensor_2d
        t_slice = t[1:, 1:]
        assert t_slice.shape == [1, 2], f"Expected shape (1, 2), got {t_slice.shape}"
        assert t_slice.tolist() == [[5, 6]], f"Data mismatch: {t_slice.tolist()}"
        # Offset testing assumes it's exposed; adjust based on actual implementation
        if hasattr(t_slice, "offset"):
            assert t_slice.offset != t.offset, "Offset should change after slicing"
    """
