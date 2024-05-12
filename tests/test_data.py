import pytest
import torch
from src.soma_integ.data import PytorchData, PytorchTrainTestSplitter


def test_PytorchTrainTestSplitter_split():
    # Create dummy data
    X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = torch.tensor([0, 1, 0, 1])
    data = PytorchData(X, y)

    # Create PytorchTrainTestSplitter object
    k = 4
    splitter = PytorchTrainTestSplitter(k, data)

    # Test split method
    i = 0
    train_data, test_data = splitter.split(i)

    assert len(train_data) == 3
    assert len(test_data) == 1
