import abc
import random

import torch

from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class Data(abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def extend(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def subset(self, indices) -> tuple:
        raise NotImplementedError


class TrainTestSpliter(abc.ABC):

    def __init__(self, k):
        self.k = k

    @abc.abstractmethod
    def split(self, i):  # Return Train and Test Data
        logger.info(f"splitting {i}th fold")
        raise NotImplementedError


class SimplePytorchData(Data):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        logger.info(
            f"Initializing SimplePytorchData with X shape : {X.shape} and y shape : {y.shape}"
        )
        self.X = X
        self.y = y

    def extend(self, X: torch.Tensor, y: torch.Tensor):
        if self.X is None or self.y is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat((self.X, X), 0)
            self.y = torch.cat((self.y, y), 0)

    def subset(self, indices) -> tuple:
        return self.y[indices], self.y[indices]


class SimplePytorchDataTrainTestSpliter(TrainTestSpliter):

    def __init__(self, k, simple_data):
        super().__init__(k)
        logger.info(f"Initializing SimplePytorchDataTrainTestSpliter")
        self.X = simple_data.X
        self.y = simple_data.y

        self.data_size = simple_data.X.shape[0]

        subsets = dict()
        subset_size = int(self.data_size / self.k)
        remain = set(range(0, self.data_size))
        for i in range(self.k - 1):
            subsets[i] = random.sample(remain, subset_size)
            remain = remain.difference(subsets[i])
        subsets[k - 1] = remain

        self.subsets = subsets

    def split(self, i):
        indices = set(range(0, self.data_size))
        test_indices = list(self.subsets[i])
        train_indices = list(indices.difference(self.subsets[i]))

        train_data = SimplePytorchData(self.X[train_indices], self.y[train_indices])
        test_data = SimplePytorchData(self.X[test_indices], self.y[test_indices])
        return train_data, test_data
