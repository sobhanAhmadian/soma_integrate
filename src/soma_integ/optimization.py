import abc
import os.path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import OptimizerConfig
from .data import Data, PytorchData, TrainTestSplitter
from .evaluation import CrossValidationResult, Result, evaluate_binary_classification
from .model import HandlerFactory, ModelHandler
from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class Trainer(abc.ABC):
    """Abstract base class for trainers.
    Trainers are responsible for training a model using a given dataset and configuration.

    Methods:
        train: Abstract method for training a model.
    """

    @abc.abstractmethod
    def train(
        self, model_handler: ModelHandler, data: Data, config: OptimizerConfig
    ) -> Result:
        """
        Train the model using the given dataset and configuration.

        Args:
            model_handler (ModelHandler): The model handler object.
            data (Data): The dataset object.
            config (OptimizerConfig): The configuration object.

        Returns:
            Result: The result of the training process.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class Tester(abc.ABC):
    """Abstract base class for testers.
    Testers are responsible for testing a model using a given dataset and configuration.

    Methods:
        test: Abstract method for testing a model.
    """

    @abc.abstractmethod
    def test(
        self, model_handler: ModelHandler, data: Data, config: OptimizerConfig
    ) -> Result:
        """
        Abstract method to perform testing.

        Args:
            model_handler (ModelHandler): The model handler object.
            data (Data): The data object.
            config (OptimizerConfig): The optimizer configuration object.

        Returns:
            Result: The result of the testing.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


def cross_validation(
    train_test_spliter: TrainTestSplitter,
    handler_factory: HandlerFactory,
    trainer: Trainer,
    tester: Tester,
    config: OptimizerConfig,
) -> CrossValidationResult:
    """
    Perform k-fold cross validation using the given components and configuration.

    Args:
        train_test_spliter (TrainTestSplitter): The train-test splitter object.
        handler_factory (HandlerFactory): The handler factory object.
        trainer (Trainer): The trainer object.
        tester (Tester): The tester object.
        config (OptimizerConfig): The optimizer configuration object.

    Returns:
        CrossValidationResult: The result of the cross-validation.
    """

    k = train_test_spliter.k
    logger.info(f"Start {k}-fold Cross Validation with config: {config.exp_name}")

    cv_result = CrossValidationResult()
    for i in range(k):
        logger.info("{:#^50}".format(f"   Fold {i + 1}   "))

        train_data, test_data = train_test_spliter.split(i)

        model_handler = handler_factory.create_handler()
        trainer.train(model_handler=model_handler, data=train_data, config=config)
        test_result = tester.test(
            model_handler=model_handler, data=test_data, config=config
        )
        logger.info(f"Result of fold {i + 1} : {test_result.get_result()}")
        model_handler.destroy()

        cv_result.add_fold_result(test_result)

    cv_result.calculate_cv_result()

    logger.info("{:#^50}".format(f"   {k}-fold Result  "))
    logger.info(
        f"Result of {k}-fold Cross Validation : {cv_result.result.get_result()}"
    )

    return cv_result


def _predict_error(X, loss_function, model_handler: ModelHandler, running_loss, y):
    pred = model_handler.model(X)
    loss = loss_function(pred, y)
    running_loss += loss.item()
    return loss, running_loss


def _backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _batch_optimize(
    loader,
    model_handler: ModelHandler,
    config: OptimizerConfig,
):
    model_handler.model.train()
    optimizer = config.optimizer(model_handler.model.parameters(), lr=config.lr)

    for epoch in range(config.n_epoch):
        running_loss = 0.0
        for j, data in enumerate(loader, 0):
            X, y = data

            loss, running_loss = _predict_error(
                X, config.criterion, model_handler, running_loss, y
            )
            _backpropagation(loss, optimizer)

            if j % config.report_size == config.report_size - 1:
                loss = running_loss / (config.report_size * config.batch_size)
                logger.info(f"loss: {loss:.4f}    [{epoch + 1}, {j + 1:5d}]")
                running_loss = 0

        if config.save:
            if epoch % 5 == 0 or epoch == config.n_epoch - 1:
                m = os.path.join(
                    config.save_path, model_handler.model_config.model_name + ".pth"
                )
                torch.save(model_handler.classifier.state_dict(), m)


def _evaluate(model: ModelHandler, loader, config: OptimizerConfig):
    model.classifier.eval()
    total_labels = []
    total_predictions = []
    total_loss = 0.0
    for data in loader:
        inputs, labels = data
        outputs = model.predict(inputs)
        total_loss += config.criterion(outputs, labels).item()
        total_labels.extend(labels.cpu().numpy())
        total_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    result = evaluate_binary_classification(
        total_labels, total_predictions, config.threshold
    )
    result.loss = total_loss / len(loader)
    return result


class PytorchTrainer(Trainer):
    def train(
        self,
        model_handler: ModelHandler,
        data: PytorchData,
        config: OptimizerConfig,
    ) -> Result:
        """
        Trains the Pytorch model using the provided data and configuration.

        Args:
            model_handler (ModelHandler): The model handler object with a PyTorch model.
            data (PytorchData): The PyTorch data object containing the training data.
            config (OptimizerConfig): The configuration object for the optimizer.

        Returns:
            Result: The result of the training process.
        """
        logger.info(
            "{:#^50}".format(f"   Running PyTorch Trainer : {config.exp_name}  ")
        )

        model_handler.model = model_handler.model.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        logger.info(f"Model and data moved to {config.device}")

        _batch_optimize(loader, model_handler, config)
        result = _evaluate(model_handler, loader, config)

        logger.info(f"Result on Train Data: {result.get_result()}")
        return result


class PytorchTester(Tester):
    def test(
        self,
        model_handler: ModelHandler,
        data: PytorchData,
        config: OptimizerConfig,
    ) -> Result:
        """Run the PyTorch Tester on the given model and data.

        Args:
            model_handler (ModelHandler): The model handler object.
            data (PytorchData): The PyTorch data object.
            config (OptimizerConfig): The optimizer configuration object.

        Returns:
            Result: The result of the test.
        """
        logger.info(
            "{:#^50}".format(f"   Running PyTorch Tester : {config.exp_name}  ")
        )

        model_handler.model = model_handler.model.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        logger.info(f"Model and data moved to {config.device}")

        result = _evaluate(model_handler, loader, config)

        logger.info(f"Result on Test Data : {result.get_result()}")
        return result
