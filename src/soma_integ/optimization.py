import abc
import os.path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import OptimizerConfig
from .data import Data, SimplePytorchData, TrainTestSpliter
from .evaluation import Result, get_prediction_results
from .model import HandlerFactory, ModelHandler
from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train(
        self, model_handler: ModelHandler, data: Data, config: OptimizerConfig
    ) -> Result:
        raise NotImplementedError


class Tester(abc.ABC):
    @abc.abstractmethod
    def test(
        self, model_handler: ModelHandler, data: Data, config: OptimizerConfig
    ) -> Result:
        raise NotImplementedError


def cross_validation(
    train_test_spliter: TrainTestSpliter,
    handler_factory: HandlerFactory,
    trainer: Trainer,
    tester: Tester,
    config: OptimizerConfig,
) -> Result:
    k = train_test_spliter.k
    logger.info(f"Start {k}-fold Cross Validation with config : {config.exp_name}")

    result = Result()
    for i in range(k):
        logger.info(f"---- Fold {i + 1} ----")

        train_data, test_data = train_test_spliter.split(i)

        handler = handler_factory.create_handler()
        trainer.train(model_handler=handler, data=train_data, config=config)
        test_result = tester.test(model_handler=handler, data=test_data, config=config)
        logger.info(f"Result of fold {i + 1} : {test_result.get_result()}")
        handler.destroy()

        result.add(test_result)

    result.divide(k=k)

    logger.info(
        f"{k}-fold result: avg_auc: {result.auc}, avg_acc: {result.acc}, avg_f1: {result.f1}, avg_aupr: {result.aupr}"
    )

    return result


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
    result = get_prediction_results(total_labels, total_predictions, config.threshold)
    result.loss = total_loss / len(loader)
    return result


class SimplePytorchTrainer(Trainer):
    def train(
        self,
        model_handler: ModelHandler,
        data: SimplePytorchData,
        config: OptimizerConfig,
    ) -> Result:
        logger.info(f"Running Simple Trainer with config : {config.exp_name}")

        logger.info(f"moving data and model to {config.device}")
        model_handler.model = model_handler.model.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        _batch_optimize(loader, model_handler, config)
        result = _evaluate(model_handler, loader, config)

        logger.info(f"Result on Train Data : {result.get_result()}")
        return result


class SimplePytorchTester(Tester):
    def test(
        self,
        model_handler: ModelHandler,
        data: SimplePytorchData,
        config: OptimizerConfig = None,
    ) -> Result:
        logger.info(f"Running Simple Tester with config : {config.exp_name}")

        logger.info(f"moving data and model to {config.device}")
        model_handler.model = model_handler.model.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        result = _evaluate(model_handler, loader, config)

        logger.info(f"Result on Test Data : {result.get_result()}")
        return result
