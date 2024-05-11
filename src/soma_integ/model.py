import abc

from .config import ModelConfig
from .utils import logging as base_logger

logger = base_logger.getLogger(__name__)


class ModelHandler(abc.ABC):
    """
    Abstract base class for model handlers.

    Args:
        model_config (ModelConfig): Configuration of the model.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.model = self._build_model()
        self.fe = self._build_feature_extractor()

    @abc.abstractmethod
    def destroy(self):
        """
        Destroys the model.
        """
        logger.info(f"Model {self.model} has been deleted.")
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, **kwargs):
        """
        Makes predictions using the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def summary(self):
        """
        Prints a summary of the model.
        """
        raise NotImplementedError

    @abs.abstractmethod
    def _build_model(self):
        """
        Builds the model.
        """
        raise NotImplementedError

    @abs.abstractmethod
    def _build_feature_extractor(self):
        """
        Builds the feature extractor.
        """
        raise NotImplementedError


class HandlerFactory(abc.ABC):
    """Abstract base class for handler factories."""

    @abc.abstractmethod
    def create_handler(self) -> ModelHandler:
        """
        Abstract method to create a ModelHandler object.

        Returns:
            ModelHandler: The created ModelHandler object.
        """
        raise NotImplementedError
