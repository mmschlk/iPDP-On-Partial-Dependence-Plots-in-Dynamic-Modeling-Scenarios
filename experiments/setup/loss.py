from river.datasets.base import BINARY_CLF, REG, MULTI_CLF
from river.metrics import MAE, CrossEntropy, MSE, Accuracy
from river.utils import Rolling

from ixai.utils.wrappers.river import RiverMetricToLossFunction

__all__ = [
    "get_loss_function",
    "get_training_metric"
]

cross_entropy_loss = RiverMetricToLossFunction(river_metric=CrossEntropy())
mse_loss = RiverMetricToLossFunction(river_metric=MSE())
accuracy_loss = RiverMetricToLossFunction(river_metric=Accuracy())


def get_loss_function(task):
    if task == BINARY_CLF or task == MULTI_CLF:
        loss_function = accuracy_loss
    elif task == REG:
        loss_function = MSE()
    else:
        raise NotImplementedError(f"No standard loss implemented for task {task}.")
    return loss_function


def get_training_metric(task, rolling_window=1000):
    if task == BINARY_CLF:
        training_metric = Accuracy()
    elif task == REG:
        training_metric = MAE()
    elif task == MULTI_CLF:
        training_metric = Accuracy()
    else:
        raise NotImplementedError(f"No standard loss implemented for task {task}.")
    if rolling_window > 0:
        rolling_training_metric = Rolling(training_metric, rolling_window)
    else:
        rolling_training_metric = training_metric
    return rolling_training_metric, training_metric
