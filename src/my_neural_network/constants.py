from enum import Enum

"""Config constants"""


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "gradient_descent"
