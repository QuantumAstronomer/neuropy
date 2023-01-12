'''
This is the implementation of various loss functions used for evaluating the performance of neural networks.
Each lossfunction is housed in a separate class that implements both the forward and backward passes of the
function, i.e. the loss function and its derivative which are useful for training purposes. The typical
protocol for a LossFunction is as follows:

class LossFunction:

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:
        ...

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...

Unlike the activation function class, loss functions need to be initialized before they can be used. This is
due to the fact that some loss functions have parameters that need to be specified, like in the Huber Loss function
or the SmoothRobust Loss Function. Technically the forward loss function does not need to be implemented as it
is not used. However, it is useful to keep for evaluating the statistics of the training process. The gradient/backwards
loss function is normalised at all times.
'''

import numpy as np
import numpy.typing as npt

from typing import Protocol


class LossFunction(Protocol):

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:
        ...

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...


class MeanSquaredError():
    '''
    This class implements the mean squared error loss function, also known as least squares error.
    While the MSE has many strengths, it is not robust against outliers. The loss is defined as:
        L(r) = 1 / n * sum(r^2)
    '''

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:

        residual = y_true - y_predicted
        return np.mean((residual) ** 2)

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        residual = y_true - y_predicted
        return -2 * residual


class MeanAbsoluteError():
    '''
    This class implements the mean absolute error loss function. The MAE has the advantage of being more
    robust against outliers compared to the MSE. However, its derivative is not smooth or continuous.
    The loss function for the MAE is defined:
        L(r) = 1 / n * sum(|r|)
    '''

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:

        residual = y_true - y_predicted
        return np.mean(np.abs(residual))

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        residual = y_true - y_predicted
        return - np.sign(residual)

class Huber():
    '''
    This class implements the Huber loss function, which is a hybrid between the MeanSquaredError and
    the MeanAbsoluteError. Advantage of the Huber loss is mainly coming from its robustness to outliers
    as these are given less weight than in a standard MeanSquaredError, while still having the nice continuity
    we so much desire around residual = 0. The Huber loss function is defined as:
                          ( 1/2 * r^2 if |r| <= delta
        L(r) = 1 / n * sum(
                          ( |r| * delta - 1/2 * delta^2 if |r| > delta

    Since the Huber loss requires a parameter, delta, to be specified, the class
    needs an initializer method.
    '''

    def __init__(self, delta: float = 1.):
        self.delta: float = delta

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:
        
        residual = y_true - y_predicted
        return np.mean(np.where(np.abs(residual) <= self.delta, .5 * residual**2, self.delta * np.abs(residual) - .5 * self.delta**2))

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        residual = y_true - y_predicted
        return np.where(np.abs(residual) <= self.delta, residual, self.delta * np.sign(residual))


class PseudoHuber:
    '''
    This class implements the Pseudo-Huber loss function. It tries to improve on the Huber function by
    being smooth everywhere, approximating an MSE for small residuals and approximating the MAE for
    large residuals/outliers. The Pseudo-Huber loss function is defined as:
        L(r) = 1/n * sum(delta * sqrt(1 + r^2 / delta^2) - delta)

    Since the Pseudo-Huber loss requires a parameter, delta, to be specified, the class
    need an initializer method.
    '''

    def __init__(self, delta: float = 1.):
        self.delta = delta

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:

        residual = y_true - y_predicted
        return np.mean(np.sqrt((self.delta**2 + residual**2)) - self.delta)

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        residual = y_true - y_predicted
        return - residual / (np.sqrt(self.delta**2 + residual**2))


class GeneralizedHuber:
    '''
    Generalized Huber loss function, also called a Smooth Robust loss function, as defined by Gokcesu and Gokcesu 2021
    (url: https://arxiv.org/abs/2108.12627). The Generalized Huber loss function improves upon the Huber and 
    Pseudo-Huber functions be guaranteeing convexity over the whole domain and is defined as:
        L(r) = 1 / a * log(exp(a * r) + exp(-a * r) + b) - 1 / a * log(2 + b)

    Again, the Generalized Huber loss function requires an initializer method as there are two parameters, 'a' and 'b',
    that need to be specified. Where 'a' controls how quickly the loss function transition from quadratic to linear,
    large 'a' leads to a very quick transition, while 'b' controls the width of the quadratic domain, larger 'b' leads to
    a broader quadratic domain.
    '''

    def __init__(self, a: float = 1., b: float = 1.):
        self.a = a
        self.b = b

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:
        
        residual = y_true - y_predicted
        return np.mean(1 / self.a * np.log(np.exp(self.a * residual) + np.exp(-self.a * residual) + self.b) - 1 / self.a * np.log(2 + self.b))

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        residual = y_true - y_predicted
        return - (np.exp(self.a * residual) - np.exp(-self.a * residual)) / (np.exp(self.a * residual) + np.exp(-self.a * residual) + self.b)


class Tukey():
    '''
    The Tukey loss function is similar in spirit to the Huber loss function by being quadratic near the origin, i.e. at small
    residual values, while being more insensitive to outliers as it is constant for large residuals. It does provide a smooth
    transition between the two regimes. The Tukey loss is defined as:
                          ( delta^2 / 6 * (1 - (1 - r^2 / delta^2)^3) if |r| <= delta
        L(r) = 1 / n * sum(
                          ( delta^2 / 6

    Since the Tukey loss requires the specification of a parameter, being delta, it needs an initializer method to
    set its value.
    '''

    def __init__(self, delta: float = 1.):
        self.delta: float = delta

    def forward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> np.float64:
        residual = y_true - y_predicted
        return np.mean(np.where(np.abs(residual) <= self.delta, self.delta**2 / 6 * (1 - (1 - residual**2 / self.delta**2)**3), self.delta**2 / 6))

    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        residual = y_true - y_predicted
        return - np.where(np.abs(residual) <= self.delta, residual * (1 - residual**2 / self.delta**2)**2, 1)