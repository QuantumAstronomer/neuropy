'''
Implementing the optimizer methods like Gradient Descent and Root Mean Squared Propagation. These
are all methods to train a neural network, each having different desirable properties.
The protocol that ALL optimizers should follow is:

class Optimizer(Protocol):

    def update_parameters(self, layer: TrainableLayer):
        ...

    def post_update(self):
        ...
'''

import numpy as np

from typing import Protocol
from .layer import TrainableLayer
from .learningrate import DecayFunction, NoDecay


class Optimizer(Protocol):

    def update_parameters(self, layer: TrainableLayer):
        ...

    def post_update(self):
        ...


class GradientDescent(Optimizer):

    def __init__(self, learning_rate: DecayFunction = NoDecay(0.1), momentum: float = 0.):

        self.learning_rate_function: DecayFunction = learning_rate
        self.learning_rate: float = self.learning_rate_function.learning_rate

        self.momentum: float = momentum

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):
        
        weights_update = - self.learning_rate * layer.dweights
        biases_update = - self.learning_rate * layer.dbiases

        if self.momentum:
            weights_update += self.momentum * layer.momentum_weights
            biases_update += self.momentum * layer.momentum_biases

            layer.momentum_weights = weights_update
            layer.momentum_biases = biases_update
        
        layer.weights += weights_update
        layer.biases += biases_update

    def post_update(self):

        self.iteration += 1
        self.learning_rate = self.learning_rate_function.calculate(self.iteration)


class NesterovGradientDescent(Optimizer):

    def __init__(self, learning_rate: DecayFunction = NoDecay(0.1), momentum: float = .9):
        self.learning_rate_function: DecayFunction = learning_rate
        self.learning_rate: float = self.learning_rate_function.learning_rate

        self.momentum: float = momentum

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        layer.momentum_weights = - self.learning_rate * layer.dweights + self.momentum * layer.momentum_weights
        layer.momentum_biases = - self.learning_rate * layer.dbiases + self.momentum * layer.momentum_biases

        layer.weights += - self.learning_rate * layer.dweights + self.momentum * layer.momentum_weights
        layer.biases += - self.learning_rate * layer.dbiases + self.momentum * layer.momentum_biases

    def post_update(self):

        self.iteration += 1
        self.learning_rate = self.learning_rate_function.calculate(self.iteration)


class AdaptiveGradientDescent(Optimizer):

    def __init__(self, learning_rate: DecayFunction = NoDecay(0.1), epsilon: float = 1e-7):

        self.learning_rate_function: DecayFunction = learning_rate
        self.learning_rate: float = self.learning_rate_function.learning_rate

        self.epsilon: float = epsilon

        self.iteration: int = 0
    
    def update_parameters(self, layer: TrainableLayer):

        layer.momentum_weights += layer.dweights**2
        layer.momentum_biases += layer.dbiases**2

        layer.weights += - self.learning_rate * layer.dweights / (np.sqrt(layer.momentum_weights) + self.epsilon)
        layer.biases += - self.learning_rate * layer.dbiases / (np.sqrt(layer.momentum_biases) + self.epsilon)

    def post_update(self):

        self.iteration += 1
        self.learning_rate = self.learning_rate_function.calculate(self.iteration)


class AdaptiveDelta(Optimizer):

    def __init__(self, learning_rate: DecayFunction = NoDecay(0.1), rho: float = 0.95, epsilon: float = 1e-7):

        self.learning_rate_function: DecayFunction = learning_rate
        self.learning_rate: float = self.learning_rate_function.learning_rate

        self.rho: float = rho
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        layer.momentum_weights = self.rho * layer.momentum_weights + (1 - self.rho) * layer.dweights**2
        layer.momentum_biases = self.rho * layer.momentum_biases + (1 - self.rho) * layer.dbiases**2

        weights_update = layer.dweights * np.sqrt(layer.memory_weights + self.epsilon) / np.sqrt(layer.momentum_weights + self.epsilon)
        biases_update = layer.dbiases * np.sqrt(layer.memory_biases + self.epsilon) / np.sqrt(layer.momentum_biases + self.epsilon)

        layer.memory_weights = self.rho * layer.memory_weights + (1 - self.rho) * weights_update**2
        layer.memory_biases = self.rho * layer.memory_biases + (1 - self.rho) * biases_update**2

        layer.weights += - self.learning_rate * weights_update
        layer.biases += - self.learning_rate * biases_update

    def post_update(self):
        self.iteration += 1
        self.learning_rate = self.learning_rate_function.calculate(self.iteration)