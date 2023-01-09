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

    def __init__(self, learning_rate: DecayFunction = NoDecay(0.05), momentum: float = 0.):

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


class NesterovGradient(Optimizer):

    def __init__(self, learning_rate: DecayFunction = NoDecay(0.01), momentum: float = .9):
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


class AdaptiveGradient(Optimizer):

    def __init__(self, learning_rate: float = 0.1, epsilon: float = 1e-7):

        self.learning_rate: float = learning_rate

        self.epsilon: float = epsilon

        self.iteration: int = 0
    
    def update_parameters(self, layer: TrainableLayer):

        layer.momentum_weights += layer.dweights**2
        layer.momentum_biases += layer.dbiases**2

        layer.weights += - self.learning_rate * layer.dweights / (np.sqrt(layer.momentum_weights) + self.epsilon)
        layer.biases += - self.learning_rate * layer.dbiases / (np.sqrt(layer.momentum_biases) + self.epsilon)

    def post_update(self):

        self.iteration += 1


class AdaptiveDelta(Optimizer):

    def __init__(self, learning_rate: float = 0.05, rho: float = .95, epsilon: float = 1e-7):

        self.learning_rate: float = learning_rate

        self.rho: float = rho
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        if len(layer.momentum_weights.shape) != 3:
            layer.momentum_weights = np.zeros(shape = (2, *layer.weights.shape), dtype = np.float64)
            layer.momentum_biases = np.zeros(shape = (2, *layer.biases.shape), dtype = np.float64)

        layer.momentum_weights = self.rho * layer.momentum_weights + (1 - self.rho) * layer.dweights**2
        layer.momentum_biases = self.rho * layer.momentum_biases + (1 - self.rho) * layer.dbiases**2

        layer.weights += - self.learning_rate * layer.dweights * np.sqrt(layer.momentum_weights[0] + self.epsilon) / np.sqrt(layer.momentum_weights[1] + self.epsilon)
        layer.biases += - self.learning_rate * layer.dbiases * np.sqrt(layer.momentum_biases[0] + self.epsilon) / np.sqrt(layer.momentum_biases[1] + self.epsilon)

    def post_update(self):
        self.iteration += 1

class RootMeanSquare(Optimizer):

    def __init__(self, learning_rate: float = 0.005, rho: float = .95, epsilon: float = 1e-7):

        self.learning_rate: float = learning_rate

        self.rho: float = rho
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        layer.momentum_weights = self.rho * layer.momentum_weights + (1 - self.rho) * layer.dweights**2
        layer.momentum_biases = self.rho * layer.momentum_biases + (1 - self.rho) * layer.dbiases**2

        layer.weights += - self.learning_rate * layer.dweights / np.sqrt(layer.momentum_weights + self.epsilon)
        layer.biases += - self.learning_rate * layer.dbiases / np.sqrt(layer.momentum_biases + self.epsilon)

    def post_update(self):
        self.iteration += 1


class AdaptiveMomentum(Optimizer):

    def __init__(self, learning_rate: float = 0.005, beta1: float = .9, beta2: float = .999, epsilon: float = 1e-7):
        self.learning_rate: float = learning_rate

        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):
        
        if len(layer.momentum_weights.shape) != 3:
            layer.momentum_weights = np.zeros(shape = (2, *layer.weights.shape), dtype = np.float64)
            layer.momentum_biases = np.zeros(shape = (2, *layer.biases.shape), dtype = np.float64)

        layer.momentum_weights[0] = self.beta1 * layer.momentum_weights[0] + (1 - self.beta1) * layer.dweights
        layer.momentum_biases[0] = self.beta1 * layer.momentum_biases[0] + (1 - self.beta1) * layer.dbiases

        layer.momentum_weights[1] = self.beta2 * layer.momentum_weights[1] + (1 - self.beta2) * layer.dweights**2
        layer.momentum_biases[1] = self.beta2 * layer.momentum_biases[1] + (1 - self.beta2) * layer.dbiases**2

        momentum_weights_corrected = layer.momentum_weights[0] / (1 - self.beta1**(self.iteration + 1))
        momentum_biases_corrected = layer.momentum_biases[0] / (1 - self.beta1**(self.iteration + 1))
        memory_weights_corrected = layer.momentum_weights[1] / (1 - self.beta2**(self.iteration + 1))
        memory_biases_corrected = layer.momentum_biases[1] / (1 -self.beta2**(self.iteration + 1))

        layer.weights += -self.learning_rate * momentum_weights_corrected / np.sqrt(memory_weights_corrected + self.epsilon)
        layer.biases += -self.learning_rate * momentum_biases_corrected / np.sqrt(memory_biases_corrected + self.epsilon)

    def post_update(self):
        self.iteration += 1


class NesterovAdaptive(Optimizer):

    def __init__(self, learning_rate: float = 0.005, beta1: float = .9, beta2: float = .999, epsilon: float = 1e-7):

        self.learning_rate: float = learning_rate

        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        if len(layer.momentum_weights.shape) != 3:
            layer.momentum_weights = np.zeros(shape = (2, *layer.weights.shape), dtype = np.float64)
            layer.momentum_biases = np.zeros(shape = (2, *layer.biases.shape), dtype = np.float64)

        layer.momentum_weights[0] = self.beta1 * layer.momentum_weights[0] + (1 - self.beta1) * layer.dweights
        layer.momentum_biases[0] = self.beta1 * layer.momentum_biases[0] + (1 - self.beta1) * layer.dbiases

        layer.momentum_weights[1] = self.beta2 * layer.momentum_weights[1] + (1 - self.beta2) * layer.dweights**2
        layer.momentum_biases[1] = self.beta2 * layer.momentum_biases[1] + (1 - self.beta2) * layer.dbiases**2

        layer.weights += -self.learning_rate / np.sqrt(layer.momentum_weights[1] + self.epsilon) * (self.beta1 * layer.momentum_weights[0] / (1 - self.beta1**(self.iteration + 1)) + \
                                                                                                   (1 - self.beta1) * layer.dweights / (1 - self.beta1**(self.iteration + 1)))
        layer.biases += -self.learning_rate / np.sqrt(layer.momentum_biases[1] + self.epsilon) * (self.beta1 * layer.momentum_biases[0] / (1 - self.beta1**(self.iteration + 1)) +\
                                                                                                 (1 - self.beta1) * layer.dbiases / (1 - self.beta1**(self.iteration + 1)))

    def post_update(self):
        self.iteration += 1


class AdaptiveMeanSquared(Optimizer):

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = .999, epsilon: float = 1e-7):

        self.learning_rate: float = learning_rate

        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        if len(layer.momentum_weights.shape)!= 3:
            layer.momentum_weights = np.zeros(shape = (2, *layer.weights.shape), dtype = np.float64)
            layer.momentum_biases = np.zeros(shape = (2, *layer.biases.shape), dtype = np.float64)

        layer.momentum_weights[1] = np.maximum(layer.momentum_weights[1], self.beta2 * layer.momentum_weights[1] + (1 - self.beta2) * layer.dweights**2)
        layer.momentum_biases[1] = np.maximum(layer.momentum_biases[1], self.beta2 * layer.momentum_biases[1] + (1 - self.beta2) * layer.dbiases**2)

        layer.momentum_weights[0] = self.beta1 * layer.momentum_weights[0] + (1 - self.beta1) * layer.dweights
        layer.momentum_biases[0] = self.beta1 * layer.momentum_biases[0] + (1 - self.beta1) * layer.dbiases

        layer.weights += -self.learning_rate / np.sqrt(layer.momentum_weights[1] + self.epsilon) * layer.momentum_weights[0]
        layer.biases += -self.learning_rate / np.sqrt(layer.momentum_biases[1] + self.epsilon) * layer.momentum_biases[0]

    def post_update(self):
        self.iteration += 1