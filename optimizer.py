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
    '''
    The Optimizer class defines an interface for optimizers that are used to update
    the parameters in the layers of a neural network.

    The update_parameters method is used to update the weights and biases of any
    trainable layer, i.e. a layer that has parameters that can be tuned to
    understand/explain the data.

    The post_update method is used to update the iteration of the optimizer and, if needed,
    the learning rate if it is set to be dynamic.

    Whether a neural network is trained on every training sample indepdently or in batches is
    controlled by the train method of the network, not by the optimizer. The optimizer is only
    concerned with the way the parameters of trainable layers are updated.
    '''

    def update_parameters(self, layer: TrainableLayer):
        ...

    def post_update(self):
        ...


# Optimizers that require the speicifcation of a decay function for the learning rate.
class GradientDescent(Optimizer):
    '''
    "Vanilla" gradient descent optimization algorithm with implementations in place to use
    a decaying learning rate as the iteration increases and to use momentum which is a parameter
    used to build up a measure of velocity towards the optimal solution and typically yields
    quicker convergence over traditional gradient descent.

    A decaying learning rate is used to prevent the optimizer to take large steps towards the
    end of training possibly moving it far away from the optimal solution.
    '''

    def __init__(self, learning_rate: float = 0.05, momentum: float = 0., decay_function = NoDecay()):

        decay_function.set_initial_learning_rate(learning_rate)
        self.learning_rate_function: DecayFunction = decay_function
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
    '''
    Nesterov accelerated gradient descent algorithm. This algorithm differs from the typical
    stochastic gradient descent with momentum by first calculating the momentum step and then
    performing the gradient calculation before updating the parameters in the trainable layer.
    '''

    def __init__(self, learning_rate: float = 0.05, momentum: float = 0., decay_function = NoDecay()):

        decay_function.set_initial_learning_rate(learning_rate)
        self.learning_rate_function: DecayFunction = decay_function
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
    '''
    Adaptive gradient descent, proposed by Duchi, Hazan and Singer in 2011
    (url: https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), is more 
    robust and efficient at optimizing parameters as it adapts the learning rate 
    based on the gradient: it limits the learning rate for parameters with 
    frequently occuring features, i.e. a large gradient, while for parameters with 
    less frequent features, larger updates are relaized by boosting the learning rate.

    In this way, adaptive gradient descent eliminates the need to specify a particular
    decay function/shape for the learning rate and reduces the need to pick an optimal
    learning rate for the most efficient convergence.
    '''

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
    '''
    Adaptive Delta algorithm proposed by Zeiler 2012 (url: https://arxiv.org/abs/1212.5701).
    Adaptive Delta is an extension of Adaptive Gradient that seeks to reduce its aggressive
    (monotonically) decreasing learning rate. Rather than accumulating the past squared
    gradients, it restricts the window of accumulated past gradients by defining a recursively
    decaying average of the past squared gradients, i.e. past squared gradients are multiplied 
    by a constant smaller than one at each iteration leading to an exponential moving average.
    '''

    def __init__(self, learning_rate: float = 0.05, rho: float = .95, epsilon: float = 1e-7):

        self.learning_rate: float = learning_rate

        self.rho: float = rho
        self.epsilon: float = epsilon

        self.iteration: int = 0

    def update_parameters(self, layer: TrainableLayer):

        if len(layer.momentum_weights.shape) != 3:
            layer.momentum_weights = np.zeros(shape = (2, *layer.weights.shape), dtype = np.float64)
            layer.momentum_biases = np.zeros(shape = (2, *layer.biases.shape), dtype = np.float64)

        layer.momentum_weights[0] = self.rho * layer.momentum_weights[0] + (1 - self.rho) * layer.dweights**2
        layer.momentum_biases[0] = self.rho * layer.momentum_biases[0] + (1 - self.rho) * layer.dbiases**2

        layer.momentum_weights[1] = self.rho* layer.momentum_weights[1] + (1 - self.rho) * self.learning_rate * layer.dweights**2 / (layer.momentum_weights[0] + self.epsilon)
        layer.momentum_biases[1] = self.rho* layer.momentum_biases[1] + (1 - self.rho) * self.learning_rate * layer.dbiases**2 / (layer.momentum_biases[0] + self.epsilon)

        layer.weights += - layer.dweights * np.sqrt(layer.momentum_weights[0] + self.epsilon) / np.sqrt(layer.momentum_weights[1] + self.epsilon)
        layer.biases += - layer.dbiases * np.sqrt(layer.momentum_biases[0] + self.epsilon) / np.sqrt(layer.momentum_biases[1] + self.epsilon)

    def post_update(self):
        self.iteration += 1

class RootMeanSquare(Optimizer):
    '''
    The Root Mean Squared Propagation optimizer is an unpublished optimizer method proposed by
    Geoff Hinton in his lectures on neural networks and optimizer functions. RMS propagation and
    Adaptive Delta have both been developed independently to resolve Adaptive Gradient's problem
    of radically diminishing learning rates.
    '''

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
    '''
    The Adaptive Momentum Estimation (AdaM), proposed by Kingma and Ba 2015 (url: https://arxiv.org/abs/1412.6980)
    optimizer is another method that computes adaptive learning rates for each parameter. In addition to storing 
    the exponentially decaying average of the past gradients squared like Adaptive Delta and RMS propagation, 
    AdaM also keeps an exponentially decaying average of past gradients, similar to a momentum term.

    AdaM differs from traditional Momentum Gradient Descent (MGD) in the sense that MGD can be
    seen as a ball rolling down a slope, AdaM behaves like a heavy ball with friction.
    '''

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
    '''
    Nesterov Adaptive Momentum Estimation (NAdaM) optimizer which is a hybrid between the AdaM optimizer
    and Nesterov Accelerated GD. Its spirit is identical to the AdaM optimizer, while its method for implemnting
    the momentum is different since it uses the Nesterov approach of taking the momentum step first after which
    a correction is applied if the momentum step overshot the optimal solution.
    '''

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
    '''
    Adaptive learning rate methods are the norm in training neural networks. However, they have a possiblity of not
    converging quick enough towards the optimal solution as at some point the learning rate becomes so small that
    steps are basically staying in the same place.

    To remedy this the Adaptive Mean Squared Gradient (AMS Gradient) optimizer was proposed by Reddi et al. at the
    ICLR 2018 (url: https://arxiv.org/abs/1904.09237) which realized that the exponential moving average of the
    past squared gradients is the cause for the poor convergence. The authors noted that some minibatches in batch
    training provide a lot of information on how to converge by generating large gradients. However, since these
    gradients occur rarely, they simply diminish in the exponential moving average. For this reason the maximum of
    the exponential moving average of past squared gradients is used, i.e. if the current exponential moving average
    is smaller than the previous step, the exponential moving average of the previous step is used. This results in
    a non-increasing step-size, avoiding the problems suffered by AdaM.
    '''

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