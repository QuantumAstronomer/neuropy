'''
Implementing the layers which make up the neural network. Layers are made up of
"neurons" and a few different types of layers exist:
    - Fully Connected or Dense layers
    - Activation layers
    - Convolution layers
    - Flattening layers
    - Pooling layers
    - Dropout layers

The layer follows a protocol which looks as follows:

class Layer(Protocol):

    def forward(self, input_data: npt.NDArray[np.float64]):
        ...

    def backward(self, grad_output: npt.NDArray[np.float64]):
        ...


This indicates that EVERY layer needs a forward and backward propagation, respectively, in order to be able
to train the neural network. Anything else is optional. The Trainable Layer is a subclass of Layer that has
some extra properties from its internal trainable parameters like the weights and biases themselves, as well
as the updates to these parameters and possible momentum terms for the optimizers to use.
'''

## TODO: Implement Convolution and Flattening layers.

import numpy as np
import numpy.typing as npt

from typing import Protocol, runtime_checkable
from activation import ActivationFunction


@runtime_checkable
class MinimalLayer(Protocol):

    input: npt.NDArray[np.float64]
    output: npt.NDArray[np.float64]
    dinput: npt.NDArray[np.float64]

    def forward(self, input_data: npt.NDArray[np.float64]):
        ...

    def backward(self, grad_output: npt.NDArray[np.float64]):
        ...


@runtime_checkable
class Layer(MinimalLayer, Protocol):

    layertype: str
    input_shape: int
    output_shape: int
    previous: MinimalLayer
    next: MinimalLayer


@runtime_checkable
class TrainableLayer(Layer, Protocol):

    weights: npt.NDArray[np.float64]
    biases: npt.NDArray[np.float64]
    dweights: npt.NDArray[np.float64]
    dbiases: npt.NDArray[np.float64]

    momentum_weights: npt.NDArray[np.float64]
    momentum_biases: npt.NDArray[np.float64]




class InputLayer(MinimalLayer):

    def __init__(self):
        # Initialize the layer type, which is the only relevant thing for the inputlayer
        # as it is merely a pass-through layer for the data.
        self.layertype = 'Input'

        self.input: npt.NDArray[np.float64]
        self.output: npt.NDArray[np.float64]
        self.dinput: npt.NDArray[np.float64]


    def forward(self, input_data: npt.NDArray[np.float64]):
        self.ouput = input_data

    def backward(self, grad_output: npt.NDArray[np.float64]):
        raise NotImplementedError('An input layer can not backpropagate, it is the first layer...')


class FullyConnected(TrainableLayer):

    def __init__(self, input_shape: int, output_shape: int, l1_regularization: float = 0., l2_regularization: float = 0.):

        # Create all variables needed to store the information about this layer
        # and where it is in the neural network.
        self.layertype = 'Fully Connected'
        self.input_shape: int = input_shape
        self.output_shape: int = output_shape
        self.shape: tuple[int, int] = (self.input_shape, self.output_shape)
        self.previous: Layer | MinimalLayer
        self.next: Layer | MinimalLayer

        self.l1_regularization: float = l1_regularization
        self.l2_regularization: float = l2_regularization

        # Create and initialize all variables for the weights and biases
        # this is a fully connected layer after all.
        self.weights: npt.NDArray[np.float64] = np.random.rand(*self.shape).astype(np.float64) - .5
        self.biases: npt.NDArray[np.float64] = np.random.rand(1, self.output_shape).astype(np.float64) -.5

        self.dweights: npt.NDArray[np.float64]  = np.zeros_like(self.weights, dtype = np.float64)
        self.dbiases: npt.NDArray[np.float64] = np.zeros_like(self.biases, dtype = np.float64)

        self.momentum_weights: npt.NDArray[np.float64] = np.zeros_like(self.weights, dtype = np.float64)
        self.momentum_biases: npt.NDArray[np.float64] = np.zeros_like(self.biases, dtype = np.float64)

    def forward(self, input_data: npt.NDArray[np.float64]):

        # Remember the input values
        self.input = input_data
        # Calculate output values
        self.output = np.dot(self.input, self.weights) + self.biases

    def backward(self, grad_output: npt.NDArray[np.float64]):

        # Gradients on the parameters connected to this layer.
        self.dweights = np.dot(self.input.T, grad_output)
        self.dbiases = grad_output.sum(axis = 0, keepdims = True)

        # Gradient on the input values
        self.dinput = np.dot(grad_output, self.weights.T)

        if self.l1_regularization > 0.:
            dl1weights = np.where(self.weights < 0, -1, 1)
            dl1biases = np.where(self.biases < 0, -1, 1)
            self.dweights += self.l1_regularization * dl1weights
            self.dbiases += self.l1_regularization * dl1biases

        if self.l2_regularization > 0.:
            self.dweights += 2 * self.weights * self.l2_regularization
            self.dbiases += 2 * self.biases * self.l2_regularization


class Activation(Layer):

    def __init__(self, activation_function: ActivationFunction, input_shape: int):

        # Create all variables needed to store the information about this layer
        # and where it is in the neural network.
        self.layertype = 'Activation'
        self.input_shape: int = input_shape
        self.output_shape: int = input_shape
        self.previous: Layer | MinimalLayer
        self.next: Layer | MinimalLayer

        self.act: ActivationFunction = activation_function

    def forward(self, input_data: npt.NDArray[np.float64]):

        # Store the input values and calculate the output values
        # by passing them through the activation function.
        self.input = input_data
        self.output = self.act.forward(self.input)

    def backward(self, grad_output: npt.NDArray[np.float64]):
        self.dinput = grad_output * self.act.backward(self.output)


class DropOut(Layer):

    def __init__(self, dropout_rate: float, input_shape: int):

        # Create all variables needed to store the information about this layer
        # and where it is in the neural network.
        self.layertype = 'Activation'
        self.input_shape: int = input_shape
        self.output_shape: int = input_shape
        self.previous: Layer | MinimalLayer
        self.next: Layer | MinimalLayer

        self.rate: float = 1 - dropout_rate

    def forward(self, input_data: npt.NDArray[np.float64]):
        
        # Store the input values and determine which 'neurons' will be 
        # disabled at random in this layer.
        self.input = input_data
        self.binary_mask = np.random.binomial(1, self.rate, size = self.input_shape) / self.rate
        self.output = self.input * self.binary_mask

    def backward(self, grad_output: npt.NDArray[np.float64]):
        self.dinput = grad_output * self.binary_mask