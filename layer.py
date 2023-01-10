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

## TODO: Implement Convolution, Flattening, and Pooling layers.

import numpy as np
import numpy.typing as npt

from typing import Protocol, runtime_checkable
from .activation import ActivationFunction


@runtime_checkable
class MinimalLayer(Protocol):

    input: npt.NDArray[np.float64]
    output: npt.NDArray[np.float64]
    dinput: npt.NDArray[np.float64]

    def forward(self, input_data: npt.NDArray[np.float64], training: bool):
        ...

    def backward(self, grad_output: npt.NDArray[np.float64]):
        ...


@runtime_checkable
class Layer(MinimalLayer, Protocol):

    # First each layer will house some information about its
    # function and shape.
    layertype: str
    input_shape: int
    output_shape: int

    # The previous and next attributes will be set when the
    # neural network is constructed. They keep track of the
    # position of the layer within the network, i.e. what
    # layers come before and after it.
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
    '''
    Implementing an input layer. As an input layer is the first layer of a neural network
    and is used to pass the input data through to the next layers for processing, it does not
    perform any calculations, it only serves as a placeholder. Furthermore, as the input
    layer is ALWAYS the first layer in a neural network, it can not perform backwards propagation
    and thus will raise a NotImplementedError when attempting to.
    '''

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
    '''
    A fully connected neural network layer takes in a 1D array of inputs and performs a matrix
    multiplication with the weights and adds biases to produce the output. This is a trainable
    layer meaning that its parameters are adjusted/learned during training.

    Two regularization techniques are available:
        - L1 and L2 regularization that penalizes neurons with large weights and/or biases
        - Dropout, which randomly selects neurons that are disabled forcing the network
          to employ more of its neurons to properly learn the patterns in the data.
    '''

    def __init__(self, input_shape: int, output_shape: int, l1_regularization: float = 0., l2_regularization: float = 0., dropout_rate: float = 0.):

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
        self.rate: float = 1 - dropout_rate

        # Create and initialize all variables for the weights and biases
        # this is a fully connected layer after all.Your location
        self.weights: npt.NDArray[np.float64] = np.random.rand(*self.shape).astype(np.float64) - .5
        self.biases: npt.NDArray[np.float64] = np.random.rand(1, self.output_shape).astype(np.float64) -.5

        self.dweights: npt.NDArray[np.float64]  = np.zeros_like(self.weights, dtype = np.float64)
        self.dbiases: npt.NDArray[np.float64] = np.zeros_like(self.biases, dtype = np.float64)

        self.momentum_weights: npt.NDArray[np.float64] = np.zeros_like(self.weights, dtype = np.float64)
        self.momentum_biases: npt.NDArray[np.float64] = np.zeros_like(self.biases, dtype = np.float64)

    def forward(self, input_data: npt.NDArray[np.float64], training: bool = True):
        '''
        Perform a forward pass through the layer done by a matrix multiplication between
        the input and the weights, adding a bias term. In other words: the output is a linear
        combinatin of the input vector.
        '''

        if self.rate > 0. and training:
            self.binary_mask = np.random.binomial(1, self.rate, size = self.input_shape) / self.rate
        else:
            self.binary_mask = np.ones_like(self.input_shape)

        # Remember the input values
        self.input = input_data.reshape(input_data.shape[0], input_data.shape[-1]) * self.binary_mask
        # Calculate output values
        self.output = np.dot(self.input, self.weights) + self.biases

    def backward(self, grad_output: npt.NDArray[np.float64]):
        '''
        The backward method calculates gradients on the input, weights, and biases.
        It updates the dweights, dbiases, dinput attributes accordingly. Additionally, 
        the backward method also applies L1 and L2 regularization to aid in the overall
        robustness of the neural network in general.
        '''
        
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
    '''
    The activation layer houses all activation functions for a single layer,
    used to introduce non-linearity into the network. The type of activation
    function is provided during the initalization of the layer.
    '''

    def __init__(self, activation_function: ActivationFunction, input_shape: int):

        # Create all variables needed to store the information about this layer
        # and where it is in the neural network.
        self.layertype = 'Activation'
        self.input_shape: int = input_shape
        self.output_shape: int = input_shape
        self.previous: Layer | MinimalLayer
        self.next: Layer | MinimalLayer

        # Set the activation function of the layer.
        self.act: ActivationFunction = activation_function

    def forward(self, input_data: npt.NDArray[np.float64], training = True):

        # Store the input values and calculate the output values
        # by passing them through the activation function.
        self.input = input_data
        self.output = self.act.forward(self.input)

    def backward(self, grad_output: npt.NDArray[np.float64]):
        self.dinput = grad_output * self.act.backward(self.output)