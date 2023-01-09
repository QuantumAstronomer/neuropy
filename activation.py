'''
This file implements the activation functions for neurons/layers in the neural network. Activation functions are housed
in classes which contain the forward and backward passes of the activation function. This makes it very easy to call
the derivative of the activation function. The protocol for activation function classes is:

class ActivationFunction:


    def forward(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...


    def backward(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
       ...

This is the structure for an activation function as it will be expected by the neural network.
'''

import numpy as np
import numpy.typing as npt

from typing import Protocol

class ActivationFunction(Protocol):

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...


# The simple linear activation function is the easiest base to work from.
class Linear():
    '''
    This class implements a simple linear activation function.
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = input_data
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.ones_like(output_error)
        return self.input_error

# Function with restrictions on negative inputs and a rough 'S'-shape are:
# binary and the minmax function.
class Binary():
    '''
    This class implements a binary activation function. The binary function
    maps inputs from [-inf, inf] to [0, 1] and has derivative = 0 everywhere.
    One could say it is a mix between a function with restricted negative inputs
    and a function which is 'S'-shaped. The binary function is defined as:
        binary(x) = (0 if x < 0, 1 if x > 0, 1/2 if x = 0)
    which has derivative = 0 everywhere.
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.heaviside(input_data, .5)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.zeros_like(output_error)
        return self.input_error

class MinMax():
    '''
    This class implements the minmax activation function defined as:
        minmax(x) = minimum(1, maximum(0, x)) = (0 if x < 0, 1 if x > 1, x if 0 < x < 1)
    The derivative of the minmax function is 0 if x < 0 or x > 1 and 1 if 0 < x < 1:
        minmax'(x) = (0 if x < 0 or x > 1, 1 if 0 < x < 1)
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.minimum(1, np.maximum(0, input_data))
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.where((output_error > 0) & (output_error < 1), 1, 0)
        return self.input_error


# Function with restrictions on negative inputs are: ReLU, ELU, and SoftPlus
class ReLU():
    '''
    This class implements the rectified linear unit activation function.
    Defined as: 
        ReLU(x) = maximum(0, x) = (0 if x < 0, x if x > 0)
    which the heaviside step function as its derivative. The heaviside step function
    is undefined/discontinuous at x = 0, therefore we choose its value to be 1/2.
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.maximum(input_data, 0)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.heaviside(output_error, .5)
        return self.input_error


class ELU():
    '''
    This class implements the exponential linear unit activation function which is similar to the
    ReLU function. The ELU function has a continuous derivative, but it is not smooth. And allows
    the function to asymptotically go to -1 for negative inputs.
    The ELU function is defined as:
        ELU(x) = (x if x > 0, exp(x) - 1 if x < 0)
    which has the derivative:
        ELU'(x) = (1 if x > 0, exp(x) if x < 0)
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.where(input_data > 0, input_data, np.exp(input_data) - 1)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.where(output_error > 0, 1, np.exp(output_error))
        return self.input_error


class SoftPlus():
    '''
    This class implements the softplus activation function, a function which is similar to
    the ReLU function, except that it is smooth and its derivative is continuous and smooth.
    Like the ReLU it goes to zero for negative inputs, but does so asymptotically.
    The softplus function is defined as:
        softplus(x) = log(1 + exp(x))
    which has the derivative:
        softplus'(x) = exp(x) / (1 + exp(x)) = sigmoid(x)
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.log(1 + np.exp(input_data))
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.exp(output_error) / (1 + np.exp(output_error))
        return self.input_error


# 'S'-shaped functions that map [-inf, inf] to [-1, 1] are: Sign, SoftSign, Sigmoid, and Hyperbolic Tangent.
class Sign():
    '''
    This class implements the sign activation function defined as:
        sign(x) = (-1 if x < 0, 1 if x > 0, 0 if x = 0)
    which has the derivative of zero everywhere.
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.sign(input_data)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = np.zeros_like(output_error)
        return self.input_error


class SoftSign():
    '''
    This class implements the softsign activation function, which tries to improve
    on the ordinary sign function by being smooth and continuous and being properly
    defined at x = 0. Furthermore, its derivative is continuous but not smooth.
    The softsign function is defined as:
        softsign(x) = x / (abs(x) + 1)
    which has the derivative:
        softsign'(x) = 1 / (abs(x) + 1)^2
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = input_data / (np.abs(input_data) + 1)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = 1 / (np.abs(output_error) + 1)**2
        return self.input_error


class Sigmoid():
    '''
    This class implements the sigmoid activation function defined as:
        sigmoid(x) = 1 / (1 + exp(-x))
    which has the derivative:
        sigmoid'(x) = s(x) * (1 - s(x))
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = self.output * (1 - self.output)
        return self.input_error


class Tanh():
    '''
    This class implements the hyperbolic tangent activation function defined as:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = sinh(x) / cosh(x)
    which has the derivative:
        tanh'(x) = 1 - tanh^2(x)
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.tanh(input_data).astype(np.float64)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_error = 1 - self.output**2
        return self.input_error

# The SoftMax function is a class of its own, it is very special in that sense.
class SoftMax():
    '''
    This class implements the softmax activation function. A function that is
    especially useful in classification problems as it ensure the sum of the resulting
    probabilities is 1, i.e. a normalized PDF. The softmax function is defined as:
        softmax(x_i) = exp(x_i - max(x_i)) / sum(exp(x_j - max(x_j)))
    Note that this is a slightly modified version of the softmax function which yields
    a more stable/robust function preventing the occurence of overflow errors. Given the
    normalisation, this makes absolutely no difference to the output. The derivative
    of the softmax function is defined as:
        softmax'(x_i) = softmax(x_i) * (delta_ij - softmax(x_j))
    where delta_ij is the kronecker delta.
    '''

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        max = np.max(input_data, axis = 1, keepdims = True)
        self.output = np.exp(input_data - max) / np.sum(np.exp(input_data - max), axis = 1, keepdims = True)
        return self.output

    def backward(self, output_error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        self.input_error = np.empty_like(self.output)

        for i, (single_output, single_output_error) in enumerate(zip(self.output, output_error)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output_error[:, np.newaxis].T)
            self.input_error[i] = jacobian.sum(axis = -1)

        return self.input_error
