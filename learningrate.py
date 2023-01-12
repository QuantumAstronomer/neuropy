'''
A decaying learning rate increases the stability of training a neural network, especially when
approaching the optimum performance where big jumps are not desirable. Each decay function
is enclosed in a DecayFunction protocol:

class DecayFunction(Protocol):

    learning_rate: float

    def calculate(self, iteration: int) -> float:
        ...
'''

import numpy as np
from typing import Protocol

class DecayFunction(Protocol):

    def set_initial_learning_rate(self, initital_learning_rate: float):
        ...
        
    def calculate(self, iteration: int) -> float:
        ...


class NoDecay(DecayFunction):
    '''
    This is the class for no decay of the learning rate.
    '''

    def set_initial_learning_rate(self, learning_rate: float):
        self.learning_rate: float = learning_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate

class StandardDecay(DecayFunction):
    '''
    This is the most common decay function for the learning rate.
    '''

    def __init__(self, decay_rate: float):
        self.decay_rate = decay_rate

    def set_initial_learning_rate(self, learning_rate: float):
        self.learning_rate: float = learning_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate / (1 + self.decay_rate * iteration)


class ExponentialDecay(DecayFunction):
    '''
    Exponential decay function for the learning rate.
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float):
        self.decay_rate = decay_rate

    def set_initial_learning_rate(self, learning_rate: float):
        self.learning_rate: float = learning_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate * self.decay_rate**iteration


class DiscreteDecay(DecayFunction):
    '''
    Discrete standard decay function for the learning rate. Instead of continuously allowing the
    learning rate to decline, this function requires discrete steps.
    '''

    def __init__(self, decay_rate: float, decay_every_n_steps: int):
        self.decay_rate = decay_rate
        self.decay_every_n_steps = decay_every_n_steps

    def set_initial_learning_rate(self, learning_rate: float):
        self.learning_rate: float = learning_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate / (1 + self.decay_rate * (iteration // self.decay_every_n_steps))


class SquareRootDecay(DecayFunction):
    '''
    The learning rate decays by the square root of the iteration number.
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float):
        self.decay_rate = decay_rate

    def set_initial_learning_rate(self, learning_rate: float):
        self.learning_rate: float = learning_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate * self.decay_rate / np.sqrt(iteration)


class InversePowerDecay(DecayFunction):
    '''
    The learning rate decays by the 1 / iteration^power
    '''

    def __init__(self, decay_rate: float, power: float | int):
        self.decay_rate = decay_rate
        self.power = power

        if self.power < 0:
            raise ValueError('The power must be non-negative.')

    def set_initial_learning_rate(self, learning_rate: float):
        self.learning_rate: float = learning_rate
    
    def calculate(self, iteration: int) -> float:
        return self.learning_rate * self.decay_rate / iteration**self.power