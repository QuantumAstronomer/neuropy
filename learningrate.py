import numpy as np
from typing import Protocol

class DecayFunction(Protocol):

    learning_rate: float

    def calculate(self, iteration: int) -> float:
        ...


class NoDecay():
    '''
    This is the class for no decay of the learning rate.
    '''

    def __init__(self, initial_learning_rate: float):
        self.learning_rate = initial_learning_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate

class StandardDecay():
    '''
    This is the most common decay function for the learning rate.
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate / (1 + self.decay_rate * iteration)


class ExponentialDecay():
    '''
    Exponential decay function for the learning rate.
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate * self.decay_rate**iteration


class DiscreteDecay():
    '''
    Discrete standard decay function for the learning rate. Instead of continuously allowing the
    learning rate to decline, this function requires discrete steps.
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float, decay_every_n_steps: int):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_every_n_steps = decay_every_n_steps

    def calculate(self, iteration: int) -> float:
        return self.learning_rate / (1 + self.decay_rate * (iteration // self.decay_every_n_steps))


class SquareRootDecay():
    '''
    The learning rate decays by the square root of the iteration number.
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def calculate(self, iteration: int) -> float:
        return self.learning_rate * self.decay_rate / np.sqrt(iteration)


class InversePowerDecay():
    '''
    The learning rate decays by the 1 / iteration^power
    '''

    def __init__(self, initial_learning_rate: float, decay_rate: float, power: float | int):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.power = power

        if self.power < 0:
            raise ValueError('The power must be non-negative.')

    def calculate(self, iteration: int) -> float:
        return self.learning_rate * self.decay_rate / iteration**self.power