import numpy as np
import numpy.typing as npt
from typing import Optional

from .layer import Layer, InputLayer, TrainableLayer
from .optimizer import Optimizer
from .lossfunction import LossFunction


class Network():

    def __init__(self):
        self.layers: list[Layer] = []
        self.trainable_layers: list[TrainableLayer] = []
        self.evaluation_layers: list[Layer] = []


    def set(self, *, lossfunction: Optional[LossFunction] = None, optimizer: Optional[Optimizer] = None):
        
        if lossfunction is not None:
            self.lossfunction = lossfunction

        if optimizer is not None:
            self.optimizer = optimizer


    def add_layer(self, layer: Layer):
        self.layers.append(layer)


    def compile(self):

        assert self.lossfunction is not None
        assert self.optimizer is not None
        
        self.number_layers = len(self.layers)
        self.input_shape = self.layers[0].input_shape
        self.output_shape = self.layers[-1].output_shape

        for i, layer in enumerate(self.layers):

            if i == 0:
                layer.previous = InputLayer()
                layer.next = self.layers[i + 1]

            elif i < self.number_layers - 1:
                layer.previous = self.layers[i - 1]
                layer.next = self.layers[i + 1]

            else:
                layer.previous = self.layers[i - 1]

            if isinstance(layer, TrainableLayer):
                self.trainable_layers.append(layer)


    def train(self, x_input: npt.NDArray[np.float64], y_input: npt.NDArray[np.float64], *, epochs: int = 1000, log_epochs: int = 100,
              batch_size: Optional[int] = None, validation_size: float = 0., randomize: bool = True):

        N_train = int(round((1 - validation_size) * len(x_input)))
        N_validation = len(x_input) - N_train
        steps_per_epoch = N_train

        if batch_size is not None:
            steps_per_epoch = N_train // batch_size

            if steps_per_epoch * batch_size < N_train:
                steps_per_epoch += 1

        x_train, x_validation, y_train, y_validation = x_input[:N_train], x_input[N_train:], y_input[:N_train], y_input[N_train:]

        for epoch in range(1, epochs):
            

            if randomize:
                shuffle = np.random.permutation(N_train)
                x_train = x_train[shuffle]
                y_train = y_train[shuffle]

            total_loss = 0.

            for step in range(steps_per_epoch):

                if batch_size is not None:
                    
                    x_batch = x_train[step * batch_size: (step + 1) * batch_size]
                    y_batch = y_train[step * batch_size: (step + 1) * batch_size]
                else:
                    x_batch = x_train
                    y_batch = y_train

                y_batch_prediction = self.forward(x_batch, training_status = True)
                total_loss += self.lossfunction.forward(y_batch, y_batch_prediction)
                self.backward(y_batch, y_batch_prediction)

                for layer in self.trainable_layers:
                    self.optimizer.update_parameters(layer)
            self.optimizer.post_update()

            validation_loss = 0.

            if validation_size:
                y_validation_prediction = self.forward(x_validation, training_status = False)
                validation_loss = self.lossfunction.forward(y_validation, y_validation_prediction)

            if (epoch + 1) % log_epochs == 0 or epoch == epochs - 1 or epoch == 0:
                if validation_size:
                    print(f'epoch: {epoch + 1}/{epochs}, training loss: {total_loss / N_train:.5f}, validation loss: {validation_loss / N_validation:.5f}', end = '\r')
                else:
                    print(f'epoch: {epoch + 1}/{epochs}, training loss: {total_loss / N_train:.5f}', end = '\r')


    def forward(self, x_data: npt.NDArray[np.float64], training_status: bool = True) -> npt.NDArray[np.float64]:

        result = np.empty_like(self.output_shape)
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(x_data, training = training_status)
            else:
                layer.forward(layer.previous.output, training = training_status)

            result = layer.output
        return result


    def backward(self, y_true: npt.NDArray[np.float64], y_predicted: npt.NDArray[np.float64]):

        loss = self.lossfunction.backward(y_true, y_predicted)

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.backward(loss)
            else:
                layer.backward(layer.next.dinput)


    def predict(self, x_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.forward(x_data, training_status = False)
