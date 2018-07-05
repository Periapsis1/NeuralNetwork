import numpy as np
import math
from ast import literal_eval

class NeuralNet:
    def __init__(self, shape, alpha=1.0, weights=None, const=True):
        """
        This constructor method returns a neural network object.
        :param shape: (list of ints) The shape of the network with each int representing the number of neurons in that layer
        :param alpha: (float) Multiplier to the weights updater. Low numbers might make it too slow.
            High numbers might make it over-correct. Defaults to 1
        :param weights: (list of np.array of floats) The values multiplied to each signal.
            Defaults to a randomly generated list of arrays that depend on the shape of the network
        :param const: (boolean) Whether or not to include an initial bias of 1 at the input.
            This is so inputs of all 0s don't mess up the network and to improve its accuracy. Defaults to True
        """
        shape = list(shape)
        self.alpha = alpha
        self.shape = shape
        self.const = const
        self.stats = {'error': 0}
        self.deltas = [np.zeros(x) for x in self.shape[1:]]
        self.errors = [np.zeros(x) for x in self.shape[1:]]

        if const:
            self.shape[0] += 1

        self.neurons = [np.zeros(x) for x in self.shape]

        if weights is None:
            self.weights = [np.random.normal(0, 1, (shape[l], shape[l + 1])) for l in range(len(shape) - 1)]
        else:
            self.weights = weights

    def change_lr(self, n):
        self.alpha = n

    def reset_nodes(self):
        self.neurons = [np.zeros(x) for x in self.shape]

    def stimulate(self, inputs, targets=None):
        """
        This function is the main function of the network. It takes an input, propagates the signal
        forward through the network, then gives the output. If targets are provided, the network will
        update the weights based on the difference between the target and the actual output.
        :param inputs: (tuple/list/ndarray) The inputs that should match the number of input neurons
            or else it will yell at you
        :param targets: (tuple/list/ndarray) The target outputs that will be used to train the network.
            If provided, the network will train based off of these. If not, it will just give an output.
            Defaults to None.
        :return: (ndarray)the output of the neural network
        """

        if type(inputs) is tuple:
            inputs = list(inputs)

        if type(inputs) is list:
            inputs = np.array([inputs])

        if type(targets) is tuple:
            targets = list(targets)

        if type(targets) is list:
            targets = np.array(targets)

        if type(inputs) != np.ndarray:
            raise Exception("Bad input in network stimulate call (inputs): \n\n%s\n\n is not a list, tuple, or ndarray" % inputs)

        if targets is not None and type(targets) != np.ndarray:
            raise Exception("Bad input in network stimulate call (targets): \n\n%s\n\n is not a list, tuple, or ndarray" % targets)

        self.neurons[0] = inputs  # first we set the first layer of neurons to the input

        if self.const:
            self.neurons[0] = np.append(self.neurons[0], [1])

        for l in range(1, len(self.neurons)):
            self.neurons[l] = activate(np.dot(self.neurons[l - 1], self.weights[l - 1]))

        output = self.neurons[-1]

        if targets is not None:
            self.stats['error'] = output - targets
            self.errors[-1] = output - targets
            self.deltas[-1] = self.errors[-1] * output * (1 - output)

            for l in range(len(self.neurons) - 3, -1, -1):
                self.errors[l] = np.dot(self.deltas[l+1], self.weights[l+1].T)
                self.deltas[l] = self.errors[l] * self.neurons[l + 1] * (1 - self.neurons[l + 1])

            for l in range(len(self.weights)):

                self.weights[l] -= self.alpha * np.dot(
                    np.expand_dims(self.deltas[l], axis=0).T,
                    np.expand_dims(self.neurons[l], axis=0)
                ).T

        return output

    def __call__(self, inputs, targets=None):
        """
        see stimulate(self, inputs, targets=None)
        """
        return self.stimulate(inputs, targets)


def store_data(file_name, data, weights=True):
    """
    This file saves data to a file. If there are weights it stores them as lists
    Keep in mind that this overwrites the entire file
    :param file_name: (string) The file to get the data from
    :param data: (object) the data to store in the file
    :param weights: (boolean) Whether to store the data as weights. This will turn all np.arrays into lists
    :return:
    """
    if weights:
        for i in range(len(data)):
            data[i] = data[i].tolist()
    data_file = open(file_name, 'w')
    data_file.write(str(data))
    data_file.close()


def read_data(file_name, weights=True):
    """
    This function extracts data from a file.
    :param file_name: (string) the file to get data from
    :param weights: (boolean) whether to interpret data as a list of np.arrays
    :return: (object) the data the file was storing
    """
    data_file = open(file_name, 'r')
    raw_data = data_file.read()
    data = literal_eval(raw_data)
    if weights:
        for i in range(len(data)):
            data[i] = np.array(data[i])

    return data


def activate(n):
    """
    shrinks a number to between 0 and 1
    @param n: (number) the number to cap

    @returns: (float) the number between 0 and 1
    """

    return 1 / (1 + np.exp(-n))
