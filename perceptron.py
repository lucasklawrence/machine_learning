import numpy as np
import random

class Perceptron:
    """
       Algorithm from Introduction to Machine Learning (2nd edition) by Miroslav Kubat
       Classification with Perceptron Learning

       Assumption: two classes, c(x) = 1 and c(x) = 0 and are linearly separable

       1. Initialize all weights, wi, to small random numbers
       Choose an appropriate learning rate learningRate: (0,1]

       2. For each training example, x = (x1, ... , xn) whose class is c(x):
            i) Let h(x) = 1 if sum from i = 1 to n of wi * xi > 0 and h(x) = 0 otherwise
            ii) Update each weight using the formula, wi = wi + learningRate*[c(x) - h(x)] * xi

        3. If c(x) = h(x) for all training examples, stop; otherwise, return to step 2
       """

    def __init__ (self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.initialized = False

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights

    def initialize_weights(self, num_attributes):
        """
        :param num_attributes: number of attributes in feature

        sets values for Perceptron weights to small random values from [.5,.5]
        """
        initial_weights = np.empty((num_attributes + 1, 1))
        # initialize weights between -.5 and .5
        for i in range(num_attributes + 1):
            initial_weights[i, 0] = random.random() - 0.5
        self.set_weights(initial_weights)
        self.set_initialized(True)

    def get_initialized(self):
        return self.initialized

    def set_initialized(self, initialized):
        self.initialized = initialized

    def add_ones_column(self, x):
        """
        :param x: feature vector
        :return: column of 1s appended to left side of feature vector
        """
        x0 = np.ones((x.shape[0], 1))
        x = np.concatenate((x0, x), axis=1)
        x.astype(float)
        return x

    def train(self, x, y, iterations=10000):
        """
        :param x: feature vector
        :param y: class vector
        :param iterations: specify max number of iterations in case classes are not linearly separable
        :return:
        """

        x = np.array(x)
        y = np.array(y)

        if len(x.shape) != 2:
            raise ValueError("Number of dimensions of feature matrix are off")

        if len(y.shape) != 2:
            raise ValueError("Number of dimensions of class vector are off")

        num_attributes = x.shape[1]
        if not self.get_initialized():
            self.initialize_weights(num_attributes)
        weights = self.get_weights()

        # xi is always 1 for weight w0, so add a column of ones to the beginning of the feature array
        x = self.add_ones_column(x)

        iteration_number = 0
        flag = 0
        while flag == 0 and iteration_number < iterations:
            updated_weights = False
            for i in range(x.shape[0]):
                feature = x[i, :]
                hypothesis = np.matmul(feature, weights)

                if hypothesis >= 0:
                    hypothesis = 1
                else:
                    hypothesis = 0
                class_value = y[i, 0]
                if class_value != hypothesis:
                    for j in range(len(weights)):
                        weights[j] = weights[j] + self.get_learning_rate() * (class_value - hypothesis) * feature[j]

                    updated_weights = True
                    self.set_weights(weights)

            # c(x) = h(x) for all examples
            if not updated_weights:
                flag = 1

            iteration_number += 1

    def predict(self, x):
        """
        :param x: feature vector
        :return: np array w/ value 1 if sum is >= 0 and value 0 if sum is < 0 for the two classes
        """
        weights = self.get_weights()
        x = self.add_ones_column(x)

        prediction = np.matmul(x, weights)

        for i in range(prediction.shape[0]):
            if prediction[i, 0] >= 0:
                prediction[i, 0] = 1
            else:
                prediction[i, 0] = 0

        return prediction

    def get_accuracy(self, prediction, actual):
        """
        :param prediction: prediction array w/ 0 or 1 as class values
        :param actual: true class values (0 or 1)
        :return: percentage of correct results
        """
        results = np.equal(prediction, actual)

        return np.count_nonzero(results)/len(results)

