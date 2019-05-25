import numpy as np


class Winnow:
    """
       Algorithm from Introduction to Machine Learning (2nd edition) by Miroslav Kubat
       Classification with WINNOW

       Assumption: two classes, c(x) = 1 and c(x) = 0 and are linearly separable

       1. Initialize classifiers weights to wi = 1

       2. Set threshold to n - 0.1 (n being the number of attributes) and choose an appropriate
            learning rate > 1 (usually learning_rate = 2)
       3. Present training example, x whose class is c(x). Classifier returns h(x)
       4. If c(x) != h(x), update weights for each attribute whose value is xi = 1
                if c(x) = 1 and h(x) = 0, wi = learning_rate * wi
                if c(x) = 0 and h(x) = 1, wi = wi / learning_rate
       5. If c(x) = h(x) for all training examples, stop; otherwise, return to step 3
    """

    def __init__(self, learning_rate=2):
        self.learning_rate = learning_rate
        self.weights = None
        self.threshold = None
        self.initialized = False

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def initialize_weights(self, num_attributes):
        """
        :param num_attributes: number of attributes in feature

        sets values for Winnow weights to 1
        """
        initial_weights = np.ones((num_attributes, 1))
        self.set_weights(initial_weights)
        self.set_initialized(True)

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_initialized(self):
        return self.initialized

    def set_initialized(self, initialized):
        self.initialized = initialized

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
            self.set_threshold(num_attributes - 0.1)
        weights = self.get_weights()

        iteration_number = 0
        flag = 0
        while flag == 0 and iteration_number < iterations:
            updated_weights = False
            for i in range(x.shape[0]):

                feature = x[i, :]
                hypothesis = np.matmul(feature, weights)

                if hypothesis >= self.get_threshold():
                    hypothesis = 1
                else:
                    hypothesis = 0
                class_value = y[i, 0]
                if class_value != hypothesis:
                    for j in range(len(weights)):
                        weights[j] = weights[j] * self.get_learning_rate() ** ((class_value - hypothesis)*feature[j])

                    updated_weights = True
                    self.set_weights(weights)

            # c(x) = h(x) for all examples
            if not updated_weights:
                flag = 1

            iteration_number += 1

    def predict(self, x):
        """
        :param x: feature vector
        :return: np array w/ value 1 if sum is >= threshold and value 0 if sum is < threshold for the two classes
        """
        weights = self.get_weights()
        prediction = np.matmul(x, weights)

        for i in range(prediction.shape[0]):
            if prediction[i, 0] >= self.get_threshold():
                prediction[i, 0] = 1
            else:
                prediction[i, 0] = 0

        return prediction

    def get_accuracy(self, predicted, actual):
        """
        :param predicted: prediction array w/ 0 or 1 as class values
        :param actual: true class values (0 or 1)
        :return: percentage of correct results
        """
        results = np.equal(predicted, actual)

        return np.count_nonzero(results)/len(results)




