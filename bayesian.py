import numpy as np


class Bayesian:
    """
    Algorithm from Introduction to Machine Learning (2nd edition) by Miroslav Kubat
    Classification with Naive-Bayes Principle
    Example to be described by x = (x1, ... , xn)
    1. For each xi and for each class cj, calculate the condition probability P(xi | cj)
            as relative frequency of xi among those training examples that belong to cj

    2. For each class, cj, carry out the following two steps
        i) estimate P(cj) as relative frequency of this class in training set
        ii) calculate the conditional probability P(x | cj) using naive
                            assumption of mutually independent attributes
                    P(x | cj) = product from i = 1 to n of P(xi | cj)

    3. Choose class with high value of P(cj) multiplied by product from i = 1 to n of P(xi | cj)
    """

    def __init__(self, m=0):
        self.m = m

    def train(self, x, y):
        """
        :param x: Feature Matrix
        :param y: Class Vector
        :return: vector with each unique class
                 vector of class probabilities
                 list of Conditional Probabilities (classed defined below), one for each unique attribute value per attribute per class
        """
        x = np.array(x)
        y = np.array(y)

        if len(x.shape) != 2:
            raise ValueError("Number of dimensions of feature matrix are off")

        if len(y.shape) != 2:
            raise ValueError("Number of dimensions of class vector are off")

        num_attributes = x.shape[1]
        num_examples = x.size / x[0].size
        if num_examples != y.size:
            raise ValueError("Number of Examples in feature matrix and class vector are not equal")

        # get probability of each class
        classes, counts = np.unique(y, return_counts=True)

        probabilities = np.empty(classes.size)
        for i in range(classes.size):
            probabilities[i] = counts[i] / num_examples

        cond_probabilities = []
        # get conditional probabilities
        # for every attribute of x
        for i in range(num_attributes):
            attribute = x[:, i]
            # for every possible output class
            for unique_class in classes:
                indices = np.where(y == unique_class)
                attribute_given_certain_class = attribute[indices[0]]
                unique_attribute_values_post, attribute_value_count = \
                    np.unique(attribute_given_certain_class, return_counts=True)
                # for every unique attribute value given the class
                for j in range(unique_attribute_values_post.size):
                    value = unique_attribute_values_post[j]
                    conditional_probability = ConditionalProb(i, value, unique_class)
                    conditional_probability.set_probability(attribute_value_count[j] / indices[0].size)
                    cond_probabilities.append(conditional_probability)

        return classes, probabilities, cond_probabilities

    def predict(self, x, classes, class_probabilities, cond_probabilities):
        """
        :param x: feature vector to be predicted
        :param classes: vector of classes
        :param class_probabilities: matrix of probabilities of each class
        :param cond_probabilities: list of conditional probabilities of each attribute value given the class
        :return:
        """
        prediction = []
        # for every example presented
        for i in range(x.shape[0]):
            current_attribute_value = x[i, :]
            probability = []
            # for every class
            for j in range(classes.size):
                product = 1
                current_class = classes[j]
                # for every attribute
                for k in range(current_attribute_value.size):
                    # find conditional probability of attribute given class
                    for cond_prob in cond_probabilities:
                        if k == cond_prob.get_attribute_number() and current_attribute_value[k] == cond_prob.get_attribute_value() and current_class == cond_prob.get_class_value():
                                    product *= cond_prob.get_probability()
                # since class_probabilities and classes are in same order, multiply by the jth class probability
                probability.append(class_probabilities[j] * product)
            index = probability.index(max(probability))
            prediction.append(classes[index])

        results = np.array(prediction).reshape(len(prediction), 1)
        return results

class ConditionalProb:
    def __init__(self, attribute_number, attribute_value, class_value):
        self.attribute_number = attribute_number
        self.attribute_value = attribute_value
        self.class_value = class_value
        self.probability = 0

    def get_attribute_number(self):
        return self.attribute_number

    def get_attribute_value(self):
        return self.attribute_value

    def get_class_value(self):
        return self.class_value

    def get_probability(self):
        return self.probability

    def set_probability(self, probability):
        self.probability = probability
