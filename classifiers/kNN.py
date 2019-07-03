import numpy as np
import pandas as pd
import sys

class kNN:
    """
        Algorithm from Introduction to Machine Learning (2nd edition) by Miroslav Kubat

        Simplest version of kNN classifier

        Suppose we have a mechanism to evaluate the similarity between attribute vectors.
        Let x denote the object whose class we want to determine.

        1. Among the training examples, identify the k nearest neighbors of x (examples that are the most similar to x)
        2. Let ci be the class most frequently found amongst these k nearest neighbors
        3. Label x with ci

        Similarity - different distance measures
    """
    def __init__(self, k=1, distance_measure='euclidian', training_data=None, training_data_labels=None):
        self.k = k
        self.distance_measure = distance_measure
        self.training_data = training_data
        self.training_data_labels = training_data_labels
        self.training_data_accuracy = None
        self.set_training_accuracy()
        self.training_data_removed = None
        self.training_data_labels_removed = None
        self.training_data_accuracy_removed = None
        self.num_examples_removed = 0


    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k

    def get_distance_measure(self):
        return self.distance_measure

    def set_distance_measure(self, distance_measure):
        self.distance_measure = distance_measure

    def get_training_data(self):
        return self.training_data

    def set_training_data(self, x):
        self.training_data = x

    def get_training_data_labels(self):
        return self.training_data_labels

    def set_training_data_labels(self, y):
        self.training_data_labels = y

    def get_training_accuracy(self):
        return self.training_data_accuracy

    def remove_tl(self):
        self.training_data_removed, self.training_data_labels_removed = remove_tomek_links(self.training_data, self.training_data_labels)
        self.num_examples_removed = self.training_data.shape[0] - self.training_data_removed.shape[0]
        self.training_data_accuracy_removed = get_accuracy(get_class_vector(self.k, self.training_data_removed, self.training_data_labels_removed), self.training_data_labels_removed)

    def get_num_examples_removed(self):
        return self.num_examples_removed

    def get_training_data_removed(self):
        return self.training_data_removed

    def get_training_data_labels_removed(self):
        return self.training_data_labels_removed

    def get_training_accuracy_removed(self):
        return self.training_data_accuracy_removed

    def set_training_accuracy(self):
        if self.training_data is None:
            self.training_data_accuracy = None
        else:
            ##########
            kNN_prediction = get_class_vector(self.k, self.training_data, self.training_data_labels)
            self.training_data_accuracy = get_accuracy(kNN_prediction, self.training_data_labels)

def get_accuracy( predicted, actual):
    """
    :param predicted: prediction array
    :param actual: true class values
    :return: percentage of correct results
    """
    results = np.equal(predicted, actual)

    return np.count_nonzero(results)/len(results)


def get_class_vector(k, x, y):
    """
    :param k: number of nearest neighbors to look at
    :param x: data to be classified
    :param y: data labels

    :return: class vector with voted label
    """
    x = np.array(x)
    y = np.array(y)
    result = np.zeros(y.shape)

    if k <= 0:
        raise ValueError("k needs to be larger than 0")

    rows = x.shape[0]
    attr = x.shape[1]

    for i in range(rows):
        distance = [None] * k
        index = [None] * k

        for j in range(rows):
            if i == j:
                continue
            dist = 0
            for z in range(attr):
                dist = dist + (x[i, z] - x[j, z])**2
            dist = np.sqrt(dist)

            for z in range(k):
                if distance[z] is None:
                    distance[z] = dist
                    index[z] = z
                    continue
                if dist < distance[z]:

                    # Shift all elements one to the right to the right of and including index z
                    shift_iteration = 0
                    for a in range(z+1, k):
                        distance[a] = distance[z + shift_iteration]
                        index[a] = index[z + shift_iteration]
                        shift_iteration = shift_iteration + 1
                    distance[z] = dist
                    index[z] = z

        # create dictionary that holds the unique labels in the k nearest indicies
        labels = {}
        for z in range(k):
            label = y[index[z]]
            if label.dtype == 'float64':
                label = np.asscalar(label)
            else:
                label = np.ndarray.item(label)

            if label not in labels:
                labels[label] = 1
            else:
                labels[label] = labels[label] + 1

        # determine which label occurs the most
        max_count = 0
        max_label = None
        for key in labels:
            if labels[key] > max_count:
                max_count = labels[key]
                max_label = key

        result[i] = max_label

    return result

"""
Algorithm from Algorithm from Introduction to Machine Learning (2nd edition) by Miroslav Kubat

To identify and remove Tomek Links
(x is nearest neighbor of y, y is nearest neighbor of x, class of x is not the same as class of y)

Input: training set of N examples

1. Let i = 1 and let T be empty set
2. Let x be the i-th training example and let y be nearest neighbor of x
3. If x and y belong to same class go to 5
4. If x is nearest neighbor of y, let T = T U {x,y}
5. Let i = i + 1, if i <= N goto 2
6. Remove from training set all examples that are now in T
"""


def remove_tomek_links(x, y):
    x = np.array(x)
    y = np.array(y)

    nearest_neighbor = np.zeros(y.shape)
    rows = x.shape[0]
    attr = x.shape[1]
    for i in range(rows):
        closest_index = -1
        closest_distance = sys.maxsize
        for j in range(rows):
            if i == j:
                continue
            dist = 0
            for z in range(attr):
                dist = dist + (x[i, z] - x[j, z]) ** 2
            dist = np.sqrt(dist)

            if dist < closest_distance:
                closest_distance = dist
                closest_index = j
        nearest_neighbor[i] = closest_index

    tomek_link_indices = list()
    for i in range(rows):
        if i in tomek_link_indices:
            continue

        closest_i = np.ndarray.item(nearest_neighbor[i])
        closest_j = nearest_neighbor[int(closest_i)]

        if closest_j == i:
            if i not in tomek_link_indices:
                tomek_link_indices.append(i)
            if closest_i not in tomek_link_indices:
                tomek_link_indices.append(closest_i)

    new_x = np.delete(x, tomek_link_indices, 0)
    new_y = np.delete(y, tomek_link_indices).astype(float)

    return new_x, new_y


def get_closest_indices(x):
    num_examples = x.shape[0]
    num_attr = x.shape[1]

    closest_indices = np.empty(num_examples)
    for i in range(num_examples):
        distances = np.zeros(num_examples)
        for j in range(num_examples):
            distance = 0
            for k in range(num_attr):
                distance = distance + (x[j, k] - x[i, k]) ** 2

            distance = distance ** 0.5
            distances[j] = distance

        # find closest example
        smallest_value = float("inf")
        smallest_index = -1

        for j in range(num_examples):
            if distances[j] < smallest_value and i != j:
                smallest_value = distances[j]
                smallest_index = j

        if smallest_index != -1:
            closest_indices[i] = smallest_index

    return closest_indices










