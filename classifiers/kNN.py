import numpy as np

class kNN:
    def __init__(self, k=1, distance_measure = 'euclidian', training_data=None):
        self.k = k
        self.distance_measure = distance_measure
        self.training_data = training_data

    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k

    def get_distance_measure(self):
        return self.distance_measure

    def set_distance_measure(self, distance_measure):
        self.distance_measure = distance_measure

    def get_closest_k_indices(self, x, y):
        x = np.array(x)
        k = self.get_k()

        rows = x.shape[0]
        attr = x.shape[1]

        for i in len(rows):
            distance = list()
            index = list()
            for num in range(k):
                distance.append(sys.maxsize)
                index.append(-1)

            for j in len(rows):
                if i == j:
                    continue
                dist = 0
                for z in range(len(attr)):
                    dist = dist + (x[i, z] - x[j, z])**2
                dist = np.sqrt(dist)

                for z in range(len(distance)):
                    if dist < distance[z]:
                        for y in range(z+1, len(distance)):
                            dist[z] = dist[y]





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
def remove_tomek_links(x,y):
    x = np.array(x)
    y = np.array(y)

    x = np.array(x)

    closest_indices = get_closest_index(x)

    tomek_link_indices = set()
    num_examples = x.shape[0]
    for i in range(num_examples):
        # closest_indices[i] == closest_indices[closest_indices[i]] -> x is nearest neighbor of y and vice versa
        # y[i] != y[closest_indices[i]] classes are not same
       if i not in tomek_link_indices and closest_indices[i] == closest_indices[closest_indices[i]] and y[i] != y[closest_indices[i]]:
            tomek_link_indices.add(closest_indices[i])
            tomek_link_indices.add(closest_indices[closest_indices[i]])

    np.delete(x, tomek_link_indices)
    np.delete(y, tomek_link_indices)




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









