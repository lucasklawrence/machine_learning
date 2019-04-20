import pandas as pd
import numpy as np
import perceptron
import matplotlib.pyplot as plt

data = pd.read_csv("datasets//parkinsons.txt", delimiter=",", header=None)

input = np.array(data)

num_rows = input.shape[0]
num_columns = input.shape[1]

np.random.shuffle(input)

training_xa = input[0:int(num_rows*.6), 0:17]
training_xb = input[0:int(num_rows*.6), 18:num_columns]
training_x = np.concatenate((training_xa, training_xb), axis=1)

training_y = input[0:int(num_rows*.6), 16:17]

testing_xa = input[int(num_rows*.6): num_rows, 0:17]
testing_xb = input[int(num_rows*.6): num_rows, 18:num_columns]
testing_x = np.concatenate((testing_xa, testing_xb), axis=1)

testing_y = input[int(num_rows*.6): num_rows, 16:17]

###### Classifier 1
classifier = perceptron.Perceptron(learning_rate=0.1)
classifier.train(training_x, training_y, iterations=10000)

prediction = classifier.predict(testing_x)
accuracy = classifier.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 1 is: ", accuracy)

accuracy_over_epochs_1 = classifier.get_accuracy_list()
error_over_epochs_1 = classifier.get_error_list()

plt.figure(1)
plt.plot(error_over_epochs_1)
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1)')
plt.show()
plt.savefig('Classifier1.png')

###### Classifier2
classifier2 = perceptron.Perceptron(learning_rate=0.01)
classifier2.train(training_x, training_y, iterations=10000)

prediction = classifier2.predict(testing_x)
accuracy = classifier2.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 2 is: ", accuracy)

accuracy_over_epochs_2 = classifier2.get_accuracy_list()
error_over_epochs_2 = classifier2.get_error_list()

plt.figure(2)
plt.plot(error_over_epochs_2)
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.01)')
plt.show()
plt.savefig('Classifier2.png')

###### Classifier3
classifier3 = perceptron.Perceptron(learning_rate=0.001)
classifier3.train(training_x, training_y, iterations=10000)

prediction = classifier3.predict(testing_x)
accuracy = classifier3.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 3 is: ", accuracy)

accuracy_over_epochs_3 = classifier3.get_accuracy_list()
error_over_epochs_3 = classifier3.get_error_list()

plt.figure(3)
plt.plot(error_over_epochs_3)
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.001)')
plt.show()
plt.savefig('Classifier3.png')

###### Classifier4
classifier4 = perceptron.Perceptron(learning_rate=0.0001)
classifier4.train(training_x, training_y, iterations=10000)

prediction = classifier4.predict(testing_x)
accuracy = classifier4.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 3 is: ", accuracy)

accuracy_over_epochs_4 = classifier3.get_accuracy_list()
error_over_epochs_4 = classifier3.get_error_list()

plt.figure(4)
plt.plot(error_over_epochs_4)
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.0001)')
plt.show()
plt.savefig('Classifier4.png')


