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
plt.plot(error_over_epochs_1, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1)')
#plt.show()
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
plt.plot(error_over_epochs_2, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.01)')
#plt.show()
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
plt.plot(error_over_epochs_3, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.001)')
#plt.show()
plt.savefig('Classifier3.png')

###### Classifier4
classifier4 = perceptron.Perceptron(learning_rate=0.0001)
classifier4.train(training_x, training_y, iterations=10000)

prediction = classifier4.predict(testing_x)
accuracy = classifier4.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 4 is: ", accuracy)

accuracy_over_epochs_4 = classifier3.get_accuracy_list()
error_over_epochs_4 = classifier3.get_error_list()

plt.figure(4)
plt.plot(error_over_epochs_4, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.0001)')
#plt.show()
plt.savefig('Classifier4.png')

# Retry with fewer number of attributes
training2 = training_x[:, 0:10]
testing2 = testing_x[:, 0:10]

###### Classifier5
classifier5 = perceptron.Perceptron(learning_rate=0.1)
classifier5.train(training2, training_y, iterations=10000)

prediction = classifier5.predict(testing2)
accuracy = classifier5.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 5 is: ", accuracy)

accuracy_over_epochs_5 = classifier5.get_accuracy_list()
error_over_epochs_5 = classifier5.get_error_list()

plt.figure(5)
plt.plot(error_over_epochs_5, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1, first 10 attributes)')
#plt.show()
plt.savefig('Classifier5.png')

# Retry with fewer number of attributes
training3 = training_x[:, 0:15]
testing3 = testing_x[:, 0:15]

###### Classifier5
classifier6 = perceptron.Perceptron(learning_rate=0.1)
classifier6.train(training3, training_y, iterations=10000)

prediction = classifier6.predict(testing3)
accuracy = classifier6.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 6 is: ", accuracy)

accuracy_over_epochs_6 = classifier6.get_accuracy_list()
error_over_epochs_6 = classifier6.get_error_list()

plt.figure(6)
plt.plot(error_over_epochs_6, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1, first 15 attributes)')
#plt.show()
plt.savefig('Classifier6.png')

# Retry with fewer number of attributes
training4 = training_x[:, 0:18]
testing4 = testing_x[:, 0:18]

###### Classifier7
classifier7 = perceptron.Perceptron(learning_rate=0.1)
classifier7.train(training4, training_y, iterations=10000)

prediction = classifier7.predict(testing4)
accuracy = classifier7.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 7 is: ", accuracy)

accuracy_over_epochs_7 = classifier7.get_accuracy_list()
error_over_epochs_7 = classifier7.get_error_list()

plt.figure(7)
plt.plot(error_over_epochs_7, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1, first 18 attributes)')
#plt.show()
plt.savefig('Classifier7.png')

# Retry with fewer number of attributes
training5 = training_x[:, training_x.shape[1]-10:training_x.shape[1]]
testing5 = testing_x[:, training_x.shape[1]-10:training_x.shape[1]]

###### Classifier8
classifier8 = perceptron.Perceptron(learning_rate=0.1)
classifier8.train(training5, training_y, iterations=10000)

prediction = classifier8.predict(testing5)
accuracy = classifier8.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 8 is: ", accuracy)

accuracy_over_epochs_8 = classifier8.get_accuracy_list()
error_over_epochs_8 = classifier8.get_error_list()

plt.figure(8)
plt.plot(error_over_epochs_8, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1, last 10 attributes)')
#plt.show()
plt.savefig('Classifier8.png')

# Retry with fewer number of attributes
training6 = training_x[:, training_x.shape[1]-15:training_x.shape[1]]
testing6 = testing_x[:, training_x.shape[1]-15:training_x.shape[1]]
###### Classifier9
classifier9 = perceptron.Perceptron(learning_rate=0.1)
classifier9.train(training6, training_y, iterations=10000)

prediction = classifier9.predict(testing6)
accuracy = classifier9.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 9 is: ", accuracy)

accuracy_over_epochs_9 = classifier9.get_accuracy_list()
error_over_epochs_9 = classifier9.get_error_list()

plt.figure(9)
plt.plot(error_over_epochs_9, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1, last 15 attributes)')
#plt.show()
plt.savefig('Classifier9.png')

# Retry with fewer number of attributes
training7 = training_x[:, training_x.shape[1]-18:training_x.shape[1]]
testing7 = testing_x[:, training_x.shape[1]-18:training_x.shape[1]]

###### Classifier10
classifier10 = perceptron.Perceptron(learning_rate=0.1)
classifier10.train(training7, training_y, iterations=10000)

prediction = classifier10.predict(testing7)
accuracy = classifier10.get_accuracy(prediction, testing_y)
print("Accuracy of classifier 10 is: ", accuracy)

accuracy_over_epochs_10 = classifier10.get_accuracy_list()
error_over_epochs_10 = classifier10.get_error_list()

plt.figure(10)
plt.plot(error_over_epochs_10, ".")
plt.ylabel('Error on training set')
plt.xlabel('Iterations')
plt.title('Error rate Vs Iterations (learning rate = 0.1, last 18 attributes)')
#plt.show()
plt.savefig('Classifier10.png')

