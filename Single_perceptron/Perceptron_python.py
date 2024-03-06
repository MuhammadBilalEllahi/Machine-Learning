import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01, num_epochs=100):
        
        self.weights = np.zeros(num_inputs + 1)  # Add one for the bias weight
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def activation(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Add bias
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.num_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)  # Update bias weight



training_inputs = np.array([[0, 0], 
                            [0, 1], 
                            [1, 0], 
                            [1, 1]])

labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(num_inputs=2)
perceptron.train(training_inputs, labels)

# Test the trained perceptron
inputs = np.array([0,0])
print(perceptron.predict(inputs))  # Output: 1

inputs = np.array([0,1])
print(perceptron.predict(inputs)) 

inputs = np.array([1,0])
print(perceptron.predict(inputs)) 
 
inputs = np.array([1,1])
print(perceptron.predict(inputs)) 