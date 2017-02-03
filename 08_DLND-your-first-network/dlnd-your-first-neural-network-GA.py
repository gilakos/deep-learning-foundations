'''
project: Udacity Deep Learning Foundations
lesson: Your first neural network
author: Gil Akos
'''

## Step 1 // Import Libraries
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Step 2 // Load and prepare the data
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
#print(rides.head())

## Step 3 // Checking out the data
rides[:24*10].plot(x='dteday', y='cnt')

## Step 4 // Dummy Variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
#print(data.head())

## Step 5 // Scaling target variables
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

## Step 6 // Splitting the data into training, testing, and validation sets
# Save the last 21 days
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

## Step 7 // Build the Network

#### Add a sigmoid function to the class
def activation(x):
    """
    Calculate sigmoid as activation
    """
    return 1 / (1 + np.exp(-x))
    #return lambda x: 1 / (1 + np.exp(-x))

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        # DEBUG to not print over each epoch
        self.DEBUG = False

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = activation

    def train(self, inputs_list, targets_list):
        """
        Train the model with inputs relative to targets
        """

        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer ## No activation function here since we are using regression, not classification

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        error = targets - final_outputs  # Output layer error is the difference between desired target and actual output.
        output_errors = error * final_outputs * (1-final_outputs)  ## compute error gradient in the output unit

        # TODO: Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)  # errors propagated to the hidden layer
        hidden_grad = hidden_errors * hidden_outputs * (1 - hidden_outputs)  ## Calculate error gradient for hidden layer

        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T) # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * np.dot(hidden_grad, inputs.T) # update input-to-hidden weights with gradient descent step

        if self.DEBUG:
            print ("self.weights_hidden_to_output.shape: ", self.weights_hidden_to_output.shape)
            print ("output_errors.shape: ", output_errors.shape)
            print ("hidden_errors.shape: ", hidden_errors.shape)
            print ("hidden_outputs.shape: ", hidden_outputs.shape)
            print ("hidden_grad.shape: ", hidden_grad.shape)
            self.DEBUG = False

    def run(self, inputs_list):
        """
        Run a single forward pass
        """
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = self.activation_function(final_inputs)  # signals from final output layer

        return final_outputs

def MSE(y, Y):
    return np.mean((y - Y) ** 2)

## Step 8 // Training the network
import sys

### Set the hyperparameters here ###
epochs = 100
learning_rate = 0.05
hidden_nodes = 3
output_nodes = 2

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values,train_targets.ix[batch]['cnt']):
        print(record)
        print(target)
        network.train(record, target)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

## Step 9 // Dummy Variables
## Step 10 // Dummy Variables
