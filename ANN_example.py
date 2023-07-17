import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the input and output data
input_data = np.array([[0.1, 0.2, 0.4, 0.15, 0.6], [0.3, 0.1, 0.25, 0.75, 0.5]])
output_data = np.array([[0.2, 0.3, 0.1, 0.5], [0.3, 0.1, 0.2, 0.7]])

# Make sure the input and output data have the same number of samples
assert input_data.shape[0] == output_data.shape[0], "Number of samples in input and output data should match"

# Define the ANN model
model = Sequential()
model.add(Dense(8, input_dim=input_data.shape[1], activation='relu'))
model.add(Dense(4, activation='sigmoid'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(input_data, output_data, epochs=1000, verbose=0)

# Test the model with slices of the input array
for test_input in input_data:
    test_input = test_input.reshape(1, -1)
    output = model.predict(test_input)
    print(output)