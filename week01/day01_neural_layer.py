import numpy as np

def neural_layer_forward(inputs, weights, biases):
    """
    inputs: shape (n_features,) - one data sample
    weights: shape (n_neurons, n_features) - weight matrix
    biases: shape (n_neurons,) - bias vector

    Returns: shape (n_neurons,) - activations before nonlinearity
    """
    # Your code here - use @ and broadcasting, no loops!
    return weights @ inputs + biases

# Test it
inputs = np.array([1.5, 2.0, 0.5])  # 3 features
weights = np.array([
    [0.2, 0.8, -0.5],  # Neuron 1 weights
    [0.5, -0.3, 0.7]  # Neuron 2 weights
])
biases = np.array([0.1, -0.2])

result = neural_layer_forward(inputs, weights, biases)
print("Neuron activations:", result)
print()

# --- --- ---

# Batch processing - multiple inputs at once!
def neural_layer_forward_batch(inputs_batch, weights, biases):
    """
    inputs_batch: shape (batch_size, n_features)
    weights: shape (n_neurons, n_features)
    biases: shape (n_neurons,)

    Returns: shape (batch_size, n_neurons)
    """
    # Can you modify your function to handle multiple inputs at once?
    # Hint: The @ operator handles this automatically!

    #return (weights @ inputs_batch.T + biases.reshape(-1, 1)).T
    return inputs_batch @ weights.T + biases

    ### From PyTorch's nn.Linear layer
    ### def forward(self, input):
    ###    return input @ self.weight.T + self.bias

# Test with 4 samples
batch_inputs = np.array([
    [1.5, 2.0, 0.5],
    [0.5, 1.0, 2.0],
    [2.0, 0.5, 1.5],
    [1.0, 1.0, 1.0]
])

# Repeat the weights and biases just for easier readability
weights = np.array([
    [0.2, 0.8, -0.5], # Neuron 1 weights
    [0.5, -0.3, 0.7]  # Neuron 2 weights
])
biases = np.array([0.1, -0.2])

batch_result = neural_layer_forward_batch(batch_inputs, weights, biases)
print("\nBatch activations shape:", batch_result.shape)
print("Batch activations:\n", batch_result)