import numpy as np

print("="*50)
print("LINEAR ALGEBRA FOR NEURAL NETWORKS")
print("="*50)

# Challenge 41: Dot product - the fundamental operation
# You did this on Day 1, but let's understand it deeper
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Three equivalent ways:
dot1 = np.dot(a, b)
dot2 = a @ b
dot3 = np.sum(a * b)

print(f"\nChallenge 41 - Dot Product:")
print(f"np.dot(a, b) = {dot1}")
print(f"a @ b = {dot2}")
print(f"sum(a * b) = {dot3}")
print(f"All equal? {dot1 == dot2 == dot3}")

# Challenge 42: Geometric interpretation
# Dot product measures "how aligned" two vectors are
# If they point same direction: large positive
# If perpendicular: zero
# If opposite: large negative

v1 = np.array([1, 0])  # Points right
v2 = np.array([1, 0])  # Points right (same direction)
v3 = np.array([0, 1])  # Points up (perpendicular)
v4 = np.array([-1, 0]) # Points left (opposite)

print(f"\nChallenge 42 - Alignment:")
print(f"v1·v2 (same direction) = {v1 @ v2}")
print(f"v1·v3 (perpendicular) = {v1 @ v3}")
print(f"v1·v4 (opposite) = {v1 @ v4}")

# Challenge 43: Matrix-vector multiplication
# This is the CORE of a neural network layer!
# Remember: weights @ inputs from Day 1

# Weight matrix: 3 neurons, each with 4 input weights
W = np.array([
    [0.1, 0.2, 0.3, 0.4],  # Neuron 1 weights
    [0.5, 0.6, 0.7, 0.8],  # Neuron 2 weights
    [0.9, 1.0, 1.1, 1.2]   # Neuron 3 weights
])

# Input vector: 4 features
x = np.array([1.0, 2.0, 3.0, 4.0])

# Compute activations
activations = W @ x  # Shape: (3,)

print(f"\nChallenge 43 - Neural Layer:")
print(f"Weight matrix shape: {W.shape}")
print(f"Input shape: {x.shape}")
print(f"Activations shape: {activations.shape}")
print(f"Activations: {activations}")

# What's happening? Each neuron computes:
# neuron_i = w_i1*x1 + w_i2*x2 + w_i3*x3 + w_i4*x4
print(f"\nManual verification:")
print(f"Neuron 0: {W[0] @ x} (matches: {activations[0]})")
print(f"Neuron 1: {W[1] @ x} (matches: {activations[1]})")
print(f"Neuron 2: {W[2] @ x} (matches: {activations[2]})")

# Challenge 44: Matrix-matrix multiplication (batch processing!)
# Multiple inputs at once

# Batch of 5 samples, each with 4 features
X_batch = np.random.randn(5, 4)

# Process entire batch at once
batch_activations = X_batch @ W.T

print()
print("Why W.T? Think about shapes!")
print("I need to transpose W so the shapes of x_batch and W are compatible")

print(f"\nChallenge 44 - Batch Processing:")
print(f"Batch input shape: {X_batch.shape}")
print(f"Weight matrix shape: {W.shape}")
print(f"Batch activations shape: {batch_activations.shape}")
print(f"Expected: (5, 3) - 5 samples, 3 neurons")

print("\n" + "="*50)
print("THE MATRIX MULTIPLICATION SHAPE RULE")
print("="*50)

# The rule: (m, n) @ (n, p) = (m, p)
#           ^^^^      ^^^^     ^^^^
#           rows  these must   result
#           in A   MATCH!      shape

# Example 1: Valid multiplication
A = np.random.randn(3, 4)  # 3 rows, 4 cols
B = np.random.randn(4, 5)  # 4 rows, 5 cols
C = A @ B                   # Result: (3, 5)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C = A @ B shape: {C.shape}")
print(f"Rule: (3,4) @ (4,5) = (3,5) ✓")

# Example 2: Your batch processing
X_batch = np.random.randn(5, 4)  # 5 samples, 4 features
W = np.random.randn(3, 4)        # 3 neurons, 4 weights each

print(f"\nBatch processing problem:")
print(f"X_batch: {X_batch.shape}")
print(f"W: {W.shape}")

# Try this - it will FAIL!
try:
    result = X_batch @ W
except ValueError as e:
    print(f"X_batch @ W fails: {e}")

# Solution: Transpose W
print(f"\nSolution: Transpose W")
print(f"X_batch: {X_batch.shape} = (5, 4)")
print(f"W.T: {W.T.shape} = (4, 3)")
result = X_batch @ W.T
print(f"X_batch @ W.T: {result.shape} = (5, 3) ✓")

print("\nWhy it works:")
print("(5, 4) @ (4, 3) = (5, 3)")
print(" ^^^      ^^^     ^^^^^^")
print(" samples  match!  samples × neurons")

print("\n" + "="*50)
print("CONNECTION: Gradient Descent (OR → ML)")
print("="*50)

# In your flow shop optimization, you probably did something like:
# solution_new = solution_old + perturbation
# if better: keep it
# if worse: maybe keep it (simulated annealing?)

# In neural networks, it's the same but with calculus:
# weights_new = weights_old - learning_rate * gradient

# Let's simulate one gradient descent step
print("Simulating one training step:\n")

# Current weights (decision variables)
weights = np.array([0.5, -0.3, 0.8])
print(f"Current weights: {weights}")

# Gradient (direction that increases loss - we want to go opposite!)
gradient = np.array([0.2, -0.1, 0.3])
print(f"Gradient (∇Loss): {gradient}")

# Learning rate (step size - like temperature in SA)
learning_rate = 0.1
print(f"Learning rate (α): {learning_rate}")

# Update step
new_weights = weights - learning_rate * gradient
print(f"\nNew weights: {new_weights}")
print(f"Change: {new_weights - weights}")

print("\n💡 This is identical to your metaheuristic updates!")
print("   OR:  x_new = x_old + step")
print("   ML:  w_new = w_old - α·∇L(w)")

# --- --- ---

print("\n" + "="*50)
print("ELEMENT-WISE vs MATRIX OPERATIONS - Don't Mix Them Up!")
print("="*50)

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Element-wise multiplication (*)
elementwise = A * B
print(f"A * B (element-wise):\n{elementwise}")
print("Each element multiplied: [[1*5, 2*6], [3*7, 4*8]]")

# Matrix multiplication (@)
matmul = A @ B
print(f"\nA @ B (matrix multiplication):\n{matmul}")
print("Dot products: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]")

print("\n⚠️ COMMON BUG in neural networks:")
print("Using * when you meant @")
print("Using @ when you meant *")

# Real example from neural networks:
weights = np.array([[0.1, 0.2], [0.3, 0.4]])
inputs = np.array([[1.0, 2.0]])

correct = inputs @ weights.T  # Matrix multiplication
wrong = inputs * weights      # Element-wise (different shape!)

print(f"\nCorrect (inputs @ weights.T): {correct}")
print(f"Shape: {correct.shape}")

try:
    print(f"\nWrong (inputs * weights): {wrong}")
    print(f"Shape: {wrong.shape}")
    print("⚠️ This broadcasts and gives wrong result!")
except:
    print("This might even crash!")