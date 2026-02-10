import numpy as np

# Two vectors
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Makes sense - same shape
result1 = a + b
print("Same shape:", result1)

# Wait, what? Different shapes!
c = np.array([100])
result2 = a + c
print("Different shapes:", result2)

# This seems impossible...
d = np.array([[1, 2, 3],
              [4, 5, 6]])
e = np.array([10, 20, 30])
result3 = d + e
print("Matrix + vector:", result3)