import numpy as np
import time

def multiplyElementWiseAndAdd(x, y):
    """Multiply each element of x and y and add them together."""
    z = 0
    for i in range(len(x)):
        z += x[i] * y[i]
    return z

# Create sample vectors with 1 million entries each
x = [i for i in range(1000000)]
y = [i for i in range(1000000)]

# Time execution duration of the method I wrote
start = time.time()
z = multiplyElementWiseAndAdd(x,y)
end = time.time()
print(z)
print("It took: ", end - start)

# Time execution duration of dot product implementation of numpy
# Using int64 to avoid silent integer overflow on some platforms
x = np.arange(0, 1000000, 1, dtype=np.int64)
y = np.arange(0, 1000000, 1, dtype=np.int64)
start = time.time()
z = np.dot(x,y)
end = time.time()
print(z)
print("It took: ", end - start)

# Result: np.dot() is 80 times faster than my own implementation.
#         This matters because in a typical neural network training run,
#         you might do billions of these operations.