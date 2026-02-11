import numpy as np

# ============= ARRAY CREATION =============
# As you solve each challenge, add a comment explaining what the function does

# Challenge 1: Create array from list
# Create: [1, 2, 3, 4, 5]
pList = [1, 2, 3, 4, 5]
# https://numpy.org/doc/stable/reference/generated/numpy.array.html
# An array, any object exposing the array interface, an object whose
# __array__ method returns an array, or any (nested) sequence. If
# object is a scalar, a 0-dimensional array containing object is returned.
#
#   It falls under the "any (nested) sequence"
#   part of that docstring. NumPy recognizes
#   Python lists (and tuples, etc.) as sequences
#   and converts them element-by-element into an
#   ndarray.
arr1 = np.array(pList)
print("Challenge 1:", arr1)

# Challenge 2: Create array of zeros
# Create: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (10 zeros)
# np.zeros(shape) creates an array filled with 0
arr2 = np.zeros(10, dtype=int)
print("Challenge 2:", arr2)

# Challenge 3: Create array of ones
# Create: [1, 1, 1, 1, 1]
# np.ones(shape) creates an array filled with 1.0
arr3 = np.ones(5, dtype=int)
print("Challenge 3:", arr3)

# Challenge 4: Create range of numbers
# Create: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr4 = np.arange(10)
print("Challenge 4:", arr4)

# Challenge 5: Create range with step
# Create: [0, 2, 4, 6, 8]
arr5 = np.arange(0, 10, 2)
print("Challenge 5:", arr5)

# Challenge 6: Create evenly spaced numbers
# Create: 5 numbers evenly spaced between 0 and 1
# Result should be: [0.  , 0.25, 0.5 , 0.75, 1.  ]
arr6 = np.linspace(0, 1, 5)
print("Challenge 6:", arr6)

# Challenge 7: Create random integers
# Create: 10 random integers between 0 and 100
arr7 = np.random.randint(1, 100, 10)
print("Challenge 7:", arr7)

# Challenge 8: Create random floats from normal distribution
# Create: 5 random numbers from standard normal (mean=0, std=1)
# Hint: This is np.random.randn() - the one you asked about!
arr8 = np.random.randn(5)
print("Challenge 8:", arr8)

# Challenge 9: Create 2D array (matrix)
# Create: [[1, 2, 3],
#          [4, 5, 6]]
arr9 = np.array([[1, 2, 3], [4, 5, 6]])
print("Challenge 9:", arr9)

# Challenge 10: Create 3x3 identity matrix
arr10 = np.eye(3)
print("Challenge 10:\n", arr10)

# Verify all are NumPy arrays
print("\n=== Type Check ===")
for i in range(1, 11):
    arr_name = f"arr{i}"
    arr = locals()[arr_name]
    print(f"{arr_name}: {type(arr)}")

print("\n" + "="*50)
print("SHAPE MANIPULATION")
print("="*50)

# --- --- ---
# ====== Shape Manipulation ======

# Challenge 11: Check shape
# Given array, what's its shape?
arr = np.array([[1, 2, 3], [4, 5, 6]])
shape = arr.shape
print(f"Challenge 11 - Array shape: {shape}")

# Challenge 12: Reshape to different dimensions
# Take: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Make it: [[1, 2, 3, 4],
#           [5, 6, 7, 8],
#           [9, 10, 11, 12]]
arr12_original = np.arange(1, 13)
arr12 = arr12_original.reshape(3, 4)
print(f"Challenge 12:\n{arr12}")

# Challenge 13: Reshape with -1 (auto-calculate dimension)
# Take: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# Make it: [[0, 1, 2],
#           [3, 4, 5],
#           [6, 7, 8]]
# But use -1 to let NumPy figure out one dimension
arr13_original = np.arange(9)
arr13 = arr13_original.reshape(-1, 3)
print(f"Challenge 13:\n{arr13}")

# Challenge 14: Flatten back to 1D
# Take the result from Challenge 13
# Make it back to: [0, 1, 2, 3, 4, 5, 6, 7, 8]
arr14 = arr13.flatten()
print(f"Challenge 14: {arr14}")

# Challenge 15: Transpose
# Take: [[1, 2, 3],
#        [4, 5, 6]]
# Make: [[1, 4],
#        [2, 5],
#        [3, 6]]
arr15_original = np.array([[1, 2, 3], [4, 5, 6]])
arr15 = arr15_original.T
print(f"Challenge 15:\n{arr15}")

# Challenge 16: Add dimension
# Take: [1, 2, 3]  (shape: (3,))
# Make: [[1, 2, 3]]  (shape: (1, 3))
#
#  axis=0 tells np.expand_dims where to insert the
#   new dimension.
#
#   For a shape (3,) array:
#   - axis=0 → inserts dimension at position 0 → shape
#    becomes (1, 3) — a row vector
#   - axis=1 → inserts dimension at position 1 → shape
#    becomes (3, 1) — a column vector
arr16_original = np.array([1, 2, 3])
arr16 = np.expand_dims(arr16_original, axis=0)
print(f"Challenge 16: {arr16}, shape: {arr16.shape}")

# Challenge 17: Remove single dimensions
# Take: [[[1, 2, 3]]]  (shape: (1, 1, 3))
# Make: [1, 2, 3]  (shape: (3,))
arr17_original = np.array([[[1, 2, 3]]])
arr17 = np.squeeze(arr17_original)
print(f"Challenge 17: {arr17}, shape: {arr17.shape}")

# --- --- ---
# ====== Reshape vs Transpose

print("\n" + "="*50)
print("EXPERIMENT: Understanding Reshape vs Transpose")
print("="*50)

# Start with same array
original = np.arange(1, 7)
print(f"Original: {original}")

# Method 1: Reshape to (2, 3)
reshaped = original.reshape(2, 3)
print(f"\nReshaped to (2, 3):\n{reshaped}")

# Method 2: Reshape to (3, 2)
reshaped2 = original.reshape(3, 2)
print(f"\nReshaped to (3, 2):\n{reshaped2}")

# Method 3: Reshape to (2, 3) then transpose
reshaped_then_transposed = original.reshape(2, 3).T
print(f"\nReshape (2,3) then transpose:\n{reshaped_then_transposed}")

# Question: Are reshaped2 and reshaped_then_transposed the same?
print(f"\nAre they equal? {np.array_equal(reshaped2, reshaped_then_transposed)}")

print("\n" + "="*50)
print("VISUALIZATION: How reshape fills")
print("="*50)

original = np.arange(1, 7)
print(f"Original (flat): {original}")
print("\nThink of it as: [1, 2, 3, 4, 5, 6]")

print("\nReshape to (3, 2) - fills ROWS first:")
print("Row 0: take next 2 → [1, 2]")
print("Row 1: take next 2 → [3, 4]")
print("Row 2: take next 2 → [5, 6]")
reshaped = original.reshape(3, 2)
print(f"Result:\n{reshaped}")

print("\nReshape to (2, 3), THEN transpose:")
print("Step 1 - Reshape to (2,3), fills rows:")
step1 = original.reshape(2, 3)
print(f"{step1}")
print("\nStep 2 - Transpose (flip rows↔columns):")
step2 = step1.T
print(f"{step2}")

print("\n💡 Key insight: Reshape and transpose are DIFFERENT operations!")
print("Reshape: redistributes elements row-by-row")
print("Transpose: flips the matrix along diagonal")