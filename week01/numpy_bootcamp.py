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

# --- --- ---
# ====== Challenge Set 3: Indexing and Slicing ======

print("\n" + "="*50)
print("DAY 3: INDEXING AND SLICING")
print("="*50)

# Challenge 18: Basic indexing
# Given: [10, 20, 30, 40, 50]
# Get: the 3rd element (30)
arr18 = np.array([10, 20, 30, 40, 50])
element = arr18[2]
print(f"Challenge 18: {element}")

# Challenge 19: Negative indexing
# Given: [10, 20, 30, 40, 50]
# Get: the LAST element (50)
arr19 = np.array([10, 20, 30, 40, 50])
last_element = arr19[-1]
print(f"Challenge 19: {last_element}")

# Challenge 20: Slicing - first 3 elements
# Given: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Get: [0, 1, 2]
arr20 = np.arange(10)
first_three = arr20[0:3]
print(f"Challenge 20: {first_three}")

# Challenge 21: Slicing - last 3 elements
# Given: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Get: [7, 8, 9]
arr21 = np.arange(10)
last_three = arr21[-3:10]
print(f"Challenge 21: {last_three}")

# Challenge 22: Slicing with step
# Given: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Get: every 2nd element: [0, 2, 4, 6, 8]
arr22 = np.arange(10)
every_second = arr22[0:10:2]
print(f"Challenge 22: {every_second}")

# Challenge 23: Reverse an array
# Given: [0, 1, 2, 3, 4]
# Get: [4, 3, 2, 1, 0]
arr23 = np.arange(5)
reversed_arr = arr23[::-1]
print(f"Challenge 23: {reversed_arr}")

# Challenge 24: 2D indexing - single element
# Given: [[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]]
# Get: the element 6 (row 1, col 2)
arr24 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
element_6 = arr24[1,2]
print(f"Challenge 24: {element_6}")

# Challenge 25: 2D slicing - entire row
# Given: [[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]]
# Get: middle row [4, 5, 6]
arr25 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
middle_row = arr25[1]
print(f"Challenge 25: {middle_row}")

# Challenge 26: 2D slicing - entire column
# Given: [[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]]
# Get: last column [3, 6, 9]
arr26 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
last_column = arr26[:,2]
print(f"Challenge 26: {last_column}")

# Challenge 27: 2D slicing - submatrix
# Given: [[1, 2, 3, 4],
#         [5, 6, 7, 8],
#         [9, 10, 11, 12]]
# Get: [[6, 7],
#       [10, 11]]
arr27 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
submatrix = arr27[1:3,1:3]
print(f"Challenge 27:\n{submatrix}")

# --- --- ---
# ====== Boolean masking ======

print("\n" + "="*50)
print("BOOLEAN MASKING - The ML Superpower")
print("="*50)

# Challenge 28: Create a boolean mask
# Given: [1, 5, 3, 8, 2, 9, 4]
# Create mask for: elements > 5
# Expected mask: [False, False, False, True, False, True, False]
arr28 = np.array([1, 5, 3, 8, 2, 9, 4])
mask = arr28 > 5
print(f"Challenge 28 - Mask: {mask}")
print(f"Type of mask: {type(mask)}")

# Challenge 29: Apply boolean mask
# Given same array, get only elements > 5
# Expected: [8, 9]
arr29 = np.array([1, 5, 3, 8, 2, 9, 4])
filtered = arr29[mask]
print(f"Challenge 29 - Filtered: {filtered}")

# Challenge 30: Multiple conditions with &
# Given: [1, 5, 3, 8, 2, 9, 4]
# Get: elements > 2 AND < 8
# Expected: [5, 3, 4]
arr30 = np.array([1, 5, 3, 8, 2, 9, 4])
filtered_range = (arr30 > 2) & (arr30 < 8)
# Note: parentheses are REQUIRED!
print(f"Challenge 30: {arr30[filtered_range]}")

# Challenge 31: Multiple conditions with |
# Given: [1, 5, 3, 8, 2, 9, 4]
# Get: elements < 3 OR > 8
# Expected: [1, 2, 9]
arr31 = np.array([1, 5, 3, 8, 2, 9, 4])
filtered_or = (arr31 < 3) | (arr31 > 8)
print(f"Challenge 31: {arr31[filtered_or]}")

# Challenge 32: Boolean indexing in 2D
# Given: [[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]]
# Get: all elements > 5
# Expected: [6, 7, 8, 9]
arr32 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
filtered_2d = arr32[arr32 > 5]
print(f"Challenge 32: {filtered_2d}")

# Challenge 33: np.where - find indices
# Given: [10, 25, 30, 15, 40]
# Find: indices where value > 20
# Expected: (array([1, 2, 4]),) - indices 1, 2, 4
arr33 = np.array([10, 25, 30, 15, 40])
indices = np.where(arr33 > 20)
print(f"Challenge 33 - Indices: {indices}")
print(f"Values at those indices: {arr33[indices]}")

# Challenge 34: np.where - conditional replacement
# Given: [1, 5, 3, 8, 2, 9, 4]
# Replace: values > 5 with 100, others with 0
# Expected: [0, 0, 0, 100, 0, 100, 0]
arr34 = np.array([1, 5, 3, 8, 2, 9, 4])
replaced = np.where(arr34 > 5, 100, 0)
print(f"Challenge 34: {replaced}")

# --- --- ---
# ====== Aggregations and Axis Operations =======

print("\n" + "="*50)
print("AGGREGATIONS AND AXIS OPERATIONS")
print("="*50)

# Challenge 35: Basic aggregations
# Given: [1, 2, 3, 4, 5]
# Compute: sum, mean, max, min, std
arr35 = np.array([1, 2, 3, 4, 5])
total = np.sum(arr35)
mean = np.mean(arr35)
maximum = np.max(arr35)
minimum = np.min(arr35)
std_dev = np.std(arr35)
print(f"Challenge 35:")
print(f"  Sum: {total}, Mean: {mean}, Max: {maximum}, Min: {minimum}, Std: {std_dev:.2f}")

# Challenge 36: Understanding axis=0 (down columns)
# Given: [[1, 2, 3],
#         [4, 5, 6]]
# Compute: sum along axis=0
# Expected: [5, 7, 9] - sum DOWN each column
arr36 = np.array([[1, 2, 3], [4, 5, 6]])
sum_axis0 = np.sum(arr36, axis=0)
print(f"Challenge 36 - Sum axis=0: {sum_axis0}")

# Challenge 37: Understanding axis=1 (across rows)
# Given: [[1, 2, 3],
#         [4, 5, 6]]
# Compute: sum along axis=1
# Expected: [6, 15] - sum ACROSS each row
arr37 = np.array([[1, 2, 3], [4, 5, 6]])
sum_axis1 = np.sum(arr37, axis=1)
print(f"Challenge 37 - Sum axis=1: {sum_axis1}")

# Challenge 38: Mean across samples (typical ML operation)
# Given batch of 3 samples with 4 features each:
# [[1, 2, 3, 4],
#  [5, 6, 7, 8],
#  [9, 10, 11, 12]]
# Compute: mean of each FEATURE across all samples
# Expected: [5, 6, 7, 8] - average of each column
batch = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
feature_means = np.mean(batch, axis=0)
print(f"Challenge 38 - Feature means: {feature_means}")

# Challenge 39: argmax - find index of maximum
# Given: [3, 7, 2, 9, 1]
# Find: index of maximum value
# Expected: 3 (the value 9 is at index 3)
arr39 = np.array([3, 7, 2, 9, 1])
max_index = np.argmax(arr39)
print(f"Challenge 39 - Index of max: {max_index}, Value: {arr39[max_index]}")

# Challenge 40: argmax with axis (ML prediction!)
# Given predictions for 3 samples across 4 classes:
# [[0.1, 0.3, 0.5, 0.1],  # Sample 0: class 2 is highest
#  [0.7, 0.1, 0.1, 0.1],  # Sample 1: class 0 is highest
#  [0.2, 0.2, 0.2, 0.4]]  # Sample 2: class 3 is highest
# Find: predicted class for each sample
# Expected: [2, 0, 3]
predictions = np.array([[0.1, 0.3, 0.5, 0.1],
                        [0.7, 0.1, 0.1, 0.1],
                        [0.2, 0.2, 0.2, 0.4]])
predicted_classes = np.argmax(predictions, axis=1)
print(f"Challenge 40 - Predicted classes: {predicted_classes}")

print("\n" + "="*50)
print("UNDERSTANDING AXIS - The Visual Way")
print("="*50)

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("Original array:")
print(arr)
print(f"Shape: {arr.shape} - (2 rows, 3 columns)")

print("\naxis=0 means 'collapse the ROWS' (go DOWN):")
print("Column 0: 1+4 =", np.sum(arr[:, 0]))
print("Column 1: 2+5 =", np.sum(arr[:, 1]))
print("Column 2: 3+6 =", np.sum(arr[:, 2]))
print("Result:", np.sum(arr, axis=0))

print("\naxis=1 means 'collapse the COLUMNS' (go ACROSS):")
print("Row 0: 1+2+3 =", np.sum(arr[0, :]))
print("Row 1: 4+5+6 =", np.sum(arr[1, :]))
print("Result:", np.sum(arr, axis=1))

print("\n💡 Memory trick:")
print("axis=0: operate DOWN the first dimension (rows)")
print("axis=1: operate ACROSS the second dimension (columns)")