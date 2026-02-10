import numpy as np

# =============================================================================
# NUMPY BROADCASTING
# =============================================================================
# Broadcasting is NumPy's way of performing operations on arrays with
# different shapes WITHOUT making copies. Instead of requiring arrays to
# have identical shapes, NumPy "stretches" the smaller array to match.
#
# THE BROADCASTING RULES (checked right-to-left on each dimension):
#   1. If arrays differ in number of dimensions, the smaller one gets
#      padded with 1s on the LEFT of its shape.
#   2. Two dimensions are compatible if they are EQUAL or one of them is 1.
#   3. If neither condition is met -> error.
# =============================================================================


# --- Rule in action: Scalar + Array ---
# Scalar shape: ()  ->  treated as (1,)  ->  stretched to (4,)
a = np.array([1, 2, 3, 4])
print("Scalar + Array")
print(f"  {a} + 10 = {a + 10}")
# NumPy conceptually expands 10 to [10, 10, 10, 10], but never allocates it.


# --- 1-D arrays of same length (no broadcasting needed) ---
b = np.array([10, 20, 30, 40])
print("\nSame shape (4,) + (4,)")
print(f"  {a} + {b} = {a + b}")


# --- 2-D + 1-D: row-wise broadcasting ---
# Shape (3,4) + (4,) -> (4,) becomes (1,4) -> stretched to (3,4)
matrix = np.array([[1,  2,  3,  4],
                   [5,  6,  7,  8],
                   [9, 10, 11, 12]])
row = np.array([100, 200, 300, 400])

print("\n2-D (3,4) + 1-D (4,)  — adds row to every row")
print(matrix + row)
# Each row of `matrix` gets `row` added element-wise.


# --- 2-D + column vector: column-wise broadcasting ---
# Shape (3,4) + (3,1) -> (3,1) stretched to (3,4)
col = np.array([[10],
                [20],
                [30]])

print("\n2-D (3,4) + column (3,1) — adds column to every column")
print(matrix + col)
# Each column of `matrix` gets the column vector added.


# --- Outer-product style: (3,1) + (1,4) -> (3,4) ---
# Both dimensions get stretched: rows repeat down, cols repeat across.
x = np.array([[0],
              [1],
              [2]])          # shape (3,1)
y = np.array([0, 1, 2, 3])  # shape (4,) -> (1,4)

print("\nOuter-product style (3,1) + (1,4) -> (3,4)")
print(x + y)
# Useful for building grids, distance matrices, etc.


# --- Practical example: centering data (zero-mean each column) ---
data = np.array([[4.0, 200, 0.5],
                 [6.0, 400, 1.5],
                 [8.0, 600, 2.5]])

col_means = data.mean(axis=0)  # shape (3,) — one mean per column
centered = data - col_means    # (3,3) - (3,) -> broadcasts over rows

print("\nPractical: center columns to zero mean")
print(f"  Original:\n{data}")
print(f"  Column means: {col_means}")
print(f"  Centered:\n{centered}")
print(f"  Verify means ≈ 0: {centered.mean(axis=0)}")


# --- Practical example: normalizing an image (H, W, 3) by per-channel stats ---
# Simulated 2x2 RGB image
image = np.random.randint(0, 256, size=(2, 2, 3)).astype(np.float32)
channel_mean = np.array([123.0, 117.0, 104.0])  # shape (3,)
# (2,2,3) - (3,) -> (3,) becomes (1,1,3) -> stretched to (2,2,3)
normalized = image - channel_mean

print("\nPractical: per-channel image normalization")
print(f"  Image shape: {image.shape}, mean shape: {channel_mean.shape}")
print(f"  Result shape: {normalized.shape}")


# --- When broadcasting FAILS ---
# (3,) + (4,) -> last dims are 3 vs 4, neither is 1 -> ERROR
try:
    bad_a = np.array([1, 2, 3])
    bad_b = np.array([1, 2, 3, 4])
    result = bad_a + bad_b
except ValueError as e:
    print(f"\nBroadcasting error: {e}")
    print("  (3,) and (4,) are incompatible — neither dim is 1 and they aren't equal.")


# =============================================================================
# QUICK REFERENCE
# =============================================================================
#  Shape A     Shape B     Result     Why
#  -------     -------     ------     ---
#  (3,)        scalar      (3,)       scalar stretched
#  (3,4)       (4,)        (3,4)      row broadcast
#  (3,4)       (3,1)       (3,4)      column broadcast
#  (3,1)       (1,4)       (3,4)      both stretched (outer product)
#  (2,2,3)     (3,)        (2,2,3)   last dim matches, prepend 1s
#  (3,)        (4,)        ERROR      3 ≠ 4 and neither is 1
# =============================================================================
