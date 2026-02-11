import numpy as np

def neural_layer_forward(inputs, weights, biases):
    """Single sample forward pass"""
    return weights @ inputs + biases

def neural_layer_forward_batch(inputs_batch, weights, biases):
    """Batch forward pass"""
    return inputs_batch @ weights.T + biases

# ============= TESTS =============
if __name__ == "__main__":
    print("Running tests...")

    # Test 1: Single sample
    inputs = np.array([1.5, 2.0, 0.5])
    weights = np.array([
        [0.2, 0.8, -0.5],
        [0.5, -0.3, 0.7]
    ])
    biases = np.array([0.1, -0.2])

    result = neural_layer_forward(inputs, weights, biases)
    expected = np.array([1.75, 0.3])

    assert result.shape == (2,), f"Wrong shape: {result.shape}"
    assert np.allclose(result, expected), f"Wrong values: {result}"
    print("✓ Test 1 passed: Single sample forward pass")

    # Test 2: Batch processing
    batch_inputs = np.array([
        [1.5, 2.0, 0.5],
        [0.5, 1.0, 2.0],
        [2.0, 0.5, 1.5],
        [1.0, 1.0, 1.0]
    ])

    batch_result = neural_layer_forward_batch(batch_inputs, weights, biases)

    assert batch_result.shape == (4, 2), f"Wrong shape: {batch_result.shape}"
    print("✓ Test 2 passed: Batch shape correct")

    # Test 3: Verify first sample in batch matches single sample
    first_from_batch = batch_result[0]
    single_result = neural_layer_forward(batch_inputs[0], weights, biases)

    assert np.allclose(first_from_batch, single_result), \
        f"Batch and single don't match: {first_from_batch} vs {single_result}"
    print("✓ Test 3 passed: Batch matches single sample")

    # Test 4: Broadcasting works correctly
    # All samples should get the same bias added
    no_bias_result = batch_inputs @ weights.T
    with_bias_result = neural_layer_forward_batch(batch_inputs, weights, biases)

    for i in range(4):
        expected_diff = biases
        actual_diff = with_bias_result[i] - no_bias_result[i]
        assert np.allclose(actual_diff, expected_diff), \
            f"Bias not applied correctly to sample {i}"
    print("✓ Test 4 passed: Bias broadcasting correct")

    print("\n✅ All tests passed!")