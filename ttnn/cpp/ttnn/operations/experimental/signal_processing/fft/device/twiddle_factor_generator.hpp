// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {
namespace detail {

// Structure to hold twiddle factors organized for efficient access
struct TwiddleFactorTensors {
    Tensor cos_factors;  // Real parts (cosine)
    Tensor sin_factors;  // Imaginary parts (sine)
};

// Generate twiddle factors for FFT and organize them into tiles
// For an N-point FFT, we need twiddle factors W_N^k = exp(-2πik/N)
inline TwiddleFactorTensors generate_twiddle_factor_tensors(
    uint32_t fft_size,
    bool inverse,
    Device* device,
    DataType dtype = DataType::BFLOAT16,
    const MemoryConfig& memory_config = MemoryConfig{}) {
    
    // Calculate total number of unique twiddle factors needed
    // For each stage s (1 to log2(N)), we need 2^(s-1) twiddle factors
    uint32_t log2_n = std::log2(fft_size);
    uint32_t total_twiddles = fft_size / 2;  // Maximum needed
    
    // Prepare host vectors for twiddle factors
    std::vector<float> cos_values;
    std::vector<float> sin_values;
    cos_values.reserve(total_twiddles);
    sin_values.reserve(total_twiddles);
    
    // Direction for FFT vs IFFT
    float direction = inverse ? 1.0f : -1.0f;
    
    // Generate twiddle factors for all stages
    for (uint32_t stage = 1; stage <= log2_n; ++stage) {
        uint32_t num_groups = fft_size >> stage;
        uint32_t group_size = 1 << stage;
        uint32_t half_group = group_size >> 1;
        
        // For each butterfly in a group
        for (uint32_t j = 0; j < half_group; ++j) {
            float angle = direction * 2.0f * M_PI * j / group_size;
            cos_values.push_back(std::cos(angle));
            sin_values.push_back(std::sin(angle));
        }
    }
    
    // Pad to tile size (32x32)
    constexpr uint32_t TILE_SIZE = 32 * 32;
    while (cos_values.size() % TILE_SIZE != 0) {
        cos_values.push_back(0.0f);
        sin_values.push_back(0.0f);
    }
    
    // Create shape for twiddle factor tensors
    uint32_t num_tiles = (cos_values.size() + TILE_SIZE - 1) / TILE_SIZE;
    Shape twiddle_shape({1, 1, num_tiles * 32, 32});
    
    // Create tensors from host data with the specified data type
    auto cos_tensor = ttnn::from_vector(
        cos_values,
        twiddle_shape,
        dtype,
        Layout::TILE,
        device,
        memory_config
    );
    
    auto sin_tensor = ttnn::from_vector(
        sin_values,
        twiddle_shape,
        dtype,
        Layout::TILE,
        device,
        memory_config
    );
    
    return {cos_tensor, sin_tensor};
}

// Generate twiddle factors specifically for 32-point FFT (single tile)
inline TwiddleFactorTensors generate_32point_twiddle_factors(
    bool inverse,
    Device* device,
    DataType dtype = DataType::BFLOAT16,
    const MemoryConfig& memory_config = MemoryConfig{}) {
    
    constexpr uint32_t N = 32;
    constexpr uint32_t TILE_SIZE = 32 * 32;
    
    std::vector<float> cos_values(TILE_SIZE, 0.0f);
    std::vector<float> sin_values(TILE_SIZE, 0.0f);
    
    float direction = inverse ? 1.0f : -1.0f;
    
    // For a 32-point FFT, we need twiddle factors W_32^k for k = 0 to 15
    // Due to symmetry: W_32^(k+16) = -W_32^k
    for (uint32_t k = 0; k < N/2; ++k) {
        float angle = direction * 2.0f * M_PI * k / N;
        cos_values[k] = std::cos(angle);
        sin_values[k] = std::sin(angle);
    }
    
    // Replicate across rows for SIMD processing
    // Each row gets the same set of twiddle factors
    for (uint32_t row = 1; row < 32; ++row) {
        for (uint32_t k = 0; k < N/2; ++k) {
            cos_values[row * 32 + k] = cos_values[k];
            sin_values[row * 32 + k] = sin_values[k];
        }
    }
    
    Shape twiddle_shape({1, 1, 32, 32});
    
    auto cos_tensor = ttnn::from_vector(
        cos_values,
        twiddle_shape,
        dtype,
        Layout::TILE,
        device,
        memory_config
    );
    
    auto sin_tensor = ttnn::from_vector(
        sin_values,
        twiddle_shape,
        dtype,
        Layout::TILE,
        device,
        memory_config
    );
    
    return {cos_tensor, sin_tensor};
}

// Helper to create bit-reversal permutation indices
inline std::vector<uint32_t> generate_bit_reversal_indices(uint32_t n) {
    std::vector<uint32_t> indices(n);
    uint32_t log2_n = std::log2(n);
    
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t reversed = 0;
        uint32_t temp = i;
        
        for (uint32_t j = 0; j < log2_n; ++j) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        indices[i] = reversed;
    }
    
    return indices;
}

}  // namespace detail
}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn