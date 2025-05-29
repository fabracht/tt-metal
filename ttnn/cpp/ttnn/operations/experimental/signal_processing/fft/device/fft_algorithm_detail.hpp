// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <complex>
#include <vector>

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {
namespace detail {

// Bit reversal permutation for FFT
inline uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2n; ++i) {
        if (x & (1 << i)) {
            result |= 1 << (log2n - 1 - i);
        }
    }
    return result;
}

// Cooley-Tukey Radix-2 Decimation-in-Time (DIT) FFT Algorithm
// This is a reference implementation showing the algorithm structure
template<typename T>
void cooley_tukey_fft(std::vector<std::complex<T>>& data, bool inverse = false) {
    size_t n = data.size();
    if (n <= 1) return;
    
    // Check if n is power of 2
    if ((n & (n - 1)) != 0) {
        throw std::runtime_error("FFT size must be a power of 2");
    }
    
    // Bit-reversal permutation
    size_t log2n = 0;
    size_t temp = n;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }
    
    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, log2n);
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
    
    // Cooley-Tukey FFT
    T direction = inverse ? 1.0 : -1.0;
    
    // Perform FFT stages
    for (size_t stage = 1; stage <= log2n; ++stage) {
        size_t m = 1 << stage;  // Size of FFT sub-problem
        size_t m2 = m >> 1;      // Half of m
        
        // Twiddle factor for this stage
        std::complex<T> w_m(std::cos(2 * M_PI / m), direction * std::sin(2 * M_PI / m));
        
        // Process all sub-problems at this stage
        for (size_t k = 0; k < n; k += m) {
            std::complex<T> w(1, 0);
            
            // Butterfly operations
            for (size_t j = 0; j < m2; ++j) {
                size_t t = k + j;
                size_t u = t + m2;
                
                // Butterfly: combine results
                std::complex<T> temp = w * data[u];
                data[u] = data[t] - temp;
                data[t] = data[t] + temp;
                
                // Update twiddle factor
                w *= w_m;
            }
        }
    }
    
    // Normalize for inverse FFT
    if (inverse) {
        T scale = 1.0 / n;
        for (auto& x : data) {
            x *= scale;
        }
    }
}

// Structure to hold FFT stage information
struct FFTStageInfo {
    uint32_t stage;           // Current stage (1 to log2(N))
    uint32_t num_groups;      // Number of butterfly groups
    uint32_t group_size;      // Size of each group
    uint32_t butterfly_span;  // Distance between butterfly pairs
};

// Get FFT stage parameters
inline FFTStageInfo get_fft_stage_info(uint32_t stage, uint32_t fft_size) {
    FFTStageInfo info;
    info.stage = stage;
    info.butterfly_span = 1 << (stage - 1);
    info.group_size = 1 << stage;
    info.num_groups = fft_size >> stage;
    return info;
}

// Compute twiddle factor index for a given butterfly
inline uint32_t get_twiddle_index(uint32_t butterfly_idx, uint32_t stage, uint32_t fft_size) {
    uint32_t group_size = 1 << stage;
    uint32_t twiddle_idx = (butterfly_idx % (group_size / 2)) * (fft_size / group_size);
    return twiddle_idx;
}

}  // namespace detail
}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn