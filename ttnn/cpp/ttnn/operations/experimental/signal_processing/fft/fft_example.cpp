// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Example usage of FFT operations

#include "fft.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace ttnn::operations::experimental::signal_processing;

// Example: Compute FFT of a simple signal
void example_basic_fft() {
    // Open device
    auto device = ttnn::open_device(0);
    
    // Create a simple test signal: sum of two sinusoids
    const size_t N = 1024;
    std::vector<float> signal(N);
    
    // Generate signal: sin(2π*5*t) + 0.5*sin(2π*12*t)
    for (size_t i = 0; i < N; ++i) {
        float t = static_cast<float>(i) / N;
        signal[i] = std::sin(2 * M_PI * 5 * t) + 0.5f * std::sin(2 * M_PI * 12 * t);
    }
    
    // Create tensor from signal
    auto shape = ttnn::Shape({1, 1, 32, 32});  // Tile-aligned shape
    auto input_tensor = ttnn::from_vector(
        signal,
        shape,
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        device
    );
    
    // Compute FFT
    auto fft_result = ttnn::experimental::fft(input_tensor);
    
    // Extract magnitude spectrum
    auto real_part = fft_result.real();
    auto imag_part = fft_result.imag();
    
    // Compute magnitude: sqrt(real^2 + imag^2)
    auto real_squared = ttnn::square(real_part);
    auto imag_squared = ttnn::square(imag_part);
    auto magnitude_squared = ttnn::add(real_squared, imag_squared);
    auto magnitude = ttnn::sqrt(magnitude_squared);
    
    // Convert back to host
    auto magnitude_host = ttnn::to_vector(magnitude);
    
    // Find peak frequencies
    std::cout << "FFT Magnitude Spectrum Peaks:" << std::endl;
    for (size_t i = 0; i < N/2; ++i) {
        if (magnitude_host[i] > 10.0f) {  // Threshold for peak detection
            float freq = static_cast<float>(i);
            std::cout << "Peak at frequency bin " << i << " (frequency = " << freq << " Hz)" << std::endl;
        }
    }
    
    // Clean up
    ttnn::close_device(device);
}

// Example: 2D FFT for image processing
void example_2d_fft() {
    auto device = ttnn::open_device(0);
    
    // Create a 64x64 test image with a pattern
    const size_t H = 64, W = 64;
    std::vector<float> image(H * W);
    
    // Generate a simple pattern (e.g., diagonal stripes)
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            image[y * W + x] = std::sin(2 * M_PI * (x + y) / 16.0f);
        }
    }
    
    // Create tensor
    auto shape = ttnn::Shape({1, 1, H, W});
    auto image_tensor = ttnn::from_vector(
        image,
        shape,
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        device
    );
    
    // Compute 2D FFT
    auto fft2d_result = ttnn::experimental::fft2d(image_tensor);
    
    // Apply frequency domain filter (e.g., low-pass)
    // This would involve multiplying by a filter mask
    
    // Compute inverse 2D FFT
    auto filtered_result = ttnn::experimental::ifft2d(fft2d_result);
    
    // Extract real part (imaginary should be near zero)
    auto filtered_image = filtered_result.real();
    
    ttnn::close_device(device);
}

// Example: FFT-based convolution
void example_fft_convolution() {
    auto device = ttnn::open_device(0);
    
    const size_t N = 512;
    
    // Create two signals to convolve
    std::vector<float> signal1(N, 0.0f);
    std::vector<float> signal2(N, 0.0f);
    
    // Simple box filters
    for (size_t i = 0; i < 32; ++i) {
        signal1[i] = 1.0f / 32.0f;
        signal2[i] = 1.0f / 32.0f;
    }
    
    // Create tensors
    auto shape = ttnn::Shape({1, 1, 16, 32});  // 512 elements in tile format
    auto tensor1 = ttnn::from_vector(signal1, shape, ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    auto tensor2 = ttnn::from_vector(signal2, shape, ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
    
    // Compute FFTs
    auto fft1 = ttnn::experimental::fft(tensor1);
    auto fft2 = ttnn::experimental::fft(tensor2);
    
    // Multiply in frequency domain (complex multiplication)
    auto real1 = fft1.real();
    auto imag1 = fft1.imag();
    auto real2 = fft2.real();
    auto imag2 = fft2.imag();
    
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    auto real_prod = ttnn::subtract(
        ttnn::multiply(real1, real2),
        ttnn::multiply(imag1, imag2)
    );
    auto imag_prod = ttnn::add(
        ttnn::multiply(real1, imag2),
        ttnn::multiply(imag1, real2)
    );
    
    // Create complex tensor from results
    auto fft_product = ttnn::complex_tensor(real_prod, imag_prod);
    
    // Inverse FFT to get convolution result
    auto convolution_result = ttnn::experimental::ifft(fft_product);
    
    // Extract real part
    auto convolution = convolution_result.real();
    
    ttnn::close_device(device);
}

// Example: Batched FFT processing
void example_batched_fft() {
    auto device = ttnn::open_device(0);
    
    // Process multiple signals in parallel
    const size_t batch_size = 32;
    const size_t signal_length = 32;  // Fits in one tile row
    
    // Create batch of signals
    std::vector<float> batch_data(batch_size * signal_length);
    
    // Fill with different frequency sinusoids
    for (size_t b = 0; b < batch_size; ++b) {
        float freq = static_cast<float>(b + 1);
        for (size_t i = 0; i < signal_length; ++i) {
            float t = static_cast<float>(i) / signal_length;
            batch_data[b * signal_length + i] = std::sin(2 * M_PI * freq * t);
        }
    }
    
    // Create tensor (each row is a separate FFT)
    auto shape = ttnn::Shape({1, 1, batch_size, signal_length});
    auto batch_tensor = ttnn::from_vector(
        batch_data,
        shape,
        ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE,
        device
    );
    
    // Compute FFT along last dimension
    // Each row is processed independently
    auto batch_fft = ttnn::experimental::fft(batch_tensor, -1, -1);
    
    ttnn::close_device(device);
}

int main() {
    std::cout << "FFT Examples" << std::endl;
    
    std::cout << "\n1. Basic FFT:" << std::endl;
    example_basic_fft();
    
    std::cout << "\n2. 2D FFT:" << std::endl;
    example_2d_fft();
    
    std::cout << "\n3. FFT Convolution:" << std::endl;
    example_fft_convolution();
    
    std::cout << "\n4. Batched FFT:" << std::endl;
    example_batched_fft();
    
    return 0;
}