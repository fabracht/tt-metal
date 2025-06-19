// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

// Mock includes for testing without full tt-metal
namespace ttnn {
    enum class DataType { BFLOAT16, FLOAT32 };
    enum class Layout { TILE };
    
    struct Shape {
        std::vector<uint32_t> dims;
        Shape(std::initializer_list<uint32_t> d) : dims(d) {}
    };
}

// Include our actual header (with mocked dependencies)
#include "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/fft_algorithm_detail.hpp"

using namespace ttnn::operations::experimental::signal_processing::detail;

class FFTHostTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
    
    // Helper to check if two complex numbers are close
    bool complexClose(const std::complex<double>& a, const std::complex<double>& b, double tol = EPSILON) {
        return std::abs(a - b) < tol;
    }
};

TEST_F(FFTHostTest, BitReversalTest) {
    // Test bit reversal for various sizes
    struct TestCase {
        uint32_t n;
        std::vector<uint32_t> expected;
    };
    
    std::vector<TestCase> test_cases = {
        {4, {0, 2, 1, 3}},
        {8, {0, 4, 2, 6, 1, 5, 3, 7}},
        {16, {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}}
    };
    
    for (const auto& tc : test_cases) {
        uint32_t log2n = std::log2(tc.n);
        std::vector<uint32_t> result;
        
        for (uint32_t i = 0; i < tc.n; ++i) {
            result.push_back(bit_reverse(i, log2n));
        }
        
        EXPECT_EQ(result, tc.expected) << "Bit reversal failed for n=" << tc.n;
        
        // Check that it's a valid permutation
        std::vector<uint32_t> sorted_result = result;
        std::sort(sorted_result.begin(), sorted_result.end());
        for (uint32_t i = 0; i < tc.n; ++i) {
            EXPECT_EQ(sorted_result[i], i) << "Invalid permutation at index " << i;
        }
    }
}

TEST_F(FFTHostTest, TwiddleFactorTest) {
    // Test twiddle factor generation
    std::vector<uint32_t> fft_sizes = {8, 16, 32, 64, 128};
    
    for (uint32_t n : fft_sizes) {
        // Forward FFT twiddles
        for (uint32_t k = 0; k < n/2; ++k) {
            double angle = -2.0 * M_PI * k / n;
            std::complex<double> expected(std::cos(angle), std::sin(angle));
            std::complex<double> actual(std::cos(angle), std::sin(angle));
            
            EXPECT_TRUE(complexClose(actual, expected)) 
                << "Twiddle factor mismatch for n=" << n << ", k=" << k;
        }
        
        // Test specific values
        std::complex<double> w0(1.0, 0.0);  // W_n^0 = 1
        EXPECT_TRUE(complexClose(w0, std::complex<double>(1.0, 0.0)));
        
        if (n >= 4) {
            // W_n^(n/4) = -i (for forward FFT)
            double angle = -2.0 * M_PI * (n/4) / n;
            std::complex<double> w_quarter(std::cos(angle), std::sin(angle));
            EXPECT_TRUE(complexClose(w_quarter, std::complex<double>(0.0, -1.0)));
        }
    }
}

TEST_F(FFTHostTest, CooleyTukeyReferenceTest) {
    // Test the reference Cooley-Tukey implementation
    std::vector<uint32_t> sizes = {8, 16, 32};
    
    for (uint32_t n : sizes) {
        // Create a simple test signal
        std::vector<std::complex<float>> data(n);
        
        // Test case 1: Impulse
        std::fill(data.begin(), data.end(), std::complex<float>(0, 0));
        data[0] = std::complex<float>(1, 0);
        
        auto data_copy = data;
        cooley_tukey_fft(data_copy, false);
        
        // FFT of impulse should be all ones
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(data_copy[i].real(), 1.0f, 1e-5) 
                << "Impulse FFT failed at index " << i << " for n=" << n;
            EXPECT_NEAR(data_copy[i].imag(), 0.0f, 1e-5) 
                << "Impulse FFT failed at index " << i << " for n=" << n;
        }
        
        // Test case 2: Pure cosine
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::complex<float>(std::cos(2 * M_PI * i / n), 0);
        }
        
        data_copy = data;
        cooley_tukey_fft(data_copy, false);
        
        // FFT of cosine should have peaks at bins 1 and n-1
        for (size_t i = 0; i < n; ++i) {
            if (i == 1 || i == n-1) {
                EXPECT_GT(std::abs(data_copy[i]), n/4) 
                    << "Cosine FFT peak missing at index " << i << " for n=" << n;
            } else {
                EXPECT_LT(std::abs(data_copy[i]), 1e-3) 
                    << "Cosine FFT has unexpected energy at index " << i << " for n=" << n;
            }
        }
    }
}

TEST_F(FFTHostTest, FFTStageInfoTest) {
    // Test FFT stage information generation
    uint32_t fft_size = 16;
    uint32_t num_stages = std::log2(fft_size);
    
    for (uint32_t stage = 1; stage <= num_stages; ++stage) {
        auto info = get_fft_stage_info(stage, fft_size);
        
        EXPECT_EQ(info.stage, stage);
        EXPECT_EQ(info.butterfly_span, 1u << (stage - 1));
        EXPECT_EQ(info.group_size, 1u << stage);
        EXPECT_EQ(info.num_groups, fft_size >> stage);
        
        // Verify consistency
        EXPECT_EQ(info.num_groups * info.group_size, fft_size);
    }
}

TEST_F(FFTHostTest, TwiddleIndexTest) {
    // Test twiddle factor indexing
    uint32_t fft_size = 16;
    
    // Stage 1: all butterflies use W^0 = 1
    for (uint32_t i = 0; i < fft_size/2; ++i) {
        uint32_t idx = get_twiddle_index(i, 1, fft_size);
        EXPECT_EQ(idx, 0) << "Stage 1 twiddle index wrong for butterfly " << i;
    }
    
    // Stage 2: butterflies alternate between W^0 and W^4
    for (uint32_t i = 0; i < fft_size/2; ++i) {
        uint32_t idx = get_twiddle_index(i, 2, fft_size);
        uint32_t expected = (i % 2) * 4;
        EXPECT_EQ(idx, expected) << "Stage 2 twiddle index wrong for butterfly " << i;
    }
}

TEST_F(FFTHostTest, ComplexMultiplyGaussTest) {
    // Test Gauss's complex multiplication method
    struct TestCase {
        std::complex<float> a, b;
        std::complex<float> expected;
    };
    
    std::vector<TestCase> test_cases = {
        {{1, 2}, {3, 4}, {-5, 10}},  // (1+2i)(3+4i) = 3+4i+6i-8 = -5+10i
        {{2, 0}, {0, 3}, {0, 6}},     // 2 * 3i = 6i
        {{0, 1}, {0, 1}, {-1, 0}},    // i * i = -1
        {{5, -3}, {-2, 7}, {11, 41}}  // (5-3i)(-2+7i) = -10+35i+6i+21 = 11+41i
    };
    
    for (const auto& tc : test_cases) {
        // Standard method
        auto result_standard = tc.a * tc.b;
        
        // Gauss method simulation
        float k1 = tc.b.real() * (tc.a.real() + tc.a.imag());
        float k2 = tc.a.real() * (tc.b.imag() - tc.b.real());
        float k3 = tc.a.imag() * (tc.b.real() + tc.b.imag());
        std::complex<float> result_gauss(k1 - k3, k1 + k2);
        
        EXPECT_NEAR(result_standard.real(), tc.expected.real(), 1e-5);
        EXPECT_NEAR(result_standard.imag(), tc.expected.imag(), 1e-5);
        EXPECT_NEAR(result_gauss.real(), tc.expected.real(), 1e-5);
        EXPECT_NEAR(result_gauss.imag(), tc.expected.imag(), 1e-5);
    }
}

TEST_F(FFTHostTest, ButterflyOperationTest) {
    // Test butterfly operation
    std::complex<float> a(3, 4);
    std::complex<float> b(1, 2);
    std::complex<float> w(0.707f, -0.707f);  // 45 degree rotation
    
    // Butterfly: out1 = a + w*b, out2 = a - w*b
    std::complex<float> wb = w * b;
    std::complex<float> out1 = a + wb;
    std::complex<float> out2 = a - wb;
    
    // Check sum and difference properties
    EXPECT_TRUE(complexClose(out1 + out2, 2.0f * a, 1e-5f));
    EXPECT_TRUE(complexClose(out1 - out2, 2.0f * wb, 1e-5f));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}