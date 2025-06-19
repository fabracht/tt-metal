// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>

// Test our actual C++ headers and implementation
#define TT_FATAL(condition, ...) if(!(condition)) { throw std::runtime_error("TT_FATAL"); }

// Mock the necessary types to test our actual code
namespace tt::tt_metal {
    struct Program {};
    struct Device {};
    struct Buffer {};
    struct Tensor {};
    struct CoreCoord {};
    struct CoreRange {};
    
    namespace operation {
        struct ProgramWithCallbacks {
            Program program;
            std::vector<std::function<void()>> callbacks;
        };
    }
}

namespace ttnn {
    enum class DataType { BFLOAT16, FLOAT32 };
    enum class Layout { TILE, ROW_MAJOR };
    struct MemoryConfig {};
    struct DeviceComputeKernelConfig {};
    struct Shape { 
        std::vector<uint32_t> shape;
        size_t rank() const { return shape.size(); }
        uint32_t operator[](size_t i) const { return shape[i]; }
    };
    
    // Mock Tensor for testing
    struct Tensor {
        DataType dtype;
        Layout layout_val;
        Shape shape_val;
        
        DataType get_dtype() const { return dtype; }
        Layout get_layout() const { return layout_val; }
        Shape get_logical_shape() const { return shape_val; }
        uint32_t volume() const {
            uint32_t v = 1;
            for (auto d : shape_val.shape) v *= d;
            return v;
        }
        MemoryConfig memory_config() const { return {}; }
        tt::tt_metal::Device* device() const { return nullptr; }
        tt::tt_metal::Buffer* buffer() const { return nullptr; }
    };
    
    // Mock ComplexTensor
    struct ComplexTensor {
        Tensor real_part;
        Tensor imag_part;
        
        const Tensor& real() const { return real_part; }
        const Tensor& imag() const { return imag_part; }
    };
}

// Now include our ACTUAL implementation files
#include "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/fft.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/fft_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/fft_algorithm_detail.hpp"

// Include the actual implementations
namespace ttnn::operations::experimental::signal_processing {
    
// Test our actual helper functions
TEST(FFTImplementationTest, NormalizeAxisTest) {
    // Test the actual normalize_axis function from our implementation
    EXPECT_EQ(normalize_axis(-1, 4), 3);
    EXPECT_EQ(normalize_axis(-2, 4), 2);
    EXPECT_EQ(normalize_axis(0, 4), 0);
    EXPECT_EQ(normalize_axis(3, 4), 3);
    
    // Test error case
    EXPECT_THROW(normalize_axis(-5, 4), std::runtime_error);
    EXPECT_THROW(normalize_axis(4, 4), std::runtime_error);
}

TEST(FFTImplementationTest, GetFFTLengthTest) {
    ttnn::Tensor tensor;
    tensor.shape_val = {{1, 1, 32, 64}};
    
    // Test default case (n = -1)
    EXPECT_EQ(get_fft_length(tensor, 3, -1), 64);
    EXPECT_EQ(get_fft_length(tensor, 2, -1), 32);
    
    // Test specified length
    EXPECT_EQ(get_fft_length(tensor, 3, 128), 128);
    EXPECT_EQ(get_fft_length(tensor, 2, 16), 16);
}

// Test our actual algorithm detail functions
TEST(FFTAlgorithmDetailTest, BitReverseTest) {
    using namespace detail;
    
    // Test our actual bit_reverse function
    EXPECT_EQ(bit_reverse(0, 3), 0);  // 000 -> 000
    EXPECT_EQ(bit_reverse(1, 3), 4);  // 001 -> 100
    EXPECT_EQ(bit_reverse(2, 3), 2);  // 010 -> 010
    EXPECT_EQ(bit_reverse(3, 3), 6);  // 011 -> 110
    EXPECT_EQ(bit_reverse(4, 3), 1);  // 100 -> 001
    EXPECT_EQ(bit_reverse(5, 3), 5);  // 101 -> 101
    EXPECT_EQ(bit_reverse(6, 3), 3);  // 110 -> 011
    EXPECT_EQ(bit_reverse(7, 3), 7);  // 111 -> 111
}

TEST(FFTAlgorithmDetailTest, FFTStageInfoTest) {
    using namespace detail;
    
    // Test our actual get_fft_stage_info function
    uint32_t fft_size = 16;
    
    auto stage1 = get_fft_stage_info(1, fft_size);
    EXPECT_EQ(stage1.stage, 1);
    EXPECT_EQ(stage1.butterfly_span, 1);
    EXPECT_EQ(stage1.group_size, 2);
    EXPECT_EQ(stage1.num_groups, 8);
    
    auto stage2 = get_fft_stage_info(2, fft_size);
    EXPECT_EQ(stage2.stage, 2);
    EXPECT_EQ(stage2.butterfly_span, 2);
    EXPECT_EQ(stage2.group_size, 4);
    EXPECT_EQ(stage2.num_groups, 4);
    
    auto stage4 = get_fft_stage_info(4, fft_size);
    EXPECT_EQ(stage4.stage, 4);
    EXPECT_EQ(stage4.butterfly_span, 8);
    EXPECT_EQ(stage4.group_size, 16);
    EXPECT_EQ(stage4.num_groups, 1);
}

TEST(FFTAlgorithmDetailTest, TwiddleIndexTest) {
    using namespace detail;
    
    // Test our actual get_twiddle_index function
    uint32_t fft_size = 16;
    
    // Stage 1: all use W^0
    EXPECT_EQ(get_twiddle_index(0, 1, fft_size), 0);
    EXPECT_EQ(get_twiddle_index(7, 1, fft_size), 0);
    
    // Stage 2: alternates
    EXPECT_EQ(get_twiddle_index(0, 2, fft_size), 0);
    EXPECT_EQ(get_twiddle_index(1, 2, fft_size), 4);
    EXPECT_EQ(get_twiddle_index(2, 2, fft_size), 0);
    EXPECT_EQ(get_twiddle_index(3, 2, fft_size), 4);
}

TEST(FFTAlgorithmDetailTest, CooleyTukeyTest) {
    using namespace detail;
    
    // Test our actual cooley_tukey_fft implementation
    std::vector<std::complex<float>> data = {
        {1, 0}, {1, 0}, {1, 0}, {1, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}
    };
    
    cooley_tukey_fft(data, false);
    
    // Check DC component (sum of inputs = 4)
    EXPECT_NEAR(std::abs(data[0]), 4.0f, 1e-5);
    
    // Test inverse
    cooley_tukey_fft(data, true);
    
    // Should get back original (within tolerance)
    EXPECT_NEAR(data[0].real(), 1.0f, 1e-5);
    EXPECT_NEAR(data[0].imag(), 0.0f, 1e-5);
}

// Test device operation validation
TEST(FFTDeviceOperationTest, ValidationTest) {
    using namespace ttnn::operations::experimental::signal_processing;
    
    // Create test tensors
    ttnn::Tensor real_tensor, imag_tensor;
    real_tensor.dtype = ttnn::DataType::BFLOAT16;
    real_tensor.layout_val = ttnn::Layout::TILE;
    real_tensor.shape_val = {{1, 1, 32, 32}};
    
    imag_tensor = real_tensor;  // Same properties
    
    FFTDeviceOperation::operation_attributes_t attrs{
        .mode = FFTMode::FFT,
        .norm = FFTNorm::BACKWARD,
        .n = 32,
        .dim = 3,
        .memory_config = {},
        .compute_kernel_config = {}
    };
    
    FFTDeviceOperation::tensor_args_t args{
        .input_real = real_tensor,
        .input_imag = imag_tensor,
        .output_real = std::nullopt,
        .output_imag = std::nullopt
    };
    
    // Should not throw
    EXPECT_NO_THROW(FFTDeviceOperation::validate_on_program_cache_miss(attrs, args));
    
    // Test mismatched shapes
    imag_tensor.shape_val = {{1, 1, 64, 64}};
    args.input_imag = imag_tensor;
    EXPECT_THROW(FFTDeviceOperation::validate_on_program_cache_miss(attrs, args), std::runtime_error);
    
    // Test invalid data type
    real_tensor.dtype = ttnn::DataType::UINT32;  // Invalid
    imag_tensor = real_tensor;
    args.input_real = real_tensor;
    args.input_imag = imag_tensor;
    EXPECT_THROW(FFTDeviceOperation::validate_on_program_cache_miss(attrs, args), std::runtime_error);
    
    // Test non-power-of-2 FFT size
    real_tensor.dtype = ttnn::DataType::FLOAT32;
    imag_tensor = real_tensor;
    attrs.n = 33;  // Not power of 2
    args.input_real = real_tensor;
    args.input_imag = imag_tensor;
    EXPECT_THROW(FFTDeviceOperation::validate_on_program_cache_miss(attrs, args), std::runtime_error);
}

// Test output shape computation
TEST(FFTDeviceOperationTest, OutputSpecsTest) {
    using namespace ttnn::operations::experimental::signal_processing;
    
    ttnn::Tensor real_tensor, imag_tensor;
    real_tensor.dtype = ttnn::DataType::FLOAT32;
    real_tensor.layout_val = ttnn::Layout::TILE;
    real_tensor.shape_val = {{2, 3, 32, 64}};
    imag_tensor = real_tensor;
    
    FFTDeviceOperation::operation_attributes_t attrs{
        .mode = FFTMode::FFT,
        .norm = FFTNorm::BACKWARD,
        .n = 128,  // Larger than input
        .dim = 3,
        .memory_config = {},
        .compute_kernel_config = {}
    };
    
    FFTDeviceOperation::tensor_args_t args{
        .input_real = real_tensor,
        .input_imag = imag_tensor,
        .output_real = std::nullopt,
        .output_imag = std::nullopt
    };
    
    auto [out_real, out_imag] = FFTDeviceOperation::compute_output_specs(attrs, args);
    
    // Check output shape
    auto out_shape = out_real.get_logical_shape();
    EXPECT_EQ(out_shape[0], 2);
    EXPECT_EQ(out_shape[1], 3);
    EXPECT_EQ(out_shape[2], 32);
    EXPECT_EQ(out_shape[3], 128);  // Changed dimension
}

// Test twiddle factor generation
TEST(FFTImplementationTest, TwiddleFactorGenerationTest) {
    using namespace detail;
    
    // Test the actual compute_all_twiddle_factors function
    auto twiddles = compute_all_twiddle_factors(8, false);
    
    // For 8-point FFT, we need twiddles for 3 stages
    // Stage 1: 1 twiddle (W^0)
    // Stage 2: 2 twiddles (W^0, W^2)  
    // Stage 3: 4 twiddles (W^0, W^1, W^2, W^3)
    // Total: 7 twiddles
    EXPECT_EQ(twiddles.real.size(), 7);
    EXPECT_EQ(twiddles.imag.size(), 7);
    
    // Check first twiddle is always (1, 0)
    EXPECT_NEAR(twiddles.real[0], 1.0f, 1e-5);
    EXPECT_NEAR(twiddles.imag[0], 0.0f, 1e-5);
    
    // Check W_8^2 = (0, -1) for forward FFT
    // This is at index 1 (second twiddle of stage 2)
    EXPECT_NEAR(twiddles.real[1], 0.0f, 1e-5);
    EXPECT_NEAR(twiddles.imag[1], -1.0f, 1e-5);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}