// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft.hpp"
#include "device/fft_device_operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {

namespace {

int64_t normalize_axis(int64_t dim, size_t rank) {
    if (dim < 0) {
        dim += rank;
    }
    TT_FATAL(dim >= 0 && dim < rank, "FFT dimension {} is out of range for tensor with {} dimensions", dim, rank);
    return dim;
}

int64_t get_fft_length(const Tensor& tensor, int64_t dim, int64_t n) {
    auto shape = tensor.get_logical_shape();
    if (n == -1) {
        return shape[dim];
    }
    return n;
}

ComplexTensor execute_fft_1d(
    const ComplexTensor& input,
    int64_t n,
    int64_t dim,
    FFTMode mode,
    FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    uint8_t queue_id) {
    
    auto& input_real = input.real();
    auto& input_imag = input.imag();
    
    // Validate inputs
    TT_FATAL(input_real.get_layout() == Layout::TILE, "FFT only supports tiled layout");
    TT_FATAL(input_real.get_dtype() == DataType::BFLOAT16 || input_real.get_dtype() == DataType::FLOAT32,
             "FFT only supports bf16 and fp32 data types");
    
    auto rank = input_real.get_logical_shape().rank();
    dim = normalize_axis(dim, rank);
    n = get_fft_length(input_real, dim, n);
    
    // Get memory config
    auto mem_config = memory_config.value_or(input_real.memory_config());
    
    // Get compute kernel config
    DeviceComputeKernelConfig compute_kernel_config;
    if (input_real.get_dtype() == DataType::BFLOAT16) {
        compute_kernel_config = ttnn::operations::primary::WormholeComputeKernelConfig{};
    } else {
        compute_kernel_config = ttnn::operations::primary::WormholeComputeKernelConfig{
            .fp32_dest_acc_en = true,
            .packer_l1_acc = true
        };
    }
    
    // Create device operation
    FFTDeviceOperation::operation_attributes_t attributes{
        .mode = mode,
        .norm = norm,
        .n = n,
        .dim = dim,
        .memory_config = mem_config,
        .compute_kernel_config = compute_kernel_config
    };
    
    FFTDeviceOperation::tensor_args_t tensor_args{
        .input_real = input_real,
        .input_imag = input_imag,
        .output_real = std::nullopt,
        .output_imag = std::nullopt
    };
    
    // Create the device operation and invoke it
    auto [device_operation_attributes, device_tensor_args] = FFTDeviceOperation::invoke(
        input_real,
        input_imag,
        mode,
        norm,
        n,
        dim,
        mem_config,
        compute_kernel_config
    );
    
    auto [output_real, output_imag] = ttnn::prim::fft(
        queue_id,
        device_operation_attributes,
        device_tensor_args
    );
    
    return ComplexTensor({output_real, output_imag});
}

}  // anonymous namespace

ComplexTensor FFT1dOperation::invoke(
    const ComplexTensor& input_tensor,
    const int64_t n,
    const int64_t dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    return execute_fft_1d(input_tensor, n, dim, FFTMode::FFT, norm, memory_config, queue_id);
}

ComplexTensor FFT1dOperation::invoke(
    const Tensor& input_tensor,
    const int64_t n,
    const int64_t dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    // Create complex tensor with zero imaginary part
    auto zero_imag = ttnn::zeros_like(input_tensor, input_tensor.get_dtype(), input_tensor.get_layout(), memory_config, queue_id);
    ComplexTensor complex_input({input_tensor, zero_imag});
    
    return execute_fft_1d(complex_input, n, dim, FFTMode::FFT, norm, memory_config, queue_id);
}

ComplexTensor IFFT1dOperation::invoke(
    const ComplexTensor& input_tensor,
    const int64_t n,
    const int64_t dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    return execute_fft_1d(input_tensor, n, dim, FFTMode::IFFT, norm, memory_config, queue_id);
}

ComplexTensor IFFT1dOperation::invoke(
    const Tensor& input_tensor,
    const int64_t n,
    const int64_t dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    // Create complex tensor with zero imaginary part
    auto zero_imag = ttnn::zeros_like(input_tensor, input_tensor.get_dtype(), input_tensor.get_layout(), memory_config, queue_id);
    ComplexTensor complex_input({input_tensor, zero_imag});
    
    return execute_fft_1d(complex_input, n, dim, FFTMode::IFFT, norm, memory_config, queue_id);
}

ComplexTensor FFT2dOperation::invoke(
    const ComplexTensor& input_tensor,
    const std::optional<std::array<int64_t, 2>>& s,
    const std::array<int64_t, 2>& dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    // 2D FFT is computed as FFT along first dimension, then FFT along second dimension
    auto n0 = s.has_value() ? (*s)[0] : -1;
    auto n1 = s.has_value() ? (*s)[1] : -1;
    
    // First FFT along dim[0]
    auto intermediate = execute_fft_1d(input_tensor, n0, dim[0], FFTMode::FFT, norm, memory_config, queue_id);
    
    // Second FFT along dim[1]
    return execute_fft_1d(intermediate, n1, dim[1], FFTMode::FFT, norm, memory_config, queue_id);
}

ComplexTensor FFT2dOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<std::array<int64_t, 2>>& s,
    const std::array<int64_t, 2>& dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    // Create complex tensor with zero imaginary part
    auto zero_imag = ttnn::zeros_like(input_tensor, input_tensor.get_dtype(), input_tensor.get_layout(), memory_config, queue_id);
    ComplexTensor complex_input({input_tensor, zero_imag});
    
    return FFT2dOperation::invoke(complex_input, s, dim, norm, memory_config, queue_id);
}

ComplexTensor IFFT2dOperation::invoke(
    const ComplexTensor& input_tensor,
    const std::optional<std::array<int64_t, 2>>& s,
    const std::array<int64_t, 2>& dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    // 2D IFFT is computed as IFFT along first dimension, then IFFT along second dimension
    auto n0 = s.has_value() ? (*s)[0] : -1;
    auto n1 = s.has_value() ? (*s)[1] : -1;
    
    // First IFFT along dim[0]
    auto intermediate = execute_fft_1d(input_tensor, n0, dim[0], FFTMode::IFFT, norm, memory_config, queue_id);
    
    // Second IFFT along dim[1]
    return execute_fft_1d(intermediate, n1, dim[1], FFTMode::IFFT, norm, memory_config, queue_id);
}

ComplexTensor IFFT2dOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<std::array<int64_t, 2>>& s,
    const std::array<int64_t, 2>& dim,
    const FFTNorm norm,
    const std::optional<MemoryConfig>& memory_config,
    const uint8_t queue_id) {
    
    // Create complex tensor with zero imaginary part
    auto zero_imag = ttnn::zeros_like(input_tensor, input_tensor.get_dtype(), input_tensor.get_layout(), memory_config, queue_id);
    ComplexTensor complex_input({input_tensor, zero_imag});
    
    return IFFT2dOperation::invoke(complex_input, s, dim, norm, memory_config, queue_id);
}

}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn