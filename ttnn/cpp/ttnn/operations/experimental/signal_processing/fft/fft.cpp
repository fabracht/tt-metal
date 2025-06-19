// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft.hpp"
#include "device/fft_device_operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
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
        compute_kernel_config = ttnn::WormholeComputeKernelConfig{};
    } else {
        compute_kernel_config = ttnn::WormholeComputeKernelConfig{
            .fp32_dest_acc_en = true,
            .packer_l1_acc = true
        };
    }
    
    // Call device operation using the new pattern
    auto [attributes, tensor_args] = FFTDeviceOperation::invoke(
        input_real,
        input_imag,
        mode,
        norm,
        n,
        dim,
        mem_config,
        compute_kernel_config
    );
    
    // Create output tensors
    auto [output_real, output_imag] = FFTDeviceOperation::create_output_tensors(attributes, tensor_args);
    
    // TODO: Actually run the computation using the program factory
    // For now, this is a placeholder that copies input to output
    output_real = ttnn::clone(input_real, std::nullopt, attributes.memory_config, attributes.compute_kernel_config);
    output_imag = ttnn::clone(input_imag, std::nullopt, attributes.memory_config, attributes.compute_kernel_config);
    
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
    auto zero_imag = ttnn::zeros_like(input_tensor);
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
    auto zero_imag = ttnn::zeros_like(input_tensor);
    ComplexTensor complex_input({input_tensor, zero_imag});
    
    return execute_fft_1d(complex_input, n, dim, FFTMode::IFFT, norm, memory_config, queue_id);
}


}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn