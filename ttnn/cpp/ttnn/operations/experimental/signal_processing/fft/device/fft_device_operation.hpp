// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {

// Forward declarations
enum class FFTMode;
enum class FFTNorm;

struct FFTDeviceOperation : public tt::tt_metal::Device_operation<FFTDeviceOperation> {
    struct operation_attributes_t {
        FFTMode mode;
        FFTNorm norm;
        int64_t n;              // FFT length
        int64_t dim;            // Dimension along which to compute FFT
        MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input_real;
        const Tensor& input_imag;
        std::optional<Tensor> output_real;
        std::optional<Tensor> output_imag;
    };

    using spec_return_value_t = std::tuple<Tensor, Tensor>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using ProgramFactory = std::variant<
        decltype(detail::fft_1d_single_core),
        decltype(detail::fft_1d_multi_core)
    >;

    static void validate_on_program_cache_miss(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static ProgramFactory select_program_factory(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_real,
        const Tensor& input_imag,
        FFTMode mode,
        FFTNorm norm,
        int64_t n,
        int64_t dim,
        const MemoryConfig& memory_config,
        const DeviceComputeKernelConfig& compute_kernel_config);
};

// Program factories for different FFT implementations
namespace detail {

tt::tt_metal::operation::ProgramWithCallbacks fft_1d_single_core(
    const Tensor& input_real,
    const Tensor& input_imag,
    Tensor& output_real,
    Tensor& output_imag,
    FFTMode mode,
    FFTNorm norm,
    int64_t n,
    int64_t dim,
    DeviceComputeKernelConfig compute_kernel_config);

tt::tt_metal::operation::ProgramWithCallbacks fft_1d_multi_core(
    const Tensor& input_real,
    const Tensor& input_imag,
    Tensor& output_real,
    Tensor& output_imag,
    FFTMode mode,
    FFTNorm norm,
    int64_t n,
    int64_t dim,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config);

}  // namespace detail

}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn