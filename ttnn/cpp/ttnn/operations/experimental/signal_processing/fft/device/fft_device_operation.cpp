// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_device_operation.hpp"
#include "../fft.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <cmath>
#include "twiddle_factor_generator.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {

void FFTDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_real = tensor_args.input_real;
    const auto& input_imag = tensor_args.input_imag;
    
    // Check that inputs have same shape and properties
    TT_FATAL(input_real.get_logical_shape() == input_imag.get_logical_shape(),
             "Real and imaginary parts must have the same shape");
    TT_FATAL(input_real.get_dtype() == input_imag.get_dtype(),
             "Real and imaginary parts must have the same data type");
    TT_FATAL(input_real.device() == input_imag.device(),
             "Real and imaginary parts must be on the same device");
    TT_FATAL(input_real.get_layout() == Layout::TILE,
             "FFT only supports TILE layout");
    
    // Check data type
    TT_FATAL(input_real.get_dtype() == DataType::BFLOAT16 || input_real.get_dtype() == DataType::FLOAT32,
             "FFT only supports BFLOAT16 and FLOAT32 data types");
    
    // Check FFT length
    auto shape = input_real.get_logical_shape();
    auto fft_length = attributes.n;
    auto dim_size = shape[attributes.dim];
    
    TT_FATAL(fft_length > 0, "FFT length must be positive");
    TT_FATAL(fft_length <= dim_size || attributes.mode == FFTMode::FFT,
             "IFFT length cannot exceed input dimension size");
    
    // Check that FFT length is a power of 2 for efficiency
    TT_FATAL((fft_length & (fft_length - 1)) == 0,
             "FFT length must be a power of 2. Got: {}", fft_length);
}

void FFTDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // No additional validation needed on cache hit
}

FFTDeviceOperation::spec_return_value_t FFTDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args) {
    
    const auto& input_real = tensor_args.input_real;
    auto output_shape = input_real.get_logical_shape();
    
    // Update the shape for the FFT dimension
    output_shape[attributes.dim] = attributes.n;
    
    // Create output specs
    auto output_spec = TensorSpec(
        output_shape,
        TensorLayout(input_real.get_dtype(), PageConfig(input_real.get_layout()), attributes.memory_config)
    );
    
    return {Tensor(output_spec), Tensor(output_spec)};
}

FFTDeviceOperation::tensor_return_value_t FFTDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args) {
    
    const auto& input_real = tensor_args.input_real;
    
    if (tensor_args.output_real.has_value() && tensor_args.output_imag.has_value()) {
        return {tensor_args.output_real.value(), tensor_args.output_imag.value()};
    }
    
    auto [output_real_spec, output_imag_spec] = compute_output_specs(attributes, tensor_args);
    
    return {
        create_device_tensor(output_real_spec, input_real.device()),
        create_device_tensor(output_imag_spec, input_real.device())
    };
}

FFTDeviceOperation::ProgramFactory FFTDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args) {
    
    const auto& input_real = tensor_args.input_real;
    auto shape = input_real.get_logical_shape();
    
    // For now, we'll use single core implementation for small FFTs
    // and multi-core for larger ones
    constexpr size_t SINGLE_CORE_THRESHOLD = 512;
    
    if (attributes.n <= SINGLE_CORE_THRESHOLD) {
        using SingleCoreType = decltype(detail::fft_1d_single_core);
        return SingleCoreType{};
    } else {
        using MultiCoreType = decltype(detail::fft_1d_multi_core);
        return MultiCoreType{};
    }
}

tt::stl::hash::hash_t FFTDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args) {
    
    const auto& input_real = tensor_args.input_real;
    
    return tt::stl::hash::hash_objects(
        attributes.mode,
        attributes.norm,
        attributes.n,
        attributes.dim,
        input_real.get_dtype(),
        input_real.get_logical_shape(),
        attributes.memory_config
    );
}

namespace detail {

// Helper function to compute twiddle factors for all FFT stages
struct TwiddleFactors {
    std::vector<float> real;
    std::vector<float> imag;
};

TwiddleFactors compute_all_twiddle_factors(int64_t n, bool inverse) {
    TwiddleFactors twiddles;
    
    // For an N-point FFT, we need twiddle factors for each stage
    // Stage s has 2^(s-1) unique twiddle factors
    float sign = inverse ? 1.0f : -1.0f;
    
    // We'll store all twiddle factors in a flat array
    // organized by stage, then by twiddle index within that stage
    for (int64_t stage = 1; stage <= std::log2(n); ++stage) {
        int64_t num_twiddles = 1 << (stage - 1);
        int64_t group_size = 1 << stage;
        
        for (int64_t k = 0; k < num_twiddles; ++k) {
            float angle = sign * 2.0f * M_PI * k / group_size;
            twiddles.real.push_back(std::cos(angle));
            twiddles.imag.push_back(std::sin(angle));
        }
    }
    
    return twiddles;
}

tt::tt_metal::operation::ProgramWithCallbacks fft_1d_single_core(
    const Tensor& input_real,
    const Tensor& input_imag,
    Tensor& output_real,
    Tensor& output_imag,
    FFTMode mode,
    FFTNorm norm,
    int64_t n,
    int64_t dim,
    DeviceComputeKernelConfig compute_kernel_config) {
    
    auto program = tt::tt_metal::CreateProgram();
    auto device = input_real.device();
    
    // Get shapes and strides
    auto shape = input_real.get_logical_shape();
    auto rank = shape.rank();
    
    // Calculate sizes
    uint32_t fft_size = n;
    uint32_t num_tiles = input_real.volume() / tt::constants::TILE_HW;
    
    // Compute twiddle factors
    auto twiddles = compute_all_twiddle_factors(fft_size, mode == FFTMode::IFFT);
    
    // Set up core
    CoreRange core_range({0, 0}, {0, 0});
    
    // Create circular buffers
    uint32_t tile_size = tt::tt_metal::detail::TileSize(input_real.get_dtype());
    
    constexpr uint32_t cb_in_real = 0;
    constexpr uint32_t cb_in_imag = 1;
    constexpr uint32_t cb_out_real = 2;
    constexpr uint32_t cb_out_imag = 3;
    constexpr uint32_t cb_twiddle_real = 4;
    constexpr uint32_t cb_twiddle_imag = 5;
    constexpr uint32_t cb_temp_real = 6;
    constexpr uint32_t cb_temp_imag = 7;
    constexpr uint32_t cb_work_real = 8;
    constexpr uint32_t cb_work_imag = 9;
    
    uint32_t num_cb_tiles = 2; // Double buffer
    
    // Input circular buffers
    tt::tt_metal::CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(num_cb_tiles * tile_size, {{cb_in_real, input_real.get_dtype()}})
        .set_page_size(cb_in_real, tile_size);
    auto cb_in_real_id = tt::tt_metal::CreateCircularBuffer(program, core_range, cb_config);
    
    cb_config = tt::tt_metal::CircularBufferConfig(num_cb_tiles * tile_size, {{cb_in_imag, input_imag.get_dtype()}})
        .set_page_size(cb_in_imag, tile_size);
    auto cb_in_imag_id = tt::tt_metal::CreateCircularBuffer(program, core_range, cb_config);
    
    // Output circular buffers
    cb_config = tt::tt_metal::CircularBufferConfig(num_cb_tiles * tile_size, {{cb_out_real, output_real.get_dtype()}})
        .set_page_size(cb_out_real, tile_size);
    auto cb_out_real_id = tt::tt_metal::CreateCircularBuffer(program, core_range, cb_config);
    
    cb_config = tt::tt_metal::CircularBufferConfig(num_cb_tiles * tile_size, {{cb_out_imag, output_imag.get_dtype()}})
        .set_page_size(cb_out_imag, tile_size);
    auto cb_out_imag_id = tt::tt_metal::CreateCircularBuffer(program, core_range, cb_config);
    
    // Work circular buffers for intermediate calculations
    cb_config = tt::tt_metal::CircularBufferConfig(num_cb_tiles * tile_size, {{cb_work_real, input_real.get_dtype()}})
        .set_page_size(cb_work_real, tile_size);
    auto cb_work_real_id = tt::tt_metal::CreateCircularBuffer(program, core_range, cb_config);
    
    cb_config = tt::tt_metal::CircularBufferConfig(num_cb_tiles * tile_size, {{cb_work_imag, input_imag.get_dtype()}})
        .set_page_size(cb_work_imag, tile_size);
    auto cb_work_imag_id = tt::tt_metal::CreateCircularBuffer(program, core_range, cb_config);
    
    // Create kernels
    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/kernels/dataflow/reader_fft.cpp",
        core_range,
        tt::tt_metal::ReaderDataMovementConfig({cb_in_real, cb_in_imag}));
    
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/kernels/dataflow/writer_fft.cpp",
        core_range,
        tt::tt_metal::WriterDataMovementConfig({cb_out_real, cb_out_imag}));
    
    // Calculate log2 of FFT size
    uint32_t log2_fft_size = 0;
    uint32_t temp = fft_size;
    while (temp > 1) {
        temp >>= 1;
        log2_fft_size++;
    }
    
    std::vector<uint32_t> compute_kernel_args = {
        num_tiles,
        fft_size,
        static_cast<uint32_t>(mode == FFTMode::IFFT),
        log2_fft_size,
        2  // radix-2 FFT
    };
    
    // Select appropriate kernel based on FFT size
    std::string kernel_path;
    if (fft_size == 32) {
        // Optimized kernel for 32-point FFT (fits in one tile)
        kernel_path = "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/kernels/compute/fft_tile_based_kernel.cpp";
    } else {
        // General kernel for arbitrary sizes
        kernel_path = "ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/kernels/compute/fft_compute_kernel.cpp";
    }
    
    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = input_real.get_dtype() == DataType::FLOAT32,
            .compile_args = compute_kernel_args
        });
    
    // Set runtime args
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_range,
        {
            input_real.buffer()->address(),
            input_imag.buffer()->address(),
            num_tiles,
            0  // start_id
        }
    );
    
    tt::tt_metal::SetRuntimeArgs(
        program,
        writer_kernel_id,
        core_range,
        {
            output_real.buffer()->address(),
            output_imag.buffer()->address(),
            num_tiles,
            0  // start_id
        }
    );
    
    return {std::move(program), {}};
}

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
    DeviceComputeKernelConfig compute_kernel_config) {
    
    auto program = tt::tt_metal::CreateProgram();
    auto device = input_real.device();
    
    // Get shapes and strides
    auto shape = input_real.get_logical_shape();
    auto rank = shape.rank();
    
    // Calculate sizes
    uint32_t fft_size = n;
    uint32_t num_ffts = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (i != dim) {
            num_ffts *= shape[i];
        }
    }
    
    // Compute twiddle factors
    auto twiddles = compute_all_twiddle_factors(fft_size, mode == FFTMode::IFFT);
    
    // TODO: Implement the actual multi-core kernel creation and configuration
    // This is a placeholder for the actual implementation
    
    // Multi-core implementation would distribute the FFT computation across cores
    // Each core would handle a subset of the butterfly operations
    // This requires careful synchronization between FFT stages
    
    TT_FATAL(false, "Multi-core FFT implementation not yet complete");
    
    return {std::move(program), {}};
}

}  // namespace detail

std::tuple<FFTDeviceOperation::operation_attributes_t, FFTDeviceOperation::tensor_args_t> FFTDeviceOperation::invoke(
    const Tensor& input_real,
    const Tensor& input_imag,
    FFTMode mode,
    FFTNorm norm,
    int64_t n,
    int64_t dim,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    
    operation_attributes_t attributes{
        .mode = mode,
        .norm = norm,
        .n = n,
        .dim = dim,
        .memory_config = memory_config,
        .compute_kernel_config = compute_kernel_config
    };
    
    tensor_args_t tensor_args{
        .input_real = input_real,
        .input_imag = input_imag,
        .output_real = std::nullopt,
        .output_imag = std::nullopt
    };
    
    return {attributes, tensor_args};
}

}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

namespace ttnn::prim {
constexpr auto fft = ttnn::register_operation<"ttnn::prim::fft", ttnn::operations::experimental::signal_processing::FFTDeviceOperation>();
}  // namespace ttnn::prim