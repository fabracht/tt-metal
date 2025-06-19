// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/command_queue.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {

enum class FFTMode {
    FFT = 0,
    IFFT = 1
};

enum class FFTNorm {
    BACKWARD = 0,   // No normalization for FFT, 1/N for IFFT (default)
    ORTHO = 1,      // 1/sqrt(N) for both FFT and IFFT
    FORWARD = 2     // 1/N for FFT, no normalization for IFFT
};

struct FFT1dOperation {
    static ComplexTensor invoke(
        const ComplexTensor& input_tensor,
        const int64_t n = -1,
        const int64_t dim = -1,
        const FFTNorm norm = FFTNorm::BACKWARD,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const uint8_t queue_id = 0);
    
    static ComplexTensor invoke(
        const Tensor& input_tensor,
        const int64_t n = -1,
        const int64_t dim = -1,
        const FFTNorm norm = FFTNorm::BACKWARD,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const uint8_t queue_id = 0);
};

struct IFFT1dOperation {
    static ComplexTensor invoke(
        const ComplexTensor& input_tensor,
        const int64_t n = -1,
        const int64_t dim = -1,
        const FFTNorm norm = FFTNorm::BACKWARD,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const uint8_t queue_id = 0);
    
    static ComplexTensor invoke(
        const Tensor& input_tensor,
        const int64_t n = -1,
        const int64_t dim = -1,
        const FFTNorm norm = FFTNorm::BACKWARD,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const uint8_t queue_id = 0);
};


}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations

// Register operations
constexpr auto fft = ttnn::register_operation<"ttnn::experimental::fft", operations::experimental::signal_processing::FFT1dOperation>();
constexpr auto ifft = ttnn::register_operation<"ttnn::experimental::ifft", operations::experimental::signal_processing::IFFT1dOperation>();

}  // namespace ttnn