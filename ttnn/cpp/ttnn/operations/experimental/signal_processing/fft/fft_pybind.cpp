// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_pybind.hpp"
#include "fft.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {


void bind_fft_operations(py::module& module) {
    // Bind enums
    py::enum_<FFTMode>(module, "FFTMode")
        .value("FFT", FFTMode::FFT)
        .value("IFFT", FFTMode::IFFT);
    
    py::enum_<FFTNorm>(module, "FFTNorm")
        .value("BACKWARD", FFTNorm::BACKWARD)
        .value("ORTHO", FFTNorm::ORTHO)
        .value("FORWARD", FFTNorm::FORWARD);
    
    // Bind FFT operations
    ttnn::bind_registered_operation(
        module,
        ttnn::fft,
        R"doc(
        Compute the 1-dimensional discrete Fourier Transform.
        
        Args:
            input_tensor: Input tensor (real or complex)
            n: Length of the transformed axis. If -1, use the entire axis
            dim: Axis along which to compute the FFT (default: -1)
            norm: Normalization mode: "backward" (default), "ortho", or "forward"
            memory_config: Memory configuration for the output tensor
            queue_id: Queue ID for the operation
            
        Returns:
            ComplexTensor: The FFT of the input
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::fft)& op,
               const ComplexTensor& input_tensor,
               const int64_t n,
               const int64_t dim,
               const FFTNorm norm,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t queue_id) {
                return op(input_tensor, n, dim, norm, memory_config, queue_id);
            },
            py::arg("input_tensor"),
            py::arg("n") = -1,
            py::arg("dim") = -1,
            py::arg("norm") = FFTNorm::BACKWARD,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
        },
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::fft)& op,
               const Tensor& input_tensor,
               const int64_t n,
               const int64_t dim,
               const FFTNorm norm,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t queue_id) {
                return op(input_tensor, n, dim, norm, memory_config, queue_id);
            },
            py::arg("input_tensor"),
            py::arg("n") = -1,
            py::arg("dim") = -1,
            py::arg("norm") = FFTNorm::BACKWARD,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
        }
    );
    
    ttnn::bind_registered_operation(
        module,
        ttnn::ifft,
        R"doc(
        Compute the 1-dimensional inverse discrete Fourier Transform.
        
        Args:
            input_tensor: Input tensor (real or complex)
            n: Length of the transformed axis. If -1, use the entire axis
            dim: Axis along which to compute the IFFT (default: -1)
            norm: Normalization mode: "backward" (default), "ortho", or "forward"
            memory_config: Memory configuration for the output tensor
            queue_id: Queue ID for the operation
            
        Returns:
            ComplexTensor: The inverse FFT of the input
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::fft)& op,
               const ComplexTensor& input_tensor,
               const int64_t n,
               const int64_t dim,
               const FFTNorm norm,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t queue_id) {
                return op(input_tensor, n, dim, norm, memory_config, queue_id);
            },
            py::arg("input_tensor"),
            py::arg("n") = -1,
            py::arg("dim") = -1,
            py::arg("norm") = FFTNorm::BACKWARD,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
        },
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::fft)& op,
               const Tensor& input_tensor,
               const int64_t n,
               const int64_t dim,
               const FFTNorm norm,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t queue_id) {
                return op(input_tensor, n, dim, norm, memory_config, queue_id);
            },
            py::arg("input_tensor"),
            py::arg("n") = -1,
            py::arg("dim") = -1,
            py::arg("norm") = FFTNorm::BACKWARD,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
        }
    );
    
}

}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn