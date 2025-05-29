// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_pybind.hpp"
#include "fft.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {

namespace detail {

template <typename T>
void bind_fft_operation(py::module& module, const std::string& name, const std::string& doc) {
    ttnn::bind_registered_operation(
        module,
        name,
        doc,
        ttnn::pybind_overload_t{
            [](const T& op,
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
            [](const T& op,
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

template <typename T>
void bind_fft2d_operation(py::module& module, const std::string& name, const std::string& doc) {
    ttnn::bind_registered_operation(
        module,
        name,
        doc,
        ttnn::pybind_overload_t{
            [](const T& op,
               const ComplexTensor& input_tensor,
               const std::optional<std::array<int64_t, 2>>& s,
               const std::array<int64_t, 2>& dim,
               const FFTNorm norm,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t queue_id) {
                return op(input_tensor, s, dim, norm, memory_config, queue_id);
            },
            py::arg("input_tensor"),
            py::arg("s") = std::nullopt,
            py::arg("dim") = std::array<int64_t, 2>{-2, -1},
            py::arg("norm") = FFTNorm::BACKWARD,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
        },
        ttnn::pybind_overload_t{
            [](const T& op,
               const Tensor& input_tensor,
               const std::optional<std::array<int64_t, 2>>& s,
               const std::array<int64_t, 2>& dim,
               const FFTNorm norm,
               const std::optional<MemoryConfig>& memory_config,
               const uint8_t queue_id) {
                return op(input_tensor, s, dim, norm, memory_config, queue_id);
            },
            py::arg("input_tensor"),
            py::arg("s") = std::nullopt,
            py::arg("dim") = std::array<int64_t, 2>{-2, -1},
            py::arg("norm") = FFTNorm::BACKWARD,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0
        }
    );
}

}  // namespace detail

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
    detail::bind_fft_operation<decltype(ttnn::experimental::fft)>(
        module,
        ttnn::experimental::fft,
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
        )doc"
    );
    
    detail::bind_fft_operation<decltype(ttnn::experimental::ifft)>(
        module,
        ttnn::experimental::ifft,
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
        )doc"
    );
    
    detail::bind_fft2d_operation<decltype(ttnn::experimental::fft2d)>(
        module,
        ttnn::experimental::fft2d,
        R"doc(
        Compute the 2-dimensional discrete Fourier Transform.
        
        Args:
            input_tensor: Input tensor (real or complex)
            s: Shape of the transformed axes. If None, use the entire axes
            dim: Axes along which to compute the FFT (default: [-2, -1])
            norm: Normalization mode: "backward" (default), "ortho", or "forward"
            memory_config: Memory configuration for the output tensor
            queue_id: Queue ID for the operation
            
        Returns:
            ComplexTensor: The 2D FFT of the input
        )doc"
    );
    
    detail::bind_fft2d_operation<decltype(ttnn::experimental::ifft2d)>(
        module,
        ttnn::experimental::ifft2d,
        R"doc(
        Compute the 2-dimensional inverse discrete Fourier Transform.
        
        Args:
            input_tensor: Input tensor (real or complex)
            s: Shape of the transformed axes. If None, use the entire axes
            dim: Axes along which to compute the IFFT (default: [-2, -1])
            norm: Normalization mode: "backward" (default), "ortho", or "forward"
            memory_config: Memory configuration for the output tensor
            queue_id: Queue ID for the operation
            
        Returns:
            ComplexTensor: The 2D inverse FFT of the input
        )doc"
    );
}

}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn