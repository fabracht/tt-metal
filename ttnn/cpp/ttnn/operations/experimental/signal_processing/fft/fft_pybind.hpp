// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ttnn {
namespace operations {
namespace experimental {
namespace signal_processing {

namespace py = pybind11;

void bind_fft_operations(py::module& module);

}  // namespace signal_processing
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn