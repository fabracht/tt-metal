# FFT Implementation for TT-NN

Fast Fourier Transform operations for TensorTorrent hardware.

## Features (Current Implementation)

- **1D FFT/IFFT operations only** (as per requirements)
- **BF16 and FP32 data types** support
- **Complex tensor support** with automatic real-to-complex conversion
- **Normalization modes**: backward (default), ortho, forward
- **Device operation architecture** complete
- **Python bindings** implemented
- **Build system integration** working

## Implementation Status

✅ **Architecture Complete**: Full device operation framework implemented
✅ **API Complete**: 1D FFT/IFFT operations with proper registration
✅ **Build System**: CMake integration and compilation working
✅ **Python Bindings**: Complete pybind11 integration
⚠️ **Compute Kernels**: Placeholder only (copy input→output, no FFT math)
❌ **Testing**: Blocked by environment issues
❌ **Multi-core**: Single-core only (multi-core disabled)

## Structure

- `fft.hpp/cpp` - Main API
- `device/` - Device operations and kernels
- `docs/` - Implementation documentation
- Python bindings in `fft_pybind.cpp`

## Architecture

The implementation follows the standard TT-NN operation pattern:

1. **High-level API** (`fft.hpp/cpp`): Provides user-facing functions
2. **Device Operation** (`device/fft_device_operation.hpp/cpp`): Handles device-specific logic
3. **Kernels** (`device/kernels/`):
   - `compute/fft_compute_kernel.cpp`: Performs the actual FFT computation
   - `dataflow/reader_fft.cpp`: Reads input data from DRAM/L1
   - `dataflow/writer_fft.cpp`: Writes output data to DRAM/L1

## Usage (once fully implemented)

```python
import ttnn
import torch

# 1D FFT
x = torch.randn(1, 1, 1024)
x_ttnn = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
y_complex = ttnn.experimental.fft(x_ttnn, dim=-1)
y_real = ttnn.to_torch(y_complex.real())
y_imag = ttnn.to_torch(y_complex.imag())

# Note: 2D FFT not implemented (not in requirements)

# Inverse FFT
x_complex = ttnn.complex_tensor(real_ttnn, imag_ttnn)
x_reconstructed = ttnn.experimental.ifft(y_complex)
```

## Algorithm Details

**Current Status**: Infrastructure ready, algorithm implementation pending

The design supports the Cooley-Tukey FFT algorithm:

1. **Device Operation Framework**: Complete implementation following TTNN patterns
2. **Memory Management**: Circular buffer configuration for input/output/work buffers
3. **Twiddle Factor Generation**: Algorithm implemented but not yet integrated
4. **Single-Core Path**: Working (placeholder kernels copy data)
5. **Multi-Core Path**: Disabled pending implementation

**Next Steps for Algorithm**:
- Implement actual FFT butterfly operations in compute kernels
- Add bit-reversal permutation
- Integrate twiddle factor computation
- Enable multi-core support for large FFTs

## Current Limitations

- **Compute Kernels**: Only placeholder implementation (no actual FFT computation)
- **Multi-Core**: Disabled (single-core threshold set to SIZE_MAX)
- **Testing**: Environment issues prevent test execution
- **Hardware**: Requires actual TT hardware for full validation

## Architecture Highlights

- **Modern TTNN Pattern**: Uses latest device operation structure (no base class inheritance)
- **Proper Memory Management**: CircularBufferConfig with correct API usage
- **Python Integration**: Complete bindings with operation registration
- **Build Integration**: CMake targets and proper linking

## References

1. Cooley, J. W.; Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series"
2. TT-NN Architecture Documentation
3. [Parent Issue #11835](https://github.com/tenstorrent/tt-metal/issues/11835)