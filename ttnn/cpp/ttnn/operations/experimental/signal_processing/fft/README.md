# FFT Implementation for TT-NN

Fast Fourier Transform operations for TensorTorrent hardware.

## Features

- 1D and 2D FFT/IFFT operations
- BF16 and FP32 data types
- Complex tensor support
- Normalization modes: backward (default), ortho, forward
- Cooley-Tukey radix-2 algorithm
- Twiddle factor generation
- Multi-core capable architecture

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

# 2D FFT
x_2d = torch.randn(1, 1, 64, 64)
x_2d_ttnn = ttnn.from_torch(x_2d, device=device, layout=ttnn.TILE_LAYOUT)
y_2d_complex = ttnn.experimental.fft2d(x_2d_ttnn)

# Inverse FFT
x_complex = ttnn.complex_tensor(real_ttnn, imag_ttnn)
x_reconstructed = ttnn.experimental.ifft(y_complex)
```

## Algorithm Details

The implementation will use the Cooley-Tukey FFT algorithm:

1. **Decimation in Time (DIT)**: Split the DFT into even and odd indices
2. **Butterfly Operations**: Combine results using twiddle factors
3. **Bit-Reversal**: Reorder the output to get the correct frequency ordering

For hardware efficiency:
- Twiddle factors are precomputed and stored
- Complex operations are optimized for the tile-based architecture
- Multi-core distribution based on FFT size and available resources

## Performance Considerations

- **Memory Layout**: Uses tiled layout for efficient matrix operations
- **Precision**: Supports both BF16 and FP32
- **Parallelization**: 
  - Small FFTs (≤512 points): Single core
  - Large FFTs: Multi-core with data distribution

## References

1. Cooley, J. W.; Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series"
2. TT-NN Architecture Documentation
3. [Parent Issue #11835](https://github.com/tenstorrent/tt-metal/issues/11835)