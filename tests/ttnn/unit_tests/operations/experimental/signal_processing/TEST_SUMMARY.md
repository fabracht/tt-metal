# FFT Testing

## Tests without hardware

**Algorithm tests** (`test_fft_algorithms.py`)
- Bit-reversal permutation
- Twiddle factor generation  
- Cooley-Tukey FFT vs NumPy
- Complex multiplication (standard and Gauss)
- FFT properties (linearity, Parseval's theorem)
- All normalization modes

**API tests** (`test_fft_api_mock.py`)
- All operations present (fft, ifft, fft2d, ifft2d)
- BF16 and FP32 support
- FFT sizes from 8 to 1024

**Algorithm equivalence** (`test_algorithm_equivalence.py`)
- Verifies Python tests match C++ implementation

## Tests requiring hardware

- Kernel execution on device
- Performance benchmarks
- Numerical accuracy
- Multi-core coordination

## Running tests

```bash
# Without hardware
python3 tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft_algorithms.py
python3 tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft_api_mock.py
python3 tests/ttnn/unit_tests/operations/experimental/signal_processing/test_algorithm_equivalence.py

# With hardware
pytest tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft.py
```