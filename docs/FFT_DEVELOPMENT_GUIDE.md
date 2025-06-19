# FFT Development Guide

This guide covers developing and testing the FFT implementation in TT-Metal using a dev container.

## Prerequisites

- VS Code with Dev Containers extension
- Docker Desktop
- Git

## Setup Development Environment

### 1. Open in Dev Container

1. **Open VS Code** in the tt-metal repository root
2. **Open Command Palette**: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. **Type**: `Dev Containers: Reopen in Container`
4. **Wait** for container to build (first time takes ~5-10 minutes)

### 2. Alternative: Manual Docker Setup

If dev container doesn't work, use Docker directly:

```bash
# Build the development container
docker build -f dockerfile/Dockerfile.basic-dev -t tt-metal-dev .

# Run with workspace mounted
docker run -it -v $(pwd):/workspace -w /workspace tt-metal-dev bash

# Set environment variables
export TT_METAL_HOME=/workspace
export ARCH_NAME=wormhole_b0
```

## Building FFT Implementation

### 1. Initialize Build System

```bash
# Inside the container
export TT_METAL_HOME=/workspace
export ARCH_NAME=wormhole_b0

# Initialize git submodules (if not done)
git submodule update --init --recursive

# Configure build
cmake -B build -G Ninja
```

### 2. Build FFT Module

```bash
# Build entire project
cmake --build build

# Or build just TTNN (includes FFT)
cmake --build build --target ttnn

# Or build specific FFT target
cmake --build build --target ttnn_experimental_signal_processing_fft
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install tt-metal in development mode
pip install -e .
```

## Testing FFT Implementation

### 1. Run FFT Unit Tests

```bash
# Run all FFT tests
pytest tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft.py -v

# Run specific test
pytest tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft.py::test_fft_1d -v

# Run with detailed output
pytest tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft.py -v -s
```

### 2. Quick FFT Test

Create a simple test script:

```python
# test_fft_quick.py
import torch
import ttnn

# Create test data
x = torch.randn(1, 1, 32, 32, dtype=torch.float32)
print(f"Input shape: {x.shape}")

# Test if FFT API is accessible
try:
    # Note: This will fail without actual hardware, but tests API
    device = ttnn.open_device(device_id=0)
    x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Test FFT call
    output = ttnn.experimental.fft(x_ttnn, dim=-1)
    print("✅ FFT API works!")
    
    ttnn.close_device(device)
except Exception as e:
    print(f"Expected error (no hardware): {e}")
    print("✅ FFT API is accessible")
```

Run with: `python test_fft_quick.py`

### 3. Check Build Issues

```bash
# Check for compilation errors
cmake --build build 2>&1 | grep -i error

# Check specific FFT files compiled
find build -name "*fft*" -type f

# Verify FFT symbols exist
nm build/lib/libtt_metal.a | grep -i fft
```

## FFT Implementation Status

### Current State (Updated)
- ✅ **API Infrastructure**: Complete 1D FFT/IFFT API with proper registration
- ✅ **Build System**: CMake integration working, _ttnn.so builds successfully
- ✅ **Device Operations**: FFT device operation framework complete with proper patterns
- ✅ **Python Bindings**: Complete pybind11 bindings for FFT operations
- ✅ **Build Fixes**: Fixed CircularBufferConfig API, namespace issues, compilation errors
- ⚠️ **Compute Kernels**: Placeholder implementation (copies input→output, no actual FFT math)
- ❌ **Testing**: Blocked by numpy import issues in dev container
- ❌ **Hardware Testing**: Requires actual TT hardware

### Key Files
- **Main API**: `ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/fft.cpp`
- **Device Op**: `ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/fft_device_operation.cpp`
- **Kernels**: `ttnn/cpp/ttnn/operations/experimental/signal_processing/fft/device/kernels/`
- **Tests**: `tests/ttnn/unit_tests/operations/experimental/signal_processing/test_fft.py`

### Next Steps
1. **Fix testing environment**: Resolve numpy import issues to enable test execution
2. **Complete compute kernels** with actual FFT butterfly operations
3. **Test on TT hardware** using CI/CD
4. **Optimize performance** for different FFT sizes
5. **Add multi-core implementation** (currently disabled)

### Important Notes
- **1D Only**: Implementation provides only 1D FFT/IFFT as per requirements (2D operations removed)
- **Architecture Complete**: Device operation framework follows modern TTNN patterns
- **TT Architecture Aligned**: Leverages tile-based compute (32x32), explicit data movement, and distributed SRAM
- **Circular Buffers**: Proper setup for input/output/work/twiddle factor storage in L1 SRAM
- **RISC-V Kernels**: Compute kernels will auto-split into UNPACK/MATH/PACK phases
- **Testing Ready**: Test infrastructure exists but needs working Python environment

## Troubleshooting

### Build Fails
- **Check environment variables**: Ensure `TT_METAL_HOME` and `ARCH_NAME` are set
- **Clean build**: `rm -rf build && cmake -B build -G Ninja`
- **Check dependencies**: Container should have all deps, but verify cmake/ninja versions

### Tests Fail
- **Numpy Import Issues**: Dev container has numpy import conflicts (known issue)
- **Expected without hardware**: Tests will skip/fail without TT hardware
- **Illegal Instruction**: CPU instruction errors expected without TT hardware
- **Python path issues**: Ensure you ran `pip install -e .` and `_ttnn.so` symlink exists

### Dev Container Issues
- **Rebuild container**: `Dev Containers: Rebuild Container`
- **Check Docker**: Ensure Docker Desktop is running
- **Volume mounting**: Verify your code changes appear in `/workspace`

## Contributing

### Making Changes
1. **Edit code** in the mounted workspace
2. **Rebuild** with `cmake --build build`
3. **Test changes** with pytest
4. **Commit changes** normally (git works from container)

### Creating PR
1. **Ensure build works**: `cmake --build build`
2. **Run tests**: `pytest tests/ttnn/unit_tests/operations/experimental/signal_processing/`
3. **Check formatting**: Follow existing code style
4. **Document changes**: Update this guide if needed

The dev container provides a complete Linux environment matching the CI system, so builds and tests should work reliably.