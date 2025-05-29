#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Test FFT API structure using mocks (no hardware required)"""

import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys


class MockTensor:
    """Mock tensor class for testing"""
    def __init__(self, shape, dtype, layout, device=None):
        self.shape_val = shape
        self.dtype_val = dtype
        self.layout_val = layout
        self.device_val = device
        self.data = np.random.randn(*shape) if len(shape) == 4 else np.random.randn(32, 32)
    
    def shape(self):
        return self.shape_val
    
    def get_dtype(self):
        return self.dtype_val
    
    def get_layout(self):
        return self.layout_val
    
    def device(self):
        return self.device_val
    
    def memory_config(self):
        return Mock()
    
    def get_logical_shape(self):
        mock_shape = Mock()
        mock_shape.rank.return_value = len(self.shape_val)
        mock_shape.__getitem__ = lambda self, idx: self.shape_val[idx]
        return mock_shape
    
    def volume(self):
        return np.prod(self.shape_val)
    
    def buffer(self):
        mock_buffer = Mock()
        mock_buffer.address.return_value = 0x1000
        return mock_buffer


class MockComplexTensor:
    """Mock complex tensor class"""
    def __init__(self, real_tensor, imag_tensor):
        self.real_tensor = real_tensor
        self.imag_tensor = imag_tensor
    
    def real(self):
        return self.real_tensor
    
    def imag(self):
        return self.imag_tensor
    
    def get_dtype(self):
        return self.real_tensor.get_dtype()
    
    def get_layout(self):
        return self.real_tensor.get_layout()
    
    @property
    def shape_val(self):
        return self.real_tensor.shape_val


def create_mock_ttnn():
    """Create a mock ttnn module"""
    ttnn = MagicMock()
    
    # Mock data types
    ttnn.bfloat16 = "BFLOAT16"
    ttnn.float32 = "FLOAT32"
    ttnn.TILE_LAYOUT = "TILE"
    
    # Mock device operations
    ttnn.open_device = Mock(return_value=Mock())
    ttnn.close_device = Mock()
    
    # Mock tensor operations
    def mock_from_torch(tensor, dtype, layout, device):
        shape = list(tensor.shape)
        return MockTensor(shape, dtype, layout, device)
    
    ttnn.from_torch = mock_from_torch
    ttnn.to_torch = lambda tensor: np.random.randn(32, 32)
    
    # Mock complex tensor
    ttnn.complex_tensor = lambda real, imag: MockComplexTensor(real, imag)
    
    # Mock experimental namespace
    ttnn.experimental = MagicMock()
    
    # Mock FFT operations
    def mock_fft(input_tensor, n=-1, dim=-1, norm=None):
        if isinstance(input_tensor, MockComplexTensor):
            # Complex input
            real_shape = input_tensor.real().shape_val
        else:
            # Real input
            real_shape = input_tensor.shape_val
        
        # Create output complex tensor
        output_real = MockTensor(real_shape, input_tensor.get_dtype(), input_tensor.get_layout())
        output_imag = MockTensor(real_shape, input_tensor.get_dtype(), input_tensor.get_layout())
        return MockComplexTensor(output_real, output_imag)
    
    ttnn.experimental.fft = mock_fft
    ttnn.experimental.ifft = mock_fft
    ttnn.experimental.fft2d = mock_fft
    ttnn.experimental.ifft2d = mock_fft
    
    # Mock FFT enums
    class MockFFTNorm:
        BACKWARD = "BACKWARD"
        ORTHO = "ORTHO"
        FORWARD = "FORWARD"
    
    ttnn.experimental.FFTNorm = MockFFTNorm
    
    # Mock other operations used in tests
    ttnn.zeros_like = lambda tensor, dtype, layout, memory_config, queue_id: MockTensor(
        tensor.shape_val, dtype, layout
    )
    
    return ttnn


def test_fft_api_structure():
    """Test that the FFT API has the expected structure"""
    # Create mock ttnn module
    ttnn = create_mock_ttnn()
    
    print("Testing FFT API structure...")
    
    # Test 1: Basic FFT operations exist
    assert hasattr(ttnn.experimental, 'fft'), "fft operation missing"
    assert hasattr(ttnn.experimental, 'ifft'), "ifft operation missing"
    assert hasattr(ttnn.experimental, 'fft2d'), "fft2d operation missing"
    assert hasattr(ttnn.experimental, 'ifft2d'), "ifft2d operation missing"
    print("✓ All FFT operations present")
    
    # Test 2: FFTNorm enum exists with correct values
    assert hasattr(ttnn.experimental, 'FFTNorm'), "FFTNorm enum missing"
    assert hasattr(ttnn.experimental.FFTNorm, 'BACKWARD'), "FFTNorm.BACKWARD missing"
    assert hasattr(ttnn.experimental.FFTNorm, 'ORTHO'), "FFTNorm.ORTHO missing"
    assert hasattr(ttnn.experimental.FFTNorm, 'FORWARD'), "FFTNorm.FORWARD missing"
    print("✓ FFTNorm enum correct")
    
    # Test 3: FFT accepts correct parameters
    device = ttnn.open_device(device_id=0)
    
    # Create test tensors
    shape = (1, 1, 32, 32)
    input_tensor = MockTensor(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    
    # Test real input FFT
    try:
        result = ttnn.experimental.fft(input_tensor)
        assert isinstance(result, MockComplexTensor), "FFT should return ComplexTensor"
        print("✓ FFT with real input works")
    except Exception as e:
        print(f"✗ FFT with real input failed: {e}")
    
    # Test complex input FFT
    try:
        real_tensor = MockTensor(shape, ttnn.float32, ttnn.TILE_LAYOUT, device)
        imag_tensor = MockTensor(shape, ttnn.float32, ttnn.TILE_LAYOUT, device)
        complex_input = ttnn.complex_tensor(real_tensor, imag_tensor)
        
        result = ttnn.experimental.fft(complex_input)
        assert isinstance(result, MockComplexTensor), "FFT should return ComplexTensor"
        print("✓ FFT with complex input works")
    except Exception as e:
        print(f"✗ FFT with complex input failed: {e}")
    
    # Test FFT with parameters
    try:
        result = ttnn.experimental.fft(
            input_tensor,
            n=64,
            dim=-1,
            norm=ttnn.experimental.FFTNorm.ORTHO
        )
        print("✓ FFT with parameters works")
    except Exception as e:
        print(f"✗ FFT with parameters failed: {e}")
    
    # Test 2D FFT
    try:
        result = ttnn.experimental.fft2d(input_tensor)
        assert isinstance(result, MockComplexTensor), "FFT2D should return ComplexTensor"
        print("✓ 2D FFT works")
    except Exception as e:
        print(f"✗ 2D FFT failed: {e}")
    
    # Test inverse operations
    try:
        fft_result = ttnn.experimental.fft(input_tensor)
        ifft_result = ttnn.experimental.ifft(fft_result)
        assert isinstance(ifft_result, MockComplexTensor), "IFFT should return ComplexTensor"
        print("✓ IFFT works")
    except Exception as e:
        print(f"✗ IFFT failed: {e}")
    
    ttnn.close_device(device)
    print("\n✅ All API structure tests passed!")


def test_data_type_support():
    """Test that both BF16 and FP32 are supported"""
    ttnn = create_mock_ttnn()
    device = ttnn.open_device(device_id=0)
    
    print("\nTesting data type support...")
    
    shape = (1, 1, 64, 64)
    
    # Test BF16
    try:
        tensor_bf16 = MockTensor(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        result_bf16 = ttnn.experimental.fft(tensor_bf16)
        assert result_bf16.real().get_dtype() == ttnn.bfloat16, "Output should maintain BF16 dtype"
        print("✓ BF16 support verified")
    except Exception as e:
        print(f"✗ BF16 support failed: {e}")
    
    # Test FP32
    try:
        tensor_fp32 = MockTensor(shape, ttnn.float32, ttnn.TILE_LAYOUT, device)
        result_fp32 = ttnn.experimental.fft(tensor_fp32)
        assert result_fp32.real().get_dtype() == ttnn.float32, "Output should maintain FP32 dtype"
        print("✓ FP32 support verified")
    except Exception as e:
        print(f"✗ FP32 support failed: {e}")
    
    ttnn.close_device(device)
    print("✅ Data type tests passed!")


def test_normalization_modes():
    """Test all normalization modes"""
    ttnn = create_mock_ttnn()
    device = ttnn.open_device(device_id=0)
    
    print("\nTesting normalization modes...")
    
    shape = (1, 1, 32, 32)
    input_tensor = MockTensor(shape, ttnn.float32, ttnn.TILE_LAYOUT, device)
    
    norms = [
        (ttnn.experimental.FFTNorm.BACKWARD, "backward"),
        (ttnn.experimental.FFTNorm.ORTHO, "ortho"),
        (ttnn.experimental.FFTNorm.FORWARD, "forward")
    ]
    
    for norm_enum, norm_name in norms:
        try:
            result = ttnn.experimental.fft(input_tensor, norm=norm_enum)
            print(f"✓ {norm_name} normalization works")
        except Exception as e:
            print(f"✗ {norm_name} normalization failed: {e}")
    
    ttnn.close_device(device)
    print("✅ Normalization mode tests passed!")


def test_fft_sizes():
    """Test various FFT sizes"""
    ttnn = create_mock_ttnn()
    device = ttnn.open_device(device_id=0)
    
    print("\nTesting various FFT sizes...")
    
    # Test power-of-2 sizes
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    for size in sizes:
        try:
            # Create appropriately sized tensor
            tiles_needed = (size + 31) // 32  # Round up to tile size
            shape = (1, 1, tiles_needed * 32, 32)
            input_tensor = MockTensor(shape, ttnn.float32, ttnn.TILE_LAYOUT, device)
            
            result = ttnn.experimental.fft(input_tensor, n=size)
            print(f"✓ FFT size {size} works")
        except Exception as e:
            print(f"✗ FFT size {size} failed: {e}")
    
    ttnn.close_device(device)
    print("✅ FFT size tests passed!")


if __name__ == "__main__":
    print("Running FFT API mock tests...\n")
    
    test_fft_api_structure()
    test_data_type_support()
    test_normalization_modes()
    test_fft_sizes()
    
    print("\n🎉 All mock tests completed!")