# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_pcc


def get_expected_transform(x, type, s=None, dim=None, norm=None):
    """Helper function to compute expected FFT/IFFT results using PyTorch"""
    if norm == "ortho":
        norm_str = "ortho"
    elif norm == "forward":
        norm_str = "forward"
    else:  # backward (default)
        norm_str = "backward"
    
    if type == "fft":
        return torch.fft.fft(x, n=s, dim=dim, norm=norm_str)
    elif type == "ifft":
        return torch.fft.ifft(x, n=s, dim=dim, norm=norm_str)
    elif type == "fft2":
        return torch.fft.fft2(x, s=s, dim=dim, norm=norm_str)
    elif type == "ifft2":
        return torch.fft.ifft2(x, s=s, dim=dim, norm=norm_str)
    else:
        raise ValueError(f"Unknown transform type: {type}")


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 2, 64, 64), (2, 4, 128, 128)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_fft_1d(device, dtype, shape, dim, norm):
    torch.manual_seed(0)
    
    # Create input tensor
    x = torch.randn(shape, dtype=torch.float32)
    
    # Compute expected output using PyTorch
    expected = get_expected_transform(x, "fft", dim=dim, norm=norm)
    
    # Convert to ttnn
    device = ttnn.open_device(device_id=0)
    x_ttnn = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Compute FFT using ttnn
    norm_enum = getattr(ttnn.experimental.FFTNorm, norm.upper())
    output_ttnn = ttnn.experimental.fft(x_ttnn, dim=dim, norm=norm_enum)
    
    # Convert back to torch
    output_real = ttnn.to_torch(output_ttnn.real())
    output_imag = ttnn.to_torch(output_ttnn.imag())
    output = torch.complex(output_real, output_imag)
    
    # Compare with tolerance appropriate for data type
    if dtype == ttnn.bfloat16:
        # BFloat16 has limited precision
        assert_with_pcc(expected.real, output.real, pcc=0.95)
        assert_with_pcc(expected.imag, output.imag, pcc=0.95)
    else:
        assert_with_pcc(expected.real, output.real, pcc=0.99)
        assert_with_pcc(expected.imag, output.imag, pcc=0.99)
    
    ttnn.close_device(device)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 2, 64, 64)])
@pytest.mark.parametrize("n", [None, 16, 64])
def test_fft_1d_with_size(device, dtype, shape, n):
    
    torch.manual_seed(0)
    
    # Create input tensor
    x = torch.randn(shape, dtype=torch.float32)
    
    # Skip if n is larger than the dimension size
    if n is not None and n > shape[-1]:
        pytest.skip("FFT size larger than dimension")
    
    # Compute expected output using PyTorch
    expected = get_expected_transform(x, "fft", s=n, dim=-1)
    
    # Convert to ttnn
    device = ttnn.open_device(device_id=0)
    x_ttnn = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Compute FFT using ttnn
    output_ttnn = ttnn.experimental.fft(x_ttnn, n=n if n else -1, dim=-1)
    
    # Convert back to torch
    output_real = ttnn.to_torch(output_ttnn.real())
    output_imag = ttnn.to_torch(output_ttnn.imag())
    output = torch.complex(output_real, output_imag)
    
    # Compare
    if dtype == ttnn.bfloat16:
        assert_with_pcc(expected.real, output.real, pcc=0.95)
        assert_with_pcc(expected.imag, output.imag, pcc=0.95)
    else:
        assert_with_pcc(expected.real, output.real, pcc=0.99)
        assert_with_pcc(expected.imag, output.imag, pcc=0.99)
    
    ttnn.close_device(device)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 2, 64, 64)])
def test_ifft_1d(device, dtype, shape):
    
    torch.manual_seed(0)
    
    # Create complex input tensor
    x_real = torch.randn(shape, dtype=torch.float32)
    x_imag = torch.randn(shape, dtype=torch.float32)
    x = torch.complex(x_real, x_imag)
    
    # Compute expected output using PyTorch
    expected = get_expected_transform(x, "ifft", dim=-1)
    
    # Convert to ttnn
    device = ttnn.open_device(device_id=0)
    x_real_ttnn = ttnn.from_torch(x_real, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x_imag_ttnn = ttnn.from_torch(x_imag, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x_complex_ttnn = ttnn.complex_tensor(x_real_ttnn, x_imag_ttnn)
    
    # Compute IFFT using ttnn
    output_ttnn = ttnn.experimental.ifft(x_complex_ttnn, dim=-1)
    
    # Convert back to torch
    output_real = ttnn.to_torch(output_ttnn.real())
    output_imag = ttnn.to_torch(output_ttnn.imag())
    output = torch.complex(output_real, output_imag)
    
    # Compare
    if dtype == ttnn.bfloat16:
        assert_with_pcc(expected.real, output.real, pcc=0.95)
        assert_with_pcc(expected.imag, output.imag, pcc=0.95)
    else:
        assert_with_pcc(expected.real, output.real, pcc=0.99)
        assert_with_pcc(expected.imag, output.imag, pcc=0.99)
    
    ttnn.close_device(device)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 2, 32, 64)])
def test_fft_2d(device, dtype, shape):
    if device == "cpu":
        pytest.skip("Skipping CPU tests until FFT kernels are implemented")
    
    torch.manual_seed(0)
    
    # Create input tensor
    x = torch.randn(shape, dtype=torch.float32)
    
    # Compute expected output using PyTorch
    expected = get_expected_transform(x, "fft2")
    
    # Convert to ttnn
    device = ttnn.open_device(device_id=0)
    x_ttnn = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Compute 2D FFT using ttnn
    output_ttnn = ttnn.experimental.fft2d(x_ttnn)
    
    # Convert back to torch
    output_real = ttnn.to_torch(output_ttnn.real())
    output_imag = ttnn.to_torch(output_ttnn.imag())
    output = torch.complex(output_real, output_imag)
    
    # Compare
    if dtype == ttnn.bfloat16:
        assert_with_pcc(expected.real, output.real, pcc=0.95)
        assert_with_pcc(expected.imag, output.imag, pcc=0.95)
    else:
        assert_with_pcc(expected.real, output.real, pcc=0.99)
        assert_with_pcc(expected.imag, output.imag, pcc=0.99)
    
    ttnn.close_device(device)


@pytest.mark.parametrize("device", ["cpu"])
def test_fft_roundtrip(device):
    """Test that FFT followed by IFFT returns the original signal"""
    if device == "cpu":
        pytest.skip("Skipping CPU tests until FFT kernels are implemented")
    
    torch.manual_seed(0)
    
    shape = (1, 1, 64, 64)
    x = torch.randn(shape, dtype=torch.float32)
    
    # Convert to ttnn
    device = ttnn.open_device(device_id=0)
    x_ttnn = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    
    # FFT then IFFT
    fft_output = ttnn.experimental.fft(x_ttnn)
    ifft_output = ttnn.experimental.ifft(fft_output)
    
    # Convert back to torch (only real part should match original)
    output_real = ttnn.to_torch(ifft_output.real())
    
    # Compare with original
    assert_with_pcc(x, output_real, pcc=0.99)
    
    ttnn.close_device(device)