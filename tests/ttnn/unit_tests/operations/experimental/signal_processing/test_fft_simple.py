#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple test to verify FFT implementation builds and runs"""

import torch
import ttnn
import numpy as np

def test_fft_basic():
    """Test that FFT operations can be imported and basic functionality works"""
    
    # Check that FFT operations are available
    assert hasattr(ttnn.experimental, 'fft'), "FFT operation not found in ttnn.experimental"
    assert hasattr(ttnn.experimental, 'ifft'), "IFFT operation not found in ttnn.experimental"
    assert hasattr(ttnn.experimental, 'fft2d'), "FFT2D operation not found in ttnn.experimental"
    assert hasattr(ttnn.experimental, 'ifft2d'), "IFFT2D operation not found in ttnn.experimental"
    
    # Check that FFT enums are available
    assert hasattr(ttnn.experimental, 'FFTNorm'), "FFTNorm enum not found"
    assert hasattr(ttnn.experimental.FFTNorm, 'BACKWARD'), "FFTNorm.BACKWARD not found"
    assert hasattr(ttnn.experimental.FFTNorm, 'ORTHO'), "FFTNorm.ORTHO not found"
    assert hasattr(ttnn.experimental.FFTNorm, 'FORWARD'), "FFTNorm.FORWARD not found"
    
    print("✓ All FFT operations and enums are available")
    
    # Try to create a simple tensor and verify complex tensor creation works
    shape = (1, 1, 32, 32)
    x_real = torch.randn(shape, dtype=torch.float32)
    x_imag = torch.zeros(shape, dtype=torch.float32)
    
    # Check if we can create a complex tensor
    device = ttnn.open_device(device_id=0)
    try:
        x_real_ttnn = ttnn.from_torch(x_real, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        x_imag_ttnn = ttnn.from_torch(x_imag, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        x_complex = ttnn.complex_tensor(x_real_ttnn, x_imag_ttnn)
        
        print("✓ Complex tensor creation works")
        print(f"  Real part shape: {x_complex.real().shape}")
        print(f"  Imag part shape: {x_complex.imag().shape}")
        
        # Note: Actual FFT computation will fail until kernels are properly implemented
        # This is just to test that the API is available
        try:
            # This will likely fail with "Single core FFT implementation not yet complete"
            # which is expected at this stage
            result = ttnn.experimental.fft(x_complex)
            print("✓ FFT computation succeeded (unexpected!)")
        except Exception as e:
            if "not yet complete" in str(e) or "not implemented" in str(e).lower():
                print(f"✓ FFT computation failed as expected: {e}")
            else:
                print(f"✗ Unexpected error: {e}")
                raise
        
    finally:
        ttnn.close_device(device)
    
    print("\n✓ Basic FFT API test passed!")

if __name__ == "__main__":
    test_fft_basic()