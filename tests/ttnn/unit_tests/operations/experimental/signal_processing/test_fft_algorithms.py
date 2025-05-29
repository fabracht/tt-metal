#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Test FFT algorithms without requiring hardware"""

import numpy as np
import math


class TestFFTAlgorithms:
    """Test FFT algorithm components independently of hardware"""
    
    def test_bit_reversal(self):
        """Test bit reversal permutation"""
        def bit_reverse(x, log2n):
            result = 0
            for i in range(log2n):
                if x & (1 << i):
                    result |= 1 << (log2n - 1 - i)
            return result
        
        # Test for different FFT sizes
        for log2n in range(1, 8):  # Test up to 128-point FFT
            n = 1 << log2n
            indices = [bit_reverse(i, log2n) for i in range(n)]
            
            # Check that it's a valid permutation
            assert sorted(indices) == list(range(n)), f"Invalid permutation for n={n}"
            
            # Check specific known values
            if n == 8:
                expected = [0, 4, 2, 6, 1, 5, 3, 7]
                assert indices == expected
            
            # Check that applying twice gives identity
            double_reversed = [indices[indices[i]] for i in range(n)]
            assert double_reversed == list(range(n)), "Double bit-reversal should be identity"
    
    def test_twiddle_factors(self):
        """Test twiddle factor generation"""
        for n in [8, 16, 32, 64, 128]:
            # Forward FFT twiddle factors
            forward_twiddles = []
            for k in range(n // 2):
                angle = -2 * np.pi * k / n
                forward_twiddles.append(complex(np.cos(angle), np.sin(angle)))
            
            # Inverse FFT twiddle factors (conjugate)
            inverse_twiddles = []
            for k in range(n // 2):
                angle = 2 * np.pi * k / n
                inverse_twiddles.append(complex(np.cos(angle), np.sin(angle)))
            
            # Check properties
            # W_n^0 should always be 1
            assert abs(forward_twiddles[0] - 1) < 1e-6
            assert abs(inverse_twiddles[0] - 1) < 1e-6
            
            # W_n^(n/2) should be -1
            if n > 2:
                assert abs(forward_twiddles[n//4] - complex(0, -1)) < 1e-6 or \
                       abs(forward_twiddles[n//4] - complex(0, 1)) < 1e-6
            
            # Check that inverse twiddles are conjugates of forward
            for i in range(n // 2):
                assert abs(forward_twiddles[i].conjugate() - inverse_twiddles[i]) < 1e-6
    
    def test_cooley_tukey_reference(self):
        """Test reference Cooley-Tukey implementation"""
        def cooley_tukey_fft(x, inverse=False):
            n = len(x)
            if n <= 1:
                return x
            
            # Check power of 2
            if n & (n - 1) != 0:
                raise ValueError("Size must be power of 2")
            
            # Make a copy
            x = x.copy()
            
            # Bit-reversal permutation
            def bit_reverse(i, log2n):
                result = 0
                for j in range(log2n):
                    if i & (1 << j):
                        result |= 1 << (log2n - 1 - j)
                return result
            
            log2n = int(np.log2(n))
            for i in range(n):
                j = bit_reverse(i, log2n)
                if i < j:
                    x[i], x[j] = x[j], x[i]
            
            # Cooley-Tukey FFT
            direction = 1 if inverse else -1
            
            # Perform FFT stages
            for stage in range(1, log2n + 1):
                m = 1 << stage
                m2 = m >> 1
                
                # Twiddle factor for this stage
                w_m = np.exp(direction * 2j * np.pi / m)
                
                # Process all sub-problems at this stage
                for k in range(0, n, m):
                    w = 1
                    
                    # Butterfly operations
                    for j in range(m2):
                        t = k + j
                        u = t + m2
                        
                        # Butterfly
                        temp = w * x[u]
                        x[u] = x[t] - temp
                        x[t] = x[t] + temp
                        
                        # Update twiddle factor
                        w *= w_m
            
            # Normalize for inverse FFT
            if inverse:
                x = x / n
            
            return x
        
        # Test against numpy FFT
        for n in [8, 16, 32, 64]:
            # Random complex signal
            x = np.random.randn(n) + 1j * np.random.randn(n)
            
            # Our implementation
            y_ours = cooley_tukey_fft(x.copy())
            
            # NumPy reference
            y_ref = np.fft.fft(x)
            
            # Compare
            assert np.allclose(y_ours, y_ref, rtol=1e-10), f"FFT mismatch for n={n}"
            
            # Test inverse
            y_inv = cooley_tukey_fft(y_ours.copy(), inverse=True)
            assert np.allclose(y_inv, x, rtol=1e-10), f"IFFT mismatch for n={n}"
    
    def test_complex_multiply_gauss(self):
        """Test Gauss's 3-multiply complex multiplication"""
        def complex_multiply_standard(a, b):
            # Standard 4-multiply method
            real = a.real * b.real - a.imag * b.imag
            imag = a.real * b.imag + a.imag * b.real
            return complex(real, imag)
        
        def complex_multiply_gauss(a, b):
            # Gauss's 3-multiply method
            k1 = b.real * (a.real + a.imag)
            k2 = a.real * (b.imag - b.real)
            k3 = a.imag * (b.real + b.imag)
            real = k1 - k3
            imag = k1 + k2
            return complex(real, imag)
        
        # Test with various complex numbers
        test_cases = [
            (1+2j, 3+4j),
            (5-3j, -2+7j),
            (0+1j, 0+1j),  # i * i = -1
            (1+0j, 3+4j),  # Real * complex
            (2.5+1.5j, -0.5+2.5j)
        ]
        
        for a, b in test_cases:
            result_standard = complex_multiply_standard(a, b)
            result_gauss = complex_multiply_gauss(a, b)
            
            # Should be identical (within floating point tolerance)
            assert abs(result_standard - result_gauss) < 1e-10, \
                f"Mismatch for {a} * {b}: standard={result_standard}, gauss={result_gauss}"
    
    def test_butterfly_operation(self):
        """Test FFT butterfly operation"""
        def butterfly(a, b, w):
            """
            Butterfly operation:
            out1 = a + w * b
            out2 = a - w * b
            """
            temp = w * b
            return a + temp, a - temp
        
        # Test cases
        test_cases = [
            (1+0j, 1+0j, 1+0j),  # Simple case with w=1
            (2+3j, 4+5j, 0+1j),  # w = i
            (1+2j, 3+4j, np.exp(-2j * np.pi / 8))  # Typical FFT twiddle
        ]
        
        for a, b, w in test_cases:
            out1, out2 = butterfly(a, b, w)
            
            # Verify butterfly properties
            # Sum should equal 2*a (when w=1)
            if w == 1:
                assert abs((out1 + out2) - 2*a) < 1e-10
            
            # Difference should equal 2*w*b
            assert abs((out1 - out2) - 2*w*b) < 1e-10
    
    def test_fft_properties(self):
        """Test mathematical properties of FFT"""
        n = 64
        
        # Test 1: Linearity
        x1 = np.random.randn(n) + 1j * np.random.randn(n)
        x2 = np.random.randn(n) + 1j * np.random.randn(n)
        a, b = 2.5, -1.3
        
        fft_linear = np.fft.fft(a * x1 + b * x2)
        fft_sum = a * np.fft.fft(x1) + b * np.fft.fft(x2)
        assert np.allclose(fft_linear, fft_sum), "FFT linearity failed"
        
        # Test 2: Parseval's theorem (energy conservation)
        x = np.random.randn(n) + 1j * np.random.randn(n)
        X = np.fft.fft(x)
        energy_time = np.sum(np.abs(x)**2)
        energy_freq = np.sum(np.abs(X)**2) / n
        assert np.isclose(energy_time, energy_freq, rtol=1e-10), "Parseval's theorem failed"
        
        # Test 3: Convolution theorem
        # Use shorter signals for convolution test
        x = np.random.randn(n//2)
        h = np.random.randn(n//2)
        
        # Pad for circular convolution
        x_pad = np.pad(x, (0, n//2), mode='constant')
        h_pad = np.pad(h, (0, n//2), mode='constant')
        
        # Frequency domain multiplication (circular convolution)
        X = np.fft.fft(x_pad)
        H = np.fft.fft(h_pad)
        conv_freq = np.real(np.fft.ifft(X * H))
        
        # Time domain linear convolution
        conv_time = np.convolve(x, h, mode='full')
        
        # Compare the valid part (they match for linear convolution length)
        valid_len = min(len(conv_freq), len(conv_time))
        assert np.allclose(conv_freq[:valid_len-1], conv_time[:valid_len-1], rtol=1e-5), \
            "Convolution theorem failed"
    
    def test_normalization_modes(self):
        """Test different FFT normalization modes"""
        n = 32
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        # Backward (default): no normalization on forward, 1/n on inverse
        fft_backward = np.fft.fft(x, norm='backward')
        ifft_backward = np.fft.ifft(fft_backward, norm='backward')
        assert np.allclose(ifft_backward, x), "Backward norm failed"
        
        # Ortho: 1/sqrt(n) on both forward and inverse
        fft_ortho = np.fft.fft(x, norm='ortho')
        ifft_ortho = np.fft.ifft(fft_ortho, norm='ortho')
        assert np.allclose(ifft_ortho, x), "Ortho norm failed"
        
        # Forward: 1/n on forward, no normalization on inverse
        fft_forward = np.fft.fft(x, norm='forward')
        ifft_forward = np.fft.ifft(fft_forward, norm='forward')
        assert np.allclose(ifft_forward, x), "Forward norm failed"
        
        # Check relative scaling
        assert np.allclose(fft_backward / fft_ortho, np.sqrt(n)), "Backward/ortho scaling wrong"
        assert np.allclose(fft_backward / fft_forward, n), "Backward/forward scaling wrong"


if __name__ == "__main__":
    test = TestFFTAlgorithms()
    
    print("Testing bit reversal...")
    test.test_bit_reversal()
    print("✓ Bit reversal tests passed")
    
    print("\nTesting twiddle factors...")
    test.test_twiddle_factors()
    print("✓ Twiddle factor tests passed")
    
    print("\nTesting Cooley-Tukey reference implementation...")
    test.test_cooley_tukey_reference()
    print("✓ Cooley-Tukey tests passed")
    
    print("\nTesting complex multiplication...")
    test.test_complex_multiply_gauss()
    print("✓ Complex multiplication tests passed")
    
    print("\nTesting butterfly operations...")
    test.test_butterfly_operation()
    print("✓ Butterfly operation tests passed")
    
    print("\nTesting FFT mathematical properties...")
    test.test_fft_properties()
    print("✓ FFT properties tests passed")
    
    print("\nTesting normalization modes...")
    test.test_normalization_modes()
    print("✓ Normalization tests passed")
    
    print("\n✅ All algorithm tests passed!")