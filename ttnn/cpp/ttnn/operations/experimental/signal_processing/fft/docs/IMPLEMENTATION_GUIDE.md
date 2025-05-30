# FFT Implementation Guide for TT Hardware

This document provides guidance on implementing the Fast Fourier Transform (FFT) on TensorTorrent hardware.

## Algorithm Overview

The implementation uses the Cooley-Tukey radix-2 Decimation-in-Time (DIT) algorithm:

1. **Bit-reversal permutation** of input data
2. **Log₂(N) stages** of butterfly operations
3. **Twiddle factor multiplication** at each stage
4. **Normalization** for inverse FFT

## TT Hardware Considerations

### Tile-Based Processing
- TT hardware operates on 32x32 tiles
- FFT data needs to be organized to maximize tile utilization
- Complex numbers stored as separate real and imaginary tiles

### Memory Hierarchy
- **L1 SRAM**: Fast local memory for active computations
- **DRAM**: Slower memory for data storage
- **NoC**: Network for inter-core communication

### Compute Resources
- **Matrix Engine**: For bulk multiply operations
- **Vector Engine**: For element-wise operations
- **SFPU**: For transcendental functions (sin/cos for twiddle factors)

## Implementation Strategy

### Single-Core Implementation (N ≤ 512)
1. Load entire dataset into L1
2. Perform in-place FFT
3. Write results back to DRAM

### Multi-Core Implementation (N > 512)
1. **Data Distribution**: 
   - Distribute FFT points across cores
   - Each core handles N/P points (P = number of cores)
   
2. **Parallel Butterfly Stages**:
   - Early stages: Local butterflies within each core
   - Later stages: Inter-core communication required
   
3. **Synchronization**:
   - Barrier synchronization between stages
   - NoC for data exchange

## Kernel Structure

### Dataflow Kernels
1. **Reader**: Loads input data with bit-reversed addressing
2. **Twiddle Reader**: Loads pre-computed twiddle factors
3. **Writer**: Stores output data

### Compute Kernel
1. **Butterfly Operations**:
   ```
   X[k] = x[k] + W * x[k + stride]
   X[k + stride] = x[k] - W * x[k + stride]
   ```
2. **Complex Multiplication**: Uses 4 real multiplies + 2 adds
3. **Stage Iteration**: Log₂(N) stages with increasing butterfly spans

## Optimization Opportunities

### Mixed Precision
- Use BF16 for general computation
- Switch to FP32 for accumulation to maintain accuracy

### Twiddle Factor Optimization
- Pre-compute and store in DRAM
- Share twiddle factors across multiple FFTs
- Use symmetry properties to reduce storage

### Memory Access Patterns
- Coalesce memory accesses
- Use double buffering to hide latency
- Optimize for cache-friendly access patterns

### Parallelization
- Stage-level parallelism for large FFTs
- Batch parallelism for multiple small FFTs
- Hybrid approach for optimal resource utilization

## Performance Targets

Based on theoretical analysis:
- **Single-core**: ~1000 FFT/s for 512-point FFT
- **Multi-core (8 cores)**: ~5000 FFT/s for 4096-point FFT
- **Efficiency**: 60-80% of theoretical peak FLOPS

## Testing Strategy

1. **Unit Tests**: Verify against NumPy/PyTorch reference
2. **Accuracy Tests**: Check numerical precision (PCC > 0.99 for FP32)
3. **Performance Tests**: Measure throughput and latency
4. **Stress Tests**: Large FFT sizes, batched operations

## Future Enhancements

1. **Mixed-radix FFT**: Support non-power-of-2 sizes
2. **Real-to-complex FFT**: Optimize for real inputs
3. **Multi-dimensional FFT**: Native 2D/3D support
4. **Convolution via FFT**: Fast convolution implementation
5. **Batched Operations**: Efficient multi-FFT processing