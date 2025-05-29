// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {

// Constants for FFT implementation
constexpr uint32_t TILE_SIZE = 32;  // 32x32 tile

// Helper to perform bit reversal on an index
FORCE_INLINE uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2n; ++i) {
        if (x & (1 << i)) {
            result |= 1 << (log2n - 1 - i);
        }
    }
    return result;
}

// Complex multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
// Using 3 multiplies instead of 4 (Gauss's method)
FORCE_INLINE void complex_multiply_gauss(
    uint32_t a_real, uint32_t a_imag,
    uint32_t b_real, uint32_t b_imag,
    uint32_t dst_real, uint32_t dst_imag,
    uint32_t temp1, uint32_t temp2, uint32_t temp3) {
    
    // k1 = c * (a + b)
    add_tiles_init();
    add_tiles(a_real, a_imag, temp1);
    mul_tiles_init();
    mul_tiles(b_real, temp1, temp1);
    
    // k2 = a * (d - c)
    sub_tiles_init();
    sub_tiles(b_imag, b_real, temp2);
    mul_tiles(a_real, temp2, temp2);
    
    // k3 = b * (c + d)
    add_tiles_init();
    add_tiles(b_real, b_imag, temp3);
    mul_tiles(a_imag, temp3, temp3);
    
    // real = k1 - k3 = ac - bd
    sub_tiles_init();
    sub_tiles(temp1, temp3, dst_real);
    
    // imag = k1 + k2 = ad + bc
    add_tiles_init();
    add_tiles(temp1, temp2, dst_imag);
}

// Butterfly operation: 
// out1 = in1 + W * in2
// out2 = in1 - W * in2
FORCE_INLINE void butterfly_operation(
    uint32_t in1_real, uint32_t in1_imag,
    uint32_t in2_real, uint32_t in2_imag,
    uint32_t w_real, uint32_t w_imag,
    uint32_t out1_real, uint32_t out1_imag,
    uint32_t out2_real, uint32_t out2_imag,
    uint32_t temp_real, uint32_t temp_imag,
    uint32_t temp1, uint32_t temp2, uint32_t temp3) {
    
    // temp = W * in2
    complex_multiply_gauss(in2_real, in2_imag, w_real, w_imag, 
                          temp_real, temp_imag, temp1, temp2, temp3);
    
    // out1 = in1 + temp
    add_tiles_init();
    add_tiles(in1_real, temp_real, out1_real);
    add_tiles(in1_imag, temp_imag, out1_imag);
    
    // out2 = in1 - temp
    sub_tiles_init();
    sub_tiles(in1_real, temp_real, out2_real);
    sub_tiles(in1_imag, temp_imag, out2_imag);
}

// Main FFT compute kernel
void MAIN {
    // Compile-time arguments
    const uint32_t num_tiles = get_compile_time_arg_val(0);
    const uint32_t fft_size = get_compile_time_arg_val(1);
    const uint32_t is_inverse = get_compile_time_arg_val(2);
    const uint32_t log2_fft_size = get_compile_time_arg_val(3);
    const uint32_t radix = get_compile_time_arg_val(4);  // 2 for radix-2
    
    // Circular buffer IDs
    constexpr uint32_t cb_in_real = 0;
    constexpr uint32_t cb_in_imag = 1;
    constexpr uint32_t cb_out_real = 2;
    constexpr uint32_t cb_out_imag = 3;
    constexpr uint32_t cb_twiddle_real = 4;
    constexpr uint32_t cb_twiddle_imag = 5;
    constexpr uint32_t cb_work1_real = 6;
    constexpr uint32_t cb_work1_imag = 7;
    constexpr uint32_t cb_work2_real = 8;
    constexpr uint32_t cb_work2_imag = 9;
    constexpr uint32_t cb_temp_real = 10;
    constexpr uint32_t cb_temp_imag = 11;
    constexpr uint32_t cb_temp1 = 12;
    constexpr uint32_t cb_temp2 = 13;
    constexpr uint32_t cb_temp3 = 14;
    
    // Initialize operations
    binary_op_init_common(cb_in_real, cb_in_imag);
    
    // Process tiles
    // Each tile contains TILE_SIZE x TILE_SIZE elements
    // For FFT, we process along rows (each row is an independent FFT)
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Wait for input
        cb_wait_front(cb_in_real, 1);
        cb_wait_front(cb_in_imag, 1);
        
        // Reserve working buffers
        cb_reserve_back(cb_work1_real, 1);
        cb_reserve_back(cb_work1_imag, 1);
        cb_reserve_back(cb_work2_real, 1);
        cb_reserve_back(cb_work2_imag, 1);
        
        // Step 1: Bit-reversal permutation
        // In a tile-based system, this is complex and might be better done
        // as a separate data movement kernel
        // For now, we'll work with the data as-is
        
        // Copy input to work buffer
        copy_tile(cb_in_real, 0, cb_work1_real);
        copy_tile(cb_in_imag, 0, cb_work1_imag);
        
        // Step 2: FFT stages
        uint32_t src_cb_real = cb_work1_real;
        uint32_t src_cb_imag = cb_work1_imag;
        uint32_t dst_cb_real = cb_work2_real;
        uint32_t dst_cb_imag = cb_work2_imag;
        
        // Perform log2(N) stages of FFT
        for (uint32_t stage = 1; stage <= log2_fft_size; ++stage) {
            uint32_t butterfly_span = 1 << (stage - 1);
            uint32_t butterfly_group_size = 1 << stage;
            uint32_t num_groups = fft_size >> stage;
            
            // Load twiddle factors for this stage
            // In a real implementation, these would be pre-loaded
            cb_wait_front(cb_twiddle_real, butterfly_span);
            cb_wait_front(cb_twiddle_imag, butterfly_span);
            
            // Process butterflies
            // This is simplified - in reality, we'd need to handle
            // the mapping between linear indices and tile positions
            
            // For demonstration, just copy through
            // A full implementation would iterate through all butterflies
            // and apply the butterfly operation with appropriate twiddle factors
            
            copy_tile(src_cb_real, 0, dst_cb_real);
            copy_tile(src_cb_imag, 0, dst_cb_imag);
            
            // Swap source and destination for next stage
            uint32_t temp = src_cb_real;
            src_cb_real = dst_cb_real;
            dst_cb_real = temp;
            
            temp = src_cb_imag;
            src_cb_imag = dst_cb_imag;
            dst_cb_imag = temp;
            
            cb_pop_front(cb_twiddle_real, butterfly_span);
            cb_pop_front(cb_twiddle_imag, butterfly_span);
        }
        
        // Step 3: Output the result
        cb_reserve_back(cb_out_real, 1);
        cb_reserve_back(cb_out_imag, 1);
        
        // Copy final result to output
        // Handle normalization for inverse FFT
        if (is_inverse) {
            // For inverse FFT, we need to divide by N
            // This would require a reciprocal operation
            // For now, just copy
            copy_tile(src_cb_real, 0, 0);
            copy_tile(src_cb_imag, 0, 1);
            
            // In a real implementation:
            // recip_tile(fft_size_tile, 2);  // 1/N in tile 2
            // mul_tiles(0, 2, 0);  // real * (1/N)
            // mul_tiles(1, 2, 1);  // imag * (1/N)
            
            pack_tile(0, cb_out_real);
            pack_tile(1, cb_out_imag);
        } else {
            copy_tile(src_cb_real, 0, 0);
            pack_tile(0, cb_out_real);
            copy_tile(src_cb_imag, 0, 0);
            pack_tile(0, cb_out_imag);
        }
        
        // Clean up
        cb_push_back(cb_out_real, 1);
        cb_push_back(cb_out_imag, 1);
        cb_pop_front(cb_in_real, 1);
        cb_pop_front(cb_in_imag, 1);
        cb_push_back(cb_work1_real, 1);
        cb_push_back(cb_work1_imag, 1);
        cb_push_back(cb_work2_real, 1);
        cb_push_back(cb_work2_imag, 1);
    }
}

} // namespace NAMESPACE