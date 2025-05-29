// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <cmath>

// This kernel would be responsible for loading pre-computed twiddle factors
// In a real implementation, these would be computed on the host and stored in DRAM
void kernel_main() {
    const uint32_t twiddle_addr_real = get_arg_val<uint32_t>(0);
    const uint32_t twiddle_addr_imag = get_arg_val<uint32_t>(1);
    const uint32_t fft_size = get_arg_val<uint32_t>(2);
    const uint32_t is_inverse = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_twiddle_real = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_twiddle_imag = get_compile_time_arg_val(1);

    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_twiddle_real);
    const DataFormat data_format = get_dataformat(cb_id_twiddle_real);

    // In a real implementation, we would:
    // 1. Load pre-computed twiddle factors from DRAM
    // 2. These would be organized by FFT stage and butterfly index
    // 3. For inverse FFT, we'd use conjugate twiddle factors
    
    // The twiddle factors for an N-point FFT are:
    // W_N^k = exp(-2πik/N) for forward FFT
    // W_N^k = exp(2πik/N) for inverse FFT
    
    // For now, this is a placeholder showing the structure
    const InterleavedAddrGenFastPacked<true> twiddle_real = {
        .bank_base_address = twiddle_addr_real,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };
    
    const InterleavedAddrGenFastPacked<true> twiddle_imag = {
        .bank_base_address = twiddle_addr_imag,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };

    // Load twiddle factors for all stages
    // In practice, we'd load these on-demand per stage
    uint32_t num_twiddle_tiles = (fft_size / 2) / 32; // Assuming 32 elements per tile
    if (num_twiddle_tiles == 0) num_twiddle_tiles = 1;
    
    for (uint32_t i = 0; i < num_twiddle_tiles; i++) {
        cb_reserve_back(cb_id_twiddle_real, 1);
        cb_reserve_back(cb_id_twiddle_imag, 1);
        
        uint32_t l1_write_addr_real = get_write_ptr(cb_id_twiddle_real);
        uint32_t l1_write_addr_imag = get_write_ptr(cb_id_twiddle_imag);
        
        noc_async_read_tile(i, twiddle_real, l1_write_addr_real);
        noc_async_read_tile(i, twiddle_imag, l1_write_addr_imag);
        
        noc_async_read_barrier();
        
        cb_push_back(cb_id_twiddle_real, 1);
        cb_push_back(cb_id_twiddle_imag, 1);
    }
}