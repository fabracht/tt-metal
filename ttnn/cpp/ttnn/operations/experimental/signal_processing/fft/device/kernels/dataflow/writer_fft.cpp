// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr_real = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr_imag = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out_real = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out_imag = get_compile_time_arg_val(1);

    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out_real);
    const DataFormat data_format = get_dataformat(cb_id_out_real);

    const InterleavedAddrGenFastPacked<true> d_real = {
        .bank_base_address = dst_addr_real,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };
    
    const InterleavedAddrGenFastPacked<true> d_imag = {
        .bank_base_address = dst_addr_imag,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        // Write real part
        cb_wait_front(cb_id_out_real, 1);
        uint32_t l1_read_addr_real = get_read_ptr(cb_id_out_real);
        noc_async_write_tile(i, d_real, l1_read_addr_real);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out_real, 1);
        
        // Write imaginary part
        cb_wait_front(cb_id_out_imag, 1);
        uint32_t l1_read_addr_imag = get_read_ptr(cb_id_out_imag);
        noc_async_write_tile(i, d_imag, l1_read_addr_imag);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out_imag, 1);
    }
}