// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#ifdef TEST_ENABLE_FABRIC_TRACING
#include "tt_metal/tools/profiler/experimental/fabric_event_profiler.hpp"
#endif

// clang-format on

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

uint32_t target_address = get_compile_time_arg_val(2);

constexpr bool is_chip_multicast = get_compile_time_arg_val(3);
constexpr bool is_noc_multicast = get_compile_time_arg_val(4);

inline void setup_header(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t packet_payload_size_bytes,
    uint32_t start_distance,
    uint32_t range,
    uint32_t dest_addr,
    uint8_t noc_x_start,
    uint8_t noc_y_start,
    uint8_t mcast_rect_size_x,
    uint8_t mcast_rect_size_y) {
    if constexpr (is_chip_multicast) {
        packet_header->to_chip_multicast(
            MulticastRoutingCommandHeader{static_cast<uint8_t>(start_distance), static_cast<uint8_t>(range)});
    } else {
        packet_header->to_chip_unicast(static_cast<uint8_t>(start_distance));
    }

    if constexpr (is_noc_multicast) {
        packet_header->to_noc_multicast(
            NocMulticastCommandHeader{dest_addr, noc_x_start, noc_y_start, mcast_rect_size_x, mcast_rect_size_y},
            packet_payload_size_bytes);
    } else {
        packet_header->to_noc_unicast_write(
            NocUnicastCommandHeader{get_noc_addr(noc_x_start, noc_y_start, dest_addr)}, packet_payload_size_bytes);
    }
}

inline void send_packet(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t source_l1_buffer_address,
    uint32_t packet_payload_size_bytes,
    uint32_t seed,
    tt::tt_fabric::WorkerToFabricEdmSender& connection) {
#ifndef BENCHMARK_MODE
    // fill packet data for sanity testing
    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
    fill_packet_data(start_addr, packet_payload_size_bytes / 16, seed);
    tt_l1_ptr uint32_t* last_word_addr =
        reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address + packet_payload_size_bytes - 4);
#endif
    connection.wait_for_empty_write_slot();
#ifdef TEST_ENABLE_FABRIC_TRACING
    RECORD_FABRIC_HEADER(packet_header);
#endif
    connection.send_payload_without_header_non_blocking_from_address(
        source_l1_buffer_address, packet_payload_size_bytes);
    connection.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// connect to edm
inline void setup_connection(tt::tt_fabric::WorkerToFabricEdmSender& connection) { connection.open(); }

inline void teardown_connection(tt::tt_fabric::WorkerToFabricEdmSender& connection) { connection.close(); }

void kernel_main() {
    using namespace tt::tt_fabric;

    size_t rt_args_idx = 0;
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t start_distance = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t range = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc_x_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc_y_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mcast_rect_size_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mcast_rect_size_y = get_arg_val<uint32_t>(rt_args_idx++);

    // create packet header
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;

    fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);

    zero_l1_buf((uint32_t*)packet_header_buffer_address, sizeof(PACKET_HEADER_TYPE) * 2);

    setup_connection(fabric_connection);

    setup_header(
        packet_header,
        packet_payload_size_bytes,
        start_distance,
        range,
        target_address,
        noc_x_start,
        noc_y_start,
        mcast_rect_size_x,
        mcast_rect_size_y);

    // initialize results
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    // loop over for num packets
    for (uint32_t i = 0; i < num_packets; i++) {
#ifndef BENCHMARK_MODE
        time_seed = prng_next(time_seed);
#endif

        send_packet(packet_header, source_l1_buffer_address, packet_payload_size_bytes, time_seed, fabric_connection);

#ifndef BENCHMARK_MODE
        target_address += packet_payload_size_bytes;
        // update header for next packet
        if (i < num_packets - 1) {
            setup_header(
                packet_header,
                packet_payload_size_bytes,
                start_distance,
                range,
                target_address,
                noc_x_start,
                noc_y_start,
                mcast_rect_size_x,
                mcast_rect_size_y);
        }
#endif
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    teardown_connection(fabric_connection);

    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    // write out results
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
