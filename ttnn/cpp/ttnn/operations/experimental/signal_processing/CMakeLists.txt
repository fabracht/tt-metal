add_library(ttnn_op_experimental_signal_processing ${LIB_TYPE})
add_library(TT::NN::Ops::Experimental::SignalProcessing ALIAS ttnn_op_experimental_signal_processing)

target_precompile_headers(ttnn_op_experimental_signal_processing REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_experimental_signal_processing
    PRIVATE
        fft/fft.cpp
        fft/fft_pybind.cpp
        fft/device/fft_device_operation.cpp
)

target_include_directories(ttnn_op_experimental_signal_processing PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_signal_processing
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_experimental_signal_processing LIBRARY COMPONENT tar)