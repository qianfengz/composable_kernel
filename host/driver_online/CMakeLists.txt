include_directories(BEFORE
    include
    ${PROJECT_BINARY_DIR}/host/online_compile/include
    ${PROJECT_SOURCE_DIR}/host/online_compile/include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/solver/include
    ${PROJECT_SOURCE_DIR}/ck/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/ck/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/ck/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/ck/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/ck/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/ck/composable_kernel/include/driver
    ${PROJECT_SOURCE_DIR}/ck/external/rocm/include
    ${PROJECT_SOURCE_DIR}/ck/external/half/include
)

##add_executable(conv_fwd_driver_online ${CONV_FWD_DRIVER_ONLINE_SOURCE})
add_executable(reduce_driver_online reduce_driver_online.cpp)

target_link_libraries(reduce_driver_online PRIVATE host_tensor)
target_link_libraries(reduce_driver_online PRIVATE online_compile)

