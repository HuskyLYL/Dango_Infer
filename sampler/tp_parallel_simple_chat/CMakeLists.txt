
add_executable(tp_parallel_simple_chat main.cpp)

target_link_directories(tp_parallel_simple_chat PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(tp_parallel_simple_chat dango_infer)


# 在多个kernel_cu的时候使用
set_target_properties(tp_parallel_simple_chat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

