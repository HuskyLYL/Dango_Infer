
add_executable(simple_chat main.cpp)

target_link_directories(simple_chat PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(simple_chat dango_infer)


# 在多个kernel_cu的时候使用
set_target_properties(simple_chat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

