cmake_minimum_required(VERSION 3.16)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")





project(dango_infer CXX CUDA)
include(cmake/cuda.cmake)
include(cmake/sentencepiece.cmake)






set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


aux_source_directory(dango_infer/src/base/ DIR_BASE)
aux_source_directory(dango_infer/src/tensor/ DIR_TENSOR)
aux_source_directory(dango_infer/src/model/ DIR_MODEL)
aux_source_directory(dango_infer/src/op/ DIR_OP)
aux_source_directory(dango_infer/src/kernel/ DIR_KERNEL)
aux_source_directory(dango_infer/src/op/cuda DIR_CUDA)

aux_source_directory(dango_infer/src/sampler DIR_SAMPLER)



add_library(dango_infer SHARED ${DIR_BASE} ${DIR_TENSOR} ${DIR_MODEL} ${DIR_OP} ${DIR_KERNEL} ${DIR_CUDA} ${DIR_SAMPLER} ${DIR_flashInfer})


find_package(glog REQUIRED)



target_link_libraries(dango_infer glog::glog sentencepiece)




target_include_directories(dango_infer PUBLIC ${PROJECT_SOURCE_DIR}/dango_infer/inc)
target_include_directories(dango_infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(dango_infer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(sampler/simple_chat)
add_subdirectory(sampler/test)





