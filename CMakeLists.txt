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
aux_source_directory(dango_infer/src/nccl DIR_NCCL)
aux_source_directory(dango_infer/src/op/flashInfer DIR_flashInfer)
aux_source_directory(dango_infer/src/sampler DIR_SAMPLER)



add_library(dango_infer SHARED ${DIR_BASE} ${DIR_TENSOR} ${DIR_MODEL} ${DIR_OP} ${DIR_KERNEL} ${DIR_CUDA} ${DIR_SAMPLER} ${DIR_flashInfer} ${DIR_NCCL})


find_package(glog REQUIRED)

find_package(NCCL QUIET)
if(NOT NCCL_FOUND)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS $ENV{NCCL_ROOT}/include $ENV{CONDA_PREFIX}/include /usr/local/include /usr/include /usr/local/cuda/include)
    find_library(NCCL_LIBRARY nccl
        HINTS $ENV{NCCL_ROOT}/lib $ENV{NCCL_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib /usr/local/lib /usr/local/lib64 /usr/lib /usr/lib64 /usr/local/cuda/lib64)
    if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
        message(FATAL_ERROR "NCCL not found: set NCCL_ROOT or NCCL_INCLUDE_DIR/NCCL_LIBRARY.")
    endif()
else()
    set(NCCL_INCLUDE_DIR ${NCCL_INCLUDE_DIRS})
    set(NCCL_LIBRARY ${NCCL_LIBRARIES})
endif()


target_link_libraries(dango_infer glog::glog sentencepiece ${NCCL_LIBRARY})


target_include_directories(dango_infer PUBLIC ${PROJECT_SOURCE_DIR}/dango_infer/inc)
target_include_directories(dango_infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(dango_infer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(dango_infer PUBLIC ${NCCL_INCLUDE_DIR})


add_subdirectory(sampler/simple_chat)
add_subdirectory(sampler/test)
add_subdirectory(sampler/tp_parallel_simple_chat)


