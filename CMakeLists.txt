cmake_minimum_required(VERSION 3.16)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")


project(dango_infer CXX CUDA)


include(cmake/cuda.cmake)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)





aux_source_directory(dango_infer/src/base/ DIR_BASE)
aux_source_directory(dango_infer/src/tensor/ DIR_TENSOR)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(dango_infer SHARED ${DIR_BASE} ${DIR_TENSOR})

target_link_libraries(dango_infer sentencepiece glog::glog)

target_include_directories(dango_infer PUBLIC ${PROJECT_SOURCE_DIR}/dango_infer/inc)
target_include_directories(dango_infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(dango_infer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(dango_infer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)





