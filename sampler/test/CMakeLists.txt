find_package(glog REQUIRED)
find_package(GTest REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

aux_source_directory(../test/unit/model DIR_TEST_MODEL)


add_executable(test_dango test_main.cpp ${DIR_TEST_MODEL} )

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_dango ${link_ext_lib})
target_include_directories(test_dango PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_dango PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_dango PUBLIC ../dango/inc)
target_link_directories(test_dango PUBLIC ${PROJECT_SOURCE_DIR}/lib)




target_link_libraries(test_dango dango_infer)

set_target_properties(test_dango PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_dango PROPERTIES CUDA_SEPARABLE_COMPILATION ON)