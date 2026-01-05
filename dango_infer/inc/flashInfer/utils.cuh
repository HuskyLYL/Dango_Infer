#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
#include <cuda_runtime.h>   // cudaGetDevice, cudaDeviceGetAttribute, cudaDevAttr...
#include <utility>          // std::pair, std::make_pair
#include <cstdint>    
#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 3) {                              \
    constexpr size_t GROUP_SIZE = 3;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else {                                                   \
    std::ostringstream err_msg;                              \
    err_msg << "Unsupported group_size: " << group_size;     \
    FLASHINFER_ERROR(err_msg.str());                         \
  }
#endif  // FLASHINFER_UTILS_CUH_

#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, ...) \
  if (compute_capacity.first >= 8) {                                                        \
    constexpr uint32_t NUM_STAGES_SMEM = 2;                                                 \
    __VA_ARGS__                                                                             \
  } else {                                                                                  \
    constexpr uint32_t NUM_STAGES_SMEM = 1;                                                 \
    __VA_ARGS__                                                                             \
  }

namespace flashinfer
{
  template <typename T1, typename T2>
  __forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) 
  {
    return (x + y - 1) / y;
  }


  inline std::pair<int, int> GetCudaComputeCapability() 
  {
    int device_id = 0;
    cudaGetDevice(&device_id);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    return std::make_pair(major, minor);
  }





}