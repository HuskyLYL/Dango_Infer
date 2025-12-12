#ifndef DANGO_INFER_REDUCE_SUM_CUH
#define DANGO_INFER_REDUCE_SUM_CUH

#include <cuda_runtime.h>

namespace base_kernel_cu 
{
  // warp-level sum reduction
  template <const int kWarpSize = 32>
  __device__ __forceinline__ float warp_reduce_sum_f32(float val) 
  {
      #pragma unroll
      for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) 
          val += __shfl_xor_sync(0xffffffff, val, mask);  
      return val;
  }

  // block-level sum reduction
  template <const int NUM_THREADS = 256>
  __device__ __forceinline__ float block_reduce_sum_f32(float val) 
  {
      constexpr int NUM_WARPS = (NUM_THREADS + 32 - 1) / 32;
      int warp = threadIdx.x / 32;
      int lane = threadIdx.x % 32;
      static __shared__ float shared[NUM_WARPS];
      val = warp_reduce_sum_f32<32>(val);
      if (lane == 0)
          shared[warp] = val;
      __syncthreads();
      val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
      val = warp_reduce_sum_f32<NUM_WARPS>(val);
      return val;
  }
}  // namespace f32x4_kernel_cu

#endif  // DANGO_INFER_REDUCE_SUM_CUH
