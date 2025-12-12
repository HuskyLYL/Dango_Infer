#include <tensor/tensor.h>
#include "kernel/cuda/swiglu.cuh"
namespace base_kernel_cu 
{
  __global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) 
  {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // 避免早退导致同步不一致，超界线程写 0 占位后同步
    extern __shared__ float shared_mem[];
    float* smem1 = shared_mem;
    float* smem2 = shared_mem + blockDim.x;

    if (idx < size) 
    {
      smem1[tid] = in1[idx];
      smem2[tid] = in2[idx];
    }
    else 
    {
      smem1[tid] = 0.f;
      smem2[tid] = 0.f;
    }
    __syncthreads();

    if (idx < size) 
    {
      float value = 1.0f / (1.0f + exp(-smem1[tid]));
      smem1[tid] = smem1[tid] * value;
      out[idx] = smem1[tid] * smem2[tid];
    }
  }

  void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output, void* stream) 
  {


    const int size = static_cast<int32_t>(input1.size());
    constexpr int threads = 128;
    const int blocks = (size + threads - 1) / threads;
    const size_t shmem = threads * sizeof(float) * 2;

    if (!stream) 
    {
      swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    }
    else 
    {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    }
  }
}  // namespace base_kernel_cu
