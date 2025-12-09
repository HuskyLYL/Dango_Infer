#include "kernel/cuda/elementwise.cuh"

namespace f32x4_kernel_cu
{


  #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

  __global__ void elementwise_add_f32x4_kernel(int32_t size,const float *in1, const float *in2, float *out) 
  {
    int32_t idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < size) 
    {
      float4 reg_a = FLOAT4(a[idx]);
      float4 reg_b = FLOAT4(b[idx]);
      float4 reg_c;
      reg_c.x = reg_a.x + reg_b.x;
      reg_c.y = reg_a.y + reg_b.y;
      reg_c.z = reg_a.z + reg_b.z;
      reg_c.w = reg_a.w + reg_b.w;
      FLOAT4(c[idx]) = reg_c;
    }
  }

  void elementwise_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,const tensor::Tensor& output, void* stream) 
  {
    int32_t size = static_cast<int32_t>(input1.size());
    int32_t thread_num = 512;
    int32_t block_num = (size + thread_num - 1) / thread_num;

    if (stream) 
      elementwise_add_f32x4_kernel<<<block_num, thread_num, 0, static_cast<CUstream_st*>(stream)>>>(size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    else 
      elementwise_add_f32x4_kernel<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),const_cast<float*>(output.ptr<float>()));
  }
}  // namespace f32x4_kernel_cu
