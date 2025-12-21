#include "kernel/cuda/elementwise.cuh"
/*
  1.elementwise的算子的输入保证了float是4的倍数
  2.bf16 或者fp16 是8的倍数
  3.已经setDevice
  4.确认了input 和 output的尺寸问题
*/


namespace f32x4_kernel_cu
{
  #define FLOAT4_READ(value) (reinterpret_cast<const float4 *>(&(value))[0])
  #define FLOAT4_WRITE(value) (reinterpret_cast<float4 *>(&(value))[0])

  __global__ void elementwise_add_f32x4_kernel(int32_t size,const float *in1, const float *in2, float *out) 
  {
    int32_t idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < size) 
    {
      float4 reg_a = FLOAT4_READ(in1[idx]);
      float4 reg_b = FLOAT4_READ(in2[idx]);
      float4 reg_c;
      reg_c.x = reg_a.x + reg_b.x;
      reg_c.y = reg_a.y + reg_b.y;
      reg_c.z = reg_a.z + reg_b.z;
      reg_c.w = reg_a.w + reg_b.w;
      FLOAT4_WRITE(out[idx]) = reg_c;
    }
  }

  void elementwise_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,const tensor::Tensor& output, void* stream) 
  {
    int32_t size = static_cast<int32_t>(input1.size());
    int32_t thread_num = 512;
    int32_t block_num = (size + 4 * thread_num - 1) / (4 * thread_num);

    if (stream) 
      elementwise_add_f32x4_kernel<<<block_num, thread_num, 0, static_cast<CUstream_st*>(stream)>>>(size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    else 
      elementwise_add_f32x4_kernel<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),const_cast<float*>(output.ptr<float>()));
  }
}  // namespace f32x4_kernel_cu

namespace bf16x8_kernel_cu
{
  __global__ void elementwise_add_bf16x8_kernel(int32_t size,
    const __nv_bfloat16* in1,const __nv_bfloat16* in2,__nv_bfloat16* out)                                       
  {
    int32_t idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x); 
    if (idx + 7 < size) 
    {
      const __nv_bfloat162* in1_2 = reinterpret_cast<const __nv_bfloat162*>(in1);
      const __nv_bfloat162* in2_2 = reinterpret_cast<const __nv_bfloat162*>(in2);
      __nv_bfloat162* out_2 = reinterpret_cast<__nv_bfloat162*>(out);

      // bf162 索引：2 个 bf16 -> 1 个 bf162，所以 /2
      int32_t j = idx >> 1;  // idx / 2

      // 一次处理 4 个 bf162 (= 8 个 bf16)
      __nv_bfloat162 a0 = in1_2[j + 0];
      __nv_bfloat162 a1 = in1_2[j + 1];
      __nv_bfloat162 a2 = in1_2[j + 2];
      __nv_bfloat162 a3 = in1_2[j + 3];

      __nv_bfloat162 b0 = in2_2[j + 0];
      __nv_bfloat162 b1 = in2_2[j + 1];
      __nv_bfloat162 b2 = in2_2[j + 2];
      __nv_bfloat162 b3 = in2_2[j + 3];

      out_2[j + 0] = __hadd2(a0, b0);
      out_2[j + 1] = __hadd2(a1, b1);
      out_2[j + 2] = __hadd2(a2, b2);
      out_2[j + 3] = __hadd2(a3, b3);
      return;
    }

    if (idx < size) 
    {
      #pragma unroll
      for (int k = 0; k < 8; ++k) 
      {
        int32_t t = idx + k;
        if (t < size) 
        {
          float a = __bfloat162float(in1[t]);
          float b = __bfloat162float(in2[t]);
          out[t] = __float2bfloat16(a + b);
        }
      }
    }
  }

  void elementwise_kernel(const tensor::Tensor& input1,
                        const tensor::Tensor& input2,
                        const tensor::Tensor& output,
                        void* stream) 
  {
    int32_t size = static_cast<int32_t>(input1.size());

    int32_t thread_num = 512;

    // 每线程处理 8 个元素，所以网格按 ceil(size/8)
    int32_t work_items = (size + 7) / 8;
    int32_t block_num  = (work_items + thread_num - 1) / thread_num;

    const __nv_bfloat16* in1 = input1.ptr<__nv_bfloat16>();
    const __nv_bfloat16* in2 = input2.ptr<__nv_bfloat16>();
    __nv_bfloat16* out = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());

    if (stream) 
      elementwise_add_bf16x8_kernel<<<block_num, thread_num, 0,static_cast<CUstream_st*>(stream)>>>(size, in1, in2, out);
    else 
      elementwise_add_bf16x8_kernel<<<block_num, thread_num>>>(size, in1, in2, out);
  
  }

}








