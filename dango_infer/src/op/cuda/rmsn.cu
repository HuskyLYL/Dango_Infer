#include "kernel/cuda/rmsn.cuh"
#include "kernel/cuda/reduce_sum.cuh"
namespace f32x4_kernel_cu 
{

  #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

  //当tensor为一维的时候
  template <int32_t BLOCK_DIM>
  static __global__ void row_f32x4_rmsnorm_kernel(float* in, float* wei, float* out, int dim_size, float eps) 
  {

    const int tid = threadIdx.x;

    constexpr int pack_size = 4;
    const int pack_num = dim_size / pack_size;
    const int pack_off = pack_size * pack_num;

    float sum = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(in);

    for (int i = tid; i < pack_num; i += blockDim.x) 
    {
      float4 in_float4 = *(in_pack + i);
      sum += in_float4.x * in_float4.x;
      sum += in_float4.y * in_float4.y;
      sum += in_float4.z * in_float4.z;
      sum += in_float4.w * in_float4.w;
    }

    //这种写法蛮顺眼的
    for (int i = pack_off + tid; i < dim_size; i += blockDim.x) 
      sum += in[i] * in[i];

    sum = base_kernel_cu::block_reduce_sum_f32<BLOCK_DIM>(sum);

    //来个共享，目的是为了求和
    __shared__ float shared_val;

    if (threadIdx.x == 0) 

      shared_val = sum;
  
    __syncthreads();

    sum = shared_val;
    
    const float scale = rsqrtf(sum / static_cast<float>(dim_size) + eps);

    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);

    for (int i = tid; i < pack_num; i += blockDim.x) 
    {
      float4 in_float4 = *(in_pack + i);
      float4 wei_float4 = *(wei_pack + i);
      *(out_pack + i) =
          make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }

    //组外的元素
    for (int i = pack_off + tid; i < dim_size; i += blockDim.x) 
      out[i] = wei[i] * in[i] * scale;
  
  }

  template <int32_t BLOCK_DIM>
  static __global__ void row_dim_f32x4_rmsnorm_kernel(float* in, float* wei, 
    float* out, int seq_len,int dim_size, float eps) 
  {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    //所以这里限制了seq的大小
    if (bid >= seq_len) 
      return;
    

    float* block_in = in + bid * dim_size;
    float* block_out = out + bid * dim_size;
    constexpr int pack_size = 4;
    const int pack_num = dim_size / pack_size;
    const int pack_off = pack_size * pack_num;

    float sum = 0.0f;
    //float 还是可以有强制转化条件的
    float4* in_pack = reinterpret_cast<float4*>(block_in);

    for (int i = tid; i < pack_num; i += blockDim.x) 
    {
      float4 in_float4 = *(in_pack + i);
      sum += in_float4.x * in_float4.x;
      sum += in_float4.y * in_float4.y;
      sum += in_float4.z * in_float4.z;
      sum += in_float4.w * in_float4.w;
    }


  //
    for (int i = pack_off + tid; i < dim_size; i += blockDim.x) 
      sum += block_in[i] * block_in[i];


    sum = base_kernel_cu::block_reduce_sum_f32<BLOCK_DIM>(sum);


    __shared__ float shared_val;

    if (threadIdx.x == 0) 

      shared_val = sum;
    
    __syncthreads();
  
    sum = shared_val;
    const float scale = rsqrtf(sum / static_cast<float>(dim_size) + eps);

    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(block_out);

    for (int i = tid; i < pack_num; i += blockDim.x) 
    {
      float4 in_float4 = *(in_pack + i);
      float4 wei_float4 = *(wei_pack + i);
      *(out_pack + i) =
          make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                      scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }

    for (int i = pack_off + tid; i < dim_size; i += blockDim.x) 
      block_out[i] = wei[i] * block_in[i] * scale;
  
  }








  void rmsnorm_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) 
  {
    const float eps = 1e-5f;

    const int32_t tensor_dim = weight.get_dim(0);

    auto cuda_stream = static_cast<cudaStream_t>(stream);


    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    if(input.dims_size()>1)
    {
      const int32_t seq_len =  input.get_dim(0);
      //这里限制了最大的输入数目
      if(cuda_stream)
        return row_dim_f32x4_rmsnorm_kernel<128><<<512,128,0,cuda_stream>>>(in_ptr, wei_ptr, out_ptr,tensor_dim ,seq_len, eps);
      else
        return row_dim_f32x4_rmsnorm_kernel<128><<<512,128>>>(in_ptr, wei_ptr, out_ptr,tensor_dim ,seq_len, eps);
    }
    else
    {
      if (cuda_stream) 
        return row_f32x4_rmsnorm_kernel<128><<<1, 128, 0, cuda_stream>>>(in_ptr, wei_ptr, out_ptr,tensor_dim , eps);
      else 
        return row_f32x4_rmsnorm_kernel<128><<<1, 128>>>(in_ptr, wei_ptr, out_ptr, tensor_dim, eps);
    }


  }
}






