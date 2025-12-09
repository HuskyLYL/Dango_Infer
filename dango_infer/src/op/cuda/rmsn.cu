#include "kernel/cuda/rmsn.cuh"
namespace f32x4_kernel_cu 
{

  #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

  //warpReduce
  template <const int kWarpSize = 32>
  __device__ __forceinline__ float warp_reduce_sum_f32(float val) 
  {
      #pragma unroll
      for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) 
          val += __shfl_xor_sync(0xffffffff, val, mask);  
      return val;
  }

  //blockReduce
  template <const int NUM_THREADS = 256>
  __device__ __forceinline__ float block_reduce_sum_f32(float val) 
  {
      constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
      int warp = threadIdx.x / WARP_SIZE;
      int lane = threadIdx.x % WARP_SIZE;
      static __shared__ float shared[NUM_WARPS];
      val = warp_reduce_sum_f32<32>(val);
      if (lane == 0)
          shared[warp] = val;
      __syncthreads();
      val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
      val = warp_reduce_sum_f32<NUM_WARPS>(val);
      return val;
  }


  //此处套用a模板类传递block的大小
  template <const int NUM_THREADS = 256 / 4>
  __global__ void rms_norm_f32x4_kernel(float *in, float *wei, float* out, 
      int dim_size,int size,float eps) 
  {
    // [seqlen,tensor_dim]

    int tid = threadIdx.x; 
    //seq_id
    int bid = blockIdx.x; 
      
      //tensor_element_id
    int idx = (bid * blockDim.x + threadIdx.x) * 4;

    //同一个block之中的缩放系数
    __shared__ float s_variance;

    float4 reg_in = FLOAT4(in[idx]);

    float4 reg_wei = FLOAT4(wei[idx]);

    float variance = 0.0f;

    if(idx+3<dim_size * size)
        variance = reg_in.x * reg_in.x + reg_in.y * reg_in.y +reg_in.z * reg_in.z + reg_in.w * reg_in.w;
    else if(idx+2<dim_size * size)
        variance = reg_in.x * reg_in.x + reg_in.y * reg_in.y +reg_in.z * reg_in.z ;
    else if(idx+1<dim_size * size)
        variance = reg_in.x * reg_in.x + reg_in.y * reg_in.y ;
    else if(idx < dim_size * size)
        variance = reg_in.x * reg_in.x;

    //块级归约
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);


//#######################################NOW###################################################

    if (tid == 0)
      
      s_variance = rsqrtf(variance / static_cast<float>(dim_size) +eps);
  
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * reg_wei.x;
    reg_y.y = reg_x.y * s_variance * reg_wei.y;
    reg_y.z = reg_x.z * s_variance * reg_wei.z;
    reg_y.w = reg_x.w * s_variance * reg_wei.w;
    if (idx+3 < size * dim_size)
      FLOAT4(out[idx]) = reg_y;
    else if(idx+2 <  size * dim_size)
    {
      out[idx] = reg_y.x;
      out[idx+1] = reg_y.y;
      out[idx+2] = reg_y.z;
    }
    else if(idx+1 <  size * dim_size)
    {
      out[idx] = reg_y.x;
      out[idx+1] = reg_y.y;
    }
    else if(idx <  size * dim_size)
      out[idx] = reg_y.x;


  }

  void rmsnorm_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, cudaStream_t stream) 
  {
    const float eps = 1e-5f;

    const int32_t tensor_dim = weight.get_dim(0);

    int32_t block_size = 1;

    int32_t thread_size = (tensor_dim+3)/4;

    if(input.dims_size()>1)
      block_size = input.get_dim(0);


    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    if (stream) 
      return rms_norm_f32x4_kernel<thread_size><<<block_size, thread_size, 0, stream>>>(in_ptr, wei_ptr, out_ptr,tensor_dim ,block_size, eps);
    else 
      return rms_norm_f32x4_kernel<thread_size><<<block_size, thread_size>>>(in_ptr, wei_ptr, out_ptr, tensor_dim,block_size, eps);
  }
}









