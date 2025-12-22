#include "kernel/cuda/rope_kernel.cuh" 
#include <cuda_bf16.h>
namespace base_kernel_cu 
{

    //两两一组，将向量进行叠加
    __device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) 
    {
        float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
        float2 vec_value = *vec_ptr;
        *vec_ptr = make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
    }

    __global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
        const float* input_q, const float* input_k,
        const float* sin_cache, const float* cos_cache) 
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        //两两一组进行计算
        idx = idx * 2;
        if (idx >= dim) 
            return;
        
        int head_dim = idx % head_size;
        float fci = *(sin_cache + pos * head_size + head_dim);
        float fcr = *(cos_cache + pos * head_size + head_dim);

        rope_calc(fcr, fci, const_cast<float*>(input_q), idx);

        //这里实际上对应了KVCache的分组，但是 70B之下的KVCache是没有分组的
        if (idx >= kv_dim) 
            return;
  
        rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
    }


    //这里应该就是最长的上下文的限制了
    __global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) 
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int head_dim = idx % head_size;
        for (int pos = 0; pos < max_seq_len; ++pos) 
        {
            float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
            float val = static_cast<float>(pos) * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            *(sin_cache + pos * head_size + head_dim) = fci;
            *(cos_cache + pos * head_size + head_dim) = fcr;
        }
    }


    //一个head有多长 然后最大的seq_len是多大
    void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache, cudaStream_t stream) 
    {

        //一个head头有多大
        int threads = head_size;
        if (stream) 
        {
            sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                const_cast<float*>(sin_cache.ptr<float>()),
                const_cast<float*>(cos_cache.ptr<float>()));
        } 
        else 
        {
            sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
        }

        
    }

    void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
        const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
        const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
        cudaStream_t  stream) 
    {
        //CPUID,调用核之前
        const int32_t pos = *input_pos.ptr<int32_t>(0);
        int threads = 128;
        int blocks = (dim + threads - 1) / threads;
        if (stream) 
        {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
            pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
            sin_cache.ptr<float>(), cos_cache.ptr<float>());
        } 
        else 
            rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                             input_k.ptr<float>(), sin_cache.ptr<float>(),
                                             cos_cache.ptr<float>());
        
    }
}  // namespace kernel

namespace bf16_kernel_cu
{
    __device__ void rope_calc_bf16(float fcr, float fci, __nv_bfloat16* vec, int32_t idx)
    {
        float2 v;
        v.x = __bfloat162float(vec[idx]);
        v.y = __bfloat162float(vec[idx + 1]);
        float2 r = make_float2(v.x * fcr - v.y * fci, v.x * fci + v.y * fcr);
        vec[idx] = __float2bfloat16(r.x);
        vec[idx + 1] = __float2bfloat16(r.y);
    }

    __global__ void rope_kernel_cu_bf16(int pos, int dim, int kv_dim, int head_size,
        __nv_bfloat16* input_q, __nv_bfloat16* input_k,
        const __nv_bfloat16* sin_cache, const __nv_bfloat16* cos_cache)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx = idx * 2;
        if (idx >= dim)
            return;

        int head_dim = idx % head_size;
        float fci = __bfloat162float(*(sin_cache + pos * head_size + head_dim));
        float fcr = __bfloat162float(*(cos_cache + pos * head_size + head_dim));

        rope_calc_bf16(fcr, fci, input_q, idx);

        if (idx >= kv_dim)
            return;

        rope_calc_bf16(fcr, fci, input_k, idx);
    }

    __global__ void sin_cos_calc_bf16(int head_size, int max_seq_len, __nv_bfloat16* sin_cache, __nv_bfloat16* cos_cache)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int head_dim = idx % head_size;
        for (int pos = 0; pos < max_seq_len; ++pos)
        {
            float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
            float val = static_cast<float>(pos) * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            *(sin_cache + pos * head_size + head_dim) = __float2bfloat16(fci);
            *(cos_cache + pos * head_size + head_dim) = __float2bfloat16(fcr);
        }
    }

    void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache, cudaStream_t stream)
    {
        int threads = head_size;
        __nv_bfloat16* sin_ptr = const_cast<__nv_bfloat16*>(sin_cache.ptr<__nv_bfloat16>());
        __nv_bfloat16* cos_ptr = const_cast<__nv_bfloat16*>(cos_cache.ptr<__nv_bfloat16>());
        if (stream)
            sin_cos_calc_bf16<<<1, threads, 0, stream>>>(head_size, max_seq_len, sin_ptr, cos_ptr);
        else
            sin_cos_calc_bf16<<<1, threads>>>(head_size, max_seq_len, sin_ptr, cos_ptr);
    }

    void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
        const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
        const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
        cudaStream_t  stream)
    {
        const int32_t pos = *input_pos.ptr<int32_t>(0);
        int threads = 128;
        int blocks = (dim + threads - 1) / threads;
        __nv_bfloat16* q_ptr = const_cast<__nv_bfloat16*>(input_q.ptr<__nv_bfloat16>());
        __nv_bfloat16* k_ptr = const_cast<__nv_bfloat16*>(input_k.ptr<__nv_bfloat16>());
        const __nv_bfloat16* sin_ptr = sin_cache.ptr<__nv_bfloat16>();
        const __nv_bfloat16* cos_ptr = cos_cache.ptr<__nv_bfloat16>();

        if (stream)
        {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            rope_kernel_cu_bf16<<<blocks, threads, 0, stream_>>>(
                pos, dim, kv_dim, head_size, q_ptr, k_ptr, sin_ptr, cos_ptr);
        }
        else
        {
            rope_kernel_cu_bf16<<<blocks, threads>>>(
                pos, dim, kv_dim, head_size, q_ptr, k_ptr, sin_ptr, cos_ptr);
        }
        //auto err = cudaGetLastError();
        //LOG(INFO)<<cudaGetErrorString(err)<<"\n";
    }
} // namespace bf16_kernel_cu
