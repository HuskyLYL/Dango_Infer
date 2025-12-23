#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "kernel/cuda/mha.cuh"
#include <base/tick.h>
#include <cuda_bf16.h>
namespace base_kernel_cu 
{
    constexpr static int thread_num = 256;


    //后面在flashAttention 这里 softmax肯定要大修
    __device__ void softmax_gpu(float* __restrict__ x, int size) 
    {
  
        int tid = threadIdx.x;
        
        int step = blockDim.x;


        float max_val = tid < size ? x[tid] : -FLT_MAX;
        for (int i = tid + step; i < size; i += step) 
            if (x[i] > max_val) 
                max_val = x[i];
    
  
        using BlockReduce = cub::BlockReduce<float, thread_num>;
        __shared__ BlockReduce::TempStorage temp;
        __shared__ float shared_val;
        max_val = BlockReduce(temp).Reduce(max_val, cub::Max());


        if (threadIdx.x == 0) 
            shared_val = max_val;
        __syncthreads();

        max_val = shared_val;
        float sum = 0.0f;
        for (int i = tid; i < size; i += step) 
        {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }
        sum = BlockReduce(temp).Sum(sum);
        if (threadIdx.x == 0) 
            shared_val = sum;
  
        __syncthreads();
        sum = shared_val;

        for (int i = tid; i < size; i += step) 
            x[i] /= sum;
  
    }


    __global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
        float* score_ptr, float* output, float* key_cache,
        float* value_cache, int32_t kv_dim, int32_t kv_mul,
        int32_t head_num, int32_t head_size,
        int32_t layer_offset) 
    {
    
        int head = blockIdx.x;
        if (head >= head_num) 
            return;
  
        //这里的block的数目是按照query来处理的
        //找到对应的head头
        extern __shared__ float s_query_head[];
        float scale = 1.f / sqrtf(float(head_size));
        float* query_head = query + head * head_size;

        // 预加载query到共享内存
        for (int i = threadIdx.x; i < head_size; i += blockDim.x) 
            s_query_head[i] = query_head[i];
  
        __syncthreads();

        //前面一个head是输入的头数目
        float* score_head = score_ptr + head * seq_len;
        // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
        // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
        // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
        // 这里不是给score 用的,这里是给k 和 v 用的
        int head_offset = (head / kv_mul) * head_size;
        // 计算自注意力分数
        //一个head 对应 pos 个seq_len
        for (int t = threadIdx.x; t <= pos; t += blockDim.x) 
        {
            float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
            float score = 0.0f;
            //emmm 这里是 key 和query 的矩阵乘
            for (int i = 0; i < head_size; i += 4) 
            {
                float4 key_val = *reinterpret_cast<float4*>(key_head + i);
                float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
                //因为是一个线程处理，所以不存在线程安全的问题
                score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
                        key_val.w * query_val.w;
            }
            score *= scale;
            score_head[t] = score;
        }
        __syncthreads();
        softmax_gpu(score_head, pos + 1);
        __syncthreads();

        float* output_head = output + head * head_size;
        // 使用自注意力分数对value矩阵加权
        //这里实际上有一些低效了吧
        for (int i = threadIdx.x; i < head_size; i += blockDim.x) 
        {
            float value = 0.0f;
            for (int t = 0; t <= pos; t++) 
            {
            float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
            //不用担心dim的问题，实际上这里只有一个维度
            float score = score_head[t];
            value += score * value_head[i];
            }
            output_head[i] = value;
        }
    }

    void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
        cudaStream_t stream) 
    {
        //thread_num 是一个固定的数目
        //[head,N.D]
        int32_t layer_offset = layer_index * seq_len * kv_dim;
        float* query = const_cast<float*>(query_tensor.ptr<float>());
        float* score = const_cast<float*>(score_tensor.ptr<float>());
        float* output = const_cast<float*>(mha_out.ptr<float>());

        float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
        float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

        if(stream)
            multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
            pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
            head_size, layer_offset);
        else
            multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float)>>>(
            pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
            head_size, layer_offset);
                
    }

}  // namespace base_kernel_cu

namespace bf16x8_kernel_cu
{
    constexpr static int thread_num = 256;

    __device__ void softmax_gpu_float(float* __restrict__ x, int size) 
    {
        int tid  = threadIdx.x;
        int step = blockDim.x;

        float max_val = tid < size ? x[tid] : -FLT_MAX;
        for (int i = tid + step; i < size; i += step) 
        {
            float v = x[i];
            if (v > max_val)
                max_val = v;
        }

        using BlockReduce = cub::BlockReduce<float, thread_num>;
        __shared__ BlockReduce::TempStorage temp;
        __shared__ float shared_val;
        max_val = BlockReduce(temp).Reduce(max_val, cub::Max());

        if (threadIdx.x == 0)
            shared_val = max_val;
        __syncthreads();

        max_val = shared_val;
        float sum = 0.0f;
        for (int i = tid; i < size; i += step) 
        {
            float v = expf(x[i] - max_val);
            x[i] = v;
            sum += v;
        }

        sum = BlockReduce(temp).Sum(sum);
        if (threadIdx.x == 0)
            shared_val = sum;
        __syncthreads();

        sum = shared_val;
        for (int i = tid; i < size; i += step) 
            x[i] = x[i] / sum;
    }

    __global__ void multi_head_attention_kernel_bf16(int32_t pos, int32_t seq_len,
        const __nv_bfloat16* query, __nv_bfloat16* score_ptr, __nv_bfloat16* output,
        const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache, int32_t kv_dim,
        int32_t kv_mul, int32_t head_num, int32_t head_size, int32_t layer_offset) 
    {
        int head = blockIdx.x;
        if (head >= head_num)
            return;

        extern __shared__ float shared[];
        float* s_query_head = shared;
        float* s_score      = shared + head_size;
        float scale = 1.f / sqrtf(float(head_size));
        const __nv_bfloat16* query_head = query + head * head_size;

        for (int i = threadIdx.x; i < head_size; i += blockDim.x)
            s_query_head[i] = __bfloat162float(query_head[i]);

        __syncthreads();

        __nv_bfloat16* score_head = score_ptr + head * seq_len;
        int head_offset = (head / kv_mul) * head_size;

        for (int t = threadIdx.x; t <= pos; t += blockDim.x) 
        {
            const __nv_bfloat16* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
            float score = 0.0f;

            int i = 0;
            for (; i + 3 < head_size; i += 4) 
            {
                float q0 = s_query_head[i];
                float q1 = s_query_head[i + 1];
                float q2 = s_query_head[i + 2];
                float q3 = s_query_head[i + 3];

                score += q0 * __bfloat162float(key_head[i]);
                score += q1 * __bfloat162float(key_head[i + 1]);
                score += q2 * __bfloat162float(key_head[i + 2]);
                score += q3 * __bfloat162float(key_head[i + 3]);
            }
            for (; i < head_size; ++i)
                score += s_query_head[i] * __bfloat162float(key_head[i]);

            score *= scale;
            s_score[t] = score;
        }

        __syncthreads();
        softmax_gpu_float(s_score, pos + 1);
        __syncthreads();

        for (int t = threadIdx.x; t <= pos; t += blockDim.x)
            score_head[t] = __float2bfloat16(s_score[t]);

        __nv_bfloat16* output_head = output + head * head_size;
        for (int i = threadIdx.x; i < head_size; i += blockDim.x) 
        {
            float value = 0.0f;
            for (int t = 0; t <= pos; t++) 
            {
                const __nv_bfloat16* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
                float score = s_score[t];
                value += score * __bfloat162float(value_head[i]);
            }
            output_head[i] = __float2bfloat16(value);
        }
    }

    void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
        cudaStream_t stream) 
    {
        int32_t layer_offset = layer_index * seq_len * kv_dim;
        const __nv_bfloat16* query = query_tensor.ptr<__nv_bfloat16>();
        __nv_bfloat16* score = const_cast<__nv_bfloat16*>(score_tensor.ptr<__nv_bfloat16>());
        __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());

        const __nv_bfloat16* key_cache = key_cache_tensor.ptr<__nv_bfloat16>();
        const __nv_bfloat16* value_cache = value_cache_tensor.ptr<__nv_bfloat16>();

        size_t shared_mem = (head_size + pos + 1) * sizeof(float);
        if (stream)
            multi_head_attention_kernel_bf16<<<head_num, thread_num, shared_mem, stream>>>(
                pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
                head_size, layer_offset);
        else
            multi_head_attention_kernel_bf16<<<head_num, thread_num, shared_mem>>>(
                pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
                head_size, layer_offset);
        //auto err = cudaGetLastError();
        //LOG(INFO)<<cudaGetErrorString(err)<<"\n";
    }
}  // namespace bf16x8_kernel_cu
