#include <tensor/tensor.h>

#include "kernel/kernels_interface.h"
#include "kernel/cuda/reduce_sum.cuh"
#include "kernel/cuda/matmul.cuh"
#include <cuda_bf16.h>

namespace f32x4_kernel_cu
{
    //线性liner kernel
    template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
    __global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output,
                                        int input_dim, int output_dim) 
    {
        __shared__ float sdata[THREAD_PER_BLOCK];
        unsigned int tid = threadIdx.x;

        int start_row = blockIdx.x * ROW_PER_BLOCK;
        int end_row = start_row + ROW_PER_BLOCK;

        
        if (start_row >= output_dim) 
            return;

        constexpr int pack_size = 4;
        const int pack_num = input_dim / pack_size;
        const int pack_off = pack_size * pack_num;

        #pragma unroll
        for (int p = start_row; p < end_row; ++p) 
        {
            sdata[tid] = 0;
            int row_offset = p * input_dim;
            float4* input_float4_ptr = (float4*)input;
            float4* weight_float4_ptr = (float4*)(weight + row_offset);

            #pragma unroll
            for (int i = tid; i < pack_num; i += blockDim.x) 
            {
                float4 input_float4 = *(input_float4_ptr + i);
                float4 weight_float4 = *(weight_float4_ptr + i);
                float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                            input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
                sdata[tid] += part_sum;
            }

            for (int i = pack_off + tid; i < input_dim; i += blockDim.x) 
                sdata[tid] += input[i] * weight[row_offset + i];
            

            __syncthreads();

            float part_sum = base_kernel_cu::block_reduce_sum_f32<THREAD_PER_BLOCK>(sdata[tid]);


            if (tid == 0) 
                output[p] = part_sum;

        }
    }

   

    void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, const float scale, cudaStream_t stream) 
    {
        const int32_t output_dim = weight.get_dim(0);
        const int32_t input_dim = weight.get_dim(1);
        (void)scale;  // 暂未使用的缩放参数
        if (stream) 
            matmul_kernel_cu_fp32<128, 1><<<output_dim, 128, 0, stream>>>(
                input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), input_dim, output_dim);
        else 
            matmul_kernel_cu_fp32<128, 1><<<output_dim, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                                    const_cast<float*>(output.ptr<float>()), input_dim, output_dim);
    }

   
}  // namespace f32x4_kernel_cu

namespace bf16x8_kernel_cu
{
    template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
    __global__ void matmul_kernel_cu_bf16(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                          __nv_bfloat16* output, int input_dim, int output_dim)
    {
        __shared__ float sdata[THREAD_PER_BLOCK];
        unsigned int tid = threadIdx.x;

        int start_row = blockIdx.x * ROW_PER_BLOCK;
        int end_row   = start_row + ROW_PER_BLOCK;

        if (start_row >= output_dim)
            return;

        constexpr int pack_size = 8;
        const int pack_num = input_dim / pack_size;
        const int pack_off = pack_size * pack_num;

        #pragma unroll
        for (int p = start_row; p < end_row && p < output_dim; ++p)
        {
            sdata[tid] = 0.f;
            int row_offset = p * input_dim;

            // vectorized chunk: 8 bf16 values per iteration
            for (int i = tid * pack_size; i < pack_off; i += blockDim.x * pack_size)
            {
                #pragma unroll
                for (int k = 0; k < pack_size; ++k)
                {
                    int idx = i + k;
                    float a = __bfloat162float(input[idx]);
                    float b = __bfloat162float(weight[row_offset + idx]);
                    sdata[tid] += a * b;
                }
            }

            // tail
            for (int i = pack_off + tid; i < input_dim; i += blockDim.x)
                sdata[tid] += __bfloat162float(input[i]) * __bfloat162float(weight[row_offset + i]);

            __syncthreads();

            //这里复用前面的float
            float part_sum = base_kernel_cu::block_reduce_sum_f32<THREAD_PER_BLOCK>(sdata[tid]);

            if (tid == 0)
                output[p] = __float2bfloat16(part_sum);

            __syncthreads();
        }
    }



    template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
    __global__ void matmul_kernel_cu_bf16(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                          float* output, int input_dim, int output_dim)
    {
        __shared__ float sdata[THREAD_PER_BLOCK];
        unsigned int tid = threadIdx.x;

        int start_row = blockIdx.x * ROW_PER_BLOCK;
        int end_row   = start_row + ROW_PER_BLOCK;

        if (start_row >= output_dim)
            return;

        constexpr int pack_size = 8;
        const int pack_num = input_dim / pack_size;
        const int pack_off = pack_size * pack_num;

        #pragma unroll
        for (int p = start_row; p < end_row && p < output_dim; ++p)
        {
            sdata[tid] = 0.f;
            int row_offset = p * input_dim;

            // vectorized chunk: 8 bf16 values per iteration
            for (int i = tid * pack_size; i < pack_off; i += blockDim.x * pack_size)
            {
                #pragma unroll
                for (int k = 0; k < pack_size; ++k)
                {
                    int idx = i + k;
                    float a = __bfloat162float(input[idx]);
                    float b = __bfloat162float(weight[row_offset + idx]);
                    sdata[tid] += a * b;
                }
            }

            // tail
            for (int i = pack_off + tid; i < input_dim; i += blockDim.x)
                sdata[tid] += __bfloat162float(input[i]) * __bfloat162float(weight[row_offset + i]);

            __syncthreads();

            //这里复用前面的float
            float part_sum = base_kernel_cu::block_reduce_sum_f32<THREAD_PER_BLOCK>(sdata[tid]);

            if (tid == 0)
                output[p] = part_sum;

            __syncthreads();
        }
    }

    void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, const float scale, cudaStream_t stream)
    {
        const int32_t output_dim = weight.get_dim(0);
        const int32_t input_dim  = weight.get_dim(1);
        (void)scale;

        const __nv_bfloat16* input_ptr  = input.ptr<__nv_bfloat16>();
        const __nv_bfloat16* weight_ptr = weight.ptr<__nv_bfloat16>();
        if(output.data_type()==base::DataType::kDataTypeBf16)
        {
            __nv_bfloat16* output_ptr       = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
            if (stream)
                matmul_kernel_cu_bf16<128, 1><<<output_dim, 128, 0, stream>>>(input_ptr, weight_ptr, output_ptr, input_dim, output_dim);
            else
                matmul_kernel_cu_bf16<128, 1><<<output_dim, 128>>>(input_ptr, weight_ptr, output_ptr, input_dim, output_dim);
        }
        else if(output.data_type()==base::DataType::kDataTypeFp32)
        {
            float* output_ptr       = const_cast<float*>(output.ptr<float>());
            if (stream)
                matmul_kernel_cu_bf16<128, 1><<<output_dim, 128, 0, stream>>>(input_ptr, weight_ptr, output_ptr, input_dim, output_dim);
            else
                matmul_kernel_cu_bf16<128, 1><<<output_dim, 128>>>(input_ptr, weight_ptr, output_ptr, input_dim, output_dim);
        }



        //auto err = cudaGetLastError();
        //LOG(INFO)<<cudaGetErrorString(err)<<"\n";
    }
}  // namespace bf16x8_kernel_cu
