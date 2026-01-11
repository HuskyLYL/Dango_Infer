#include "kernel/cuda/parallel_embedding.cuh"
namespace f32x4_kernel_cu
{
    #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


    __global__ void parallel_embedding_f32x4_kernel(const int32_t *input_ptr,
                                       const float *weight_ptr,
                                       float *output_ptr,
                                       int emb_size,
                                       int vocab_start,
                                       int vocab_size) 
    {
        int tx = threadIdx.x * 4;
        int bx = blockIdx.x;
        int token = input_ptr[bx];
        bool in_range = (token >= vocab_start) && (token < vocab_start + vocab_size);
        int local_offset = (token - vocab_start) * emb_size;
        int out_base = bx * emb_size + tx;
        if(tx+3 <emb_size)
        {
            output_ptr[out_base]     = in_range ? weight_ptr[local_offset + tx]     : 0.f;
            output_ptr[out_base + 1] = in_range ? weight_ptr[local_offset + tx + 1] : 0.f;
            output_ptr[out_base + 2] = in_range ? weight_ptr[local_offset + tx + 2] : 0.f;
            output_ptr[out_base + 3] = in_range ? weight_ptr[local_offset + tx + 3] : 0.f;
        }
        else if(tx+2 <emb_size)
        {
            output_ptr[out_base]     = in_range ? weight_ptr[local_offset + tx]     : 0.f;
            output_ptr[out_base + 1] = in_range ? weight_ptr[local_offset + tx + 1] : 0.f;
            output_ptr[out_base + 2] = in_range ? weight_ptr[local_offset + tx + 2] : 0.f;
        }
        else if(tx+1 <emb_size)
        {
            output_ptr[out_base]     = in_range ? weight_ptr[local_offset + tx]     : 0.f;
            output_ptr[out_base + 1] = in_range ? weight_ptr[local_offset + tx + 1] : 0.f;
        }
        else if(tx < emb_size)
            output_ptr[out_base] = in_range ? weight_ptr[local_offset + tx] : 0.f;



    }


    void parallel_embedding_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, void* stream) 
    {
        const int32_t input_num = static_cast<int32_t>(input.size());
   
        const int32_t weight_rows = weight.get_dim(0);
        const int32_t weight_dim = weight.get_dim(1);

 
 
        const int32_t* in_ptr = input.ptr<int32_t>();
    
        float* wei_ptr = const_cast<float*>(weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());
        
        const int vocab_start = nccl::G_MPI_RANK * weight_rows;
        const int vocab_size = weight_rows;

        if (stream) 
            parallel_embedding_f32x4_kernel<<<input_num, (weight_dim+3)/4, 0, static_cast<cudaStream_t>(stream)>>>(
                in_ptr, wei_ptr, out_ptr, weight_dim, vocab_start, vocab_size);
        else
            parallel_embedding_f32x4_kernel<<<input_num, (weight_dim+3)/4>>>(
                in_ptr, wei_ptr, out_ptr, weight_dim, vocab_start, vocab_size);
    }


}

namespace bf16x8_kernel_cu
{
    __global__ void parallel_embedding_bf16x8_kernel(const int32_t* input_ptr,
                                            const __nv_bfloat16* weight_ptr,
                                            __nv_bfloat16* output_ptr,
                                            int emb_size,
                                            int vocab_start,
                                            int vocab_size)
    {
        int tx = threadIdx.x * 8;
        int bx = blockIdx.x;
        int token = input_ptr[bx];
        bool in_range = (token >= vocab_start) && (token < vocab_start + vocab_size);
        int offset = (token - vocab_start) * emb_size + tx;
        int out_base = bx * emb_size + tx;

        if (tx + 7 < emb_size)
        {
            // vectorized path: 4 x bf162 = 8 elements
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base);
            if (in_range)
            {
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight_ptr + offset);
                o2[0] = w2[0];
                o2[1] = w2[1];
                o2[2] = w2[2];
                o2[3] = w2[3];
            }
            else
            {
                const __nv_bfloat16 zero = __nv_bfloat16(0.0f);
                o2[0] = make_bfloat162(zero, zero);
                o2[1] = make_bfloat162(zero, zero);
                o2[2] = make_bfloat162(zero, zero);
                o2[3] = make_bfloat162(zero, zero);
            }
        }
        else if (tx < emb_size)
        {
            #pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                int idx = tx + k;
                if (idx < emb_size)
                    output_ptr[out_base + k] = in_range ? weight_ptr[offset + k] : __nv_bfloat16(0.0f);
            }
        }
    }

    void parallel_embedding_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, void* stream)
    {
        const int32_t input_num = static_cast<int32_t>(input.size());
        const int32_t weight_rows = weight.get_dim(0);
        const int32_t weight_dim = weight.get_dim(1);

        const int32_t* in_ptr = input.ptr<int32_t>();
        const __nv_bfloat16* wei_ptr = weight.ptr<__nv_bfloat16>();
        __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());

        const int vocab_start = nccl::G_MPI_RANK * weight_rows;
        const int vocab_size = weight_rows;

        int32_t threads = (weight_dim + 7) / 8;
        if (stream)
            parallel_embedding_bf16x8_kernel<<<input_num, threads, 0, static_cast<cudaStream_t>(stream)>>>(
                in_ptr, wei_ptr, out_ptr, weight_dim, vocab_start, vocab_size);
        else
            parallel_embedding_bf16x8_kernel<<<input_num, threads>>>(
                in_ptr, wei_ptr, out_ptr, weight_dim, vocab_start, vocab_size);

        //auto err = cudaGetLastError();
        //LOG(INFO)<<cudaGetErrorString(err)<<"\n";
    }
}
