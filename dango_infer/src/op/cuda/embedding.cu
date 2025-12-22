#include "kernel/cuda/embedding.cuh"
namespace f32x4_kernel_cu
{
    #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


    __global__ void embedding_f32x4_kernel(const int32_t *input_ptr, const float *weight_ptr,
                                       float *output_ptr, int emb_size) 
    {
        int tx = threadIdx.x * 4;
        int bx = blockIdx.x;
        int offset = input_ptr[bx] * emb_size;
        if(tx+3 <emb_size)
        {
            output_ptr[bx * emb_size + tx] = weight_ptr[offset + tx];
            output_ptr[bx * emb_size + tx + 1] = weight_ptr[offset + tx + 1];
            output_ptr[bx * emb_size + tx + 2] = weight_ptr[offset + tx + 2];
            output_ptr[bx * emb_size + tx + 3] = weight_ptr[offset + tx + 3];
        }
        else if(tx+2 <emb_size)
        {
            output_ptr[bx * emb_size + tx] = weight_ptr[offset + tx];
            output_ptr[bx * emb_size + tx + 1] = weight_ptr[offset + tx + 1];
            output_ptr[bx * emb_size + tx + 2] = weight_ptr[offset + tx + 2];
        }
        else if(tx+1 <emb_size)
        {
            output_ptr[bx * emb_size + tx] = weight_ptr[offset + tx];
            output_ptr[bx * emb_size + tx + 1] = weight_ptr[offset + tx + 1];
        }
        else if(tx < emb_size)
            output_ptr[bx * emb_size + tx] = weight_ptr[offset + tx];



    }


    void embedding_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, void* stream) 
    {
        const int32_t input_num = static_cast<int32_t>(input.size());
   
        const int32_t weight_dim = weight.get_dim(1);

 
 
        const int32_t* in_ptr = input.ptr<int32_t>();
    
        float* wei_ptr = const_cast<float*>(weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());
        
        if (stream) 
            embedding_f32x4_kernel<<<input_num, (weight_dim+3)/4, 0, static_cast<cudaStream_t>(stream)>>>(in_ptr,wei_ptr,out_ptr,weight_dim);
        else
            embedding_f32x4_kernel<<<input_num, (weight_dim+3)/4>>>(in_ptr,wei_ptr,out_ptr,weight_dim);
    }


}

namespace bf16x8_kernel_cu
{
    __global__ void embedding_bf16x8_kernel(const int32_t* input_ptr,
                                            const __nv_bfloat16* weight_ptr,
                                            __nv_bfloat16* output_ptr,
                                            int emb_size)
    {
        int tx = threadIdx.x * 8;
        int bx = blockIdx.x;
        int offset = input_ptr[bx] * emb_size + tx;
        int out_base = bx * emb_size + tx;

        if (tx + 7 < emb_size)
        {
            // vectorized path: 4 x bf162 = 8 elements
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight_ptr + offset);
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(output_ptr + out_base);
            o2[0] = w2[0];
            o2[1] = w2[1];
            o2[2] = w2[2];
            o2[3] = w2[3];
        }
        else if (tx < emb_size)
        {
            #pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                int idx = tx + k;
                if (idx < emb_size)
                    output_ptr[out_base + k] = weight_ptr[offset + k];
            }
        }
    }

    void embedding_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, void* stream)
    {
        const int32_t input_num = static_cast<int32_t>(input.size());
        const int32_t weight_dim = weight.get_dim(1);

        const int32_t* in_ptr = input.ptr<int32_t>();
        const __nv_bfloat16* wei_ptr = weight.ptr<__nv_bfloat16>();
        __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());

        int32_t threads = (weight_dim + 7) / 8;
        if (stream)
            embedding_bf16x8_kernel<<<input_num, threads, 0, static_cast<cudaStream_t>(stream)>>>(in_ptr, wei_ptr, out_ptr, weight_dim);
        else
            embedding_bf16x8_kernel<<<input_num, threads>>>(in_ptr, wei_ptr, out_ptr, weight_dim);

        auto err = cudaGetLastError();
        LOG(INFO)<<cudaGetErrorString(err)<<"\n";
    }
}
