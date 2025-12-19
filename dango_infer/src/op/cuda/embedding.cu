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