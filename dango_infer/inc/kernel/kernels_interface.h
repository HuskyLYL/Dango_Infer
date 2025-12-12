#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H
#include "tensor/tensor.h"
namespace f32x4_kernel_cu
{

    typedef void (*Elementwise)(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output, void* stream);

    typedef void (*Embedding)(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output, void* stream );

    typedef void (*Rmsnorm)(const tensor::Tensor& input, const tensor::Tensor& weight,
        const tensor::Tensor& output, void* stream );

    typedef void (*Matmul)(const tensor::Tensor& input, const tensor::Tensor& weight,
        const tensor::Tensor& output, float scale ,cudaStream_t  stream);



                   
    Elementwise get_elementwise_kernel();

    Embedding get_embedding_kernel();

    Rmsnorm get_rmsn_kernel();

    Matmul get_matmul_kernel();




}  // namespace f32x4_kernel_cu


namespace base_kernel_cu 
{

    typedef void(*Swiglu)(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output, void* stream);
    Swiglu get_swiglu_kernel();







    typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
        const tensor::Tensor& input_q, const tensor::Tensor& input_k,
        const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
        const tensor::Tensor& cos_cache, void* stream);

    RoPEKernel get_rope_kernel();



    typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size,
        const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
        const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor,
        const tensor::Tensor& value_cache_tensor, 
        cudaStream_t stream);

    MHAKernel get_mha_kernel();


}// namespace base_kernel_cu





#endif  // KERNELS_INTERFACE_H
