#ifndef MHA_KERNEL_H
#define MHA_KERNEL_H
#include "tensor/tensor.h"
#include <cuda_bf16.h>
namespace base_kernel_cu 
{
    void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
        cudaStream_t stream = nullptr);
}

namespace bf16x8_kernel_cu
{
    void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
        cudaStream_t stream = nullptr);
}
#endif  // MHA_KERNEL_H
