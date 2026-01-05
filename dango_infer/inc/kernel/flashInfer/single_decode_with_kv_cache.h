#ifndef SINGLE_DECODE_WITH_KV_CACHE_H_
#define SINGLE_DECODE_WITH_KV_CACHE_H_

#include <cuda_runtime.h>
#include <cstdint>

#include "tensor/tensor.h"

namespace flashinfer {

// Launch helper for single-step decode with KV cache.


// High-level entry for single token decode with KV cache.
template <typename T>
void single_decode_with_kv_cache(int32_t pos, int32_t head_num, int32_t layer_index,
                                 int32_t seq_len, int32_t kv_dim, int32_t kv_mul,
                                 int32_t head_size, const tensor::Tensor& mha_out,
                                 const tensor::Tensor& query_tensor,
                                 const tensor::Tensor& score_tensor,
                                 const tensor::Tensor& key_cache_tensor,
                                 const tensor::Tensor& value_cache_tensor,
                                 cudaStream_t stream);

}  // namespace flashinfer

#endif  // SINGLE_DECODE_WITH_KV_CACHE_H_
