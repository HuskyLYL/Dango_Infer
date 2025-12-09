#ifndef EMBEDDING_CU_H
#define EMBEDDING_CU_H
#include <cuda_runtime.h>
#include "tensor/tensor.h"
namespace f32x4_kernel_cu
{



    //void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
    //               const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);

    
    void embedding_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace f32x4_kernel_cu
#endif  // EMBEDDING_CU_H
