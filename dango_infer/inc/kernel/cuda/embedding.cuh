#ifndef EMBEDDING_CU_H
#define EMBEDDING_CU_H
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tensor/tensor.h"
namespace f32x4_kernel_cu
{

    void embedding_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace f32x4_kernel_cu
namespace bf16x8_kernel_cu
{

    void embedding_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace bf16x8_kernel_cu



#endif  // EMBEDDING_CU_H
