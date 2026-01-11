#ifndef PARALLEL_EMBEDDING_CU_H
#define PARALLEL_EMBEDDING_CU_H
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tensor/tensor.h"
#include "nccl/base.h"
namespace f32x4_kernel_cu
{

    void parallel_embedding_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace f32x4_kernel_cu
namespace bf16x8_kernel_cu
{

    void parallel_embedding_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace bf16x8_kernel_cu



#endif  // EMBEDDING_CU_H
