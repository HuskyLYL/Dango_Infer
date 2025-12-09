#ifndef ELEMENTWISE_CU_H
#define ELEMENTWISE_CU_H
#include <cuda_runtime.h>
#include "tensor/tensor.h"
namespace f32x4_kernel_cu
{
    void elementwise_kernel(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // ADD_CU_H
