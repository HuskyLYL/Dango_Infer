#ifndef DANGO_KERNEL_CU_CUH
#define DANGO_KERNEL_CU_CUH
#include <tensor/tensor.h>
#include <cuda_bf16.h>
namespace base_kernel_cu 
{
    void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output, void* stream=nullptr);
}

namespace bf16_kernel_cu
{
    void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output, void* stream=nullptr);
}
#endif  // DANGO_KERNEL_CU_CUH
