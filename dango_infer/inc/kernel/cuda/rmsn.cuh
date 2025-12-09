#ifndef RMSN_KERNEL_CU_CUH
#define RMSN_KERNEL_CU_CUH
#include <tensor/tensor.h>
#include <cuda_runtime.h>
namespace f32x4_kernel_cu
{
    void rmsnorm_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream = nullptr);
} 
#endif  