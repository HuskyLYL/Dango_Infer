#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace f32x4_kernel_cu
{
    void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      cudaStream_t  stream = nullptr);

}  // namespace f32x4_kernel_cu

#endif  // MATMUL_KERNEL_CU_CUH
