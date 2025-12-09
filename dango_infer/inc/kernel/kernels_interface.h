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

                   
    Elementwise get_elementwise_kernel();

    Embedding get_embedding_kernel();

    Rmsnorm get_rmsnorm_kernel();

}  // namespace kernel
#endif  // KERNELS_INTERFACE_H
