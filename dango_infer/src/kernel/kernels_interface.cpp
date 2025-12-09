#include <base/base.h>
#include "kernel/cuda/elementwise.cuh"
#include "kernel/cuda/embedding.cuh"
#include "kernel/cuda/rmsn.cuh"
#include "kernel/kernels_interface.h"
namespace f32x4_kernel_cu
{
    Elementwise get_elementwise_kernel()
    {
        return elementwise_kernel;
    }

    Embedding get_embedding_kernel()
    {
        return embedding_kernel;
    }

    Rmsnorm get_rmsn_kernel()
    {
        return rmsnorm_kernel;
    }





} 




