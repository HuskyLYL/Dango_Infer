#include <base/base.h>
#include "kernel/cuda/elementwise.cuh"
#include "kernel/cuda/embedding.cuh"
#include "kernel/cuda/rmsn.cuh"
#include "kernel/cuda/mha.cuh"
#include "kernel/cuda/matmul.cuh"
#include "kernel/cuda/swiglu.cuh"
#include "kernel/cuda/rope_kernel.cuh"
#include "kernel/kernels_interface.h"
#include "kernel/flashInfer/single_decode_with_kv_cache.cuh"
#include <cuda_bf16.h>

namespace f32x4_kernel_cu
{
    Elementwise get_elementwise_kernel() { return elementwise_kernel; }

    Embedding get_embedding_kernel() { return embedding_kernel; }

    Rmsnorm get_rmsn_kernel() { return rmsnorm_kernel; }

    Matmul get_matmul_kernel() { return  matmul_kernel_cu; }

} 

namespace bf16x8_kernel_cu
{
    Elementwise get_elementwise_kernel(){ return elementwise_kernel; }

    Embedding get_embedding_kernel(){ return embedding_kernel; }

    Matmul get_matmul_kernel(){ return matmul_kernel_cu; }

    MHAKernel get_mha_kernel() { return mha_kernel_cu; }

    Rmsnorm get_rmsn_kernel() { return rmsnorm_kernel; }

   


} 







namespace base_kernel_cu 
{
    Swiglu get_swiglu_kernel() { return swiglu_kernel_cu; }

    RoPEKernel get_rope_kernel() { return rope_kernel_cu; }

    MHAKernel get_mha_kernel() { return mha_kernel_cu; }
}


namespace bf16_kernel_cu 
{
    RoPEKernel get_rope_kernel() { return rope_kernel_cu; }

    Swiglu get_swiglu_kernel() { return swiglu_kernel_cu; }
}

namespace flashinfer
{

    SingleDecodeWithKVCache get_single_decode_with_kv_cache_kernel(){return  single_decode_with_kv_cache;}
}
