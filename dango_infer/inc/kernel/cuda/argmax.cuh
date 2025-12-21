#ifndef DANGO_KERNEL_CUH
#define DANGO_KERNEL_CUH
#include <cuda_bf16.h>
namespace base_kernel_cu 
{
    size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream=nullptr);
}

namespace bf16_kernel_cu
{
    size_t argmax_kernel_cu(const __nv_bfloat16* input_ptr, size_t size, void* stream=nullptr);
}
#endif  // ARGMAX_KERNEL_CUH
