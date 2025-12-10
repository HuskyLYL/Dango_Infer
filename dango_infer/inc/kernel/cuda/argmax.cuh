#ifndef DANGO_KERNEL_CUH
#define DANGO_KERNEL_CUH
namespace base_kernel_cu 
{
    size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream=nullptr);
}
#endif  // ARGMAX_KERNEL_CUH
