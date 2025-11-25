#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"
namespace base 
{

    //这里不涉及内存的分配可以弄得简单一些
    CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(CPUID) {}

    void* CPUDeviceAllocator::allocate(size_t byte_size) const 
    {
        if (!byte_size)

            return nullptr;
  
        void* data = malloc(byte_size);
    
        return data;

    }

    void CPUDeviceAllocator::release(void* ptr) const 
    {
        if (ptr) 
            free(ptr);
    }



}

