#include "kernel/kernels_interface.h"
#include "kernel/cuda/argmax.cuh"
#include "kernel/cuda/argmax_reduce.cuh"
#include "tensor/tensor.h"
#include <cuda_bf16.h>
namespace base_kernel_cu 
{

    //这里只是对单个元素的
    __global__ void argmax_kernel_fp32(const float* input_ptr, size_t size, size_t* output_idx) 
    {
        //这里是以warp为单位，所以这里实际上是统计归约到的单位？？？
        __shared__ size_t shared_max_ptr[32];
        __shared__ float shared_max_value[32];
        uint32_t tid = threadIdx.x;
  
        size_t max_index;
        float max_value;


        if (tid >= size) 
            max_index = SIZE_MAX;
        else
        {
            max_index = threadIdx.x;
            max_value = input_ptr[max_index];
        }

        for (size_t i = tid; i < size; i += blockDim.x) 
        {
            if (input_ptr[i] > max_value) 
            {
                max_index = i;
                max_value = input_ptr[i];
            }
        }

        block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
        //高风险地方，同步屏障n的古代return线程
        __syncthreads();
        if (threadIdx.x == 0) 
            *output_idx = max_index;
  
    }

    //所以这里是默认获得一个输出
    //只负责调用，不负责检查
    //进来之前最好检查一下input_ptr的大小
    //这里的 argmax_kernel_cu是会阻塞的，影响后面函数的调用与执行
    size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream) 
    {
        base::deviceId device_id;
        CUDA_CALL(cudaGetDevice(&device_id));
        std::shared_ptr<base::DeviceAllocator> alloc_cu =
            base::DeviceAllocatorFactory::get_instance(device_id);
  
        size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
  
        size_t output_index = 0;
        if (!stream) 
        {
            argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, index);
            cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
        } 
        else 
        {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            argmax_kernel_fp32<<<1, 512, 0, stream_>>>(input_ptr, size, index);
            cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
            CUDA_CALL(cudaStreamSynchronize(stream_));
        }

        alloc_cu->release(index);
        return output_index;
    }
}  // namespace base_kernel_cu 


namespace bf16_kernel_cu
{
    __global__ void argmax_kernel_bf16(const __nv_bfloat16* input_ptr, size_t size, size_t* output_idx)
    {
        __shared__ size_t shared_max_ptr[32];
        __shared__ float shared_max_value[32];
        uint32_t tid = threadIdx.x;

        size_t max_index;
        float max_value;

        if (tid >= size)
        {
            max_index = SIZE_MAX;
        }
        else
        {
            max_index = threadIdx.x;
            max_value = __bfloat162float(input_ptr[max_index]);
        }

        for (size_t i = tid; i < size; i += blockDim.x)
        {
            float val = __bfloat162float(input_ptr[i]);
            if (val > max_value)
            {
                max_index = i;
                max_value = val;
            }
        }

        base_kernel_cu::block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
        __syncthreads();
        if (threadIdx.x == 0)
            *output_idx = max_index;
    }

    size_t argmax_kernel_cu(const __nv_bfloat16* input_ptr, size_t size, void* stream)
    {
        base::deviceId device_id;
        CUDA_CALL(cudaGetDevice(&device_id));
        std::shared_ptr<base::DeviceAllocator> alloc_cu =
            base::DeviceAllocatorFactory::get_instance(device_id);

        size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));

        size_t output_index = 0;
        if (!stream)
        {
            argmax_kernel_bf16<<<1, 512>>>(input_ptr, size, index);
            cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            argmax_kernel_bf16<<<1, 512, 0, stream_>>>(input_ptr, size, index);
            cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
            CUDA_CALL(cudaStreamSynchronize(stream_));
        }

        auto err = cudaGetLastError();
        LOG(INFO)<<cudaGetErrorString(err)<<"\n";

        alloc_cu->release(index);
        return output_index;
    }

}
