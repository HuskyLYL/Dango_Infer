#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "kernel/cuda/argmax.cuh"
#include <cuda_bf16.h>
namespace sampler 
{

    base::Status ArgmaxSampler::check(const tensor::Tensor& tensor) const
    {
        if (tensor.is_empty() || tensor.size() == 0) 
            return base::error::InvalidArgument("The tensor parameter is empty.");

        if (tensor.data_type() != base::DataType::kDataTypeFp32 &&
            tensor.data_type() != base::DataType::kDataTypeBf16)
            return base::error::InvalidArgument("The tensor data type must be fp32 or bf16.");

        if (tensor.getDeviceId() == base::CPUID) 
            return base::error::InvalidArgument("The tensor must be on GPU.");

        base::setDevice(tensor.getDeviceId());
        return base::error::Success();
    }


    size_t ArgmaxSampler::sample(const tensor::Tensor& input_tensor, void* stream)
    {
        base::Status status = check(input_tensor);

        if (!status) 
        {
            LOG(ERROR) << status.get_err_msg();
            return 0;
        }

        size_t size = input_tensor.size();

        size_t next = 0;
        if (input_tensor.data_type() == base::DataType::kDataTypeFp32)
            next = base_kernel_cu::argmax_kernel_cu(input_tensor.ptr<float>(), size, stream);
        else
            next = bf16_kernel_cu::argmax_kernel_cu(input_tensor.ptr<__nv_bfloat16>(), size, stream);

        return next;
        
    }
}  // namespace sampler
