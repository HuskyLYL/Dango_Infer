#include "op/rope.h"
#include <cmath>
#include "kernel/kernels_interface.h"
namespace op 
{
    RoPELayer::RoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer("RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) 
    {
        reset_input_size(5);
        reset_output_size(1);
    }


    base::Status RoPELayer::check() const 
    {

        const auto& input_q = get_input(0);
        const auto& input_k = get_input(1);
        const auto& input_pos = get_input(2);
        const auto& sin_cache = get_input(3);
        const auto& cos_cache = get_input(4);

        base::deviceId device_id = input_q.getDeviceId();
        base::DataType data_type = input_q.data_type();

        base::Status status = check_tensor_with_dim(input_pos, base::CPUID,
            base::DataType::kDataTypeInt32, 1);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 2 error in the rope layer.";
            return status;
        }

        if (input_q.dims_size() != 1) 
            return base::error::InvalidArgument("The rope layer requires 1-D query tensor.");
        status = check_tensor_with_dim(input_q, device_id, data_type, dim_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 0 error in the rope layer.";
            return status;
        }

        if (input_k.dims_size() != 1) 
            return base::error::InvalidArgument("The rope layer requires 1-D key tensor.");
        status = check_tensor_with_dim(input_k, device_id, data_type, kv_dim_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 1 error in the rope layer.";
            return status;
        }

        if (sin_cache.dims_size() != 2 || cos_cache.dims_size() != 2) 
            return base::error::InvalidArgument("The rope layer requires 2-D sin/cos cache tensors.");
        int32_t max_seq_len = sin_cache.get_dim(0);
        status = check_tensor_with_dim(sin_cache, device_id, data_type, max_seq_len, head_size_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 3 error in the rope layer.";
            return status;
        }

        status = check_tensor_with_dim(cos_cache, device_id, data_type, max_seq_len, head_size_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 4 error in the rope layer.";
            return status;
        }

        base::setDevice(device_id);
        return base::error::Success();
    }





    base::Status RoPELayer::forward(cudaStream_t stream) 
    {
        base::Status status = check();
        if (!status) 
            return status;

        tensor::Tensor input_q = this->get_input(0);
        tensor::Tensor input_k = this->get_input(1);
        tensor::Tensor input_pos = this->get_input(2);

        tensor::Tensor sin_cache = this->get_input(3);
        tensor::Tensor cos_cache = this->get_input(4);


        base_kernel_cu::get_rope_kernel()(dim_, kv_dim_, head_size_, input_q, input_k, input_pos,
                                        sin_cache, cos_cache, stream);
        return base::error::Success();
    }



}  // namespace op
