#include "op/matmul.h"
#include "kernel/kernels_interface.h"
#include "kernel/cuda/matmul.cuh"
namespace op 
{
    MatmulLayer::MatmulLayer(bool has_bias)
    : LayerParam("Matmul"),has_bias_(has_bias) 
    {
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
        if (has_bias_) 
            bias_.resize(1);
  
    }

    base::Status MatmulLayer::check() const 
    {
        const tensor::Tensor& input_tensor = get_input(0);
        const tensor::Tensor& weight_tensor = get_weight(0);
        const tensor::Tensor& output_tensor = get_output(0);

        if (weight_tensor.dims_size() != 2) 
            return base::error::InvalidArgument("The weight tensor of matmul must be 2-D.");
        
        int32_t output_dim = weight_tensor.get_dim(0);
        int32_t input_dim = weight_tensor.get_dim(1);

        base::deviceId device_id = weight_tensor.getDeviceId();

        base::DataType data_type = weight_tensor.data_type();

        if (input_tensor.dims_size() != 1) 
            return base::error::InvalidArgument("The input tensor of matmul must be 1-D.");

        base::Status status = check_tensor_with_dim(input_tensor, device_id, data_type, input_dim);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor error in the matmul layer.";
            return status;
        }

        status = check_tensor_with_dim(weight_tensor, device_id, data_type, output_dim, input_dim);
        if (!status) 
        {
            LOG(ERROR) << "The weight tensor error in the matmul layer.";
            return status;
        }

        if (output_tensor.dims_size() != 1) 
            return base::error::InvalidArgument("The output tensor of matmul must be 1-D.");

        status = check_tensor_with_dim(output_tensor, device_id, data_type, output_dim);
        if (!status) 
        {
            LOG(ERROR) << "The output tensor error in the matmul layer.";
            return status;
        }

        if (has_bias_) 
        {
            status = check_tensor_with_dim(get_bias(0), device_id, data_type, output_dim);
            if (!status) 
            {
                LOG(ERROR) << "The bias tensor error in the matmul layer.";
                return status;
            }
        }

        base::setDevice(device_id);

        return base::error::Success();
    }

    base::Status MatmulLayer::forward(cudaStream_t stream) 
    {
        auto status = check();
        if (!status) 
            return status;
  

        f32x4_kernel_cu::get_matmul_kernel()(get_input(0), get_weight(0), get_output(0), 1.f, stream);
  
        if (has_bias_) 
            f32x4_kernel_cu::get_elementwise_kernel()(get_output(0), get_bias(0), get_output(0), stream);


        return base::error::Success();
    }

    base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
        base::deviceId device_id) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, bias_.size());
        CHECK_NE(bias_ptr, nullptr);


        tensor::Tensor bias(dim, device_id, base::DataType::kDataTypeFp32,
                            const_cast<void*>(bias_ptr));
        bias_.at(idx) = bias;


        return base::error::Success();
    }

    tensor::Tensor& MatmulLayer::get_bias(int32_t idx)  
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, bias_.size());
        return bias_.at(idx);
    }

    const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, bias_.size());
        return bias_.at(idx);
    }



}  // namespace op
