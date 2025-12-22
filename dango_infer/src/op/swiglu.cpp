#include "op/swiglu.h"
#include "kernel/kernels_interface.h"
#include "op/layer.h"
namespace op 
{
    SwiGLULayer::SwiGLULayer()
        : Layer("SwiGLU")
    {
        reset_input_size(2);
        reset_output_size(1);
    }

    base::Status SwiGLULayer::check() const 
    {
        const auto& input1 = get_input(0);
        const auto& input2 = get_input(1);
        const auto& output = get_output(0);

        base::deviceId device_id = input1.getDeviceId();
        base::DataType data_type = input1.data_type();

        base::Status status = check_tensor(input1, device_id, data_type);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 1 error in the swiglu layer.";
            return status;
        }

        status = check_tensor(input2, device_id, data_type);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 2 error in the swiglu layer.";
            return status;
        }

        status = check_tensor(output, device_id, data_type);
        if (!status) 
        {
            LOG(ERROR) << "The output tensor error in the swiglu layer.";
            return status;
        }

        if (input1.dims_size() != input2.dims_size() ||
            input1.dims_size() != output.dims_size()) 
        {
            return base::error::InvalidArgument("The swiglu layer requires same dims size for inputs and output.");
        }

        for (int32_t i = 0; i < input1.dims_size(); ++i) 
        {
            int32_t dim = input1.get_dim(i);
            if (input2.get_dim(i) != dim || output.get_dim(i) != dim) 
            {
                return base::error::InvalidArgument(
                    "The swiglu layer tensors have different dimensions in dim " + std::to_string(i));
            }
        }

        base::setDevice(device_id);
        return base::error::Success();
    }

    base::Status SwiGLULayer::forward(cudaStream_t stream) 
    {
        auto status = check();
        if (!status) 
            return status;
  
        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);
        auto output = this->get_output(0);

        auto data_type = input1.data_type();
        if (data_type == base::DataType::kDataTypeFp32)
        {
            base_kernel_cu::get_swiglu_kernel()(input1, input2, output, stream);
        }
        else if (data_type == base::DataType::kDataTypeBf16)
        {
            bf16_kernel_cu::get_swiglu_kernel()(input1, input2, output, stream);
        }
        else
        {
            return base::error::InvalidArgument("Unsupported data type in the swiglu layer.");
        }
        return base::error::Success();
}

}  // namespace op
