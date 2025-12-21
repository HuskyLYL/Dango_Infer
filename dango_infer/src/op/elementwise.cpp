#include "op/elementwise.h"
#include "kernel/kernels_interface.h"
namespace op 
{
  //在初始化的时候规划
  VecAddLayer::VecAddLayer():Layer("Add") 
  {
    reset_input_size(2);
    reset_output_size(1);
  }

  //check中检查数据类型
  //检查当前程序的上下文
  base::Status VecAddLayer::check() const 
  {
    tensor::Tensor input1 = this->get_input(0);
    tensor::Tensor input2 = this->get_input(1);


    base::DataType data_type = input1.data_type();
    int32_t size = input1.size();
    switch (data_type) 
    {
      case base::DataType::kDataTypeFp32:
        if (size % 4 != 0) 
          return base::error::InvalidArgument("Elementwise fp32 input size must be a multiple of 4.");
        break;
      case base::DataType::kDataTypeBf16:
        if (size % 8 != 0) 
          return base::error::InvalidArgument("Elementwise bf16 input size must be a multiple of 8.");
        break;
      default:
        return base::error::InvalidArgument("Unsupported data type in the add layer.");
    }

    base::deviceId device_id = input1.getDeviceId();


    base::Status status;
    status = check_tensor_with_dim(input1, device_id, data_type, size);
    if (!status) 
    {
      LOG(ERROR) << "The input tensor 1 error in the add layer.";
      return status;
    }

    status = check_tensor_with_dim(input2, device_id, data_type, size);
    if (!status) 
    {
      LOG(ERROR) << "The input tensor 2 error in the add layer.";
      return status;
    }

    status = check_tensor_with_dim(get_output(0), device_id, data_type, size);
    if (!status) 
    {
      LOG(ERROR) << "The output tensor error in the add layer.";
      return status;
    }

    

    base::setDevice(device_id);


    return base::error::Success();
  }

  base::Status VecAddLayer::forward(cudaStream_t stream) 
  {
    auto status = this->check();
    if (!status) 
      return status;
  
    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);
    auto output = this->get_output(0);

    base::DataType data_type = input1.data_type();
    if(data_type == base::DataType::kDataTypeFp32)
      f32x4_kernel_cu::get_elementwise_kernel()(input1, input2, output,stream ? stream : nullptr);
    else if(data_type == base::DataType::kDataTypeBf16)
      bf16x8_kernel_cu::get_elementwise_kernel()(input1, input2, output,stream ? stream : nullptr);
      
    
    return base::error::Success();
}

}  // namespace op
