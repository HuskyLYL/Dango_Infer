#include "op/rmsnorm.h"
#include <cuda_runtime_api.h>
#include "kernel/kernels_interface.h"

namespace op 
{
    RmsNormLayer::RmsNormLayer(): LayerParam("RMSNorm") 
    {
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
    }

    //首先，rmsNorm的检查：
    //根据weight的数据类型和向量的大小判定input
    /*
        1.判断输入设备是否相同
        2.判断向量的长短是否相同
        3.判断向量非空
    
    */
    base::Status RmsNormLayer::check() const 
    {
        const auto & input_tensor = this->get_input(0);
        const auto & weight_tensor = this->get_weight(0);
        const auto & output_tensor = this->get_output(0);


        CHECK(!input_tensor.is_empty());
        CHECK(!weight_tensor.is_empty());
        CHECK(!output_tensor.is_empty());

        int32_t dim_head_size = weight_tensor.get_dim(0);

        base::deviceId device_id = weight_tensor.getDeviceId();

        //这里我们要支持二维和一维
        int32_t dim_size = get_input(0).dims_size();

        int32_t tensor_dim_head_size = input_tensor.get_dim(dim_size-1);

        CHECK_EQ(tensor_dim_head_size,dim_head_size)
            <<"[RmsNorm]: input_tensor dim must be equal with weight";

        CHECK_EQ(weight_tensor.getDeviceId(),input_tensor.getDeviceId())
            <<"[RmsNorm]: input_tensor device must be same as  weight";

        base::setDevice(device_id);

        return base::error::Success();
    } 
    

    base::Status RmsNormLayer::forward(cudaStream_t stream) 
    {
        auto status = check();
        if (!status)
            return status;
  
        auto input = this->get_input(0);
        auto weight = this->get_weight(0);
        auto output = this->get_output(0);

        f32x4_kernel_cu::get_rmsn_kernel()(input, weight, output, stream);

        return base::error::Success();
    }




   
} // namespace op








