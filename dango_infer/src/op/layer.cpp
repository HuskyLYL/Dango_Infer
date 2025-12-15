#include "op/layer.h"
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op 
{
    BaseLayer::BaseLayer(std::string layer_name)
        : layer_name_(std::move(layer_name)) {}

    base::Status BaseLayer::check() const
    {
        return base::error::Success();
    }


    base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) { return base::error::FunctionNotImplement(); }

    base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
            base::deviceId device_id ,base::DataType weight_data_type )
    {
        return base::error::FunctionNotImplement();
    }






    const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

    void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }



    base::Status BaseLayer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                            base::deviceId device_id, base::DataType data_type,
                                            ...) const 
    {
        std::va_list args;
        if (tensor.is_empty()) 
            return base::error::InvalidArgument("The tensor parameter is empty.");
  
        if (tensor.getDeviceId() != device_id) 
            return base::error::InvalidArgument("The tensor has a wrong device type.");
  
        if (tensor.data_type() != data_type) 
            return base::error::InvalidArgument("The tensor has a wrong data type.");
  

        va_start(args, data_type);
        
        int32_t dims = tensor.dims_size();
  
        for (int32_t i = 0; i < dims; ++i) 
        {
            int32_t dim = va_arg(args, int32_t);
            if (dim != tensor.get_dim(i)) 
                return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
        }
        va_end(args);
        return base::error::Success();
    }

    base::Status BaseLayer::check_tensor(const tensor::Tensor& tensor, 
        base::deviceId device_id,base::DataType data_type) const 
    {
        if (tensor.is_empty()) 
            return base::error::InvalidArgument("The tensor parameter is empty.");
  
        if (tensor.getDeviceId() != device_id) 
            return base::error::InvalidArgument("The tensor has a wrong device type.");
  
        if (tensor.data_type() != data_type) 
            return base::error::InvalidArgument("The tensor has a wrong data type.");
  
        return base::error::Success();
    }


    Layer::Layer(std::string layer_name)
        : BaseLayer(std::move(layer_name)) {}

    base::Status Layer::init() { return base::error::Success(); }

    base::Status Layer::forward(cudaStream_t stream) { return base::error::FunctionNotImplement(""); }



    void Layer::set_input(int32_t idx, const tensor::Tensor& input) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        this->inputs_.at(idx) = input;
    }

    void Layer::set_output(int32_t idx, const tensor::Tensor& output) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        this->outputs_.at(idx) = output;
    }

    const tensor::Tensor& Layer::get_input(int32_t idx) const 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_input(int32_t idx) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_output(int32_t idx) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }



    const tensor::Tensor& Layer::get_output(int32_t idx) const 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

    void Layer::reset_output_size(size_t size) { outputs_.resize(size); }


    size_t Layer::input_size() const { return inputs_.size(); }

    size_t Layer::output_size() const { return outputs_.size(); }

    void Layer::to_device(base::deviceId device_id ,cudaStream_t stream) 
    {
        for (auto& input : inputs_) 
            if (!input.is_empty()) 
                input.to_device(device_id,stream);
        for (auto& output : outputs_) 
            if (!output.is_empty()) 
            output.to_device(device_id,stream);
    }

    void LayerParam::to_device(base::deviceId device_id,cudaStream_t stream) 
    {
        Layer::to_device(device_id,stream);
        for (auto& weight : weights_) 
            weight.to_device(device_id,stream);
        
        if (!scales_.is_empty()) 
            scales_.to_device(device_id,stream);
        
    }




    LayerParam::LayerParam(std::string layer_name)
        : Layer(std::move(layer_name)) {}

    base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        weights_.at(idx) = weight;
        return base::error::Success();
    }

    const tensor::Tensor& LayerParam::get_weight(int32_t idx) const 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }



    base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                        const void* weight_ptr, base::deviceId device_id,base::DataType weight_data_type) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK_NE(weight_ptr, nullptr);

        
        //传入的指针非空，进入托管
        //托管的值一般为模型参数映射过来的值
        //所以这里不用担心释放的风险
        tensor::Tensor weight(dims,device_id,weight_data_type,const_cast<void*>(weight_ptr));

        weights_.at(idx) = weight;
     
        return base::error::Success();
    }





    void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

    size_t LayerParam::weight_size() const { return weights_.size(); }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1,cudaStream_t stream) 
    {
        this->set_input(0, input1);
        this->set_output(0, output1);
        return this->forward(stream);
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& output1,cudaStream_t stream) 
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_output(0, output1);
        return this->forward(stream);
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& output1,cudaStream_t stream) 
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);

        this->set_output(0, output1);
        return this->forward(stream);
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& output1,cudaStream_t stream) 
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);

        this->set_output(0, output1);
        return this->forward(stream);
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& input5, const tensor::Tensor& output1,cudaStream_t stream) 
    {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);

        this->set_output(0, output1);
        return this->forward(stream);
    }

    tensor::Tensor& LayerParam::get_weight(int32_t idx) 
    {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

}  // namespace op