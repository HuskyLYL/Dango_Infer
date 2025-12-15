#ifndef DANGO_INCLUDE_OP_LAYER_H_
#define DANGO_INCLUDE_OP_LAYER_H_
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op 
{
  class Layer;


    //这里是我们的算子层
    class BaseLayer 
    {
    public:
        explicit BaseLayer(std::string layer_name = "");

        virtual base::Status init() = 0;

        virtual base::Status forward(cudaStream_t stream=nullptr) = 0;

        virtual base::Status check() const ;



        virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1,cudaStream_t stream=nullptr) = 0;

        virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                    const tensor::Tensor& output1,cudaStream_t stream=nullptr) = 0;

        virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                    const tensor::Tensor& input3, const tensor::Tensor& output1,cudaStream_t stream=nullptr) = 0;

        virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                    const tensor::Tensor& input3, const tensor::Tensor& input4,
                                    const tensor::Tensor& output1,cudaStream_t stream=nullptr) = 0;


                                    
        //这里算子层的计算应该就是我们的forward
        virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                    const tensor::Tensor& input3, const tensor::Tensor& input4,
                                    const tensor::Tensor& input5, const tensor::Tensor& output1,cudaStream_t stream=nullptr) = 0;

        virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

        virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

        virtual size_t input_size() const = 0;

        virtual size_t output_size() const = 0;

        virtual tensor::Tensor& get_input(int32_t idx) = 0;

        virtual tensor::Tensor& get_output(int32_t idx) = 0;

        virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

        virtual const tensor::Tensor& get_output(int32_t idx) const = 0;


        //这里先写着,主要目的是为了好让基础指针调用
        //这里不可以直接整为0
        virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

        virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
            base::deviceId device_id ,base::DataType weight_data_type = base::DataType::kDataTypeFp32) ;



        const std::string& get_layer_name() const;

        void set_layer_name(const std::string& layer_name);


        base::Status check_tensor(const tensor::Tensor& tensor,
            base::deviceId device_id,base::DataType data_type) const;

        base::Status check_tensor_with_dim(const tensor::Tensor& tensor, 
            base::deviceId device_id,base::DataType data_type, ...) const;


    protected:
        std::vector<tensor::Tensor> inputs_;
        std::vector<tensor::Tensor> outputs_;
        std::string layer_name_;
    };




    class Layer : public BaseLayer 
    {
    public:
        explicit Layer(std::string layer_name = "");

        base::Status init() override;


        base::Status forward(cudaStream_t stream=nullptr) override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1,cudaStream_t stream=nullptr) override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1,cudaStream_t stream=nullptr) override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1,cudaStream_t stream=nullptr) override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1,cudaStream_t stream=nullptr) override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1,cudaStream_t stream=nullptr) override;

        void set_input(int32_t idx, const tensor::Tensor& input) override;

        void set_output(int32_t idx, const tensor::Tensor& output) override;

        const tensor::Tensor& get_input(int32_t idx) const override;

        const tensor::Tensor& get_output(int32_t idx) const override;

        tensor::Tensor& get_input(int32_t idx) override;

        tensor::Tensor& get_output(int32_t idx) override;

        size_t input_size() const override;

        size_t output_size() const override;

        void reset_input_size(size_t size);

        void reset_output_size(size_t size);

        virtual void to_device(base::deviceId device_id,cudaStream_t stream=nullptr);



    };

    class LayerParam : public Layer 
    {
    public:
        explicit LayerParam(std::string layer_name = "");

        size_t weight_size() const;

        void reset_weight_size(size_t size);

        tensor::Tensor& get_weight(int32_t idx);

        const tensor::Tensor& get_weight(int32_t idx) const;

        void to_device(base::deviceId device_id,cudaStream_t stream=nullptr) override;


        base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

        base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          base::deviceId device_id ,base::DataType weight_data_type = base::DataType::kDataTypeFp32) override;


    protected:

        tensor::Tensor scales_;

        std::vector<tensor::Tensor> weights_;
    };
}  // namespace op
#endif  // DANGO_INCLUDE_OP_LAYER_H_
