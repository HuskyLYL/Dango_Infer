
#ifndef DANGO_INCLUDE_OP_ROW_PARALLEL_MATMUL_H_
#define DANGO_INCLUDE_OP_ROW_PARALLEL_MATMUL_H_
#include <cuda_runtime_api.h>
#include <op/matmul.h>
#include "nccl/collective.h"
#include "layer.h"
namespace op 
{
    //重写了权重设置与forward
    //为了多卡，设置权重读取的偏移
    class RowMatmulLayer : public MatmulLayer
    {
    public:
        explicit RowMatmulLayer(bool has_all_gather,bool has_bias = false);

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1,cudaStream_t stream=nullptr) override;

        base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,base::deviceId device_id);

        base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

        base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          base::deviceId device_id ,base::DataType weight_data_type = base::DataType::kDataTypeFp32) override;


    private:

        bool has_all_gather = false;
    };
}  // namespace op
#endif  // DANGO_INCLUDE_OP_ROW_PARALLEL_MATMUL_H_
