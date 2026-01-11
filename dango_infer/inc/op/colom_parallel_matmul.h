
#ifndef DANGO_INCLUDE_OP_COLOM_PARALLEL_MATMUL_H_
#define DANGO_INCLUDE_OP_COLOM_PARALLEL_MATMUL_H_
#include <cuda_runtime_api.h>
#include <op/matmul.h>
#include "nccl/collective.h"
#include "layer.h"
namespace op 
{
    // 列并行：按列切分权重/输入，输出在各 rank 上求和（可选 all-reduce）。
    class ColomParallelMatmulLayer : public MatmulLayer
    {
    public:
        explicit ColomParallelMatmulLayer(bool has_all_reduce, bool has_bias = false);

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1,
                             cudaStream_t stream = nullptr) override;

        base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                              base::deviceId device_id);

        base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

        base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                          const void* weight_ptr, base::deviceId device_id,
                          base::DataType weight_data_type = base::DataType::kDataTypeFp32) override;

    private:
        bool has_all_reduce = false;
    };
}  // namespace op
#endif  // DANGO_INCLUDE_OP_COLOM_PARALLEL_MATMUL_H_
