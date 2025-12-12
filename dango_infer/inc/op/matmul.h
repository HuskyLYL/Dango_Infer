
#ifndef DANGO_INCLUDE_OP_MATMUL_H_
#define DANGO_INCLUDE_OP_MATMUL_H_
#include <cuda_runtime_api.h>
#include "layer.h"
namespace op 
{
    class MatmulLayer : public LayerParam 
    {
    public:
        explicit MatmulLayer(bool has_bias = false);

        base::Status check() const override;

        base::Status forward(cudaStream_t stream=nullptr) override;

        base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                                base::deviceId device_id);

        tensor::Tensor& get_bias(int32_t idx);

        const tensor::Tensor& get_bias(int32_t idx) const;

    private:
        bool has_bias_ = false;
        
        std::vector<tensor::Tensor> bias_;
    };
}  // namespace op
#endif  // DANGO_INCLUDE_OP_MATMUL_H_
