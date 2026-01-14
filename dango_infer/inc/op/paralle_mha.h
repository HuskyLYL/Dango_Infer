#ifndef DANGO_INCLUDE_PARALLE_MHA_H_
#define DANGO_INCLUDE_PARALLE_MHA_H_

#include "op/mha.h"

namespace op
{
    // Parallel variant of MHA; implementation to be filled in next step.
    class Paralle_MultiHeadAttenton : public MultiHeadAttention
    {
    public:
        explicit Paralle_MultiHeadAttenton(int32_t layer_index,
            int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
            int32_t head_num, int32_t head_size);

        base::Status check() const override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
            const tensor::Tensor& input3, const tensor::Tensor& input4,
            const tensor::Tensor& output1, cudaStream_t stream = nullptr) override;

    private:
        int32_t global_head_num_ = 0;
    };
}  // namespace op

#endif  // DANGO_INCLUDE_PARALLE_MHA_H_
