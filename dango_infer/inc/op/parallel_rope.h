#ifndef DANGO_INCLUDE_OP_PARALLEL_ROPE_H_
#define DANGO_INCLUDE_OP_PARALLEL_ROPE_H_

#include "op/rope.h"

namespace op
{
    // Parallel variant placeholder; override hooks for rank-aware RoPE logic.
    class ParallelRoPELayer : public RoPELayer
    {
    public:
        explicit ParallelRoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size);

        base::Status check() const override;
        base::Status forward(cudaStream_t stream = nullptr) override;

        // Overload to accept explicit inputs/outputs before dispatching.
        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& input4,
                             const tensor::Tensor& input5, const tensor::Tensor& output1,
                             cudaStream_t stream = nullptr);
    };
}  // namespace op

#endif  // DANGO_INCLUDE_OP_PARALLEL_ROPE_H_
