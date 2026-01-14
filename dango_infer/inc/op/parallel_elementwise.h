#ifndef DANGO_INCLUDE_PARALLEL_ELEMENTWISE_H_
#define DANGO_INCLUDE_PARALLEL_ELEMENTWISE_H_

#include "op/elementwise.h"

namespace op
{
    // Parallel VecAdd placeholder; logic to be filled in later.
    class ParallelVecAddLayer : public VecAddLayer
    {
    public:
        ParallelVecAddLayer();

        base::Status check() const override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
            const tensor::Tensor& output1, cudaStream_t stream = nullptr) override;
    };
}  // namespace op

#endif  // DANGO_INCLUDE_PARALLEL_ELEMENTWISE_H_
