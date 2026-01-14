#ifndef DANGO_INCLUDE_PARALLEL_SWIGLU_H_
#define DANGO_INCLUDE_PARALLEL_SWIGLU_H_

#include "op/swiglu.h"

namespace op
{
    // Parallel SwiGLU placeholder; logic to be filled in later.
    class ParallelSwiGLULayer : public SwiGLULayer
    {
    public:
        ParallelSwiGLULayer();

        base::Status check() const override;

        base::Status forward(cudaStream_t stream = nullptr) override;

        base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
            const tensor::Tensor& output1, cudaStream_t stream = nullptr) override;
    };
}  // namespace op

#endif  // DANGO_INCLUDE_PARALLEL_SWIGLU_H_
