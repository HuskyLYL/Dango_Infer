#ifndef DANGO_INFER_INCLUDE_OP_SWIGLU_H_
#define DANGO_INFER_INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op 
{
    class SwiGLULayer : public op::Layer 
    {
    public:
        explicit SwiGLULayer();

        base::Status check() const override;

        base::Status forward(cudaStream_t stream = nullptr) override;

    private:
    };
}  // namespace op
#endif  // DANGO_INFER_INCLUDE_OP_SWIGLU_H_
