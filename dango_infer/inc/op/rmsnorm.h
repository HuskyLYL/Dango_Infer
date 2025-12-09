#ifndef DANGO_INCLUDE_OP_RMSNORM_H_
#define DANGO_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op 
{
    class RmsNormLayer : public LayerParam 
    {
    public:
        explicit RmsNormLayer();

        base::Status check() const override;

        base::Status forward(cudaStream_t stream = nullptr) override;
    };
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_RMSNORM_H_
