#ifndef DANGO_INCLUDE_OP_ADD_H
#define DANGO_INCLUDE_OP_ADD_H
#include "base/base.h"
#include "layer.h"
namespace op 
{
    class VecAddLayer : public Layer 
    {
    public:
        explicit VecAddLayer();

        base::Status check() const override;

        base::Status forward(cudaStream_t stream=nullptr) override;
    };
}  // namespace op
#endif  // DANGO_INCLUDE_OP_ADD_H