#ifndef DANGO_INFER_SAMPLER_H
#define DANGO_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
#include "base/base.h"
#include "tensor/tensor.h"
namespace sampler 
{
    class Sampler 
    {
    public:
        explicit Sampler() = default;

        virtual size_t sample(const tensor::Tensor& input_tensor, void* stream = nullptr) = 0;

        virtual base::Status check(const tensor::Tensor& tensor) const = 0;
    };
}  // namespace sampler
#endif  // DANGO_INFER_SAMPLER_H
