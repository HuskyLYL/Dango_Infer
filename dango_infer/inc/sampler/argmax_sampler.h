//
// Created by fss on 24-6-9.
//

#ifndef DANGO_INFER_NON_SAMPLER_H
#define DANGO_INFER_NON_SAMPLER_H
#include <base/base.h>
#include "sampler.h"

namespace sampler 
{
    class ArgmaxSampler : public Sampler 
    {
    public:
        explicit ArgmaxSampler() : Sampler() {}

        size_t sample(const tensor::Tensor& input_tensor, void* stream = nullptr) override;

        base::Status check(const tensor::Tensor& tensor) const override;
    };
}  // namespace sampler
#endif  // DANGO_INFER_NON_SAMPLER_H
