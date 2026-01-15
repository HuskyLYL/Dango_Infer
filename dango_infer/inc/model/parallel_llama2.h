#ifndef DANGO_INCLUDE_MODEL_PARALLEL_LLAMA2_H_
#define DANGO_INCLUDE_MODEL_PARALLEL_LLAMA2_H_

#include "model/llama2.h"
#include "op/parallel_embedding.h"
#include "op/colom_parallel_matmul.h"
#include "op/row_parallel_matmul.h"
#include "op/parallel_rope.h"
#include "op/parallel_swiglu.h"
#include "op/paralle_mha.h"



namespace model
{
    // Parallel LLama2 placeholder; implementation to be provided later.
    class ParallelLLama2Model : public LLama2Model
    {
    public:
        explicit ParallelLLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
            std::string model_path, base::DataType data_type, bool is_quant_model);

        void init_mem() override;

    private:
        void create_param_layers() override;
        void create_nonparam_layers() override;

    protected:
        void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const override;
        std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
            int32_t token_pos) const override;
    };
}  // namespace model

#endif  // DANGO_INCLUDE_MODEL_PARALLEL_LLAMA2_H_
