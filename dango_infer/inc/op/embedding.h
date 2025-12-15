
#ifndef DANGO_INCLUDE_OP_EMBEDDING_H_
#define DANGO_INCLUDE_OP_EMBEDDING_H_
#include <utility>
#include "layer.h"
namespace op 
{
    struct EmbeddingOutput 
    {
        tensor::Tensor input_tokens;
        tensor::Tensor input_embeddings;

        explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings)
            : input_tokens(std::move(input_tokens)),
            input_embeddings(std::move(input_embeddings))
        {}
    };

    class EmbeddingLayer : public LayerParam 
    {
    public:
        explicit EmbeddingLayer();
        
        base::Status check() const override;

        base::Status forward(cudaStream_t stream=nullptr) override;

    //private:
    //    int32_t dim_ = 0;
    //    int32_t seq_len_ = 0;
    //    int32_t vocab_size_ = 0;
    };
}  // namespace op
#endif  // DANGO_INCLUDE_OP_EMBEDDING_H_
