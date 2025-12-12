#ifndef DANGO_INLCUDE_MHA_H
#define DANGO_INLCUDE_MHA_H
#include "layer.h"
namespace op 
{
    class MultiHeadAttention : public op::Layer 
    {
    public:
        explicit MultiHeadAttention(int32_t layer_index,
            int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
            int32_t head_num, int32_t head_size);

        base::Status check() const override;

        void set_pos(int32_t pos);
        void set_layer_idx(int32_t layer_idx);

        base::Status forward(cudaStream_t stream=nullptr) override;

    private:
        
        int32_t layer_index_ = 0;
        int32_t pos_ = 0;
        int32_t kv_mul_ = 0;
        int32_t kv_dim_ = 0;
        int32_t seq_len_ = 0;
        int32_t head_num_ = 0;
        int32_t head_size_ = 0;
    };
}  // namespace op
#endif  // DANGO_INLCUDE_MHA_H