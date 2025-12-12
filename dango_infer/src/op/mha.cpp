#include "op/mha.h"
#include "kernel/kernels_interface.h"
namespace op 
{
    MultiHeadAttention::MultiHeadAttention( int32_t layer_index,
        int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
        int32_t head_num, int32_t head_size)
        : Layer("MultiHead"),
        layer_index_(layer_index),
        kv_mul_(kv_mul),
        kv_dim_(kv_dim),
        seq_len_(seq_len),
        head_num_(head_num),
        head_size_(head_size) 
    {
        reset_input_size(5);
        reset_output_size(1);
    }


    base::Status MultiHeadAttention::check() const 
    {
        const auto& query_tensor = get_input(0);
        const auto& score_tensor = get_input(1);
        const auto& key_cache_tensor = get_input(2);
        const auto& value_cache_tensor = get_input(3);
        const auto& output_tensor = get_output(0);

        base::deviceId device_id = query_tensor.getDeviceId();
        base::DataType data_type = query_tensor.data_type();

        if (pos_ < 0 || pos_ >= seq_len_) 
            return base::error::InvalidArgument("The mha layer position is out of range.");

        if (query_tensor.dims_size() != 2) 
            return base::error::InvalidArgument("The mha layer requires 2-D query tensor.");
        base::Status status = check_tensor_with_dim(query_tensor, device_id, data_type, head_num_, head_size_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 0 error in the mha layer.";
            return status;
        }

        if (score_tensor.dims_size() != 2) 
            return base::error::InvalidArgument("The mha layer requires 2-D score tensor.");
        status = check_tensor_with_dim(score_tensor, device_id, data_type, head_num_, seq_len_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 1 error in the mha layer.";
            return status;
        }

        status = check_tensor(key_cache_tensor, device_id, data_type);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 2 error in the mha layer.";
            return status;
        }
        if (value_cache_tensor.dims_size() != key_cache_tensor.dims_size()) 
            return base::error::InvalidArgument("The mha layer requires same dims size for key/value cache tensors.");
        status = check_tensor(value_cache_tensor, device_id, data_type);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor 3 error in the mha layer.";
            return status;
        }

        if (key_cache_tensor.dims_size() != 3) 
            return base::error::InvalidArgument("The mha layer requires 3-D cache tensors.");
        int32_t cache_layer = key_cache_tensor.get_dim(0);
        int32_t cache_seq = key_cache_tensor.get_dim(1);
        int32_t cache_dim = key_cache_tensor.get_dim(2);
        if (cache_layer <= layer_index_) 
            return base::error::InvalidArgument("The mha layer cache tensor layer index is out of range.");
        if (cache_seq != seq_len_) 
            return base::error::InvalidArgument("The mha layer requires cache seq_len equal to seq_len_.");
        if (cache_dim != kv_dim_) 
            return base::error::InvalidArgument("The mha layer requires cache dim equal to kv_dim_.");

        if (value_cache_tensor.get_dim(0) != cache_layer ||
            value_cache_tensor.get_dim(1) != cache_seq ||
            value_cache_tensor.get_dim(2) != cache_dim) 
        {
            return base::error::InvalidArgument("The mha layer key/value cache tensors have different dims.");
        }

        if (output_tensor.dims_size() != 2) 
            return base::error::InvalidArgument("The mha layer requires 2-D output tensor.");
        status = check_tensor_with_dim(output_tensor, device_id, data_type, head_num_, head_size_);
        if (!status) 
        {
            LOG(ERROR) << "The output tensor error in the mha layer.";
            return status;
        }

        base::setDevice(device_id);
        return base::error::Success();
    }

    base::Status MultiHeadAttention::forward(cudaStream_t stream) 
    {
        auto status = check();
        if (!status) 
            return status;
  
        const tensor::Tensor& mha_out = this->get_output(0);
        const tensor::Tensor& query_tensor = this->get_input(0);
        const tensor::Tensor& score_tensor = this->get_input(1);
        const tensor::Tensor& key_cache_tensor = this->get_input(2);
        const tensor::Tensor& value_cache_tensor = this->get_input(3);


        base_kernel_cu::get_mha_kernel()(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,
            head_size_, mha_out, query_tensor, score_tensor,
            key_cache_tensor, value_cache_tensor, stream);
  
        return base::error::Success();
    }

    void MultiHeadAttention::set_pos(int32_t pos) { this->pos_ = pos; }

    void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }



}  // namespace op
