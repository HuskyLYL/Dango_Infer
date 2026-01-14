#include "op/paralle_mha.h"
#include "nccl/base.h"
#include "nccl/collective.h"
#include "base/base.h"

namespace op
{
    Paralle_MultiHeadAttenton::Paralle_MultiHeadAttenton(int32_t layer_index,
        int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
        int32_t head_num, int32_t head_size)
        : MultiHeadAttention(layer_index, kv_mul, kv_dim, seq_len,
              head_num / nccl::G_MPI_SIZE, head_size),
          global_head_num_(head_num)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        CHECK_EQ(head_num % nccl::G_MPI_SIZE, 0)
            << "head_num must be divisible by world size for Paralle_MultiHeadAttenton.";
    }

    base::Status Paralle_MultiHeadAttenton::check() const
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);

        if (global_head_num_ % nccl::G_MPI_SIZE != 0)
            return base::error::InvalidArgument("head_num must be divisible by world size in Paralle_MultiHeadAttenton.");

        if (head_num_ != global_head_num_ / nccl::G_MPI_SIZE)
            return base::error::InvalidArgument("local head_num does not match global_head_num/world_size.");

        if (kv_mul_ != 1)
            return base::error::InvalidArgument("Paralle_MultiHeadAttenton requires kv_mul == 1.");

        // Reuse base validation after parallel-specific checks.
        return MultiHeadAttention::check();
    }

    base::Status Paralle_MultiHeadAttenton::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& input3, const tensor::Tensor& input4,
        const tensor::Tensor& output1, cudaStream_t stream)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);

        // Slice query across ranks.
        const size_t total_elems = input1.size();
        CHECK_EQ(total_elems % static_cast<size_t>(nccl::G_MPI_SIZE), 0u)
            << "Paralle_MultiHeadAttenton query size must be divisible by world size";
        const size_t per_rank = total_elems / static_cast<size_t>(nccl::G_MPI_SIZE);
        const size_t elem_bytes = base::DataTypeSize(input1.data_type());
        const size_t offset_bytes = per_rank * elem_bytes * static_cast<size_t>(nccl::G_MPI_RANK);
        void* offset_ptr = static_cast<char*>(const_cast<void*>(input1.get_buffer()->ptr())) + offset_bytes;
        tensor::Tensor local_query(static_cast<int32_t>(per_rank),
            input1.getDeviceId(), input1.data_type(), offset_ptr);

        // Slice score storage by head dimension: [head_num_, seq_len_].
        CHECK_EQ(input2.dims_size(), 2);
        CHECK_EQ(input2.get_dim(0), global_head_num_);
        const int32_t seq_len = input2.get_dim(1);
        const int32_t heads_per_rank = head_num_;
        CHECK_GT(heads_per_rank, 0);
        const size_t score_elem_bytes = base::DataTypeSize(input2.data_type());
        const size_t score_offset_bytes =
            static_cast<size_t>(heads_per_rank) * static_cast<size_t>(seq_len) *
            score_elem_bytes * static_cast<size_t>(nccl::G_MPI_RANK);
        void* score_offset_ptr =
            static_cast<char*>(const_cast<void*>(input2.get_buffer()->ptr())) + score_offset_bytes;
        tensor::Tensor local_score({heads_per_rank, seq_len},
            input2.getDeviceId(), input2.data_type(), score_offset_ptr);

        // Slice output to match local query layout.
        CHECK_EQ(output1.size(), input1.size());
        void* out_offset_ptr = static_cast<char*>(const_cast<void*>(output1.get_buffer()->ptr())) + offset_bytes;
        tensor::Tensor local_output(static_cast<int32_t>(per_rank),
            output1.getDeviceId(), output1.data_type(), out_offset_ptr);

        this->set_input(0, local_query);
        this->set_input(1, local_score);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_output(0, local_output);

        // Run local forward.
        auto status = MultiHeadAttention::forward(stream);
        if (!status)
            return status;

        return base::error::Success();
    }
}  // namespace op
