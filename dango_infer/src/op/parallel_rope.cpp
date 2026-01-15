#include "op/parallel_rope.h"
#include "nccl/base.h"
#include "base/base.h"

namespace op
{
    ParallelRoPELayer::ParallelRoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size)
        : RoPELayer(dim/nccl::G_MPI_SIZE, kv_dim/nccl::G_MPI_SIZE, head_size){}

    base::Status ParallelRoPELayer::check() const
    {
        // TODO: add rank-aware validation if RoPE inputs are sharded.
        // Example: validate local dims vs global and world size.
        return RoPELayer::check();
    }

    base::Status ParallelRoPELayer::forward(cudaStream_t stream)
    {
        // TODO: add parallel-specific behavior (e.g., slice sin/cos cache per rank) before/after base call.
        // Example:
        // 1) assert/check world size and per-rank shapes
        // 2) create per-rank views of sin/cos cache if needed
        // 3) call base implementation on local shards
        return RoPELayer::forward(stream);
    }

    base::Status ParallelRoPELayer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                                            const tensor::Tensor& input5, const tensor::Tensor& output1,
                                            cudaStream_t stream)
    {
        // Slice input1 per rank if it is sharded by rows.
        tensor::Tensor local_input1 = input1;
        if (!input1.is_empty())
        {
            CHECK_GT(nccl::G_MPI_SIZE, 0);
            const size_t total_elems = input1.size();
            CHECK_EQ(total_elems % static_cast<size_t>(nccl::G_MPI_SIZE), 0u)
                << "ParallelRoPELayer input1 size must be divisible by world size.";

            const size_t per_rank = total_elems / static_cast<size_t>(nccl::G_MPI_SIZE);
            const size_t elem_bytes = base::DataTypeSize(input1.data_type());
            const size_t offset_bytes =
                per_rank * elem_bytes * static_cast<size_t>(nccl::G_MPI_RANK);
            void* offset_ptr =
                static_cast<char*>(const_cast<void*>(input1.get_buffer()->ptr())) + offset_bytes;

            local_input1 = tensor::Tensor(static_cast<int32_t>(per_rank),
                                          input1.getDeviceId(),
                                          input1.data_type(),
                                          offset_ptr);
        }

        this->set_input(0, local_input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);
        this->set_output(0, output1);

        return RoPELayer::forward(stream);
    }
}  // namespace op
