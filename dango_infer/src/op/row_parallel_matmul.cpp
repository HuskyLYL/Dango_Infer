#include "op/row_parallel_matmul.h"
#include "nccl/base.h"
#include "nccl/collective.h"
#include "base/base.h"


namespace op
{

    RowMatmulLayer::RowMatmulLayer(bool has_all_gather_flag, bool has_bias)
        : MatmulLayer(has_bias), has_all_gather(has_all_gather_flag){}



    base::Status RowMatmulLayer::forward(const tensor::Tensor& input1,
                                    const tensor::Tensor& output1,
                                    cudaStream_t stream)
    {
        //RowMatmulLayer input输入直接不需要进行切分
        this->set_input(0, input1);

        // Create a view for this rank's output slice.
        const size_t total_out_elems = output1.size();
        CHECK_EQ(total_out_elems % static_cast<size_t>(nccl::G_MPI_SIZE), 0u)
            << "RowMatmul output size must be divisible by world size";
        
        const size_t out_per_rank = total_out_elems / static_cast<size_t>(nccl::G_MPI_SIZE);
        const size_t out_elem_bytes = base::DataTypeSize(output1.data_type());

        const size_t out_offset_bytes =
            out_per_rank * out_elem_bytes * static_cast<size_t>(nccl::G_MPI_RANK);
        void* out_offset_ptr =
            static_cast<char*>(const_cast<void*>(output1.get_buffer()->ptr())) + out_offset_bytes;
        tensor::Tensor output_view(static_cast<int32_t>(out_per_rank),
                                   output1.getDeviceId(),
                                   output1.data_type(),
                                   out_offset_ptr);

        this->set_output(0, output_view);

        base::Status status = MatmulLayer::forward(stream);

        if (!status)
            return status;

        if (has_all_gather)
            nccl::TensorAllReduce(output1);

        return base::error::Success();
    }

    base::Status RowMatmulLayer::set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                                        base::deviceId device_id)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        CHECK_EQ(dims % nccl::G_MPI_SIZE, 0) << "Bias length must be divisible by world size";

        const int32_t per_rank_dim = dims / nccl::G_MPI_SIZE;
        const size_t offset_bytes =
            static_cast<size_t>(per_rank_dim) * sizeof(float) * static_cast<size_t>(nccl::G_MPI_RANK);
        const void* offset_ptr = static_cast<const char*>(bias_ptr) + offset_bytes;

        int32_t per_rank_dim_copy = per_rank_dim;
        return MatmulLayer::set_bias(idx, per_rank_dim_copy, offset_ptr, device_id);
    }

    base::Status RowMatmulLayer::set_weight(int32_t idx, const tensor::Tensor& weight)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        const int32_t rows = weight.get_dim(0);
        const int32_t cols = weight.get_dim(1);
        CHECK_EQ(rows % nccl::G_MPI_SIZE, 0)
            << "Weight rows must be divisible by world size";

        const int32_t rows_per_rank = rows / nccl::G_MPI_SIZE;
        const size_t elem_bytes = base::DataTypeSize(weight.data_type());
        const size_t offset_bytes =
            static_cast<size_t>(rows_per_rank) * static_cast<size_t>(cols) * elem_bytes *
            static_cast<size_t>(nccl::G_MPI_RANK);
        void* offset_ptr =
            static_cast<char*>(const_cast<void*>(weight.get_buffer()->ptr())) + offset_bytes;

        std::vector<int32_t> dims = {rows_per_rank, cols};
        tensor::Tensor weight_view(std::move(dims), weight.getDeviceId(), weight.data_type(), offset_ptr);

        return MatmulLayer::set_weight(idx, weight_view);
    }

    base::Status RowMatmulLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                            const void* weight_ptr, base::deviceId device_id,
                                            base::DataType weight_data_type)
    {
        CHECK_EQ(dims.size(), 2u);
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        const int32_t rows = dims[0];
        const int32_t cols = dims[1];
        CHECK_EQ(rows % nccl::G_MPI_SIZE, 0)
            << "Weight rows must be divisible by world size";

        const int32_t rows_per_rank = rows / nccl::G_MPI_SIZE;
        const size_t elem_bytes = base::DataTypeSize(weight_data_type);
        const size_t offset_bytes =
            static_cast<size_t>(rows_per_rank) * static_cast<size_t>(cols) * elem_bytes *
            static_cast<size_t>(nccl::G_MPI_RANK);
        const void* offset_ptr = static_cast<const char*>(weight_ptr) + offset_bytes;

        std::vector<int32_t> local_dims = {rows_per_rank, cols};
        return MatmulLayer::set_weight(idx, local_dims, offset_ptr, device_id, weight_data_type);
    }

}  // namespace op
