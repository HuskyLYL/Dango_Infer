#include "op/colom_parallel_matmul.h"
#include "nccl/base.h"
#include "nccl/collective.h"
#include "base/base.h"

namespace op
{

    ColomParallelMatmulLayer::ColomParallelMatmulLayer(bool has_all_reduce_flag, bool has_bias)
        : MatmulLayer(has_bias), has_all_reduce(has_all_reduce_flag) {}

    base::Status ColomParallelMatmulLayer::forward(const tensor::Tensor& input1,
                                                const tensor::Tensor& output1,
                                                cudaStream_t stream)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);

        const size_t total_in = input1.size();
        CHECK_EQ(total_in % static_cast<size_t>(nccl::G_MPI_SIZE), 0u)
            << "ColomParallelMatmul input size must be divisible by world size";

        const size_t per_rank_in = total_in / static_cast<size_t>(nccl::G_MPI_SIZE);
        const size_t elem_bytes = base::DataTypeSize(input1.data_type());
        const size_t offset_bytes =
            per_rank_in * elem_bytes * static_cast<size_t>(nccl::G_MPI_RANK);

        void* offset_ptr =
            static_cast<char*>(const_cast<void*>(input1.get_buffer()->ptr())) + offset_bytes;
        tensor::Tensor input_view(static_cast<int32_t>(per_rank_in),
                                input1.getDeviceId(),
                                input1.data_type(),
                                offset_ptr);

        this->set_input(0, input_view);
        this->set_output(0, output1);

        base::Status status = MatmulLayer::forward(stream);
        if (!status)
            return status;

        if (has_all_reduce)
            nccl::TensorAllReduce(output1);

        return base::error::Success();
    }

    base::Status ColomParallelMatmulLayer::set_bias(int32_t idx, int32_t& dims,
                                                    const void* bias_ptr, base::deviceId device_id)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        CHECK_NE(bias_ptr, nullptr);

        // Scale bias by world size so that post all-reduce the sum equals the original bias.
        const float scale = 1.0f / static_cast<float>(nccl::G_MPI_SIZE);
        std::vector<float> scaled_bias(dims);
        const float* src = static_cast<const float*>(bias_ptr);
        for (int i = 0; i < dims; ++i)
            scaled_bias[i] = src[i] * scale;

        // Store scaled bias on device via base setter.
        return MatmulLayer::set_bias(idx, dims, scaled_bias.data(), device_id);
    }

    base::Status ColomParallelMatmulLayer::set_weight(int32_t idx, const tensor::Tensor& weight)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        const int32_t rows = weight.get_dim(0);
        const int32_t cols = weight.get_dim(1);
            CHECK_EQ(cols % nccl::G_MPI_SIZE, 0)
                << "Weight cols must be divisible by world size";

        const int32_t cols_per_rank = cols / nccl::G_MPI_SIZE;
        const size_t elem_bytes = base::DataTypeSize(weight.data_type());
        const int32_t cols_offset = cols_per_rank * nccl::G_MPI_RANK;

        // Allocate a contiguous buffer for this rank's column slice.
        tensor::Tensor local_weight({rows, cols_per_rank}, weight.getDeviceId(),
                                    weight.data_type(), nullptr);

        void* dst_ptr = const_cast<void*>(local_weight.get_buffer()->ptr());
        const void* src_ptr = weight.get_buffer()->ptr();

        const size_t src_pitch = static_cast<size_t>(cols) * elem_bytes;
        const size_t dst_pitch = static_cast<size_t>(cols_per_rank) * elem_bytes;

        const size_t width = static_cast<size_t>(cols_per_rank) * elem_bytes;
        const size_t height = static_cast<size_t>(rows);

        if (weight.getDeviceId() == base::CPUID)
        {
            // Host copy row by row.
            const char* src = static_cast<const char*>(src_ptr) + cols_offset * elem_bytes;
            char* dst = static_cast<char*>(dst_ptr);
            for (int32_t r = 0; r < rows; ++r)
            {
                memcpy(dst, src, width);
                src += src_pitch;
                dst += dst_pitch;
            }
        }
        else
        {
            const char* src = static_cast<const char*>(src_ptr) + cols_offset * elem_bytes;
            CUDACHECK(cudaMemcpy2D(dst_ptr, dst_pitch, src, src_pitch, width, height,
                                cudaMemcpyDeviceToDevice));
        }

        return MatmulLayer::set_weight(idx, local_weight);
    }

    base::Status ColomParallelMatmulLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                                  const void* weight_ptr, base::deviceId device_id,
                                                  base::DataType weight_data_type)
    {
        CHECK_EQ(dims.size(), 2u);
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        const int32_t rows = dims[0];
        const int32_t cols = dims[1];
        CHECK_EQ(cols % nccl::G_MPI_SIZE, 0)
            << "Weight cols must be divisible by world size";

        const int32_t cols_per_rank = cols / nccl::G_MPI_SIZE;
        const size_t elem_bytes = base::DataTypeSize(weight_data_type);
        const int32_t cols_offset = cols_per_rank * nccl::G_MPI_RANK;

        tensor::Tensor local_weight({rows, cols_per_rank}, device_id, weight_data_type, nullptr);

        void* dst_ptr = const_cast<void*>(local_weight.get_buffer()->ptr());
        const void* src_ptr = weight_ptr;
        const size_t src_pitch = static_cast<size_t>(cols) * elem_bytes;
        const size_t dst_pitch = static_cast<size_t>(cols_per_rank) * elem_bytes;
        const size_t width = static_cast<size_t>(cols_per_rank) * elem_bytes;
        const size_t height = static_cast<size_t>(rows);

        if (device_id == base::CPUID)
        {
            const char* src = static_cast<const char*>(src_ptr) + cols_offset * elem_bytes;
            char* dst = static_cast<char*>(dst_ptr);
            for (int32_t r = 0; r < rows; ++r)
            {
                memcpy(dst, src, width);
                src += src_pitch;
                dst += dst_pitch;
            }
        }
        else
        {
            const char* src = static_cast<const char*>(src_ptr) + cols_offset * elem_bytes;
            CUDACHECK(cudaMemcpy2D(dst_ptr, dst_pitch, src, src_pitch, width, height,
                                cudaMemcpyDeviceToDevice));
        }


        return MatmulLayer::set_weight(idx, local_weight);
    }

}  // namespace op
