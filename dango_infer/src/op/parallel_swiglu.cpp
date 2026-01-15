#include "op/parallel_swiglu.h"
#include "nccl/base.h"
#include "nccl/collective.h"
#include "base/base.h"

namespace op
{
    ParallelSwiGLULayer::ParallelSwiGLULayer()
        : SwiGLULayer()
    {
        this->set_layer_name("ParallelSwiGLU");
    }

    base::Status ParallelSwiGLULayer::check() const
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);
        return SwiGLULayer::check();
    }

    base::Status ParallelSwiGLULayer::forward(cudaStream_t stream)
    {
        return SwiGLULayer::forward(stream);
    }


    base::Status ParallelSwiGLULayer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
        const tensor::Tensor& output1, cudaStream_t stream)
    {
        CHECK_GT(nccl::G_MPI_SIZE, 0);

        // Slice input and output evenly across ranks.
        const size_t total_elems = input1.size();
        CHECK_EQ(total_elems % static_cast<size_t>(nccl::G_MPI_SIZE), 0u)
            << "ParallelSwiGLU input size must be divisible by world size";
        CHECK_EQ(output1.size(), input1.size());
        CHECK_EQ(input2.size(), input1.size());

        const size_t per_rank = total_elems / static_cast<size_t>(nccl::G_MPI_SIZE);
        const size_t elem_bytes = base::DataTypeSize(input1.data_type());
        const size_t offset_bytes = per_rank * elem_bytes * static_cast<size_t>(nccl::G_MPI_RANK);

        void* in1_offset_ptr = static_cast<char*>(const_cast<void*>(input1.get_buffer()->ptr())) + offset_bytes;
        tensor::Tensor local_in1(static_cast<int32_t>(per_rank),
            input1.getDeviceId(), input1.data_type(), in1_offset_ptr);

        void* in2_offset_ptr = static_cast<char*>(const_cast<void*>(input2.get_buffer()->ptr())) + offset_bytes;
        tensor::Tensor local_in2(static_cast<int32_t>(per_rank),
            input2.getDeviceId(), input2.data_type(), in2_offset_ptr);

        void* out_offset_ptr = static_cast<char*>(const_cast<void*>(output1.get_buffer()->ptr())) + offset_bytes;
        tensor::Tensor local_out(static_cast<int32_t>(per_rank),
            output1.getDeviceId(), output1.data_type(), out_offset_ptr);

        this->set_input(0, local_in1);
        this->set_input(1, local_in2);
        this->set_output(0, local_out);

        auto status = SwiGLULayer::forward(stream);
        if (!status)
            return status;

        nccl::TensorAllGather(output1);


        return base::error::Success();
    }
}  // namespace op
